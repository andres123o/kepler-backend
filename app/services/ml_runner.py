"""
Servicio ML: lee datos de Supabase y corre el mismo pipeline que los comandos del domingo.

  run_prediction()           → equivale a: python scripts/prediccion_ultima_semana.py --csv ...
  run_training()             → equivale a: python -m ml_pipeline.train --csv ...
  get_training_status()      → estado del último modelo en models/
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ml_pipeline.config import BENCHMARK_WINDOW_WEEKS
from ml_pipeline.data_contracts import validate_data_contracts
from ml_pipeline.feature_engineering import build_feature_matrix, prepare_base_df
from ml_pipeline.funnel_config import load_funnel_ml_config
from ml_pipeline.train import run_training as _pipeline_train

from datetime import date, timedelta

from app.services.supabase_client import FunnelClient, _default_fc

logger = logging.getLogger("kepler.ml_runner")

LOCAL_MODELS_ROOT = _BACKEND / "models"

# Caché en memoria: evita re-descargar el modelo en cada request dentro del mismo proceso
_MODEL_CACHE: dict[str, tuple[xgb.Booster, dict, float | None]] = {}

_MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def _semana_label(semana_datos: str) -> str:
    """
    Genera 'X al Y de mes AAAA' a partir de semana_datos (YYYY-MM-DD o DD/MM/YYYY).
    Asume semanas de lunes a domingo (7 días).
    """
    try:
        if "/" in semana_datos:
            parts = semana_datos.split("/")
            inicio = date(int(parts[2]), int(parts[1]), int(parts[0]))
        else:
            inicio = date.fromisoformat(semana_datos[:10])
        fin = inicio + timedelta(days=6)
        if inicio.month == fin.month:
            return f"{inicio.day} al {fin.day} de {_MESES_ES[inicio.month]} {inicio.year}"
        return (
            f"{inicio.day} de {_MESES_ES[inicio.month]}"
            f" al {fin.day} de {_MESES_ES[fin.month]} {inicio.year}"
        )
    except Exception:
        return semana_datos


# ─────────────────────────────────────────────
# Modelo local
# ─────────────────────────────────────────────

def _load_model_from_storage(fc: FunnelClient, ml_cfg: dict) -> tuple[xgb.Booster, dict, float | None]:
    """
    Descarga model.json + model_meta.json + training_summary.json desde Supabase Storage
    al directorio temporal, los carga en memoria y cachea por (org/funnel/version).
    ml_cfg debe tener 'model_storage' (bucket) y 'model_version' (int).
    """
    bucket  = ml_cfg["model_storage"]
    version = int(ml_cfg["model_version"])
    prefix  = f"{fc.org_slug}/{fc.funnel_slug}/v{version}"
    cache_key = prefix

    if cache_key in _MODEL_CACHE:
        logger.info("Modelo desde caché: %s", cache_key)
        return _MODEL_CACHE[cache_key]

    logger.info("Descargando modelo desde Storage: bucket=%s path=%s", bucket, prefix)
    client = fc._client

    with tempfile.TemporaryDirectory() as tmpdir:
        vdir = Path(tmpdir) / f"v{version}"
        vdir.mkdir()
        for fname in ("model.json", "model_meta.json", "training_summary.json"):
            try:
                data = client.storage.from_(bucket).download(f"{prefix}/{fname}")
                (vdir / fname).write_bytes(data)
            except Exception as exc:
                if fname == "training_summary.json":
                    logger.warning("training_summary.json no encontrado en Storage — sin MAE")
                else:
                    raise RuntimeError(
                        f"No se pudo descargar {prefix}/{fname} del bucket '{bucket}': {exc}"
                    ) from exc
        result = _load_latest_model(Path(tmpdir))

    _MODEL_CACHE[cache_key] = result
    return result


def _upload_model_to_storage(fc: FunnelClient, bucket: str, model_dir: Path, version: int) -> None:
    """Sube los 3 archivos del modelo al bucket y actualiza model_version en el config."""
    prefix = f"{fc.org_slug}/{fc.funnel_slug}/v{version}"
    client = fc._client
    for fname in ("model.json", "model_meta.json", "training_summary.json"):
        fpath = model_dir / fname
        if not fpath.exists():
            continue
        path_in_bucket = f"{prefix}/{fname}"
        try:
            client.storage.from_(bucket).remove([path_in_bucket])
        except Exception:
            pass
        client.storage.from_(bucket).upload(path_in_bucket, fpath.read_bytes())
        logger.info("Subido a Storage: %s/%s", bucket, path_in_bucket)
    fc.update_ml_version(version)
    logger.info("Config actualizado: ml.model_version=%d", version)


def _load_latest_model(model_root: Path | None = None) -> tuple[xgb.Booster, dict, float | None]:
    """Carga el modelo más reciente de {model_root}/v{N}/ (por defecto models/)."""
    root = model_root or LOCAL_MODELS_ROOT
    if not root.exists():
        raise ValueError(f"No hay modelos en {root}. Ejecuta el entrenamiento primero.")

    dirs = sorted(
        [d for d in root.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    if not dirs:
        raise ValueError(f"No hay versiones de modelo en {root}.")

    latest = dirs[-1]
    booster = xgb.Booster()
    booster.load_model(str(latest / "model.json"))
    meta = json.loads((latest / "model_meta.json").read_text(encoding="utf-8"))

    mae_wf = None
    summary_path = latest / "training_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        mae_wf = float(s.get("mae_walk_forward") or 0) or None

    logger.info("Modelo cargado: %s | MAE walk-forward: %s", latest.name, mae_wf)
    return booster, meta, mae_wf


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _feature_action_type(feat: str, feature_action_map: dict[str, str]) -> str:
    """
    Resuelve el tipo de acción para una feature SHAP.
    feature_action_map viene del config JSONB del funnel (ml.feature_action_map).
    Soporta exact match, pattern match con * (ej. "*dias_habiles*"), y fallback "investigar".
    """
    if feat in feature_action_map:
        return feature_action_map[feat]
    for pattern, action_type in feature_action_map.items():
        if "*" in pattern:
            prefix, _, suffix = pattern.partition("*")
            if feat.startswith(prefix) and feat.endswith(suffix):
                return action_type
    return "investigar"


# ─────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────

def run_prediction(fc: FunnelClient | None = None) -> dict[str, Any]:
    """
    Lee master + ultima_semana desde Supabase, corre el pipeline completo.
    fc: FunnelClient del request. Si None, usa KEPLER_DEFAULT_ORG/FUNNEL del .env.
    """
    if fc is None:
        fc = _default_fc()
    logger.info("=== INICIO Predicción [%s/%s] ===", fc.org_slug, fc.funnel_slug)

    # 1. Leer datos de Supabase
    df_hist = fc.get_master_df()
    df_ultima = fc.get_ultima_semana_df()

    if df_hist.empty:
        raise ValueError("La tabla master está vacía. Verifica que haya datos históricos.")
    if df_ultima.empty:
        raise ValueError("ultima_semana está vacía. Ingresa los datos de la semana primero.")

    date_col = "semana" if "semana" in df_hist.columns else "fecha_inicio"
    row_ultima = df_ultima.iloc[0]

    # 2. Detectar fecha de la semana desde ultima_semana
    excel_date_col = next(
        (c for c in ("semana", "fecha_inicio", "fecha") if c in df_ultima.columns), None
    )
    if excel_date_col and not pd.isna(row_ultima.get(excel_date_col)):
        raw_fecha = str(row_ultima[excel_date_col]).strip()
        if "/" in raw_fecha:
            parts = raw_fecha.split("/")
            if len(parts) == 3:
                try:
                    d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                    semana_str = f"{y:04d}-{m:02d}-{d:02d}"
                except (ValueError, TypeError):
                    semana_str = raw_fecha
            else:
                semana_str = raw_fecha
        else:
            semana_str = raw_fecha[:10] if len(raw_fecha) >= 10 else raw_fecha
    else:
        # Fallback: siguiente lunes al último del historial
        df_tmp = df_hist.copy()
        df_tmp["_d"] = pd.to_datetime(df_tmp[date_col], dayfirst=True, errors="coerce")
        last_date = df_tmp["_d"].max()
        semana_str = (last_date + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")

    logger.info("Semana a predecir: %s", semana_str)

    # Determinar fuente del modelo desde config del funnel
    funnel_cfg = fc.get_funnel_config()
    ml_cfg     = funnel_cfg.get("ml") or {}
    funnel_ml_cfg = load_funnel_ml_config(ml_cfg)

    # 3. Construir nueva fila alineada con columnas del historial
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    ultima_by_norm = {_norm(c): c for c in df_ultima.columns}
    new_row: dict[str, Any] = {date_col: semana_str}

    for col in df_hist.columns:
        if col == date_col:
            continue
        val = None
        if col in df_ultima.columns:
            val = row_ultima[col]
        else:
            key = _norm(col)
            if key in ultima_by_norm:
                val = row_ultima[ultima_by_norm[key]]

        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            try:
                new_row[col] = float(val)
            except (TypeError, ValueError):
                new_row[col] = np.nan
        else:
            new_row[col] = np.nan

    # Advertencia si full_users_aprobados no tiene dato
    if pd.isna(new_row.get("full_users_aprobados", float("nan"))):
        logger.warning("full_users_aprobados es NaN → full_users_aprobados_lag1 será NaN.")

    # 4. Validar contratos de datos
    df_new = pd.DataFrame([new_row])
    ok, violations = validate_data_contracts(df_new, funnel_ml_cfg.non_negative_columns, funnel_ml_cfg.target_name)
    if not ok:
        logger.warning("Contratos fallidos en ultima_semana: %s", violations)

    # 5. Combinar historial + nueva fila y correr pipeline
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_prepared = prepare_base_df(df_combined)
    X_full, y_full, feature_names = build_feature_matrix(df_prepared, funnel_ml_cfg, drop_na=False)

    if len(X_full) == 0:
        raise ValueError("No quedaron filas tras el pipeline de features.")

    # Referencia al DataFrame crudo (pre-FE) — contiene features nativas del funnel
    # que build_feature_matrix() no incluye en X si no están listadas en ml.funnel_features/macro_features.
    # Se usa como fallback para recuperar current_value y trailing_12w_mean en el contexto SHAP.
    _df_raw_hist = df_prepared.iloc[:-1]   # historial sin la fila de predicción
    _raw_last    = df_prepared.iloc[-1]    # fila de predicción actual (valores del usuario)

    # 6. Cargar modelo (Storage si existe model_storage, local como fallback)
    if ml_cfg.get("model_storage"):
        booster, meta, mae_modelo = _load_model_from_storage(fc, ml_cfg)
    else:
        model_dir_rel = ml_cfg.get("model_dir")
        model_root    = (_BACKEND / model_dir_rel) if model_dir_rel else LOCAL_MODELS_ROOT
        booster, meta, mae_modelo = _load_latest_model(model_root)
    model_features = meta.get("feature_names", [])
    target_name = meta.get("target_name", funnel_ml_cfg.target_name)
    version = meta.get("version", "?")

    # Extender X_full con features nativas del funnel que están en df_prepared
    # pero que build_feature_matrix() excluye (no listadas en ml.funnel_features/macro_features).
    # Red de seguridad ante drift entre el config actual y meta.json del modelo entrenado.
    missing_in_X = [f for f in model_features if f not in X_full.columns and f in df_prepared.columns]
    if missing_in_X:
        logger.info("Añadiendo %d features del funnel a X_full desde df_prepared: %s", len(missing_in_X), missing_in_X)
        X_full = X_full.copy()
        for col in missing_in_X:
            X_full[col] = df_prepared[col].values

    last_row = X_full.iloc[-1:].copy()

    # Alinear features restantes con NaN (features que no están en df_prepared ni en X_full)
    for col in model_features:
        if col not in last_row.columns:
            last_row[col] = np.nan
    last_row = last_row[model_features].copy()

    # 7. Predicción
    dmat = xgb.DMatrix(last_row.values.astype(np.float64), feature_names=model_features)
    prediccion = float(booster.predict(dmat)[0])
    logger.info("Predicción: %.0f %s", prediccion, target_name)

    # 8. Baseline 12 semanas
    target_hist = y_full.iloc[:-1].dropna()
    baseline_12w = float(target_hist.tail(BENCHMARK_WINDOW_WEEKS).mean()) if not target_hist.empty else 0.0
    brecha = round(prediccion - baseline_12w, 0)

    # 9. SHAP
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmat)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    contribs = shap_values[0]

    def _safe_contrib(v) -> float | None:
        """float() seguro — devuelve None si NaN/Inf (SHAP sobre inputs NaN)."""
        try:
            f = float(v)
            return None if (f != f or f == float("inf") or f == float("-inf")) else f
        except (TypeError, ValueError):
            return None

    def _get_val(name: str) -> float | None:
        """Valor actual de la feature — primero X_full, fallback a df_prepared crudo."""
        v = last_row[name].iloc[0] if name in last_row.columns else np.nan
        if pd.isna(v) and name in _raw_last.index and not pd.isna(_raw_last[name]):
            v = _raw_last[name]
        return None if pd.isna(v) else float(v)

    shap_list = sorted(
        [
            {
                "feature": name,
                "value": _get_val(name),
                "contribution": _safe_contrib(contribs[i]) or 0.0,
            }
            for i, name in enumerate(model_features)
        ],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )

    # Filtrar features no accionables del display (funnel interno + autoregresivo)
    # El modelo las usa para predecir pero Marketing no puede actuar sobre ellas vía CIO.
    # Lista viene de ml.non_actionable_features en el config del funnel.
    shap_display = [s for s in shap_list if s["feature"] not in funnel_ml_cfg.non_actionable_features]

    # 10. Contexto histórico (z-scores vs. últimas 12 semanas)
    X_hist_ctx = X_full.iloc[:-1]
    contexto = []
    for s in shap_display[:20]:
        f = s["feature"]
        val = s["value"]
        if f in X_hist_ctx.columns:
            hist = X_hist_ctx[f].dropna().tail(BENCHMARK_WINDOW_WEEKS)
        elif f in _df_raw_hist.columns:
            # Feature nativa del funnel no incluida en X por build_feature_matrix
            # (ej. features PE como bvl_var_semanal, pen_usd_var_semanal)
            hist = _df_raw_hist[f].dropna().tail(BENCHMARK_WINDOW_WEEKS)
        else:
            hist = pd.Series([], dtype=float)
        if not hist.empty:
            mean_12w = float(hist.mean())
            std_12w = float(hist.std()) if len(hist) > 1 else 0.0
        else:
            mean_12w = std_12w = 0.0
        delta = val - mean_12w if val is not None else None
        z = float(delta / std_12w) if (delta is not None and std_12w > 0) else None
        contexto.append({
            "feature": f,
            "current_value": round(val, 4) if val is not None else None,
            "trailing_12w_mean": round(mean_12w, 4),
            "z_score": round(z, 2) if z is not None else None,
            "shap_contribution": round(s["contribution"], 4),
        })

    # 11. Prescripción — evalúa todos los features del contexto, no solo el top 5
    # Mapa feature→tipo_accion viene de ml.feature_action_map en el config del funnel
    feature_action_map: dict[str, str] = ml_cfg.get("feature_action_map") or {}

    prescripcion = []
    for item in contexto:
        feat = item["feature"]
        z = item["z_score"]
        contrib = item["shap_contribution"]
        if z is None or abs(z) < 0.5:
            continue
        accion: dict[str, Any] = {
            "variable": feat,
            "z_score": z,
            "contribucion_depositos": round(contrib, 0),
            "direccion": "negativa" if contrib < 0 else "positiva",
            "severidad": "crítica" if abs(z) > 2.0 else "alerta" if abs(z) > 1.5 else "monitorear",
        }
        if feat in funnel_ml_cfg.non_actionable_features:
            accion["tipo_accion"] = "monitorear_no_accionable"
        else:
            # Buscar en el mapa de acción del funnel (feature_action_map en config ML)
            accion["tipo_accion"] = _feature_action_type(feat, feature_action_map)
        prescripcion.append(accion)

    # Ordenar por z-score absoluto desc para que las más anómalas lleguen primero al slice del frontend
    prescripcion.sort(key=lambda x: abs(x["z_score"]), reverse=True)

    logger.info("=== FIN Predicción: %.0f | brecha: %.0f ===", prediccion, brecha)

    result = {
        "semana_datos": semana_str,
        "prediccion_siguiente_semana": round(prediccion, 0),
        "target_name": target_name,
        "modelo_version": f"v{version}",
        "mae_modelo": round(mae_modelo, 0) if mae_modelo else None,
        "baseline_12w": round(baseline_12w, 0),
        "brecha_vs_baseline": brecha,
        "shap_top": shap_display[:20],
        "contexto_historico_top_features": contexto,
        "prescripcion": prescripcion,
    }

    predicted_start = (date.fromisoformat(semana_str[:10]) + timedelta(weeks=1)).strftime("%Y-%m-%d")
    label = _semana_label(predicted_start)
    result["semana_label"] = label

    try:
        fc.save_prediction_result(result, semana_label=label)
    except Exception as exc:
        logger.error("❌ FALLO AL GUARDAR prediction_results: %s", exc, exc_info=True)

    return result


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def run_training(fc: FunnelClient | None = None) -> dict[str, Any]:
    """
    Descarga master desde Supabase → CSV temporal → entrena el pipeline.
    fc: FunnelClient del request. Si None, usa KEPLER_DEFAULT_ORG/FUNNEL del .env.
    """
    if fc is None:
        fc = _default_fc()
    logger.info("=== INICIO Entrenamiento [%s/%s] ===", fc.org_slug, fc.funnel_slug)

    df = fc.get_master_df()
    if df.empty:
        raise ValueError("La tabla master está vacía. No se puede entrenar.")

    # Escribir a CSV temporal con el formato que espera el pipeline (delimitador ;)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        df.to_csv(f, sep=";", index=False)
        tmp_path = Path(f.name)

    funnel_cfg = fc.get_funnel_config()
    ml_cfg     = funnel_cfg.get("ml") or {}
    funnel_ml_cfg = load_funnel_ml_config(ml_cfg)

    use_storage = bool(ml_cfg.get("model_storage"))

    # Si usa Storage, entrena en directorio temporal y luego sube
    if use_storage:
        train_output_dir = Path(tempfile.mkdtemp(prefix="kepler_model_"))
    else:
        model_dir_rel    = ml_cfg.get("model_dir")
        train_output_dir = (_BACKEND / model_dir_rel) if model_dir_rel else None

    try:
        summary = _pipeline_train(tmp_path, funnel_ml_cfg, output_dir=train_output_dir)
    finally:
        tmp_path.unlink(missing_ok=True)

    if use_storage:
        bucket  = ml_cfg["model_storage"]
        version = summary["version"]
        model_vdir = train_output_dir / f"v{version}"
        _upload_model_to_storage(fc, bucket, model_vdir, version)
        # Invalidar caché del modelo anterior
        _MODEL_CACHE.pop(f"{fc.org_slug}/{fc.funnel_slug}/v{version - 1}", None)
        import shutil; shutil.rmtree(train_output_dir, ignore_errors=True)

    logger.info("=== FIN Entrenamiento: v%s | MAE_wf=%.2f ===",
                summary.get("version"), summary.get("mae_walk_forward", 0))
    return summary


# ─────────────────────────────────────────────
# Estado del modelo
# ─────────────────────────────────────────────

def get_training_status(fc: FunnelClient | None = None) -> dict[str, Any]:
    """Estado del último modelo. Lee desde Supabase Storage si model_storage está en config."""
    if fc is None:
        model_root = LOCAL_MODELS_ROOT
        return _training_status_from_local(model_root)

    cfg    = fc.get_funnel_config()
    ml_cfg = cfg.get("ml") or {}

    if ml_cfg.get("model_storage"):
        bucket  = ml_cfg["model_storage"]
        version = ml_cfg.get("model_version")
        if not version:
            return {"has_model": False, "message": "model_version no está en el config del funnel."}
        version = int(version)
        prefix  = f"{fc.org_slug}/{fc.funnel_slug}/v{version}/training_summary.json"
        try:
            data = fc._client.storage.from_(bucket).download(prefix)
            s    = json.loads(data.decode("utf-8"))
            return {
                "has_model":        True,
                "version":          version,
                "source":           "storage",
                "mae_walk_forward": s.get("mae_walk_forward"),
                "r2_train":         s.get("r2_train"),
                "mae_train":        s.get("mae_train"),
                "ratio_wf_train":   s.get("ratio_wf_train"),
                "overfitting_flag": s.get("overfitting_flag"),
                "n_samples":        s.get("n_samples"),
                "n_features":       s.get("n_features"),
                "best_params":      s.get("best_params"),
                "feature_importance": s.get("feature_importance"),
            }
        except Exception as exc:
            return {"has_model": False, "message": f"Error leyendo Storage: {exc}"}

    model_dir_str = ml_cfg.get("model_dir")
    model_root    = (_BACKEND / model_dir_str) if model_dir_str else LOCAL_MODELS_ROOT
    return _training_status_from_local(model_root)


def _training_status_from_local(model_root: Path) -> dict[str, Any]:
    if not model_root.exists():
        return {"has_model": False, "message": "No hay modelos entrenados."}
    dirs = sorted(
        [d for d in model_root.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    if not dirs:
        return {"has_model": False, "message": f"No hay versiones en {model_root}."}
    latest = dirs[-1]
    out: dict[str, Any] = {"has_model": True, "version": int(latest.name[1:]), "source": "local"}
    summary_path = latest / "training_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        out.update({
            "mae_walk_forward": s.get("mae_walk_forward"),
            "r2_train":         s.get("r2_train"),
            "mae_train":        s.get("mae_train"),
            "ratio_wf_train":   s.get("ratio_wf_train"),
            "overfitting_flag": s.get("overfitting_flag"),
            "n_samples":        s.get("n_samples"),
            "n_features":       s.get("n_features"),
            "best_params":      s.get("best_params"),
            "feature_importance": s.get("feature_importance"),
        })
    return out
