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

from ml_pipeline.config import BENCHMARK_WINDOW_WEEKS, TARGET_NAME
from ml_pipeline.data_contracts import validate_data_contracts
from ml_pipeline.feature_engineering import build_feature_matrix, prepare_base_df
from ml_pipeline.train import run_training as _pipeline_train

from datetime import date, timedelta

from app.services.supabase_client import get_master_df, get_ultima_semana_df, save_prediction_result

logger = logging.getLogger("kepler.ml_runner")

LOCAL_MODELS_ROOT = _BACKEND / "models"

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

def _load_latest_model() -> tuple[xgb.Booster, dict, float | None]:
    """Carga el modelo más reciente de models/v{N}/."""
    if not LOCAL_MODELS_ROOT.exists():
        raise ValueError("No hay modelos entrenados. Ejecuta el entrenamiento primero.")

    dirs = sorted(
        [d for d in LOCAL_MODELS_ROOT.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    if not dirs:
        raise ValueError("No hay versiones de modelo en models/.")

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
# Predicción
# ─────────────────────────────────────────────

def run_prediction() -> dict[str, Any]:
    """
    Lee master_consolidado_final + ultima_semana desde Supabase,
    combina ambos, corre el pipeline de features, predice y calcula SHAP.
    Devuelve el mismo JSON que producía prediccion_ultima_semana.py.
    """
    logger.info("=== INICIO Predicción ===")

    # 1. Leer datos de Supabase
    df_hist = get_master_df()
    df_ultima = get_ultima_semana_df()

    if df_hist.empty:
        raise ValueError("master_consolidado_final está vacía.")
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
    ok, violations = validate_data_contracts(df_new)
    if not ok:
        logger.warning("Contratos fallidos en ultima_semana: %s", violations)

    # 5. Combinar historial + nueva fila y correr pipeline
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_prepared = prepare_base_df(df_combined)
    X_full, y_full, feature_names = build_feature_matrix(df_prepared, drop_na=False)

    if len(X_full) == 0:
        raise ValueError("No quedaron filas tras el pipeline de features.")

    last_row = X_full.iloc[-1:].copy()

    # 6. Cargar modelo
    booster, meta, mae_modelo = _load_latest_model()
    model_features = meta.get("feature_names", [])
    target_name = meta.get("target_name", TARGET_NAME)
    version = meta.get("version", "?")

    # Alinear features (rellenar faltantes con NaN, nunca con 0)
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

    shap_list = sorted(
        [
            {
                "feature": name,
                "value": None if pd.isna(last_row[name].iloc[0]) else float(last_row[name].iloc[0]),
                "contribution": float(contribs[i]),
            }
            for i, name in enumerate(model_features)
        ],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )

    # 10. Contexto histórico (z-scores vs. últimas 12 semanas)
    X_hist_ctx = X_full.iloc[:-1]
    contexto = []
    for s in shap_list[:20]:
        f = s["feature"]
        val = s["value"]
        if f in X_hist_ctx.columns:
            hist = X_hist_ctx[f].dropna().tail(BENCHMARK_WINDOW_WEEKS)
            mean_12w = float(hist.mean()) if not hist.empty else 0.0
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

    # 11. Prescripción
    prescripcion = []
    for item in contexto[:5]:
        feat = item["feature"]
        z = item["z_score"]
        contrib = item["shap_contribution"]
        if z is None or abs(z) < 1.0:
            continue
        accion: dict[str, Any] = {
            "variable": feat,
            "z_score": z,
            "contribucion_depositos": round(contrib, 0),
            "direccion": "negativa" if contrib < 0 else "positiva",
            "severidad": "crítica" if abs(z) > 2.0 else "alerta" if abs(z) > 1.5 else "monitorear",
        }
        if "push_mail" in feat:
            accion["tipo_accion"] = "revisar_campanas_customer_io"
        elif "tasa_" in feat and "kyc" not in feat:
            accion["tipo_accion"] = "campana_ayuda_onboarding"
        elif "rechazo" in feat or "kyc" in feat or "cx_friccion" in feat:
            accion["tipo_accion"] = "alerta_tech_soporte"
        elif "cx_bloqueos" in feat:
            accion["tipo_accion"] = "alerta_soporte"
        elif "registro" in feat or "aprobados" in feat:
            accion["tipo_accion"] = "monitorear_adquisicion"
        elif feat in ("TRM", "Variacion_COLCAP", "Tasa_Intervencion_Mensual"):
            accion["tipo_accion"] = "contexto_macro"
        elif "dias_habiles" in feat or "semana_del_mes" in feat or "mes_prima" in feat:
            accion["tipo_accion"] = "estacionalidad"
        else:
            accion["tipo_accion"] = "investigar"
        prescripcion.append(accion)

    logger.info("=== FIN Predicción: %.0f | brecha: %.0f ===", prediccion, brecha)

    result = {
        "semana_datos": semana_str,
        "prediccion_siguiente_semana": round(prediccion, 0),
        "target_name": target_name,
        "modelo_version": f"v{version}",
        "mae_modelo": round(mae_modelo, 0) if mae_modelo else None,
        "baseline_12w": round(baseline_12w, 0),
        "brecha_vs_baseline": brecha,
        "shap_top": shap_list[:20],
        "contexto_historico_top_features": contexto,
        "prescripcion": prescripcion,
    }

    predicted_start = (date.fromisoformat(semana_str[:10]) + timedelta(weeks=1)).strftime("%Y-%m-%d")
    label = _semana_label(predicted_start)
    result["semana_label"] = label

    try:
        save_prediction_result(result, semana_label=label)
    except Exception as exc:
        logger.warning("No se pudo guardar en prediction_results: %s", exc)

    return result


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def run_training() -> dict[str, Any]:
    """
    Descarga master_consolidado_final de Supabase → CSV temporal → entrena el pipeline.
    Equivale a: python -m ml_pipeline.train --csv Master_Consolidado_Final.csv
    """
    logger.info("=== INICIO Entrenamiento desde Supabase ===")

    df = get_master_df()
    if df.empty:
        raise ValueError("master_consolidado_final está vacía. No se puede entrenar.")

    # Escribir a CSV temporal con el formato que espera el pipeline (delimitador ;)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as f:
        df.to_csv(f, sep=";", index=False)
        tmp_path = Path(f.name)

    try:
        summary = _pipeline_train(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("=== FIN Entrenamiento: v%s | MAE_wf=%.2f ===",
                summary.get("version"), summary.get("mae_walk_forward", 0))
    return summary


# ─────────────────────────────────────────────
# Estado del modelo
# ─────────────────────────────────────────────

def get_training_status() -> dict[str, Any]:
    """Estado del último modelo entrenado localmente en models/."""
    if not LOCAL_MODELS_ROOT.exists():
        return {"has_model": False, "message": "No hay modelos entrenados."}

    dirs = sorted(
        [d for d in LOCAL_MODELS_ROOT.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    if not dirs:
        return {"has_model": False, "message": "No hay versiones en models/."}

    latest = dirs[-1]
    out: dict[str, Any] = {"has_model": True, "version": int(latest.name[1:])}

    summary_path = latest / "training_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        out.update({
            "mae_walk_forward": s.get("mae_walk_forward"),
            "r2_train": s.get("r2_train"),
            "mae_train": s.get("mae_train"),
            "ratio_wf_train": s.get("ratio_wf_train"),
            "overfitting_flag": s.get("overfitting_flag"),
            "n_samples": s.get("n_samples"),
            "n_features": s.get("n_features"),
            "best_params": s.get("best_params"),
            "feature_importance": s.get("feature_importance"),
        })

    return out
