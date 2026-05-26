"""
Predicción + SHAP para la siguiente semana — todo local, sin Supabase.

Inputs:
  1. Master_Consolidado_Final.csv  — historial completo (lags, contexto, EWMA)
  2. ultima semana.xlsx            — datos de la semana más reciente a predecir
  3. kepler-backend/models/v{N}/   — modelo entrenado localmente

Uso:
  python scripts/prediccion_ultima_semana.py --csv Master_Consolidado_Final.csv
  python scripts/prediccion_ultima_semana.py --csv Master_Consolidado_Final.csv --excel "ultima semana.xlsx"
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv
load_dotenv(_BACKEND / ".env")

LOCAL_MODELS_ROOT = _BACKEND / "models"
LOG_DIR = _BACKEND / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "prediccion_ultima_semana.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("prediccion_ultima_semana")


def _find_excel(root: Path) -> Path:
    """Busca 'ultima semana.xlsx' en la raíz del backend o del proyecto."""
    candidates = [
        "ultima semana.xlsx", "Ultima semana.xlsx", "ultima_semana.xlsx",
        "Ultima_semana.xlsx", "ultima semana.xls",
    ]
    for base in (root, root.parent):
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"No se encontró 'ultima semana.xlsx' en {root} ni en {root.parent}.\n"
        "Usa --excel <ruta> para indicar la ubicación."
    )


def _load_latest_model():
    """Carga el modelo XGBoost más reciente desde models/v{N}/."""
    import xgboost as xgb

    if not LOCAL_MODELS_ROOT.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de modelos: {LOCAL_MODELS_ROOT}\n"
            "Ejecuta primero: python -m ml_pipeline.train --csv Master_Consolidado_Final.csv"
        )
    dirs = sorted(
        [d for d in LOCAL_MODELS_ROOT.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda d: int(d.name[1:]),
    )
    if not dirs:
        raise FileNotFoundError(f"No hay versiones de modelo en {LOCAL_MODELS_ROOT}.")

    latest = dirs[-1]
    model_file = latest / "model.json"
    meta_file = latest / "model_meta.json"
    summary_file = latest / "training_summary.json"

    if not model_file.exists():
        raise FileNotFoundError(f"model.json no encontrado en {latest}.")
    if not meta_file.exists():
        raise FileNotFoundError(f"model_meta.json no encontrado en {latest}.")

    booster = xgb.Booster()
    booster.load_model(str(model_file))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    mae_wf = None
    if summary_file.exists():
        try:
            s = json.loads(summary_file.read_text(encoding="utf-8"))
            mae_wf = float(s.get("mae_walk_forward", 0)) or None
        except Exception:
            pass

    logger.info("Modelo cargado: %s | MAE walk-forward: %s", latest.name, mae_wf)
    return booster, meta, mae_wf


def main():
    import argparse
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap

    from ml_pipeline.config import BENCHMARK_WINDOW_WEEKS, TARGET_NAME
    from ml_pipeline.data_contracts import validate_data_contracts
    from ml_pipeline.feature_engineering import (
        build_feature_matrix,
        csv_bytes_to_dataframe,
        prepare_base_df,
    )

    parser = argparse.ArgumentParser(
        description="Predicción + SHAP: historial CSV + ultima semana.xlsx + modelo local."
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Ruta a Master_Consolidado_Final.csv (historial completo).",
    )
    parser.add_argument(
        "--excel", type=Path, default=None,
        help="Ruta a 'ultima semana.xlsx' (por defecto se busca automáticamente).",
    )
    # Backwards compatibility silenciosa
    parser.add_argument("--organization-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--semana", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ingesta-reentreno", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # --- Rutas ---
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = csv_path.resolve() if csv_path.exists() else _BACKEND / csv_path
    if not csv_path.is_file():
        logger.error("CSV no encontrado: %s", csv_path)
        sys.exit(1)

    if args.excel:
        excel_path = Path(args.excel)
        if not excel_path.is_absolute():
            excel_path = excel_path.resolve() if excel_path.exists() else _BACKEND / excel_path
    else:
        try:
            excel_path = _find_excel(_BACKEND)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

    if not excel_path.exists():
        logger.error("Excel no encontrado: %s", excel_path)
        sys.exit(1)

    logger.info("========== INICIO Predicción última semana ==========")
    logger.info("CSV historial : %s", csv_path)
    logger.info("Excel semana  : %s", excel_path)

    # 1) Cargar historial CSV
    df_hist = csv_bytes_to_dataframe(csv_path.read_bytes())
    logger.info("Historial: %d filas, %d columnas.", len(df_hist), len(df_hist.columns))

    # 2) Detectar columna temporal del historial
    date_col = "semana" if "semana" in df_hist.columns else "fecha_inicio"

    # 3) Leer Excel (primera hoja)
    try:
        df_excel = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl")
    except Exception:
        df_excel = pd.read_excel(excel_path, sheet_name=0)
    df_excel.columns = [str(c).strip() for c in df_excel.columns]
    if len(df_excel) == 0:
        logger.error("El Excel no tiene filas de datos.")
        sys.exit(1)
    row_excel = df_excel.iloc[0]
    logger.info("Excel: %d fila(s), columnas: %s", len(df_excel), list(df_excel.columns))

    # 4) Detectar fecha de la semana desde el Excel
    # Buscar columna de fecha en el Excel (semana, fecha_inicio, fecha)
    excel_date_col = None
    for candidate in ("semana", "fecha_inicio", "fecha"):
        for col in df_excel.columns:
            if str(col).strip().lower() == candidate:
                excel_date_col = col
                break
        if excel_date_col:
            break

    if excel_date_col and not pd.isna(row_excel[excel_date_col]):
        raw_fecha = str(row_excel[excel_date_col]).strip()
        # Normalizar a YYYY-MM-DD
        if hasattr(row_excel[excel_date_col], "strftime"):
            semana_str = row_excel[excel_date_col].strftime("%Y-%m-%d")
        elif "/" in raw_fecha:
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
        logger.info("Fecha semana tomada del Excel: %s", semana_str)
    else:
        # Fallback: siguiente semana después de la última en el historial
        df_tmp = df_hist.copy()
        df_tmp["_d"] = pd.to_datetime(df_tmp[date_col], dayfirst=True, errors="coerce")
        last_date = df_tmp["_d"].max()
        next_date = last_date + pd.Timedelta(weeks=1)
        semana_str = next_date.strftime("%Y-%m-%d")
        logger.info("Fecha semana no encontrada en Excel; usando semana siguiente al historial: %s", semana_str)

    logger.info(">>> SEMANA A PREDECIR: %s <<<", semana_str)

    # 5) Construir nueva fila alineada con las columnas del historial
    # Normalizar nombres Excel para emparejar con columnas del CSV
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "_")

    excel_by_norm = {_norm(c): c for c in df_excel.columns}

    new_row = {date_col: semana_str}
    for col in df_hist.columns:
        if col == date_col:
            continue
        val = None
        # Buscar el valor en el Excel por nombre exacto o normalizado
        if col in df_excel.columns:
            val = row_excel[col]
        else:
            key = _norm(col)
            if key in excel_by_norm:
                val = row_excel[excel_by_norm[key]]

        # NaN si no hay dato (NO rellenar con 0)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            try:
                new_row[col] = float(val)
            except (TypeError, ValueError):
                new_row[col] = np.nan
        else:
            new_row[col] = np.nan

    logger.info("Nueva fila construida: %d columnas mapeadas.", len(new_row))

    # Verificar que full_users_aprobados viene en el Excel (fuente de full_users_aprobados_lag1)
    if "full_users_aprobados" not in df_excel.columns:
        logger.warning(
            "ADVERTENCIA: 'full_users_aprobados' no esta en el Excel. "
            "full_users_aprobados_lag1 sera NaN para esta prediccion. "
            "Agrega la columna al Excel con los aprobados reales de la semana (por date_full_user)."
        )
    elif pd.isna(row_excel.get("full_users_aprobados", float("nan"))):
        logger.warning(
            "ADVERTENCIA: 'full_users_aprobados' esta en el Excel pero es NaN. "
            "full_users_aprobados_lag1 sera NaN para esta prediccion."
        )

    # 6) Combinar historial + nueva fila
    df_new = pd.DataFrame([new_row])
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    logger.info("DataFrame combinado: %d filas.", len(df_combined))

    # 6b) Validar contratos de datos sobre la nueva fila del Excel
    contracts_ok, violations = validate_data_contracts(df_new)
    if not contracts_ok:
        logger.error(
            "CONTRATOS FALLIDOS en datos del Excel (%d violación/es) — "
            "la predicción puede ser basura:",
            len(violations),
        )
        for v in violations:
            logger.error("  · %s", v)
        logger.error(
            "Corrige el Excel antes de usar esta predicción. "
            "El script continúa pero el resultado NO es fiable."
        )
    else:
        logger.info("Contratos de datos del Excel: OK.")

    # 7) Pipeline de features (drop_na=False para conservar la última fila sin target)
    df_prepared = prepare_base_df(df_combined)
    X_full, y_full, feature_names_full = build_feature_matrix(df_prepared, drop_na=False)
    if len(X_full) == 0:
        logger.error("No quedaron filas útiles tras el pipeline de features.")
        sys.exit(1)
    last_row = X_full.iloc[-1:].copy()
    logger.info("Vector última semana: %d features.", len(feature_names_full))

    # 8) Cargar modelo local
    try:
        booster, meta, mae_modelo = _load_latest_model()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    model_feature_names = meta.get("feature_names", [])
    target_name = meta.get("target_name", TARGET_NAME)
    version = meta.get("version", "?")

    # Alinear features faltantes con NaN (nunca con 0)
    missing = [c for c in model_feature_names if c not in last_row.columns]
    if missing:
        logger.warning("%d features faltantes → NaN: %s", len(missing), missing)
        for col in missing:
            last_row[col] = np.nan
    last_row = last_row[model_feature_names].copy()

    # 9) Predicción
    dmat = xgb.DMatrix(last_row.values.astype(np.float64), feature_names=model_feature_names)
    prediccion = float(booster.predict(dmat)[0])
    logger.info("Predicción siguiente semana (%s): %.0f", target_name, prediccion)

    # 10) Baseline 12 semanas
    X_hist_ctx = X_full.iloc[:-1]
    target_hist = y_full.iloc[:-1].dropna()
    baseline_12w = float(target_hist.tail(BENCHMARK_WINDOW_WEEKS).mean()) if not target_hist.empty else 0.0
    brecha = round(prediccion - baseline_12w, 0)
    logger.info("Baseline 12w: %.0f | Brecha vs baseline: %.0f", baseline_12w, brecha)

    # 11) SHAP
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(dmat)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    contribs = shap_values[0]

    shap_list = sorted(
        [
            {
                "feature": name,
                "value": (
                    None if pd.isna(last_row[name].iloc[0])
                    else float(last_row[name].iloc[0])
                ),
                "contribution": float(contribs[i]),
            }
            for i, name in enumerate(model_feature_names)
        ],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )

    logger.info("--- SHAP top 20 ---")
    for i, s in enumerate(shap_list[:20]):
        logger.info(
            "  %2d. %-45s value=%-12s contrib=%.2f",
            i + 1, s["feature"], s["value"], s["contribution"],
        )

    # 12) Contexto histórico
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
        if val is not None:
            delta = val - mean_12w
            z = float(delta / std_12w) if std_12w > 0 else None
        else:
            delta = z = None
        contexto.append({
            "feature": f,
            "current_value": round(val, 4) if val is not None else None,
            "trailing_12w_mean": round(mean_12w, 4),
            "z_score": round(z, 2) if z is not None else None,
            "shap_contribution": round(s["contribution"], 4),
        })

    # 13) Prescripción
    prescripcion = []
    for item in contexto[:5]:
        feat = item["feature"]
        z = item["z_score"]
        contrib = item["shap_contribution"]
        if z is None or abs(z) < 1.0:
            continue
        accion = {
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
        elif "contenido" in feat:
            accion["tipo_accion"] = "alerta_growth_contenido"
        elif "registro" in feat or "aprobados" in feat:
            accion["tipo_accion"] = "monitorear_adquisicion"
        elif feat in ("TRM", "Variacion_COLCAP", "Tasa_Intervencion_Mensual"):
            accion["tipo_accion"] = "contexto_macro"
        elif "dias_habiles" in feat or "semana_del_mes" in feat or "mes_prima" in feat:
            accion["tipo_accion"] = "estacionalidad"
        else:
            accion["tipo_accion"] = "investigar"
        prescripcion.append(accion)

    logger.info("--- Prescripción (%d acciones) ---", len(prescripcion))
    for p in prescripcion:
        logger.info(
            "  [%s] %s (z=%.2f, contrib=%.0f) → %s",
            p["severidad"], p["variable"], p["z_score"], p["contribucion_depositos"], p["tipo_accion"],
        )

    logger.info("========== FIN ==========")

    out = {
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
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
