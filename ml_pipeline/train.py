"""
Entrenamiento del modelo ML (pipeline v2) — todo local, sin Supabase.

Uso:
  python -m ml_pipeline.train --csv Master_Consolidado_Final.csv

El modelo se guarda en:
  kepler-backend/models/v{N}/model.json
  kepler-backend/models/v{N}/model_meta.json
  kepler-backend/models/v{N}/training_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND = _SCRIPT_DIR.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from dotenv import load_dotenv
load_dotenv(_BACKEND / ".env")

import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_pipeline.config import (
    ALL_CSV_FEATURES,
    COMPUTED_FEATURE_NAMES,
    INITIAL_TRAIN_WEEKS,
    LOSS_OBJECTIVE,
    MULTICOLLINEARITY_THRESHOLD,
    OPTUNA_MAX_DEPTH_RANGE,
    OPTUNA_N_TRIALS,
    OPTUNA_NUM_BOOST_ROUND_RANGE,
    OPTUNA_REG_ALPHA_RANGE,
    OPTUNA_REG_LAMBDA_RANGE,
    OPTUNA_SUBSAMPLE_RANGE,
    OPTUNA_COLSAMPLE_RANGE,
    OPTUNA_LEARNING_RATE_RANGE,
    OPTUNA_MIN_CHILD_WEIGHT_RANGE,
    OPTUNA_TIMEOUT_SECONDS,
    TARGET_CLIP_PERCENTILE,
    TARGET_NAME,
)
from ml_pipeline.data_contracts import (
    clip_target_to_percentile,
    prune_multicollinearity,
    validate_data_contracts,
)
from ml_pipeline.feature_engineering import (
    build_feature_matrix,
    csv_bytes_to_dataframe,
    load_csv,
    prepare_base_df,
)
from ml_pipeline.validation import walk_forward_splits

LOCAL_MODELS_ROOT = _BACKEND / "models"

_LOG_DIR = _BACKEND / "logs"
_LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("kepler.ml.train")


def _next_version(root: Path | None = None) -> int:
    """Retorna la siguiente versión leyendo carpetas v* existentes en root (default: models/)."""
    target = root or LOCAL_MODELS_ROOT
    if not target.exists():
        return 1
    versions = []
    for d in target.glob("v*"):
        if d.is_dir() and d.name[1:].isdigit():
            versions.append(int(d.name[1:]))
    return max(versions) + 1 if versions else 1


def train_one_fold(X_train, y_train, X_test, y_test, params: dict, num_boost_round: int):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dtest, "test")], early_stopping_rounds=20, verbose_eval=False,
    )
    pred = model.predict(dtest)
    return float(mean_absolute_error(y_test, pred)), model.best_iteration


def run_training(csv_path: Path, output_dir: Path | None = None) -> dict:
    logger.info("========== INICIO ENTRENAMIENTO ==========")
    logger.info("[Fase 0] Leyendo CSV: %s", csv_path)
    data = csv_path.read_bytes()
    df = csv_bytes_to_dataframe(data)
    df = prepare_base_df(df)
    logger.info("[Fase 0] DataFrame listo: %d filas.", len(df))

    # Excluir semanas exógenas (campañas internas, outliers confirmados).
    # ml_runner.py NO aplica este filtro: usa el CSV completo para calcular lags.
    # Acepta tanto 'es_exogena' como 'es_exogeno' (variante del CSV)
    _exo_col = next((c for c in ("es_exogena", "es_exogeno") if c in df.columns), None)
    if _exo_col:
        n_antes = len(df)
        df = df[df[_exo_col] != 1].copy()
        logger.info(
            "[Fase 0] Semanas exógenas excluidas (%s): %d → %d filas de entrenamiento.",
            _exo_col, n_antes, len(df),
        )

    # Fase 1: Contratos
    logger.info("[Fase 1] Validando contratos de datos...")
    ok, violations = validate_data_contracts(df)
    if not ok:
        raise ValueError("Contratos fallidos: " + "; ".join(violations))

    # Fase 2: Features
    logger.info("[Fase 2] Construyendo matriz de features...")
    X_full, y_full, feature_names = build_feature_matrix(df, drop_na=True)
    if len(X_full) < INITIAL_TRAIN_WEEKS + 5:
        raise ValueError(
            f"Filas útiles ({len(X_full)}) insuficientes para walk-forward (mínimo {INITIAL_TRAIN_WEEKS}+5)."
        )
    logger.info("[Fase 2] Aplicando target clipping (p%d)...", TARGET_CLIP_PERCENTILE)
    y_full = clip_target_to_percentile(y_full, TARGET_CLIP_PERCENTILE)

    # Fase 2.3: Poda multicolinealidad (solo ALL_CSV_FEATURES — tiempo t)
    # Las COMPUTED_FEATURE_NAMES quedan fuera de esta poda (ver data_contracts.prune_multicollinearity).
    logger.info("[Fase 2.3] Poda multicolinealidad (threshold=%.2f)...", MULTICOLLINEARITY_THRESHOLD)
    csv_time_t = [f for f in feature_names if f in ALL_CSV_FEATURES]
    computed_cols = [f for f in feature_names if f in COMPUTED_FEATURE_NAMES]
    surviving = prune_multicollinearity(X_full, csv_time_t, MULTICOLLINEARITY_THRESHOLD)
    feature_names = surviving + computed_cols
    X_full = X_full[feature_names]

    X = X_full.values
    y = y_full.values.astype(float)
    n_samples = len(y)
    logger.info("Matriz final: %d filas, %d features.", n_samples, len(feature_names))
    logger.info("Features: %s", feature_names)

    # Fase 3-4: Optuna + Walk-Forward
    logger.info("[Fase 3-4] Walk-Forward + Optuna (%d trials)...", OPTUNA_N_TRIALS)
    depth_lo, depth_hi = OPTUNA_MAX_DEPTH_RANGE
    alpha_lo, alpha_hi = OPTUNA_REG_ALPHA_RANGE
    lambda_lo, lambda_hi = OPTUNA_REG_LAMBDA_RANGE
    sub_lo, sub_hi = OPTUNA_SUBSAMPLE_RANGE
    col_lo, col_hi = OPTUNA_COLSAMPLE_RANGE
    lr_lo, lr_hi = OPTUNA_LEARNING_RATE_RANGE
    mcw_lo, mcw_hi = OPTUNA_MIN_CHILD_WEIGHT_RANGE
    boost_lo, boost_hi = OPTUNA_NUM_BOOST_ROUND_RANGE

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": LOSS_OBJECTIVE,
            "max_depth": trial.suggest_int("max_depth", depth_lo, depth_hi),
            "min_child_weight": trial.suggest_int("min_child_weight", mcw_lo, mcw_hi),
            "subsample": trial.suggest_float("subsample", sub_lo, sub_hi),
            "colsample_bytree": trial.suggest_float("colsample_bytree", col_lo, col_hi),
            "reg_alpha": trial.suggest_float("reg_alpha", alpha_lo, alpha_hi),
            "reg_lambda": trial.suggest_float("reg_lambda", lambda_lo, lambda_hi),
            "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi),
            "tree_method": "hist", "seed": 42,
        }
        num_boost_round = trial.suggest_int("num_boost_round", boost_lo, boost_hi)
        maes = [
            train_one_fold(X[ti], y[ti], X[vi], y[vi], params, num_boost_round)[0]
            for ti, vi in walk_forward_splits(n_samples, INITIAL_TRAIN_WEEKS)
        ]
        return float(np.mean(maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SECONDS, show_progress_bar=True)
    best_params = study.best_params
    best_mae_wf = study.best_value
    logger.info("[Fase 4] MAE walk-forward=%.4f | params=%s", best_mae_wf, best_params)

    # Fase 5: Entrenamiento final
    logger.info("[Fase 5] Entrenamiento final...")
    num_boost_round_final = best_params.pop("num_boost_round", 100)
    xgb_params = {k: v for k, v in best_params.items()}
    xgb_params.update({"objective": LOSS_OBJECTIVE, "tree_method": "hist", "seed": 42})
    best_params["num_boost_round"] = num_boost_round_final

    dtrain_full = xgb.DMatrix(X, label=y, feature_names=feature_names)
    final_model = xgb.train(xgb_params, dtrain_full, num_boost_round=num_boost_round_final, verbose_eval=False)

    pred_full = final_model.predict(dtrain_full)
    r2 = float(r2_score(y, pred_full))
    mae_full = float(mean_absolute_error(y, pred_full))
    rmse_full = float(np.sqrt(mean_squared_error(y, pred_full)))
    logger.info("[Fase 5] R²=%.4f | MAE=%.4f | RMSE=%.4f", r2, mae_full, rmse_full)

    # Diagnóstico de overfitting
    ratio_overfitting = best_mae_wf / mae_full if mae_full > 0 else float("inf")
    if ratio_overfitting > 5.0:
        logger.error(
            "Overfitting severo: ratio=%.2f — revisar hiperparámetros "
            "(mae_wf=%.2f, mae_train=%.2f)",
            ratio_overfitting, best_mae_wf, mae_full,
        )
    elif ratio_overfitting > 3.0:
        logger.warning(
            "Posible overfitting detectado: ratio=%.2f "
            "(mae_wf=%.2f, mae_train=%.2f)",
            ratio_overfitting, best_mae_wf, mae_full,
        )
    else:
        logger.info("[Overfitting] ratio_wf_train=%.2f — OK", ratio_overfitting)

    if ratio_overfitting > 5.0:
        overfitting_flag = "critical"
    elif ratio_overfitting > 3.0:
        overfitting_flag = "warning"
    else:
        overfitting_flag = "ok"

    importance = final_model.get_score(importance_type="gain")
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("=== Feature Importance (gain) ===")
    for feat, gain in importance_sorted:
        logger.info("  %s: %.2f", feat, gain)

    # Guardar localmente
    models_root = output_dir or LOCAL_MODELS_ROOT
    version = _next_version(models_root)
    version_dir = models_root / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    meta = {"feature_names": feature_names, "target_name": TARGET_NAME, "version": version}
    summary = {
        "version": version,
        "mae_walk_forward": best_mae_wf,
        "r2_train": r2,
        "mae_train": mae_full,
        "rmse_train": rmse_full,
        "ratio_wf_train": round(ratio_overfitting, 4),
        "overfitting_flag": overfitting_flag,
        "best_params": best_params,
        "feature_names": feature_names,
        "n_samples": n_samples,
        "n_features": len(feature_names),
        "feature_importance": {k: round(v, 4) for k, v in importance_sorted},
        "status": "ok",
    }

    final_model.save_model(str(version_dir / "model.json"))
    (version_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (version_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Modelo guardado en: %s", version_dir)
    logger.info("========== ENTRENAMIENTO FINALIZADO v%d ==========", version)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo XGBoost — todo local.")
    parser.add_argument("--csv", type=Path, required=True, help="Ruta a Master_Consolidado_Final.csv")
    # Backwards compatibility: --organization-id se acepta pero se ignora
    parser.add_argument("--organization-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.is_absolute():
        csv_path = csv_path.resolve() if csv_path.exists() else _BACKEND / csv_path
    if not csv_path.is_file():
        print(f"Error: CSV no encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        summary = run_training(csv_path)
        print(json.dumps(summary, indent=2))
    except Exception as e:
        logger.exception("Entrenamiento fallido: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
