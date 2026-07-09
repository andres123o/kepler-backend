"""
Pipeline ML — motor de entrenamiento genérico, sin asunciones de dominio.

Ejecución: python -m ml_pipeline.train --csv <ruta_al_csv> --org <org> --funnel <funnel>

Todo lo específico de un funnel vive en funnels.config.ml en Supabase — ver
ml_pipeline/funnel_config.py (FunnelMLConfig) y ml_pipeline/enrichers.py
(catálogo de transformaciones de dominio opcionales, declaradas por el
funnel que las necesite). Este paquete solo expone hiperparámetros globales
de modelado.
"""

from ml_pipeline.config import (
    MULTICOLLINEARITY_THRESHOLD,
    CSV_DELIMITER,
    INITIAL_TRAIN_WEEKS,
    TARGET_CLIP_PERCENTILE,
    ANOMALY_JUMP_PCT,
    OPTUNA_N_TRIALS,
    OPTUNA_MAX_DEPTH_RANGE,
    OPTUNA_REG_ALPHA_RANGE,
    OPTUNA_NUM_BOOST_ROUND_RANGE,
    LOSS_OBJECTIVE,
    BENCHMARK_WINDOW_WEEKS,
)
from ml_pipeline.enrichers import ENRICHER_REGISTRY
from ml_pipeline.funnel_config import FunnelMLConfig, load_funnel_ml_config, load_funnel_ml_config_from_json

__all__ = [
    "MULTICOLLINEARITY_THRESHOLD",
    "CSV_DELIMITER",
    "INITIAL_TRAIN_WEEKS",
    "TARGET_CLIP_PERCENTILE",
    "ANOMALY_JUMP_PCT",
    "OPTUNA_N_TRIALS",
    "OPTUNA_MAX_DEPTH_RANGE",
    "OPTUNA_REG_ALPHA_RANGE",
    "OPTUNA_NUM_BOOST_ROUND_RANGE",
    "LOSS_OBJECTIVE",
    "BENCHMARK_WINDOW_WEEKS",
    "ENRICHER_REGISTRY",
    "FunnelMLConfig",
    "load_funnel_ml_config",
    "load_funnel_ml_config_from_json",
]
