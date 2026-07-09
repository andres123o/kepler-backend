"""
Configuración GLOBAL del pipeline ML — hiperparámetros de modelado que aplican
igual a cualquier funnel, sin importar el dominio o el país.

Todo lo específico de un funnel (columnas del CSV, target, transformaciones
de dominio como festivos o ciclos de conversión) vive en funnels.config.ml en
Supabase — ver ml_pipeline/funnel_config.py (FunnelMLConfig) y
ml_pipeline/enrichers.py (catálogo de transformaciones opcionales). Montar un
funnel nuevo, sea del dominio que sea, no requiere editar este archivo.
"""

# === CSV (formato de archivo, no cambia por funnel) ===
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8"

# === Target ===
TARGET_CLIP_PERCENTILE = 95

# === Multicolinealidad (default global — overridable por funnel) ===
MULTICOLLINEARITY_THRESHOLD = 0.92

# === Contratos de datos (default global) ===
ANOMALY_JUMP_PCT = 200

# === Walk-forward ===
INITIAL_TRAIN_WEEKS = 52

# === Optuna ===
OPTUNA_N_TRIALS = 60
OPTUNA_TIMEOUT_SECONDS = 600
OPTUNA_MAX_DEPTH_RANGE = (3, 5)
OPTUNA_REG_ALPHA_RANGE = (3.0, 10.0)
OPTUNA_REG_LAMBDA_RANGE = (3.0, 10.0)
OPTUNA_NUM_BOOST_ROUND_RANGE = (50, 300)
OPTUNA_SUBSAMPLE_RANGE = (0.5, 0.85)
OPTUNA_COLSAMPLE_RANGE = (0.5, 0.85)
OPTUNA_LEARNING_RATE_RANGE = (0.01, 0.05)
OPTUNA_MIN_CHILD_WEIGHT_RANGE = (5, 20)

# === Loss function ===
# reg:absoluteerror (MAE puro): robusto a outliers (semanas de campaña),
# usa hessianos unitarios → min_child_weight y reg_* mantienen su escala normal.
LOSS_OBJECTIVE = "reg:absoluteerror"

# === Benchmarking histórico ===
BENCHMARK_WINDOW_WEEKS = 12
