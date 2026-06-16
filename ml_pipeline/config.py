"""
Configuración central del pipeline ML — Kepler v7.
Features validadas: NZV → VIF → MI → MRMR → Granger → Estabilidad CV.
"""

# === CSV ===
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8"
DATE_COLUMN = "semana"

# === Target ===
TARGET_NAME = "usuarios_primer_cashin"
TARGET_CLIP_PERCENTILE = 95

# === Features del CSV (tiempo t) ===
FUNNEL_FEATURES = [
    "step_09_full_account",
    "tasa_basic_a_risk",
    "tasa_risk_a_fulldata",
    "tasa_fulldata_a_video",
    "tasa_video_a_review",
    "pct_perfil_conservador",
    "pct_perfil_arriesgado",
    "full_users_aprobados",
]

MACRO_FEATURES = [
    "TRM",
    "sp500_cambio_semanal_pct",
    "brent_cambio_semanal_pct",
    "colcap_cambio_semanal_pct",
    "spread_tes_banrep",
    "trends_cdt",
    "trends_acciones",
    "pct_dias_quincena",
]

ALL_CSV_FEATURES = FUNNEL_FEATURES + MACRO_FEATURES

# === Weighted pipeline — ciclo aprobación → depósito ===
#
# Fuente: cicloUsuario.md (BigQuery, Colombia, datos desde 2022):
#   lag-0 (aprobados semana t)    → días  7-13 en t+1 : ~6%  → peso 0.214
#   lag-1 (aprobados semana t-1)  → días 14-20 en t+1 : ~9%  → peso 0.321  ← MÁXIMO (repunte día 14)
#   lag-2 (aprobados semana t-2)  → días 21-27 en t+1 : ~6%  → peso 0.214
#   lag-3                         → días 28-34 en t+1 : ~3%  → peso 0.107
#   lag-4                         → días 35-41 en t+1 : ~2%  → peso 0.071
#   lag-5                         → días 42-48 en t+1 : ~2%  → peso 0.072
#   lag-6                         → días 49+   en t+1 :  0%  → peso 0.000
CONVERSION_CYCLE_WEIGHTS = [0.214, 0.321, 0.214, 0.107, 0.071, 0.072, 0.000]

WEIGHTED_PIPELINE_VARIABLES = [
    "full_users_aprobados",
]

# === Features computadas (calculadas en Python — no vienen del CSV) ===
COMPUTED_FEATURE_NAMES = [
    "dias_habiles_semana",        # días hábiles lun-vie semana t (festivos colombianos exactos)
    "dias_habiles_proyeccion",    # días hábiles lun-vie semana t+1 (la semana predicha)
    "aprobados_ponderados",       # weighted pipeline full_users_aprobados lags 0-5
    "lag_1_target",               # target t-1 (autoregresivo)
    "full_users_aprobados_lag1",  # aprobaciones semana t-1 (pasado covariante)
    "tendencia_aprobados_4w",     # pendiente OLS 4w de full_users_aprobados
    "tendencia_depositos_4w",     # pendiente OLS 4w del target
]

# === Grupos para SHAP y prescripción ===
#
# FUNNEL_INFORMATIVO — tasas KYC y volumen crudo.
#   El modelo las necesita para predecir pero Marketing no las puede optimizar vía CIO.
#
# PIPELINE_AUTOREG — estado histórico del pipeline y el target.
#   Capturan inercia y ciclo aprobados→depósito. Descriptivos, no levers de campaña.
#
# NON_ACTIONABLE_FEATURES = FUNNEL_INFORMATIVO | PIPELINE_AUTOREG
#   El resto (pct_perfil_*, macro, trends, timing) son accionables vía campaña.

FUNNEL_INFORMATIVO: set[str] = {
    "step_09_full_account",
    "tasa_basic_a_risk",
    "tasa_risk_a_fulldata",
    "tasa_fulldata_a_video",
    "tasa_video_a_review",
    "full_users_aprobados",
}

PIPELINE_AUTOREG: set[str] = {
    "lag_1_target",
    "full_users_aprobados_lag1",
    "aprobados_ponderados",
    "tendencia_aprobados_4w",
}

NON_ACTIONABLE_FEATURES: set[str] = FUNNEL_INFORMATIVO | PIPELINE_AUTOREG

# === Contratos de datos ===
NON_NEGATIVE_COLUMNS = [
    "step_09_full_account",
    "full_users_aprobados",
    "usuarios_primer_cashin",
]
ANOMALY_JUMP_PCT = 200

# === Multicolinealidad ===
MULTICOLLINEARITY_THRESHOLD = 0.92

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
