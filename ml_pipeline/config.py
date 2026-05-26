"""
Configuración central del pipeline ML — Activation Co-Pilot v2.
"""

# === CSV ===
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8"
DATE_COLUMN = "semana"  # Formato YYYY-MM-DD (lunes de cada semana)

# === Target ===
TARGET_NAME = "usuarios_primer_cashin"
TARGET_CLIP_PERCENTILE = 95

# === Features del CSV (tiempo t) ===
# Funnel del usuario (por semana de registro)
FUNNEL_FEATURES = [
    "usuarios_registro_base",
    "step_09_full_account",
    "tasa_basic_a_risk",
    "tasa_risk_a_fulldata",
    "tasa_fulldata_a_video",
    "tasa_video_a_review",
    "tasa_review_a_aprobado",
    "tasa_registro_a_aprobado",
    "tasa_rechazo_implicita_kyc",
    "mediana_dias_registro_a_full",
    "pct_perfil_conservador",
    "pct_perfil_arriesgado",
]

# Canales Trii (cobertura parcial — NaN antes de su fecha de inicio)
CHANNEL_FEATURES = [
    "push_mail_delivered_pre_deposito",
    "push_mail_converted_pre_deposito",
    "cx_friccion_kyc",
    "cx_bloqueos",
]

# Macroeconómicas
MACRO_FEATURES = [
    "Tasa_Intervencion_Mensual",
    "TRM",
    "Variacion_COLCAP",
]

# Estacionales (calculadas en Python desde la fecha — NO vienen del CSV)
SEASONAL_FEATURES = [
    "semana_del_mes",                # semana del mes — semana t
    "semana_del_mes_proyeccion",     # semana del mes — semana t+1 (la que se predice)
    "dias_habiles_semana",           # dias habiles (todos los festivos CO) — semana t
    "mes_prima",
    "dias_habiles_proyeccion",       # dias habiles (todos los festivos CO) — semana t+1
]

# Todas las features del CSV en tiempo t
# SEASONAL_FEATURES se calculan en Python desde la fecha — NO vienen del CSV
ALL_CSV_FEATURES = FUNNEL_FEATURES + CHANNEL_FEATURES + MACRO_FEATURES

# === Lags comprimidos (weighted pipeline) ===
# Pesos del ciclo de conversión (validado desde BigQuery)
CONVERSION_CYCLE_WEIGHTS = [0.350, 0.293, 0.112, 0.070, 0.052, 0.040, 0.083]
# 7 valores: lag_0 a lag_5 + lag_6_8 agrupado
# Estos pesos suman 1.0

# Variables a las que se aplica el weighted pipeline
WEIGHTED_PIPELINE_VARIABLES = [
    "usuarios_registro_base",
    "step_09_full_account",
]
# Resultado: registros_ponderados, aprobados_ponderados

# Lag autoregresivo del target
LAG_TARGET_PERIODS = [1]  # Solo lag 1 del target

# Nombres de las features calculadas (para referencia)
COMPUTED_FEATURE_NAMES = [
    "registros_ponderados",
    "aprobados_ponderados",
    "lag_1_target",
    "full_users_aprobados_lag1",   # aprobaciones reales t-1 agrupadas por date_full_user
    "tendencia_registros_4w",      # slope OLS 4w de registros (top funnel)
    "tendencia_aprobados_4w",      # slope OLS 4w de aprobados reales (solo si existe)
    "tendencia_depositos_4w",      # slope OLS 4w del target autoregresivo
]

# === EWMA (Exponential Weighted Moving Average) ===
# Para capturar tendencia en variables de tendencia del funnel
EWMA_VARIABLES = [
    "tasa_registro_a_aprobado",
    "tasa_rechazo_implicita_kyc",
]
EWMA_SPAN = 4  # ~1 mes de ventana

# === Contratos de datos ===
NON_NEGATIVE_COLUMNS = [
    "usuarios_registro_base",
    "step_09_full_account",
    "push_mail_delivered_pre_deposito",
    "push_mail_converted_pre_deposito",
    "cx_friccion_kyc",
    "cx_bloqueos",
    "usuarios_primer_cashin",
    "dias_habiles_semana",
    "dias_habiles_proyeccion",
    "full_users_aprobados",
]
ANOMALY_JUMP_PCT = 200  # Saltos >200% en target semana a semana = alerta

# === Multicolinealidad ===
MULTICOLLINEARITY_THRESHOLD = 0.92

# === Validación ===
INITIAL_TRAIN_WEEKS = 52  # 1 año mínimo para empezar walk-forward

# === Optuna ===
OPTUNA_N_TRIALS = 60
OPTUNA_TIMEOUT_SECONDS = 600
OPTUNA_MAX_DEPTH_RANGE = (3, 5)           # ← reducido de (3,8): evita árboles demasiado profundos
OPTUNA_REG_ALPHA_RANGE = (3.0, 10.0)     # ← mínimo aumentado de 0.01: fuerza L1 mínimo
OPTUNA_REG_LAMBDA_RANGE = (3.0, 10.0)    # ← mínimo aumentado de 0.1: fuerza L2 mínimo
OPTUNA_NUM_BOOST_ROUND_RANGE = (50, 300)
OPTUNA_SUBSAMPLE_RANGE = (0.5, 0.85)     # ← máximo reducido de 1.0: evita ver todo el dataset
OPTUNA_COLSAMPLE_RANGE = (0.5, 0.85)     # ← máximo reducido de 1.0
OPTUNA_LEARNING_RATE_RANGE = (0.01, 0.05) # ← máximo reducido de 0.3: aprendizaje más lento
OPTUNA_MIN_CHILD_WEIGHT_RANGE = (5, 20)  # ← máximo reducido de 50: rango de regularización razonable

# === Almacenamiento (Supabase) ===
BUCKET_ML_DATASETS = "ml_datasets"
BUCKET_ML_MODELS = "ml_models"
CSV_PATH_TEMPLATE = "{organization_id}/Master_Consolidado_Final.csv"
MODEL_PATH_TEMPLATE = "{organization_id}/v{version}/model.json"
META_PATH_TEMPLATE = "{organization_id}/v{version}/model_meta.json"
SUMMARY_PATH_TEMPLATE = "{organization_id}/v{version}/training_summary.json"

# === Benchmarking histórico (para SHAP contextual) ===
BENCHMARK_WINDOW_WEEKS = 12  # Ventana de comparación para z-scores
