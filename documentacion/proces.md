  config.py — el cerebro de configuración

  Define todas las listas de features, los rangos de Optuna y los parámetros del pipeline. Las features actuales en FUNNEL_FEATURES son las viejas
  (incluye usuarios_registro_base, tasa_registro_a_aprobado, etc.) — esto es lo que hay que actualizar para v2. También define MACRO_FEATURES con
  solo Tasa_Intervencion_Mensual, TRM y Variacion_COLCAP — muy diferente al set nuevo.

  Hay algo importante: además de las features del CSV, el config define features computadas en Python que no vienen del CSV:
  - registros_ponderados y aprobados_ponderados — weighted pipeline con pesos [0.350, 0.293, 0.112, 0.070, 0.052, 0.040, 0.083] del ciclo de      
  conversión
  - lag_1_target — lag autoregresivo del propio target
  - full_users_aprobados_lag1 — aprobados de la semana anterior (ya está lagueado por diseño)
  - tendencia_registros_4w, tendencia_aprobados_4w, tendencia_depositos_4w — pendiente OLS de 4 semanas normalizada por la media
  - EWMA span=4 de tasa_registro_a_aprobado y tasa_rechazo_implicita_kyc
  - Variables estacionales: semana_del_mes, semana_del_mes_proyeccion, dias_habiles_semana, dias_habiles_proyeccion, mes_prima

  feature_engineering.py — el pipeline de features

  1. Parsea fechas (acepta DD/MM/YYYY y YYYY-MM-DD)
  2. Calcula festivos colombianos completos (Ley Emiliani + Pascua) para dias_habiles
  3. Aplica el weighted pipeline — combina lags 0-6 de registros y aprobados con los pesos del ciclo de conversión
  4. Crea lag_1_target (autoregresivo) y full_users_aprobados_lag1
  5. Calcula tendencias OLS de 4 semanas
  6. EWMA span=4
  7. El target se shiftea -1 (df["target"] = df[TARGET_NAME].shift(-1)) — la fila de la semana t predice la semana t+1

  data_contracts.py — validación previa

  Valida negativos en columnas críticas y alerta saltos >200% en el target. No bloquea el entrenamiento por saltos, solo por negativos.

  También tiene prune_multicollinearity — elimina automáticamente una de cada par de features con correlación > 0.92. Corre solo sobre features   
  del CSV en tiempo t, no sobre las computadas.

  validation.py — walk-forward CV

  Walk-forward puro de 1 paso: entrena con 52 semanas mínimas, predice la siguiente, luego entrena con 53, predice la siguiente, y así hasta el   
  final. Cada fold tiene un solo punto de test.

  train.py — orquestador principal

  1. Lee CSV → prepara → valida contratos → construye features → clipping del target al p95
  2. Poda multicolinealidad (threshold 0.92) en las features del CSV
  3. Optuna con 60 trials, timeout 600s — busca los mejores hiperparámetros minimizando el MAE promedio del walk-forward
  4. Hiperparámetros buscados: max_depth (3-5), min_child_weight (5-20), subsample (0.5-0.85), colsample_bytree (0.5-0.85), reg_alpha (3-10),     
  reg_lambda (3-10), learning_rate (0.01-0.05), num_boost_round (50-300)
  5. Entrena el modelo final con todos los datos y los mejores params
  6. Detecta overfitting: ratio MAE_wf / MAE_train > 5 = crítico, > 3 = warning
  7. Guarda model.json, model_meta.json, training_summary.json en models/v{N}/

  ml_runner.py — predicción + SHAP

  Para predecir: combina master_consolidado_final (historial) + ultima_semana (nueva fila) → corre el mismo pipeline de features → predice con el 
  modelo cargado → calcula SHAP con shap.TreeExplainer → genera z-scores vs. media 12w → genera prescripción con categorías de acción → guarda en 
  Supabase.

  ---
  Lo que implica para el v2: hay que tocar principalmente config.py — actualizar FUNNEL_FEATURES, MACRO_FEATURES, agregar pct_dias_quincena a la  
  lista de features del CSV, y ajustar los EWMA_VARIABLES y WEIGHTED_PIPELINE_VARIABLES según las variables que entran o salen.