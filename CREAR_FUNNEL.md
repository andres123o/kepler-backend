# Cómo crear un nuevo funnel en Kepler

Guía paso a paso basada en la creación de `activacion_pe` (Perú) sobre la arquitectura multi-tenant de Trii.

---

## Arquitectura de referencia

```
Supabase (toda la configuración y datos)
  organizations          — org por cliente
  funnels                — un funnel por país/producto, config JSONB completo
  {org}_{funnel}_master          — historial semanal completo
  {org}_{funnel}_ultima_semana   — fila única de la semana actual
  {org}_{funnel}_prediction_results — historial de predicciones

Storage bucket: models
  {org}/{funnel}/v{N}/model.json
  {org}/{funnel}/v{N}/model_meta.json
  {org}/{funnel}/v{N}/training_summary.json

Código Python (kepler-backend)
  skills/conectores genéricos en CONNECTOR_REGISTRY  — sin parámetros hardcodeados
  ml_runner.py           — descarga el modelo de Storage al predecir
  data.py                — endpoints genéricos via FunnelClient
```

**Principio:** Agregar un funnel = cero cambios de código. Todo vive en Supabase.

---

## Paso 1 — Preparar los datos históricos

El funnel necesita un master consolidado histórico para entrenar el modelo.

### 1.1 Estructura del master
Un CSV/Excel con una fila por semana y las columnas:
- `semana` — fecha inicio de semana en formato `DD/MM/YYYY`
- Variables del funnel (internas): registro, tasas de conversión, aprobados
- Variables externas (macro, mercado, calendario)
- Target: `usuarios_primer_cashin` (o el equivalente del funnel)

### 1.2 Decidir qué variables son auto-fetch vs manuales

| Tipo | Ejemplos | Cómo se consigue |
|---|---|---|
| Auto-fetch (conector) | PEN/USD, BVL, S&P500, Cobre, Google Trends, quincena, CTS, gratificación, días hábiles | Script automático basado en la semana |
| Manual input | tasa_bcrp, is_ventana_afp, variables internas del funnel | El usuario las escribe en el formulario |
| Target | usuarios_primer_cashin | Se completa después de la semana real |

### 1.3 Crear script de seed del master
Usar `scripts/seed_master_peru.py` como referencia. El script:
- Lee el Excel histórico
- Limpia y normaliza columnas
- Hace upsert por semana en `{org}_{funnel}_master`

```bash
# Con el venv activo, desde kepler-backend/
python scripts/seed_master_{funnel}.py --dry-run   # verificar primero
python scripts/seed_master_{funnel}.py             # ejecutar
```

---

## Paso 2 — Crear las tablas en Supabase

Ejecutar en el SQL editor de Supabase:

> **Nota sobre nombre del master:** Si el master se llama `{prefix}_master_consolidado_final`
> en lugar de `{prefix}_master`, el código lo detecta automáticamente.
> `_resolve_master_table()` en `supabase_client.py` prueba ambas variantes.

```sql
-- Tabla master (historial completo)
CREATE TABLE IF NOT EXISTS trii_activacion_pe_master (
  id                          bigserial PRIMARY KEY,
  semana                      text,
  -- variables funnel interno
  usuarios_registro_base      float8,
  tasa_fulldata_a_video       float8,
  tasa_review_a_aprobado      float8,
  full_users_aprobados        float8,
  usuarios_primer_cashin      float8,
  -- variables manuales
  tasa_bcrp                   float8,
  is_ventana_afp              float8,
  -- variables auto-fetch
  pen_usd_var_semanal         float8,
  pen_usd_volatilidad_4w      float8,
  bvl_var_semanal             float8,
  cobre_var_semanal           float8,
  sp500_var_semanal           float8,
  trends_fondos_mutuos        float8,
  trends_invertir             float8,
  is_ventana_quincena         float8,
  is_ventana_cts              float8,
  is_ventana_gratificacion    float8,
  dias_habiles_semana         float8,
  created_at                  timestamptz DEFAULT now()
);

-- Tabla ultima_semana (fila única de la semana actual)
CREATE TABLE IF NOT EXISTS trii_activacion_pe_ultima_semana (
  -- mismas columnas que master
  id                          bigserial PRIMARY KEY,
  semana                      text,
  usuarios_registro_base      float8,
  tasa_fulldata_a_video       float8,
  tasa_review_a_aprobado      float8,
  full_users_aprobados        float8,
  usuarios_primer_cashin      float8,
  tasa_bcrp                   float8,
  is_ventana_afp              float8,
  pen_usd_var_semanal         float8,
  pen_usd_volatilidad_4w      float8,
  bvl_var_semanal             float8,
  cobre_var_semanal           float8,
  sp500_var_semanal           float8,
  trends_fondos_mutuos        float8,
  trends_invertir             float8,
  is_ventana_quincena         float8,
  is_ventana_cts              float8,
  is_ventana_gratificacion    float8,
  dias_habiles_semana         float8,
  created_at                  timestamptz DEFAULT now()
);

-- Tabla prediction_results (historial de predicciones del modelo)
CREATE TABLE IF NOT EXISTS trii_activacion_pe_prediction_results (
  id              bigserial PRIMARY KEY,
  semana_datos    text,
  semana_label    text,
  prediccion      float8,
  baseline_12w    float8,
  brecha          float8,
  mae_modelo      float8,
  modelo_version  text,
  full_result     jsonb,
  created_at      timestamptz DEFAULT now()
);
```

---

## Paso 3 — Registrar la organización y el funnel

```sql
-- Insertar organización (si no existe)
INSERT INTO organizations (slug, name)
VALUES ('trii', 'Trii')
ON CONFLICT (slug) DO NOTHING;

-- Insertar el funnel (config queda NULL por ahora — se llena en Paso 6)
INSERT INTO funnels (org_id, slug, name, country, is_active, config)
VALUES (
  (SELECT id FROM organizations WHERE slug = 'trii'),
  'activacion_pe',
  'Activación Perú',
  'PE',
  true,
  NULL
);
```

**Nota:** `is_active = true` hace que el funnel aparezca en el selector del sidebar al login.

---

## Paso 4 — Entrenar el modelo ML

El pipeline es `ml_pipeline/train.py`. Necesita el master como CSV.

### 4.1 Exportar master a CSV y entrenar localmente

```bash
# Exportar master desde Supabase a CSV, luego:
python -m ml_pipeline.train --csv ruta/al/master.csv
# Genera models/v{N}/ con model.json, model_meta.json, training_summary.json
```

### 4.2 Validar el modelo antes de subir

Revisar `training_summary.json`:
- `ratio_wf_train` < 3.0 (sin overfitting severo)
- `overfitting_flag` = "ok"
- `mae_walk_forward` razonable para el volumen del negocio

Ejemplo Peru v2:
- MAE walk-forward: 48 usuarios
- R² train: 0.93
- ratio: 2.45 — ok
- 23 features, 218 semanas de entrenamiento

### 4.3 Subir el modelo a Supabase Storage

```bash
# Primero crear el bucket 'models' en Supabase Dashboard → Storage (si no existe)
# Luego correr:
python scripts/upload_model.py \
  --org trii \
  --funnel activacion_pe \
  --version 2 \
  --dir ruta/a/models/v2

# El script sube a: models/trii/activacion_pe/v2/
# y muestra el SQL a ejecutar para conectar el modelo al funnel
```

El script `upload_model.py` es genérico — sirve para cualquier org/funnel.

---

## Paso 5 — Configurar los conectores (data_sources)

Los conectores son los skills genéricos de auto-fetch. Sus parámetros viven en Supabase.

Ver `CONECTORES.md` para el catálogo completo de conectores disponibles.

Los conectores de Peru usan:

| Campo | Conector | Params clave |
|---|---|---|
| `pen_usd_var_semanal` | `yfinance_weekly_pct` | ticker: `PEN=X` |
| `pen_usd_volatilidad_4w` | `yfinance_rolling_std` | ticker: `PEN=X`, weeks: 4 |
| `bvl_var_semanal` | `yfinance_weekly_pct` | ticker: `EPU` (^SPBLPGPT delisted) |
| `cobre_var_semanal` | `yfinance_weekly_pct` | ticker: `HG=F` |
| `sp500_var_semanal` | `yfinance_weekly_pct` | ticker: `^GSPC` |
| `trends_fondos_mutuos` | `google_trends` | keyword: `fondos mutuos`, geo: `PE` |
| `trends_invertir` | `google_trends` | keyword: `invertir`, geo: `PE` |
| `is_ventana_quincena` | `quincena_binary` | quincena_days: `[1,15,16,28,29,30,31]` |
| `is_ventana_cts` | `calendar_window_binary` | May 1-15 y Nov 1-15 |
| `is_ventana_gratificacion` | `calendar_window_binary` | Jul 1-20 y Dic 1-20 |
| `dias_habiles_semana` | `working_days_count` | country: `PE` |

**Campos que NO tienen conector (manuales):**
- `tasa_bcrp` — BCRP no tiene API pública
- `is_ventana_afp` — ventanas extraordinarias del Congreso, no predecibles
- Variables internas del funnel (`usuarios_registro_base`, etc.)

---

## Paso 6 — Escribir el config completo del funnel

El config JSONB del funnel tiene 6 secciones. **Nunca usar SQL con `||` si el config es NULL** — usar Python directo o un UPDATE con valor completo.

```python
# Usar el cliente Supabase Python para escribir el config completo:
config = {
    "data_sources": [
        # Array de conectores (ver Paso 5)
        {"field": "pen_usd_var_semanal", "connector": "yfinance_weekly_pct", "params": {"ticker": "PEN=X"}},
        # ... resto de conectores
    ],
    "ingestion_groups": [
        # Grupos del formulario de ingreso de datos
        {
            "id": "funnel",          # IMPORTANTE: el grupo con vars internas DEBE llamarse "funnel"
            "title": "Funnel Interno",
            "description": "Datos del embudo de activacion de la semana",
            "auto_fetch": False,
            "fields": [
                {"key": "usuarios_registro_base", "label": "Usuarios Registro Base", "type": "integer"},
                # ... resto de campos
            ]
        },
        {
            "id": "referencia_manual",
            "title": "Referencia Manual",
            "auto_fetch": False,
            "fields": [
                {"key": "tasa_bcrp", "label": "Tasa BCRP (%)", "type": "number"},
                {"key": "is_ventana_afp", "label": "Ventana AFP? (0=No 1=Si)", "type": "integer"},
            ]
        },
        {
            "id": "mercado_macro",
            "title": "Variables Macro y Mercado",
            "description": "Se obtienen automaticamente con el boton auto-fetch",
            "auto_fetch": True,  # activa el botón ⚡ en el formulario
            "fields": [
                # campos auto-fetch — uno por conector del data_sources
            ]
        },
    ],
    "ml": {
        "model_storage": "models",   # nombre del bucket de Supabase Storage
        "model_version": 2,          # versión del modelo a usar
        "target_label": "Usuarios Primer Deposito",  # label en la UI
    },
    "derived_vars": [
        # Features calculadas por el pipeline (lags, tendencias, adstocks)
        # Se ocultan del SHAP display — no son accionables
        "lag_1_target", "tendencia_aprobados_4w", "tendencia_registro_4w",
        "tendencia_depositos_4w", "gratif_adstock_geom", "cts_adstock_geom",
    ],
    "feature_labels": {
        # Nombres legibles para el SHAP display
        "tasa_bcrp": "Tasa BCRP (%)",
        "bvl_var_semanal": "BVL Lima Var. Semanal (%)",
        # ... resto
    },
    "market": {
        "locale":   "es-PE",   # formato de números y fechas
        "currency": "PEN",
        "country":  "pe",
        # "rate_label": "Tasa BanRep (%)"  # solo si el funnel tiene conector tradingview_minus_manual
    },
    "validation_rules": {
        # REQUERIDO para Fase 2 (validate_and_send bloqueará con 422 si falta)
        "forbidden_sfc_terms": [
            "rentabilidad garantizada", "sin riesgo", "capital garantizado",
            "rendimiento asegurado", "inversion segura", "ganancias garantizadas",
            # ... adaptar por país (CO = SFC, PE = SMV/SBS)
        ],
        "market_patterns": [
            # Términos de mercado que SOLO se permiten en el paso de revisión backend
            "BVL", "BCRP", "PEN", "cobre", "spread"
            # CO usa: "COLCAP", "TRM", "BanRep"
        ],
        "char_limits": {
            "push_subject": 65,
            "push_cuerpo":  240,
            "email_subject": 80,
            "email_preheader": 120,
            "email_cuerpo":  5000
        },
        "revision_backend_steps": ["step_05_photo_validation"],
        # CO: ["befullusercreated", "photo_validation_completed"]
        "brand_fallback_name": "Trii",
        "voseo_check": {
            "enabled": True,   # TRUE para todos los funnels — CO y PE usan tuteo
            "pattern": "\\b(podés|tenés|invertís|abrís|empezás|hacés|querés|sabés|venís|entrás|completás|subís|mirás|buscás|encontrás|traés|traigás)\\b"
        }
    },
}

client.table('funnels').update({'config': config}).eq('id', funnel_id).execute()
```

### Estructura del SHAP display en la UI

El frontend filtra las features del SHAP en dos capas:
1. **INTERNAL_VARS** — fields del grupo con `id: "funnel"` → nunca accionables vía CIO
2. **DERIVED_VARS** — lista en `config.derived_vars` → calculadas por el pipeline, no son features del CSV

Lo que muestra el SHAP = todas las features del modelo MENOS INTERNAL_VARS MENOS DERIVED_VARS.

---

## Paso 7 — Verificar todo antes de lanzar

Correr el script de verificación:

```bash
python scripts-get-data/verify_setup.py  # o el equivalente para el funnel nuevo
```

Checklist:
- [ ] Storage: los 3 archivos del modelo existen y se descargan OK
- [ ] Config: `ml.model_storage`, `ml.model_version`, `ingestion_groups`, `derived_vars`, `market` presentes
- [ ] Carga del modelo: `_load_model_from_storage` retorna booster + meta + MAE sin error
- [ ] Tablas: `_master`, `_ultima_semana`, `_prediction_results` existen
- [ ] Funnel en tabla `funnels` con `is_active = true`

---

## Paso 8 — Flujo semanal operativo (domingo)

Una vez configurado, el flujo semanal es idéntico para todos los funnels:

```
1. Login → selector de funnel → elegir el funnel nuevo
2. /app/ingresar → "Nueva proyección" (archiva semana anterior → master)
3. Ingresar semana nueva (DD/MM/YYYY)
4. Click ⚡ "Obtener datos automáticos" → se llenan las 11 variables macro
5. Completar variables manuales (tasa_bcrp, is_ventana_afp, vars internas)
6. "Guardar y proyectar" → guarda en ultima_semana → llama POST /api/ml/predict
7. /app/predecir → muestra predicción + SHAP + prescripción
```

---

## Resumen de archivos relevantes

| Archivo | Rol |
|---|---|
| `app/services/supabase_client.py` | FunnelClient genérico, `update_ml_version()` |
| `app/services/ml_runner.py` | `_load_model_from_storage()`, `_upload_model_to_storage()` |
| `app/services/market_data_fetcher.py` | CONNECTOR_REGISTRY + `fetch_auto_variables()` |
| `app/routers/data.py` | Endpoints genéricos: auto-variables, ultima-semana, nueva-proyeccion |
| `app/routers/ml.py` | Endpoints genéricos: predict, train, training-status |
| `scripts/upload_model.py` | Subir modelo entrenado a Supabase Storage |
| `CONECTORES.md` | Documentación de todos los conectores disponibles |

---

## Errores comunes

| Error | Causa | Solución |
|---|---|---|
| Config vacío tras SQL | `config` era NULL — `NULL \|\| jsonb = NULL` en PostgreSQL | Usar Python directo o `COALESCE(config, '{}') \|\| jsonb` |
| SHAP muestra vars internas | Grupo del funnel no se llama `"funnel"` | Renombrar `id` del grupo a `"funnel"` en `ingestion_groups` |
| Storage 400 Bad Request | Bucket no existe | Crear bucket en Supabase Dashboard → Storage antes de subir |
| Modelo carga pero features no alinean | El master del funnel tiene nombres de columnas distintos al modelo | Revisar `ml_column_mapping` en config — mapea Supabase → pipeline |
| `Bucket not found` en predicción | `ml.model_storage` no está en el config del funnel | Correr script `set_{funnel}_config.py` con el config completo |
| Formulario de ingreso vacío | `ingestion_groups` no está en el config | Agregar la sección al config del funnel |
| Auto-fetch no llena campos | `data_sources` no está en el config | Agregar la sección al config del funnel |
| `validate_and_send` devuelve 422 "sin validation_rules" | `validation_rules` no está en el config del funnel | Agregar la sección completa al config (ver Paso 6) |
| Agente básico genera 0 acciones | Los journeys en CIO tienen 0 nodos de mensaje | Juanita debe agregar nodos email/push a los journeys en CIO antes de correr el agente |
| `get_master_df()` falla con "table not found" | La tabla master se llama `_master_consolidado_final` en vez de `_master` | El código lo resuelve automáticamente. Si persiste, crear SQL VIEW: `CREATE VIEW {prefix}_master AS SELECT * FROM {prefix}_master_consolidado_final` |
| Voseo no bloqueado en PE | `voseo_check.enabled` estaba en `false` | Actualizar a `true` con el mismo patrón de CO (ver Paso 6) |
| Perplexity 400 en generate_basic/premium | Endpoint `/v1/responses` no acepta formato `messages` | Ya corregido en `perplexity_client.py` — usa `/chat/completions` |

---

---

# Fase 2 — Agente de Estrategia (Customer.io)

> **Estado:** En construcción. Los pasos 9–15 aplican solo cuando el funnel necesita el agente de estrategia.
> El código de Fase 2 está implementado para `activacion_co`. Para un funnel nuevo, solo faltan los datos.

---

## Paso 9 — Crear campañas en Customer.io ✅ COMPLETADO (activacion_pe — 2026-06-20)

Customer.io es una sola cuenta por organización. Colombia y Perú comparten la misma instancia.

1. Crear cada campaña en el dashboard de CIO como "Journey"
2. Anotar el **Journey ID numérico** (visible en la URL: `.../journeys/4596/...`)
3. Anotar el **nombre exacto** del journey (se usa como clave de mapeo)

**Convención de nombres recomendada:** `Nombre | País`
- Ejemplo Colombia: `Primer depósito | Colombia` (id=4596)
- Ejemplo Perú: `Primer depósito | Perú` (id=4674)

**Peru — campañas creadas (draft, triggers y goals ya configurados):**

| ID CIO | Nombre | Tier | Paso del funnel |
|--------|--------|------|-----------------|
| 4670 | Datos básicos \| Perú | basic | step_00_registro → step_01_basic_data |
| 4671 | Datos completos \| Perú | basic | step_01_basic_data → step_03_data_validation |
| 4672 | Fotos KYC \| Perú | basic | step_03_data_validation → step_05_photo_validation |
| 4673 | Validation cuenta \| Perú | basic | step_05_photo_validation → step_07_full_user |
| 4674 | Primer depósito \| Perú | premium | step_07_full_user → step_08_becashin |

---

## Paso 10 — Registrar funnel_steps en Supabase ✅ COMPLETADO (activacion_pe — 2026-06-20)

La tabla `{org}_{funnel}_funnel_steps` define los pasos del funnel (no los nodos de cada journey).
Columnas: `step_order`, `step_code`, `step_name`, `entry_event`, `exit_event`, `benchmark_conversion_rate`.

Insertar via script Python (no SQL directo — evita problemas de encoding):

```python
# Ver scripts/seed_peru_fase2.py como referencia
steps = [
    {"step_order": 0, "step_code": "usuarios_registro_base", "step_name": "Registro", ...},
    {"step_order": 1, "step_code": "step_01_basic_data",      "step_name": "Datos básicos", ...},
    # ... un step por cada estado del funnel
]
client.table(f'{PREFIX}_funnel_steps').insert(steps).execute()
```

**Peru — 6 pasos registrados:**
`usuarios_registro_base` → `step_01_basic_data` → `step_03_data_validation` → `step_05_photo_validation` → `step_07_full_user` → `step_08_becashin`

**Nota:** También se deben crear 7 tablas de Fase 2 antes de este paso:
`campaigns_cache`, `strategy_results`, `knowledge_base`, `funnel_steps`, `node_update_log`, `user_campaign_assignments`, `bq_measurement_snapshots`.

SQL para `bq_measurement_snapshots` (necesaria para el panel de Medición):
```sql
CREATE TABLE IF NOT EXISTS {prefix}_bq_measurement_snapshots (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at    timestamptz NOT NULL DEFAULT now(),
  semana_label  text NOT NULL,
  inicio_semana text,
  fin_semana    text,
  model_version text NOT NULL DEFAULT '',
  html_content  text NOT NULL
);
```
Ver SQL completo de las otras 6 tablas en la sesión 2026-06-20.

---

## Paso 11 — Llenar la knowledge base ⬜ PENDIENTE (activacion_pe)

La KB contiene el contexto de negocio que el agente usa para generar copy.
Tabla: `{org}_{funnel}_knowledge_base`, columnas: `tipo`, `titulo`, `contenido`, `activo`.

- **Desde la UI:** `/app/estrategia` → sección Knowledge Base → agregar entradas
- **O directamente en Supabase:** insert con `tipo`, `titulo`, `contenido`, `activo=true`

**Tipos necesarios para Peru (en orden de importancia):**
- `PRODUCTO` — cada producto de trii PE con nombre exacto, monto mínimo, perfil objetivo
- `REGULATORIO` — marco SMV/SBS, Kallpa SAB como aliado regulado, disclaimer obligatorio
- `estrategia` — argumentos primer depósito por perfil (Conservador/Moderado/Arriesgado)
- `contexto` — montos mínimos, barreras de entrada, custodia CAVALI
- `MARCA` — voz y tono de trii (puede copiarse de Colombia)
- `GUIA_COPY` — guía técnica push/email CIO (puede copiarse de Colombia)
- `CAMPANAS` — descripción del funnel PE y sus 5 campañas

---

## Paso 12 — Crear los system prompts 🔄 EN PROGRESO (activacion_pe)

Los prompts viven en Supabase tabla `funnel_prompts`, NO en archivos locales.
Cada funnel necesita 5 filas por agente: `system`, `kb_preamble`, `user_template`, `perplexity_system`, `perplexity_query`.

**Proceso:**
1. Redactar el prompt en un `.txt` local para revisión (ej. `prompts_pe_draft.txt`)
2. Sembrar en Supabase via insert directo (Python + supabase_client)

**Estado Peru — todos los prompts SEMBRADOS ✅ (2026-06-20/21):**

| Prompt | Estado | Chars |
|--------|--------|-------|
| `[premium] system` | ✅ SEMBRADO | 37,120 |
| `[premium] kb_preamble` | ✅ SEMBRADO | 267 |
| `[premium] user_template` | ✅ SEMBRADO | 208 |
| `[premium] perplexity_system` | ✅ SEMBRADO | 1,359 |
| `[premium] perplexity_query` | ✅ SEMBRADO | 1,721 |
| `[basic] system` | ✅ SEMBRADO | 12,754 |
| `[basic] kb_preamble` | ✅ SEMBRADO | 306 |
| `[basic] user_template` | ✅ SEMBRADO | 147 |
| `[basic] perplexity_system` | ✅ SEMBRADO | 1,291 |
| `[basic] perplexity_query` | ✅ SEMBRADO | 1,843 |

Notas del prompt premium PE:
- Ciclo real Peru: W0=29.2%, W1=35.3% pico. Objetivo: comprimir a W0.
- Variables Peru: BVL/EPU, cobre, PEN/USD, BCRP, trends fondos mutuos/invertir
- Señales calendario: AFP=override máximo > gratificación (jul/dic) > CTS (may/nov) > quincena
- Compliance: SMV/SBS, Kallpa SAB (supervisada SMV), CAVALI (custodia). Sin crypto.
- step_code: `step_07_full_user`, campaign ID: 4674, prefijo: `PE_Kepler_PrimerDeposito_`

---

## Paso 12b — Agregar nodos email/push a los journeys en CIO ⬜ PENDIENTE (activacion_pe)

**Bloqueante para el agente básico.** El agente lee el copy actual de cada nodo en CIO vía `build_journey()`. Si los journeys existen en CIO pero no tienen nodos de mensaje, `build_journey()` retorna 0 mensajes y Claude no tiene nada que optimizar → devuelve 0 acciones.

**Quién lo hace:** Juanita (equipo CIO).

Para cada journey (4670-4673 para PE):
1. Abrir el journey en CIO Dashboard
2. Agregar al menos un nodo email o push
3. Configurar subject, preheader (emails), cuerpo
4. Guardar el journey en estado "activo" o "borrador"

Una vez que los journeys tengan nodos, correr el agente básico desde `/app/estrategia` y verificar que `acciones` sea > 0.

---

## Paso 13 — Sincronizar campaigns_cache ⬜ PENDIENTE (activacion_pe)

El agente lee las campañas desde una cache en Supabase, no directamente de CIO en cada request.

```bash
# Llamar el endpoint de sync (requiere el backend corriendo):
curl -X POST http://localhost:8000/api/strategy/sync \
  -H "X-Org-Slug: trii" \
  -H "X-Funnel-Slug: activacion_co"
```

Esto:
1. Llama a CIO para obtener las campañas activas
2. Descarga la estructura de cada journey (nodos, edges, triggers)
3. Guarda el resultado en `{org}_{funnel}_campaigns_cache`

Verificar que la tabla tenga filas después del sync.

---

## Paso 14 — Dry-run del agente premium ⬜ PENDIENTE

Con `CIO_DRY_RUN=true` en el `.env`, el agente genera la estrategia pero **no escribe nada en CIO**.

```bash
# En .env del backend:
CIO_DRY_RUN=true

# Desde la UI:
# /app/estrategia → seleccionar campaña → "Generar estrategia (Premium)"
# Revisar el preview: segmento, copy email/push, timing, variantes A/B
# Si todo se ve bien → pasar al Paso 15
```

Checklist del dry-run:
- [ ] El agente devuelve una estrategia sin errores 500
- [ ] El preview muestra copy coherente con el mercado y productos del funnel
- [ ] Los campos SHAP que el agente cita existen en la última predicción
- [ ] Los nombres de campaña en la estrategia coinciden con los de CIO

---

## Paso 15 — Habilitar escrituras reales en CIO ⬜ PENDIENTE

Solo cuando el dry-run está validado:

```bash
# En .env del backend:
CIO_DRY_RUN=false
```

Reiniciar el servidor. A partir de este momento, aprobar una estrategia en la UI **ejecuta cambios reales** en Customer.io.

**Checklist antes de activar:**
- [ ] Dry-run revisado y aprobado
- [ ] KB con al menos una entrada por categoría principal
- [ ] funnel_steps con todos los pasos de las campañas activas
- [ ] campaigns_cache sincronizado (Paso 13)
- [ ] system prompts sembrados (Paso 12)
- [ ] Juanita / equipo CIO notificado — Kepler puede modificar journeys

---

## Apéndice A — Scripts de seeding (ya corridos, solo para referencia)

Estos scripts se eliminaron del repo porque ya cumplieron su función. Se documentan aquí para saber cómo rehacerlos si se necesita para un funnel nuevo.

### seed_master_v2.py (Colombia)
Cargaba `master_consolidado_final_v2.csv` a la tabla `master_consolidado_final` en Supabase.
- Renombraba columnas (TRM → trm), convertía NaN → None, borraba todo y reinsertaba en batches de 100.
- Para un funnel nuevo: exportar el master desde BigQuery/Excel → mismo patrón con el nombre de tabla correcto `{org}_{funnel}_master`.

### seed_master_peru.py (Perú)
Mismo patrón que seed_master_v2.py pero para el Excel `master_consolidado_peru_full_v2.xlsx` → tabla `trii_activacion_pe_master_consolidado_final`.

### seed_prediction_results.py
Parseaba los archivos MD de `logs-historico-predicciones/` y los cargaba a la tabla `prediction_results`.
- Extraía `semana_label` del nombre del archivo (`proyeccion-6-12-202603.md` → "6 al 12 de marzo 2026").
- Solo relevante si se quiere reconstruir historial de predicciones desde archivos MD locales.

---

## Apéndice B — Scripts de análisis de features (ya corridos, solo para referencia)

Se eliminaron del repo. La documentación del proceso y decisiones finales está en `documentacion/FEATURE_SELECTION_PIPELINE.md`.

### feature_analysis.py (Colombia) y feature_analysis_peru.py (Perú)
Pipeline de 8 etapas para selección de features:
1. Near-zero variance — eliminar features sin varianza
2. VIF — detectar multicolinealidad entre features
3. PCA — explorar estructura latente
4. Mutual Information vs target
5. MRMR — máxima relevancia, mínima redundancia
6. Granger Causality — causalidad temporal (clave para lags)
7. Walk-forward CV stability — features estables en el tiempo
8. Tabla resumen de decisión

Para un funnel nuevo: correr el mismo pipeline sobre el master del nuevo país. Ver decisiones finales en `FEATURE_SELECTION_PIPELINE.md`.

### lag_analysis_peru.py
Analizó el ciclo real del usuario peruano (registro → primer depósito):
- Semana 0: 29.2%, Semana 1: 35.3% (pico), Semana 2: 12.5%, ... Semana 8: 1.9%
- Determinó que `full_users_aprobados` con lag 1 es el predictor clave de volumen.
- Para un funnel nuevo: entrevistar a producto para obtener la distribución real del ciclo usuario → correr el mismo análisis de correlación con distintos lags.
