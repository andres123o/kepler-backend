# Kepler — Motor de Conectores Auto-fetch

## Arquitectura (2 niveles)

```
Level 1 — Código Python          Level 2 — Supabase JSONB
─────────────────────────────    ─────────────────────────────────────────
market_data_fetcher.py           funnels.config → "data_sources": [ ... ]
  CONNECTOR_REGISTRY             
  ├── yfinance_weekly_pct        { "field": "sp500_var_semanal",
  ├── yfinance_rolling_std         "connector": "yfinance_weekly_pct",
  ├── tradingview_perf_w           "params": { "ticker": "^GSPC" } }
  ├── tradingview_minus_manual   
  ├── socrata_daily_avg          Agregar un funnel = insertar su array
  ├── google_trends              data_sources en Supabase.
  ├── quincena_fraction          Cero cambios de código Python.
  ├── quincena_binary            
  ├── calendar_window_binary     
  ├── working_days_count         
  └── ...                        
```

**Principio:** Los conectores son genéricos (leen cualquier ticker, cualquier keyword, cualquier país). Los parámetros específicos de cada funnel viven en Supabase. Para agregar un funnel nuevo, solo se edita el config JSONB.

---

## Conectores disponibles

### `yfinance_weekly_pct`
**Qué devuelve:** % cambio viernes→viernes para la semana de `monday`.

| Param | Tipo | Descripción |
|---|---|---|
| `ticker` | string | Símbolo yfinance (ej. `"^GSPC"`, `"BZ=F"`, `"PEN=X"`) |

**Lógica:** Descarga datos diarios via yfinance → resamplea a semanal (último viernes) → calcula pct_change → mapea al lunes de la semana.

**Ejemplos de uso por funnel:**
```json
{ "field": "sp500_var_semanal",   "connector": "yfinance_weekly_pct", "params": {"ticker": "^GSPC"} }
{ "field": "brent_var_semanal",   "connector": "yfinance_weekly_pct", "params": {"ticker": "BZ=F"} }
{ "field": "pen_usd_var_semanal", "connector": "yfinance_weekly_pct", "params": {"ticker": "PEN=X"} }
{ "field": "bvl_var_semanal",     "connector": "yfinance_weekly_pct", "params": {"ticker": "EPU"} }         // EPU = iShares MSCI Peru ETF; ^SPBLPGPT y ^IGBVL estan delisted en yfinance
{ "field": "cobre_var_semanal",   "connector": "yfinance_weekly_pct", "params": {"ticker": "HG=F"} }
{ "field": "colcap_cambio_semanal_pct", "connector": "yfinance_weekly_pct", "params": {"ticker": "^SPBLPGPT"} }
```

---

### `yfinance_rolling_std`
**Qué devuelve:** Desviación estándar de los últimos N retornos semanales de un ticker. Mide volatilidad reciente.

| Param | Tipo | Default | Descripción |
|---|---|---|---|
| `ticker` | string | requerido | Símbolo yfinance |
| `weeks` | int | `4` | Número de semanas para el rolling window |

**Lógica:** Mismo pipeline que `yfinance_weekly_pct` → toma las últimas `weeks` observaciones → `std()`.

**Ejemplos de uso:**
```json
{ "field": "pen_usd_volatilidad_4w", "connector": "yfinance_rolling_std", "params": {"ticker": "PEN=X", "weeks": 4} }
```

---

### `tradingview_perf_w`
**Qué devuelve:** % rendimiento semanal desde el snapshot actual de TradingView (campo `Perf.W`).

| Param | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Símbolo TradingView (ej. `"BVC:ICOLCAP"`) |

**Nota:** Devuelve el valor del momento del fetch, no histórico. Útil para índices que yfinance no tiene.

**Ejemplos de uso:**
```json
{ "field": "colcap_cambio_semanal_pct", "connector": "tradingview_perf_w", "params": {"symbol": "BVC:ICOLCAP"} }
```

---

### `tradingview_minus_manual`
**Qué devuelve:** `valor_TV − parámetro_manual`. Diseñado para calcular spreads (ej. TES − BanRep).

| Param | Tipo | Descripción |
|---|---|---|
| `symbol` | string | Símbolo TradingView |
| `tv_field` | string | Campo del snapshot (default: `"close"`) |
| `manual_param` | string | Clave del dict `manual_params` pasado al fetch |

**Lógica:** El usuario ingresa la tasa manual en el query string del endpoint (`?banrep_tasa=11.25`). El conector la resta al valor que trae TradingView.

**Ejemplos de uso:**
```json
{
  "field": "spread_tes_banrep",
  "connector": "tradingview_minus_manual",
  "params": {"symbol": "TVC:CO10Y", "tv_field": "close", "manual_param": "banrep_tasa"}
}
```

---

### `socrata_daily_avg`
**Qué devuelve:** Promedio semanal (lun-dom) de un campo en una API Socrata con vigencia por día.

| Param | Tipo | Descripción |
|---|---|---|
| `url` | string | Endpoint Socrata |
| `value_field` | string | Campo del valor numérico (ej. `"valor"`) |
| `date_from_field` | string | Campo inicio de vigencia (ej. `"vigenciadesde"`) |
| `date_to_field` | string | Campo fin de vigencia (ej. `"vigenciahasta"`) |

**Lógica:** Consulta diaria para cada día de la semana → promedia los que responden. Maneja fines de semana (reutiliza viernes) y festivos (busca hasta 5 días atrás).

**Ejemplos de uso (Colombia TRM):**
```json
{
  "field": "trm",
  "connector": "socrata_daily_avg",
  "params": {
    "url": "https://www.datos.gov.co/resource/mcec-87by.json",
    "value_field": "valor",
    "date_from_field": "vigenciadesde",
    "date_to_field": "vigenciahasta"
  }
}
```

---

### `google_trends`
**Qué devuelve:** Índice Google Trends (0–100) para la semana que contiene `monday`.

| Param | Tipo | Default | Descripción |
|---|---|---|---|
| `keyword` | string | requerido | Término de búsqueda |
| `geo` | string | `"CO"` | Código de país |
| `hl` | string | `"es-CO"` | Locale |

**Notas:** Incluye pausa automática de 30s entre dos llamadas consecutivas a Trends (anti-429).

**Ejemplos de uso:**
```json
{ "field": "trends_cdt",          "connector": "google_trends", "params": {"keyword": "CDT",          "geo": "CO", "hl": "es-CO"} }
{ "field": "trends_acciones",     "connector": "google_trends", "params": {"keyword": "acciones",     "geo": "CO", "hl": "es-CO"} }
{ "field": "trends_fondos_mutuos","connector": "google_trends", "params": {"keyword": "fondos mutuos","geo": "PE", "hl": "es-PE"} }
{ "field": "trends_invertir",     "connector": "google_trends", "params": {"keyword": "invertir",     "geo": "PE", "hl": "es-PE"} }
```

---

### `quincena_fraction`
**Qué devuelve:** Fracción (0.0–1.0) de los 7 días de la semana que caen en ventana post-quincena. Preserva la intensidad de la señal (vs binaria).

| Param | Tipo | Default | Descripción |
|---|---|---|---|
| `quincena_days` | array[int] | `[1,2,3,15,16,17,28,29,30]` | Días del mes que pertenecen a la ventana |

**Ejemplos de uso (Colombia):**
```json
{ "field": "pct_quincena", "connector": "quincena_fraction", "params": {"quincena_days": [1,2,3,15,16,17,28,29,30]} }
```

---

### `quincena_binary`
**Qué devuelve:** `1.0` si algún día de la semana lun-dom cae en días de quincena, `0.0` si no.

| Param | Tipo | Default | Descripción |
|---|---|---|---|
| `quincena_days` | array[int] | `[1,2,3,15,16,17,28,29,30]` | Días del mes que pertenecen a la ventana |

**Cuándo usar `quincena_binary` vs `quincena_fraction`:** Si el modelo fue entrenado con variable 0/1, usar `quincena_binary`. Si fue entrenado con fracción continua, usar `quincena_fraction`.

**Ejemplos de uso (Perú):**
```json
{ "field": "is_ventana_quincena", "connector": "quincena_binary", "params": {"quincena_days": [1,2,3,15,16,17,28,29,30]} }
```

---

### `calendar_window_binary`
**Qué devuelve:** `1.0` si algún día de la semana cae dentro de alguna ventana calendario definida, `0.0` si no. Diseñado para eventos de nómina predecibles con fechas fijas por ley.

| Param | Tipo | Descripción |
|---|---|---|
| `windows` | array | Lista de `{"month": int, "day_start": int, "day_end": int}` |

**Eventos peruanos cubiertos:**
- **CTS:** depósitos May 1–15 y Nov 1–15 (Ley 25129)
- **Gratificación:** depósitos Jul 1–20 y Dic 1–20 (Ley 27735)

**Ejemplos de uso (Perú):**
```json
{
  "field": "is_ventana_cts",
  "connector": "calendar_window_binary",
  "params": { "windows": [{"month": 5, "day_start": 1, "day_end": 15}, {"month": 11, "day_start": 1, "day_end": 15}] }
}

{
  "field": "is_ventana_gratificacion",
  "connector": "calendar_window_binary",
  "params": { "windows": [{"month": 7, "day_start": 1, "day_end": 20}, {"month": 12, "day_start": 1, "day_end": 20}] }
}
```

---

### `working_days_count`
**Qué devuelve:** Número de días hábiles (lun-vie) en la semana descontando festivos nacionales. Retorna float (1.0–5.0).

| Param | Tipo | Default | Descripción |
|---|---|---|---|
| `country` | string | `"CO"` | Código ISO del país. Usa librería `holidays` (100+ países soportados) |

**Dependencia:** `pip install holidays>=0.59`

**Ejemplos de uso:**
```json
{ "field": "dias_habiles_semana", "connector": "working_days_count", "params": {"country": "PE"} }
{ "field": "dias_habiles_semana", "connector": "working_days_count", "params": {"country": "CO"} }
```

---

## Campos manuales — no automatizables

Estos campos no tienen conector porque requieren input humano o son extraordinarios:

| Campo | Razón |
|---|---|
| `banrep_tasa` / `tasa_bcrp` | Sin API pública. BanRep y BCRP no exponen tasa vigente en tiempo real. Input manual en UI. |
| `is_ventana_afp` (Perú) | Las ventanas AFP son medidas extraordinarias aprobadas por el Congreso peruano sin calendario fijo. No predecibles algorítmicamente. Input manual. |
| Variables de funnel interno | `usuarios_registro_base`, `tasa_*`, `full_users_aprobados` — datos internos de Trii que no están en ninguna API pública. |

---

## Cómo agregar un funnel nuevo

1. Insertar fila en `funnels` con `config JSONB` que incluya el array `data_sources`
2. Cada elemento del array: `{"field": "nombre_columna", "connector": "tipo", "params": {...}}`
3. El endpoint `GET /api/data/auto-variables` ya lee el array dinámicamente — **cero código Python nuevo**
4. Si el campo requiere un `manual_param` (como `banrep_tasa`), el frontend lo pasa como query param al endpoint

## Cómo agregar un conector nuevo

1. Agregar función `_run_nombre_conector(monday, params, manual_params) → float | None` en `market_data_fetcher.py`
2. Registrarla en `CONNECTOR_REGISTRY`
3. Documentarla en este archivo

---

## Configuración actual por funnel

### `trii/activacion_co` (Colombia)

| Campo auto | Conector | Params clave |
|---|---|---|
| `trm` | `socrata_daily_avg` | API datos.gov.co |
| `colcap_cambio_semanal_pct` | `tradingview_perf_w` | `BVC:ICOLCAP` |
| `sp500_var_semanal` | `yfinance_weekly_pct` | `^GSPC` |
| `brent_var_semanal` | `yfinance_weekly_pct` | `BZ=F` |
| `spread_tes_banrep` | `tradingview_minus_manual` | `TVC:CO10Y` − `banrep_tasa` |
| `trends_cdt` | `google_trends` | `"CDT"` geo=CO |
| `trends_acciones` | `google_trends` | `"acciones"` geo=CO |
| `pct_quincena` | `quincena_fraction` | days=[1,2,3,15,16,17,28,29,30] |

| Campo manual | Descripción |
|---|---|
| `banrep_tasa` | Tasa BanRep vigente (%) — input en UI |
| `pct_dias_festivos` | Calculado via `working_days_count` "CO" (o manual) |

### `trii/activacion_pe` (Perú)

| Campo auto | Conector | Params clave |
|---|---|---|
| `pen_usd_var_semanal` | `yfinance_weekly_pct` | `PEN=X` |
| `pen_usd_volatilidad_4w` | `yfinance_rolling_std` | `PEN=X`, weeks=4 |
| `bvl_var_semanal` | `yfinance_weekly_pct` | `EPU` (iShares MSCI Peru ETF — `^SPBLPGPT` delisted) |
| `cobre_var_semanal` | `yfinance_weekly_pct` | `HG=F` |
| `sp500_var_semanal` | `yfinance_weekly_pct` | `^GSPC` |
| `trends_fondos_mutuos` | `google_trends` | `"fondos mutuos"` geo=PE |
| `trends_invertir` | `google_trends` | `"invertir"` geo=PE |
| `is_ventana_quincena` | `quincena_binary` | days=[1,2,3,15,16,17,28,29,30] |
| `is_ventana_cts` | `calendar_window_binary` | May 1-15, Nov 1-15 |
| `is_ventana_gratificacion` | `calendar_window_binary` | Jul 1-20, Dic 1-20 |
| `dias_habiles_semana` | `working_days_count` | country="PE" |

| Campo manual | Descripción |
|---|---|
| `tasa_bcrp` | Tasa BCRP vigente (%) — input en UI |
| `is_ventana_afp` | Ventana extraordinaria AFP — no predecible, checkbox en UI |
| Variables internas | `usuarios_registro_base`, `tasa_fulldata_a_video`, `tasa_review_a_aprobado`, `full_users_aprobados` |
