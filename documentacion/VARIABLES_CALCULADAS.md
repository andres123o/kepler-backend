# Variables Calculadas — Kepler v2

Referencia completa de todas las variables nuevas del modelo v2.
Para cada una: qué es, cómo se calcula históricamente y cómo sacar el dato semanal nuevo.

**Estado (junio 2026):** El flujo del domingo está automatizado. En `/app/ingresar` → ingresar semana (DD/MM/YYYY) + tasa BanRep → botón ⚡ → el formulario se llena automáticamente. El script subyacente es `kepler-backend/app/services/market_data_fetcher.py`. Las secciones "Cada domingo para proyectar" abajo son referencia de respaldo en caso de error del auto-fetch.

---

## Variables ya calculadas

### 1. `sp500_cambio_semanal_pct`
**Qué mide:** Cambio porcentual semanal del S&P 500 (sentimiento del mercado global)

**Cálculo histórico:** Automático vía yfinance (`^GSPC`)
```
sp500_cambio_semanal_pct = (cierre_viernes / cierre_viernes_anterior - 1) × 100
```

**Cada domingo para proyectar:**
1. Ir a Google y buscar "S&P 500 semana"
2. O en investing.com → S&P 500 → Información histórica → última semana
3. El valor es el % de variación de esa semana (ej: `+1.24` o `-2.10`)

**Archivo generado:** `kepler_yfinance_variables.csv`

---

### 2. `brent_cambio_semanal_pct`
**Qué mide:** Cambio porcentual semanal del petróleo Brent (relevante para Colombia por Ecopetrol ~25% COLCAP)

**Cálculo histórico:** Automático vía yfinance (`BZ=F`)
```
brent_cambio_semanal_pct = (cierre_viernes / cierre_viernes_anterior - 1) × 100
```

**Cada domingo para proyectar:**
- Mismo proceso que S&P 500, buscar "Brent crudo variación semanal"
- O en investing.com → Petróleo Brent → Información histórica

**Archivo generado:** `kepler_yfinance_variables.csv`

---

### 3. `colcap_cambio_semanal_pct`
**Qué mide:** Cambio porcentual semanal del índice COLCAP (mercado accionario colombiano)

**Fuente:** ETF iShares COLCAP (`ICOLCAP.CL`) en Yahoo Finance. El índice real no expone histórico vía API.

**Cálculo histórico:** Automático vía yfinance (`ICOLCAP.CL`)
```
colcap_cambio_semanal_pct = (cierre_viernes / cierre_viernes_anterior - 1) × 100
```

**Cada domingo para proyectar:**
- investing.com → COLCAP → variación semanal
- O Google: "COLCAP variación semana"

**Archivo generado:** `kepler_yfinance_variables.csv`

---

### 4. `spread_tes_banrep` ⭐
**Qué mide:** Diferencia entre la tasa de largo plazo del mercado (TES 10 años) y la tasa de política monetaria del BanRep. Captura cuánto exige el mercado por encima de la política actual — señal de apetito de inversión y expectativas de tasas futuras.

**Por qué esta resta y no solo el TES:**
- TES solo sube mecánicamente cuando BanRep sube → no da información nueva
- La RESTA elimina ese efecto y captura la "prima de plazo" pura
- Spread alto (+3 o más): mercado optimista, apetito inversor alto → más primeros depósitos en Trii
- Spread bajo o negativo (curva invertida): mercado defensivo → menos inversión

**Cálculo:**
```
spread_tes_banrep = TES_10Y (%) − Tasa_Intervencion_Mensual (%)
```

**Ejemplo semana 20/04/2026:**
```
TES 10Y         = 12.682%
BanRep rate     = 11.25%
spread          = 12.682 − 11.25 = 1.432
```

**Cálculo histórico:**
- TES 10Y diario: descargado de investing.com → Colombia 10 años → Información histórica
  - Archivo fuente: `tes10años.xlsx`
  - Agregación: cierre del VIERNES de cada semana (último día de trading disponible)
  - Mapeado al LUNES de esa semana (consistente con el resto del modelo)
- Tasa_Intervencion_Mensual: ya está en el master (Supabase), ingresada manualmente cada semana
- Script que lo procesó: `procesar_tes.py`

**Archivo generado:** `kepler_spread_tes_banrep.csv`

**Cada domingo para proyectar la semana siguiente:**

1. **Obtener TES 10Y del viernes:**
   - Ir a: https://es.investing.com/rates-bonds/colombia-10-year-bond-yield
   - El número grande en pantalla es el yield actual (ej: `12.250`)
   - Usar ese valor directo en porcentaje (NO dividir entre 100)

2. **Obtener Tasa_Intervencion_Mensual:**
   - Ya la tienes porque la ingresas en el formulario semanal de Kepler
   - Si hubo reunión BanRep esa semana, usar la NUEVA tasa
   - Si no hubo reunión, repetir la tasa anterior

3. **Calcular:**
   ```
   spread_tes_banrep = TES_10Y_viernes − Tasa_Intervencion_Mensual
   ```

4. **Ejemplo semana del 1 al 7 de junio 2026:**
   ```
   TES 10Y viernes 5-jun-2026  = 12.250
   BanRep rate                 = 11.25  (sin cambio esa semana)
   spread_tes_banrep           = 12.250 − 11.25 = 1.000
   ```

---

## Variables en producción (auto-fetch)

### 5. `trends_cdt` ✅ AUTO-FETCH
**Qué mide:** Interés de búsqueda en Google para "CDT" en Colombia (intención de ahorro conservador)
**Fuente:** Google Trends vía pytrends — `market_data_fetcher.py`
**Nota:** pytrends introduce pausa de 30s anti-429. El endpoint `/api/data/auto-variables` tarda ~40-60s en total por esto.

### 6. `trends_acciones` ✅ AUTO-FETCH
**Qué mide:** Interés de búsqueda en Google para "acciones Colombia" (intención de inversión en equity)
**Fuente:** Google Trends vía pytrends — `market_data_fetcher.py`

---

## Variables evaluadas y descartadas del modelo v14

### `icc_fedesarrollo` — **No incluida en v14**
**Por qué:** Mensual con forward-fill semanal. El análisis de feature selection mostró señal débil para el objetivo de cashin semanal. La carga manual también genera fricción operativa.

### `inflacion_ipc` — **No incluida en v14**
**Por qué:** Variable mensual, muy lenta para capturar variaciones semanales de comportamiento. Correlación baja con el target en el análisis pre-model. Potencialmente útil para un lag de 4-8 semanas en futura versión.

---

## Resumen de archivos

| Archivo | Contenido |
|---|---|
| `kepler_yfinance_variables.csv` | sp500, brent, colcap (semanales con signo) |
| `kepler_spread_tes_banrep.csv` | tes10y_pct, banrep_rate_pct, spread_tes_banrep |
| `nuevas_variables_historicas.csv` | Todas las variables consolidadas (generado por fetch_variables.py) |
| `tes10años.xlsx` | Fuente TES diario descargada de investing.com |
| `Master_Consolidado_Final.csv` | Master histórico con todas las variables originales |

---

## Receta completa para el domingo de proyección

Cada domingo antes de correr el modelo, necesitas tener listos estos valores para la semana que empieza:

| Variable | Dónde conseguirla | Tipo |
|---|---|---|
| `sp500_cambio_semanal_pct` | Yahoo Finance / investing.com | % con signo |
| `brent_cambio_semanal_pct` | Yahoo Finance / investing.com | % con signo |
| `colcap_cambio_semanal_pct` | BVC / investing.com | % con signo |
| `TES 10Y` (para el spread) | investing.com → Colombia 10 años | % absoluto |
| `Tasa_Intervencion_Mensual` | Ya la tienes — formulario Kepler | % absoluto |
| `spread_tes_banrep` | TES 10Y − Tasa_Intervencion_Mensual | Calculado |
| `trends_cdt` | Google Trends (script automático) | Índice 0-100 |
| `trends_acciones` | Google Trends (script automático) | Índice 0-100 |
| `icc_fedesarrollo` | Fedesarrollo (mensual, repetir si no hay nuevo) | Número |
| `inflacion_ipc` | DANE (mensual, repetir si no hay nuevo) | % mensual |
