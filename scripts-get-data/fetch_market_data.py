# ============================================================
# keplerv4 — Variables yfinance (S&P500 + Brent + COLCAP)
# Ejecutar desde kepler-backend/:
#   python experimentacion/fetch_market_data.py
# Output: experimentacion/kepler_yfinance_variables.csv
# Formato: porcentaje con signo (-3.2, 1.5)
# COLCAP via ICOLCAP.CL (ETF iShares BVC, COP) — el índice real no expone histórico por API
# ============================================================

from pathlib import Path
OUT_CSV = Path(__file__).parent / 'kepler_yfinance_variables.csv'

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── SEMANAS DEL MODELO ────────────────────────────────────
SEMANAS_RAW = """
25/05/2026
18/05/2026
11/05/2026
4/05/2026
27/04/2026
"""

# ─── PARSEAR FECHAS ────────────────────────────────────────
semanas = pd.to_datetime(
    [s.strip() for s in SEMANAS_RAW.strip().split('\n')],
    format='%d/%m/%Y'
)
semanas_sorted = sorted(semanas)
start_dl = (min(semanas_sorted) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
end_dl   = (max(semanas_sorted) + pd.Timedelta(days=8)).strftime('%Y-%m-%d')

print(f"Período descarga: {start_dl} → {end_dl}")
print(f"Semanas: {len(semanas)}")

# ─── DESCARGAR ─────────────────────────────────────────────
# En yfinance >= 0.2.x, ['Close'] devuelve un DataFrame con el ticker
# como columna (no un Series). safe_close() normaliza siempre a Series.
def safe_close(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=start_dl, end=end_dl,
                      progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw['Close']
    # MultiIndex / DataFrame → extraer primera columna → Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()

print("\nDescargando S&P500, Brent y COLCAP...")

sp500 = safe_close('^GSPC')
brent = safe_close('BZ=F')

# COLCAP: ICOLCAP.CL — ETF iShares MSCI COLCAP, BVC, COP. Replica el índice en COP sin ruido divisa.
# El índice real (^737809-COP-STRD) solo tiene cotización en tiempo real, sin histórico vía API.
colcap = safe_close('ICOLCAP.CL')

print(f"S&P500:  {len(sp500)} días descargados")
print(f"Brent:   {len(brent)} días descargados")
print(f"COLCAP:  {len(colcap.dropna())} días con datos")

# ─── CALCULAR RETORNO SEMANAL ──────────────────────────────
# Lógica: cierre del viernes / cierre del viernes anterior - 1
# Resultado en PORCENTAJE con signo: -3.24, 1.57 (no decimal)

def weekly_return_pct(daily_series: pd.Series) -> pd.Series:
    """
    Resamplea a viernes y calcula pct_change * 100.
    Resultado: porcentaje con signo, 2 decimales.
    Ejemplo: -3.24 significa -3.24%
    """
    # Asegurar índice sin timezone
    s = daily_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)

    # Cierre semanal = último precio disponible hasta el viernes
    weekly = s.resample('W-FRI').last()

    # Retorno porcentual respecto al viernes anterior
    ret = weekly.pct_change() * 100
    return ret.round(2)


sp500_weekly  = weekly_return_pct(sp500)
brent_weekly  = weekly_return_pct(brent)
colcap_weekly = weekly_return_pct(colcap)

# ─── MAPEAR VIERNES → LUNES ────────────────────────────────
# Cada viernes pertenece a la semana que empezó el lunes anterior (4 días antes)
def shift_to_monday(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index - pd.to_timedelta(4, unit='d')
    s.index = s.index.strftime('%d/%m/%Y')
    return s

sp500_weekly  = shift_to_monday(sp500_weekly)
brent_weekly  = shift_to_monday(brent_weekly)
colcap_weekly = shift_to_monday(colcap_weekly)

# ─── FILTRAR Y ORDENAR ─────────────────────────────────────
semanas_str = [s.strftime('%d/%m/%Y') for s in semanas]

# Construir DataFrame columna por columna para evitar problemas de índice
df = pd.DataFrame(index=semanas_str)
df.index.name = 'semana'
df['sp500_cambio_semanal_pct']  = sp500_weekly.reindex(semanas_str)
df['brent_cambio_semanal_pct']  = brent_weekly.reindex(semanas_str)
df['colcap_cambio_semanal_pct'] = colcap_weekly.reindex(semanas_str)

# Orden descendente (más reciente primero)
df = df.sort_index(
    ascending=False,
    key=lambda x: pd.to_datetime(x, format='%d/%m/%Y')
)

# ─── VALIDACIÓN ────────────────────────────────────────────
print("\nValidación de rangos:")
for col in df.columns:
    s = df[col]
    print(f"  {col:<35}  min: {s.min():+.2f}%  max: {s.max():+.2f}%  NaN: {s.isna().sum()}")

print("\nPrimeras 5 filas:")
print(df.head(5).to_string())

# ─── EXPORTAR ──────────────────────────────────────────────
df.to_csv(OUT_CSV)
print(f"\n✅ Archivo guardado en: {OUT_CSV}")
print(f"   Columnas: {list(df.columns)}")
print(f"   Semanas:  {len(df)}")