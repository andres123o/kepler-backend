# procesar_tes.py — genera kepler_spread_tes_banrep.csv
# Ejecutar desde kepler-backend/: python experimentacion/procesar_tes.py

import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ── 1. Leer TES 10Y diario (investing.com) ─────────────────────────────────────
tes_path = SCRIPT_DIR / 'tes10ãos.xlsx'
if not tes_path.exists():
    # fallback: buscar cualquier xlsx que empiece con 'tes'
    candidatos = list(SCRIPT_DIR.glob('tes*.xlsx'))
    if not candidatos:
        raise FileNotFoundError(f"No se encontro archivo TES en {SCRIPT_DIR}")
    tes_path = candidatos[0]

print(f"Leyendo: {tes_path.name}")
tes_df = pd.read_excel(tes_path)
tes_df.columns = ['fecha', 'tes10y', 'apertura', 'maximo', 'minimo', 'var_pct']
tes_df['fecha'] = pd.to_datetime(tes_df['fecha'], format='%d.%m.%Y')
tes_df = tes_df.set_index('fecha').sort_index()
tes_daily = tes_df['tes10y']
print(f"  {len(tes_daily)} dias: {tes_daily.index.min().date()} -> {tes_daily.index.max().date()}")
print(f"  Rango yield: {tes_daily.min():.3f}% -> {tes_daily.max():.3f}%")

# ── 2. Agregar a semanal ────────────────────────────────────────────────────────
# Tomar el ultimo valor disponible de cada semana (viernes o dia anterior si festivo)
# y mapearlo al lunes de esa semana (mismo criterio que fetch_market_data.py)
tes_weekly_fri = tes_daily.resample('W-FRI').last()
tes_weekly_mon = tes_weekly_fri.copy()
tes_weekly_mon.index = tes_weekly_fri.index - pd.Timedelta(days=4)
print(f"\nTES semanal (mapeado a lunes): {len(tes_weekly_mon)} semanas")

# ── 3. Leer master para obtener Tasa_Intervencion_Mensual ──────────────────────
master_path = SCRIPT_DIR / 'Master_Consolidado_Final.csv'
master = pd.read_csv(master_path, sep=None, engine='python')
master['semana_dt'] = pd.to_datetime(master['semana'], format='%d/%m/%Y', errors='coerce')
master = master.dropna(subset=['semana_dt']).set_index('semana_dt').sort_index()
banrep = master['Tasa_Intervencion_Mensual']
print(f"Master: {len(master)} semanas")
print(f"BanRep rate rango: {banrep.min():.2f}% -> {banrep.max():.2f}%")

# ── 4. Alinear TES al indice semanal del master ────────────────────────────────
# Paso 1: nearest match desde la serie semanal (tolerancia 7 dias)
tes_aligned = tes_weekly_mon.reindex(master.index, method='nearest',
                                      tolerance=pd.Timedelta('7D'))

# Paso 2: para los NaN restantes, buscar en la serie diaria (tolerancia 5 dias)
mask_nan = tes_aligned.isna()
if mask_nan.any():
    fill = tes_daily.reindex(master.index[mask_nan], method='nearest',
                              tolerance=pd.Timedelta('5D'))
    tes_aligned[mask_nan] = fill
    print(f"\nRellenados desde daily: {mask_nan.sum()} semanas")

nan_count = tes_aligned.isna().sum()
print(f"TES alineado - NaN: {nan_count} / {len(tes_aligned)}")
if nan_count > 0:
    print("  Semanas sin dato:", master.index[tes_aligned.isna()].strftime('%d/%m/%Y').tolist())

# ── 5. Calcular spread_tes_banrep ──────────────────────────────────────────────
# spread = TES_10Y (%) - Tasa_Intervencion_Mensual (%)
# Mide cuanto exige el mercado de largo plazo por encima de la politica monetaria
spread = tes_aligned - banrep

# ── 6. DataFrame final ────────────────────────────────────────────────────────
result = pd.DataFrame({
    'tes10y_pct':        tes_aligned.round(3),
    'banrep_rate_pct':   banrep.round(2),
    'spread_tes_banrep': spread.round(3),
}, index=master.index)

# Formato semana dd/mm/yyyy, orden descendente (mas reciente primero)
result.index = result.index.strftime('%d/%m/%Y')
result.index.name = 'semana'
result = result.sort_index(ascending=False,
                           key=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))

# ── 7. Validacion ─────────────────────────────────────────────────────────────
print("\nPrimeras 8 filas (mas recientes):")
print(result.head(8).to_string())
print("\nUltimas 5 filas (mas antiguas):")
print(result.tail(5).to_string())

s = result['spread_tes_banrep'].dropna()
print(f"\nSpread  min: {s.min():.3f}  max: {s.max():.3f}  media: {s.mean():.3f}  NaN: {result['spread_tes_banrep'].isna().sum()}")

# Spot check: semanas clave
spot = ['20/04/2026', '3/01/2022', '13/04/2020']
print("\nSpot check:")
for s_date in spot:
    if s_date in result.index:
        row = result.loc[s_date]
        print(f"  {s_date}: TES={row['tes10y_pct']}%  BanRep={row['banrep_rate_pct']}%  spread={row['spread_tes_banrep']}")

# ── 8. Guardar ────────────────────────────────────────────────────────────────
out_path = SCRIPT_DIR / 'kepler_spread_tes_banrep.csv'
result.to_csv(out_path)
print(f"\nGuardado: {out_path}")
print(f"Columnas: {list(result.columns)}")
print(f"Semanas:  {len(result)}")
