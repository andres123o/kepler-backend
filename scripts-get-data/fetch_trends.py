# ============================================================
# kepler v2 — Google Trends: CDT + bolsa de valores
# Ejecutar desde kepler-backend/:
#   python experimentacion/fetch_trends.py
# Output: experimentacion/kepler_trends_variables.csv
#
# Keywords (seleccionados con datos reales - ver check_trends.py):
#   trends_cdt      → "CDT"              vol:48.4  CV:37.7%
#   trends_acciones → "bolsa de valores" vol:12.7  CV:28.8%
#   Correlacion entre ellos: -0.44 (risk-off vs risk-on)
#
# Estrategia rango largo (6 anos):
#   Intento 1: request directo 2020→2026 (pytrends devuelve semanal)
#   Intento 2: 2 ventanas con overlap largo (18 meses) + normalizacion robusta
# ============================================================

import time
import sys
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pytrends.request import TrendReq

sys.stdout.reconfigure(encoding='utf-8')

OUT_CSV = Path(__file__).parent / 'kepler_trends_variables.csv'

KEYWORDS = {
    'trends_cdt':      'CDT',
    'trends_acciones': 'bolsa de valores',
}

SEMANAS_RAW = """25/05/2026
18/05/2026
11/05/2026
4/05/2026
27/04/2026"""

semanas = pd.to_datetime(
    [s.strip() for s in SEMANAS_RAW.strip().split('\n')],
    format='%d/%m/%Y'
)
semanas_sorted = sorted(semanas)

TODAY = date.today().strftime('%Y-%m-%d')

pt = TrendReq(hl='es-CO', tz=300, timeout=(15, 45))


def fetch_range(keyword, timeframe, label='', retries=3):
    """Descarga una ventana de Google Trends. Retorna Serie semanal limpia.
    Reintenta hasta `retries` veces ante 429 con espera exponencial."""
    for attempt in range(1, retries + 1):
        try:
            pt.build_payload([keyword], geo='CO', timeframe=timeframe)
            df = pt.interest_over_time()
            if df.empty or keyword not in df.columns:
                print(f'    {label}: respuesta vacía (df.empty={df.empty})')
                return pd.Series(dtype=float)
            s = df[keyword].astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[s > 0]
            if label:
                print(f'    {label}: {len(s)} semanas  rango {s.min():.1f}–{s.max():.1f}')
            return s
        except Exception as e:
            exc_type = type(e).__name__
            msg = str(e)
            # Intentar extraer status code HTTP si el objeto tiene response
            status_code = None
            response_text = None
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                response_text = e.response.text[:300]
            elif hasattr(e, 'args') and e.args:
                # pytrends envuelve el error en args[0] a veces
                inner = str(e.args[0])
                if '429' in inner or '500' in inner or 'code' in inner.lower():
                    msg = inner

            is_429 = ('429' in msg or status_code == 429)

            print(f'    ERROR [{exc_type}] {label} (intento {attempt}/{retries}):')
            print(f'      mensaje  : {msg[:200]}')
            if status_code:
                print(f'      HTTP code: {status_code}')
            if response_text:
                print(f'      respuesta: {response_text}')

            if is_429 and attempt < retries:
                wait = 45 * attempt  # 45s, 90s
                print(f'      → 429 rate limit — esperando {wait}s antes de reintentar...')
                time.sleep(wait)
            else:
                if not is_429:
                    print(f'      → error no-429, traceback completo:')
                    traceback.print_exc()
                return pd.Series(dtype=float)
    return pd.Series(dtype=float)


def stitch_two(old_s, new_s, min_overlap=8):
    """
    Une dos series solapadas escalando la VIEJA para que su media en el overlap
    coincida con la media de la NUEVA. La nueva no se toca.
    Retorna serie combinada con la nueva como referencia de escala.
    """
    overlap = old_s.index.intersection(new_s.index)
    if len(overlap) < min_overlap:
        print(f'    Overlap insuficiente ({len(overlap)} semanas), concatenando sin escalar')
        combined = pd.concat([old_s, new_s])
        return combined[~combined.index.duplicated(keep='last')].sort_index()

    mean_old = old_s.loc[overlap].mean()
    mean_new = new_s.loc[overlap].mean()

    print(f'    Overlap: {len(overlap)} semanas | '
          f'media_vieja={mean_old:.2f}  media_nueva={mean_new:.2f}  '
          f'factor={mean_new/mean_old:.3f}')

    if mean_old < 1 or mean_new < 1:
        print('    AVISO: media muy baja en overlap — skipping scale para evitar explosion')
        scale = 1.0
    else:
        scale = mean_new / mean_old

    # Solo escalar la parte VIEJA (no la nueva)
    old_scaled = old_s * scale
    # Combinar: vieja escalada + nueva (nueva tiene prioridad en el overlap)
    combined = pd.concat([old_scaled, new_s])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined


def fetch_keyword(keyword, col_name):
    """
    Estrategia:
    1. Pedir rango directo 2020-2026 (pytrends lo retorna semanal)
    2. Si los valores del overlap se ven mal, usar 2 ventanas con stitch robusto
    """
    print(f'\n  Intento 1: request directo 2020-04-01 → {TODAY}')
    s_direct = fetch_range(keyword, f'2020-04-01 {TODAY}', 'directo')
    time.sleep(8)

    if len(s_direct) >= 200:
        print(f'    OK — usando serie directa ({len(s_direct)} semanas)')
        return s_direct

    # Fallback: 2 ventanas con overlap largo
    print(f'  Intento 2: 2 ventanas con 18 meses de overlap')
    # Ventana reciente: 2022-01-01 → hoy (la mas confiable, es la referencia)
    s_new = fetch_range(keyword, f'2022-01-01 {TODAY}', f'nueva(2022-{TODAY[:4]})')
    time.sleep(10)

    # Ventana antigua: 2020-04-01 → 2023-06-30 (overlap = ene2022-jun2023 = 18 meses)
    s_old = fetch_range(keyword, '2020-04-01 2023-06-30', 'vieja(2020-2023)')
    time.sleep(10)

    if s_new.empty and s_old.empty:
        return pd.Series(dtype=float, name=col_name)
    if s_new.empty:
        return s_old
    if s_old.empty:
        return s_new

    # Debug: mostrar valores del overlap antes de escalar
    overlap_idx = s_old.index.intersection(s_new.index)
    print(f'    Overlap: {overlap_idx.min().date()} → {overlap_idx.max().date()}')
    print(f'    Muestra overlap (vieja): {s_old.loc[overlap_idx].head(5).round(1).tolist()}')
    print(f'    Muestra overlap (nueva): {s_new.loc[overlap_idx].head(5).round(1).tolist()}')

    stitched = stitch_two(s_old, s_new, min_overlap=12)
    return stitched


# ─── PIPELINE ──────────────────────────────────────────────
result_df = pd.DataFrame()

for col_name, keyword in KEYWORDS.items():
    print(f'\n[{col_name}] keyword: "{keyword}"')
    s = fetch_keyword(keyword, col_name)

    if s.empty:
        print(f'  Sin datos')
        result_df[col_name] = np.nan
        continue

    # Clip valores fuera de rango razonable (proteccion ante scaling bugs)
    p99 = s.quantile(0.99)
    n_outliers = (s > p99 * 3).sum()
    if n_outliers > 0:
        print(f'  AVISO: {n_outliers} valores outlier clippeados (>{p99*3:.1f})')
        s = s.clip(upper=p99 * 3)

    result_df[col_name] = s
    print(f'  Serie final: {len(s)} semanas  '
          f'rango {s.min():.1f}–{s.max():.1f}  '
          f'media {s.mean():.1f}')
    time.sleep(30)  # pausa entre keywords para evitar 429

# ─── ALINEAR A SEMANAS DEL MODELO ─────────────────────────
print('\nAlineando a fechas del modelo...')
target_idx = pd.DatetimeIndex(semanas_sorted)
aligned = pd.DataFrame(index=target_idx)

for col in result_df.columns:
    s = result_df[col].dropna()
    if s.empty:
        aligned[col] = np.nan
        continue
    # Google Trends indexa al domingo — modelo usa lunes (+/- 4 dias)
    aligned[col] = s.reindex(target_idx, method='nearest',
                              tolerance=pd.Timedelta('4D'))
    mask = aligned[col].isna()
    if mask.any():
        fill = s.reindex(target_idx[mask], method='nearest',
                         tolerance=pd.Timedelta('8D'))
        aligned.loc[mask, col] = fill

# ─── FORMATO Y VALIDACION ──────────────────────────────────
aligned.index = aligned.index.strftime('%d/%m/%Y')
aligned.index.name = 'semana'
aligned = aligned.sort_index(
    ascending=False,
    key=lambda x: pd.to_datetime(x, format='%d/%m/%Y')
)

print('\nValidacion:')
print(f'{"Columna":<25} {"Min":>6} {"Max":>6} {"Media":>6} {"NaN":>5}')
print('-' * 50)
for col in aligned.columns:
    s = aligned[col].dropna()
    nan_n = aligned[col].isna().sum()
    if len(s):
        print(f'  {col:<23} {s.min():6.1f} {s.max():6.1f} {s.mean():6.1f} {nan_n:5}')
    else:
        print(f'  {col:<23}  TODO NaN')

print('\nPrimeras 10 filas:')
print(aligned.head(10).to_string())
print('\nMuestra 2022-2023 (zona de stitching):')
mask_2022 = aligned.index.str.contains('2022|2023')
print(aligned[mask_2022].head(10).to_string())
print('\nUltimas 5 filas:')
print(aligned.tail(5).to_string())

# ─── GUARDAR ───────────────────────────────────────────────
aligned.to_csv(OUT_CSV)
print(f'\nGuardado: {OUT_CSV}')
print(f'Semanas:  {len(aligned)}')
