"""
Bloque D — Google Trends geo=PE (fondos mutuos, invertir).

Fuente : pytrends (Google Trends API no oficial)
Input  : primer_master_peru/semanas_peru.csv
Output : primer_master_peru/bloque_d_trends.csv

Variables:
  trends_fondos_mutuos — interes relativo semanal keyword "fondos mutuos" en Peru
  trends_invertir      — interes relativo semanal keyword "invertir" en Peru

Estrategia: ventanas anuales de 12 meses para evitar muestreo mensual de Google.
Cada ventana normaliza de 0-100; se re-escalan entre ventanas usando solapamiento de
4 semanas entre ventanas contiguas para preservar la escala relativa cross-periodo.

Si no hay solapamiento valido, se usa concatenacion directa (sin re-escalar).
Pausa de 60s entre llamadas para evitar rate-limit 429.
"""

import time
import warnings
from datetime import date, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

KEYWORDS = ["fondos mutuos", "invertir"]
GEO = "PE"

WINDOWS = [
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-07-01"),
]

OVERLAP_WEEKS = 4   # semanas de solapamiento para re-escalar
SLEEP_SEC     = 65  # pausa entre llamadas pytrends


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def fetch_window(keyword: str, start: str, end: str, retries: int = 3) -> pd.Series:
    """
    Descarga Google Trends para un keyword y ventana temporal.
    Retorna Serie indexada por fecha (weekly), valores 0-100.
    """
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="es-PE", tz=300, timeout=(10, 30))

    for attempt in range(1, retries + 1):
        try:
            pytrends.build_payload(
                [keyword],
                cat=0,
                timeframe=f"{start} {end}",
                geo=GEO,
                gprop="",
            )
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                print(f"    WARN: respuesta vacia para '{keyword}' {start}-{end}")
                return pd.Series(dtype=float)
            s = df[keyword].astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            return s
        except Exception as e:
            print(f"    Intento {attempt}/{retries} error: {e}")
            if attempt < retries:
                time.sleep(SLEEP_SEC)
    return pd.Series(dtype=float)


def rescale_and_join(series_list: list[pd.Series]) -> pd.Series:
    """
    Concatena ventanas anuales re-escalando por el solapamiento entre ventanas.
    Si no hay solapamiento, concatena directamente.
    """
    if not series_list:
        return pd.Series(dtype=float)

    result = series_list[0].copy()

    for nxt in series_list[1:]:
        if nxt.empty:
            continue
        # Semanas en comun entre result y nxt
        common = result.index.intersection(nxt.index)

        if len(common) >= 2:
            ref_vals = result.loc[common]
            nxt_vals = nxt.loc[common]
            # Factor de escala: media ref / media nxt (evitar division por cero)
            nxt_mean = nxt_vals.mean()
            ref_mean = ref_vals.mean()
            if nxt_mean > 0:
                scale = ref_mean / nxt_mean
            else:
                scale = 1.0
            # Aplicar escala solo a los indices nuevos de nxt
            new_idx = nxt.index.difference(result.index)
            scaled_new = nxt.loc[new_idx] * scale
        else:
            # Sin solapamiento — concatenar sin escalar
            new_idx = nxt.index.difference(result.index)
            scaled_new = nxt.loc[new_idx]

        result = pd.concat([result, scaled_new]).sort_index()

    return result


def fetch_keyword(keyword: str) -> pd.Series:
    """Descarga todas las ventanas para un keyword y las une."""
    all_windows = []
    for i, (start, end) in enumerate(WINDOWS):
        print(f"  Ventana {i+1}/{len(WINDOWS)}: {start} -> {end}")
        s = fetch_window(keyword, start, end)
        if not s.empty:
            print(f"    {len(s)} semanas, max={s.max():.0f}, mean={s.mean():.1f}")
        all_windows.append(s)
        if i < len(WINDOWS) - 1:
            print(f"    Pausa {SLEEP_SEC}s ...")
            time.sleep(SLEEP_SEC)

    combined = rescale_and_join(all_windows)
    return combined


def match_trend(series: pd.Series, monday: date) -> float | None:
    """Busca el valor de trends mas cercano al lunes dado (±7 dias)."""
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return None
    ts = pd.Timestamp(monday)
    diffs = abs(series.index - ts)
    idx_min = int(diffs.argmin())
    if diffs[idx_min] <= pd.Timedelta(days=7):
        v = series.iloc[idx_min]
        return float(v) if not pd.isna(v) else None
    return None


def calcular_bloque_d(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)
    mondays = [_parse_monday(s) for s in semanas["semana"]]

    keyword_series = {}
    for kw in KEYWORDS:
        col = "trends_" + kw.replace(" ", "_")
        print(f"\nDescargando Google Trends: '{kw}' ...")
        s = fetch_keyword(kw)
        n_ok = s.notna().sum() if not s.empty else 0
        print(f"  Total semanas recuperadas: {n_ok}")
        keyword_series[col] = s

    results = []
    for semana_str, monday in zip(semanas["semana"], mondays):
        row = {"semana": semana_str}
        for col, s in keyword_series.items():
            v = match_trend(s, monday)
            if v is None:
                print(f"  WARN: sin dato {col} para {semana_str}")
            row[col] = v
        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")

    print(f"\nOK Guardado: {output_csv} ({len(out)} filas)")
    for col in ["trends_fondos_mutuos", "trends_invertir"]:
        n_ok  = out[col].notna().sum()
        n_nan = out[col].isna().sum()
        print(f"   {col}: {n_ok} ok / {n_nan} NaN")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "primer_master_peru", "semanas_peru.csv")
    output_csv = os.path.join(base, "primer_master_peru", "bloque_d_trends.csv")

    df = calcular_bloque_d(input_csv, output_csv)
    print()
    print("--- Primeras 5 filas ---")
    print(df.head(5).to_string())
    print("--- Ultimas 5 filas ---")
    print(df.tail(5).to_string())
    print()
    nan_fm = df["trends_fondos_mutuos"].isna().sum()
    nan_inv = df["trends_invertir"].isna().sum()
    if nan_fm + nan_inv > 0:
        print(f"ATENCION: {nan_fm} NaN en fondos_mutuos, {nan_inv} NaN en invertir")
    else:
        print("Sin NaN en ninguna variable Trends.")
