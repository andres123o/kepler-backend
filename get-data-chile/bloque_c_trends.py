"""
Bloque C — Google Trends geo=CL (deposito a plazo, fondos mutuos).

Fuente : pytrends (Google Trends API no oficial)
Input  : chile/semanas_chile.csv
Output : chile/bloque_c_trends.csv

Variables:
  trends_deposito_plazo — interes relativo semanal "deposito a plazo" (DAP) en Chile
  trends_fondos_mutuos   — interes relativo semanal "fondos mutuos" en Chile

Estrategia identica a Peru: ventanas anuales para evitar muestreo mensual de
Google, re-escaladas por solapamiento de 4 semanas entre ventanas contiguas.
Pausa de 65s entre llamadas para evitar rate-limit 429.
"""

import time
import warnings
from datetime import date, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

KEYWORDS = ["deposito a plazo", "fondos mutuos"]
GEO = "CL"

WINDOWS = [
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-07-01"),
]

OVERLAP_WEEKS = 4
SLEEP_SEC     = 65


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def fetch_window(keyword: str, start: str, end: str, retries: int = 3) -> pd.Series:
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="es-CL", tz=240, timeout=(10, 30))

    for attempt in range(1, retries + 1):
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=f"{start} {end}", geo=GEO, gprop="")
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
    if not series_list:
        return pd.Series(dtype=float)

    result = series_list[0].copy()
    for nxt in series_list[1:]:
        if nxt.empty:
            continue
        common = result.index.intersection(nxt.index)
        if len(common) >= 2:
            nxt_mean = nxt.loc[common].mean()
            ref_mean = result.loc[common].mean()
            scale = ref_mean / nxt_mean if nxt_mean > 0 else 1.0
            new_idx = nxt.index.difference(result.index)
            scaled_new = nxt.loc[new_idx] * scale
        else:
            new_idx = nxt.index.difference(result.index)
            scaled_new = nxt.loc[new_idx]
        result = pd.concat([result, scaled_new]).sort_index()
    return result


def fetch_keyword(keyword: str) -> pd.Series:
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
    return rescale_and_join(all_windows)


def match_trend(series: pd.Series, monday: date):
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return None
    ts = pd.Timestamp(monday)
    diffs = abs(series.index - ts)
    idx_min = int(diffs.argmin())
    if diffs[idx_min] <= pd.Timedelta(days=7):
        v = series.iloc[idx_min]
        return float(v) if not pd.isna(v) else None
    return None


def calcular_bloque_c(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)
    mondays = [_parse_monday(s) for s in semanas["semana"]]

    keyword_series = {}
    for kw in KEYWORDS:
        col = "trends_" + kw.replace(" ", "_")
        print(f"\nDescargando Google Trends: '{kw}' (geo={GEO}) ...")
        s = fetch_keyword(kw)
        print(f"  Total semanas recuperadas: {s.notna().sum() if not s.empty else 0}")
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
    for col in ["trends_deposito_a_plazo", "trends_fondos_mutuos"]:
        if col in out.columns:
            print(f"   {col}: {out[col].notna().sum()} ok / {out[col].isna().sum()} NaN")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "chile", "semanas_chile.csv")
    output_csv = os.path.join(base, "chile", "bloque_c_trends.csv")

    df = calcular_bloque_c(input_csv, output_csv)
    print()
    print("--- Primeras 5 filas ---")
    print(df.head(5).to_string())
    print("--- Ultimas 5 filas ---")
    print(df.tail(5).to_string())
