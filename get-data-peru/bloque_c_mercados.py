"""
Bloque C — Retornos semanales BVL, Cobre y S&P500.

Fuente : yfinance
Input  : primer_master_peru/semanas_peru.csv
Output : primer_master_peru/bloque_c_mercados.csv

Variables:
  bvl_var_semanal    — % cambio semanal iShares MSCI All Peru ETF (EPU, proxy BVL)
  cobre_var_semanal  — % cambio semanal Cobre COMEX (HG=F)
  sp500_var_semanal  — % cambio semanal S&P 500 (^GSPC)

Nota BVL: ^SPBLPGPT no disponible en yfinance. EPU (iShares MSCI All Peru Capped ETF)
es el proxy liquido mas cercano — correlacion alta con el indice BVL.

Logica comun: cierre diario -> resample W-FRI -> pct_change*100 -> shift viernes->lunes.
Fallback nearest +-3 dias cuando el lunes exacto no tiene dato (festivos, etc.).
"""

import warnings
from datetime import date, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

TICKERS = {
    "bvl_var_semanal":   "EPU",    # iShares MSCI All Peru ETF — proxy BVL
    "cobre_var_semanal": "HG=F",
    "sp500_var_semanal": "^GSPC",
}


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def fetch_weekly_return(ticker: str, start: str, end: str) -> pd.Series:
    """
    Descarga ticker diario, resamplea a W-FRI, calcula retorno % semanal.
    Retorna Serie indexada por lunes (Timestamp).
    """
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance no devolvio datos para {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    weekly = close.resample("W-FRI").last()
    returns = (weekly.pct_change() * 100).round(4)

    # Mover indice viernes -> lunes (restar 4 dias)
    returns.index = returns.index - pd.to_timedelta(4, unit="d")
    returns.name = ticker
    return returns


def match_value(series: pd.Series, monday: date):
    """Busca el valor exacto para un lunes; fallback nearest +-3 dias."""
    if series.empty or not isinstance(series.index, pd.DatetimeIndex):
        return None
    ts = pd.Timestamp(monday)
    if ts in series.index:
        v = series.loc[ts]
        return float(v) if not pd.isna(v) else None
    diffs = abs(series.index - ts)
    idx_min = int(diffs.argmin())
    if diffs[idx_min] <= pd.Timedelta(days=3):
        v = series.iloc[idx_min]
        return float(v) if not pd.isna(v) else None
    return None


def calcular_bloque_c(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)
    mondays = [_parse_monday(s) for s in semanas["semana"]]

    min_date = (min(mondays) - timedelta(days=14)).strftime("%Y-%m-%d")
    max_date = (max(mondays) + timedelta(days=8)).strftime("%Y-%m-%d")

    # Descargar todas las series
    series_map = {}
    for col, ticker in TICKERS.items():
        print(f"Descargando {ticker} ({col}) ...")
        try:
            s = fetch_weekly_return(ticker, min_date, max_date)
            n_nan = int(s.isna().sum())
            print(f"  {len(s)} semanas, {n_nan} NaN")
            series_map[col] = s
        except Exception as e:
            print(f"  ERROR: {e}")
            series_map[col] = pd.Series(dtype=float)

    # Mapear a semanas del master
    results = []
    for semana_str, monday in zip(semanas["semana"], mondays):
        row = {"semana": semana_str}
        for col, s in series_map.items():
            row[col] = match_value(s, monday)
            if row[col] is None:
                print(f"  WARN: sin dato {col} para {semana_str}")
        results.append(row)

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")

    print(f"\nOK Guardado: {output_csv} ({len(out)} filas)")
    for col in TICKERS:
        n_ok  = out[col].notna().sum()
        n_nan = out[col].isna().sum()
        print(f"   {col}: {n_ok} ok / {n_nan} NaN")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "primer_master_peru", "semanas_peru.csv")
    output_csv = os.path.join(base, "primer_master_peru", "bloque_c_mercados.csv")

    df = calcular_bloque_c(input_csv, output_csv)
    print()
    print("--- Primeras 5 filas ---")
    print(df.head(5).to_string())
    print("--- Ultimas 5 filas ---")
    print(df.tail(5).to_string())
