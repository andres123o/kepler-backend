"""
Bloque A — Retornos semanales USD/CLP, IPSA (proxy), Cobre y S&P500.

Fuente : yfinance
Input  : chile/semanas_chile.csv
Output : chile/bloque_a_mercados.csv

Variables:
  usd_clp_var_semanal — % cambio semanal USD/CLP (positivo = peso se deprecia)
  ipsa_var_semanal     — % cambio semanal indice bursatil chileno
  cobre_var_semanal    — % cambio semanal Cobre COMEX (HG=F)
  sp500_var_semanal    — % cambio semanal S&P 500 (^GSPC)

Nota IPSA: ^IPSA no devuelve datos en yfinance (probado, vacio). Se usa ECH
(iShares MSCI Chile Capped ETF, NYSE, USD) como proxy — mismo criterio que
bvl_var_semanal en Peru (proxy EPU). Introduce ruido cambiario USD/CLP.

Logica comun: cierre diario -> resample W-FRI -> pct_change*100 -> shift viernes->lunes.
Fallback nearest +-3 dias cuando el lunes exacto no tiene dato (festivos, etc.).
"""

import warnings
from datetime import date, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

TICKERS = {
    "usd_clp_var_semanal": "CLP=X",
    "ipsa_var_semanal":    "ECH",     # proxy — ver nota arriba
    "cobre_var_semanal":   "HG=F",
    "sp500_var_semanal":   "^GSPC",
}


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def fetch_weekly_return(ticker: str, start: str, end: str) -> pd.Series:
    """Descarga ticker diario, resamplea a W-FRI, calcula retorno % semanal, shift a lunes."""
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
    returns.index = returns.index - pd.to_timedelta(4, unit="d")
    returns.name = ticker
    return returns


def match_value(series: pd.Series, monday: date):
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


def calcular_bloque_a(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)
    mondays = [_parse_monday(s) for s in semanas["semana"]]

    min_date = (min(mondays) - timedelta(days=14)).strftime("%Y-%m-%d")
    max_date = (max(mondays) + timedelta(days=8)).strftime("%Y-%m-%d")

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
    input_csv  = os.path.join(base, "chile", "semanas_chile.csv")
    output_csv = os.path.join(base, "chile", "bloque_a_mercados.csv")

    df = calcular_bloque_a(input_csv, output_csv)
    print()
    print("--- Primeras 5 filas ---")
    print(df.head(5).to_string())
    print("--- Ultimas 5 filas ---")
    print(df.tail(5).to_string())
