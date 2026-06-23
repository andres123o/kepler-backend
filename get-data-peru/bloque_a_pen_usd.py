"""
Bloque A — PEN/USD semanal y volatilidad 4 semanas.

Fuente : yfinance ticker PEN=X (USD/PEN tipo de cambio diario)
Input  : primer_master_peru/semanas_peru.csv
Output : primer_master_peru/bloque_a_pen_usd.csv

Variables:
  pen_usd_var_semanal     — % cambio semanal PEN/USD (viernes→viernes)
                            positivo = sol se deprecia (1 USD compra mas PEN)
  pen_usd_volatilidad_4w  — std de los ultimos 4 retornos semanales
                            mide incertidumbre cambiaria
"""

import warnings
from datetime import date, timedelta
import pandas as pd

warnings.filterwarnings("ignore")

TICKER = "PEN=X"


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def fetch_pen_usd(start: str, end: str) -> pd.DataFrame:
    """
    Descarga PEN=X diario, resamplea a W-FRI, calcula retorno % y vol 4w.
    Retorna DataFrame indexado por lunes (Timestamp) con las dos columnas.
    """
    import yfinance as yf

    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"yfinance no devolvio datos para {TICKER} ({start} - {end})")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)

    # Semanal cierre viernes, retorno % viernes-a-viernes
    weekly = close.resample("W-FRI").last()
    returns = (weekly.pct_change() * 100).round(4)
    vol4w = returns.rolling(4).std().round(4)

    # Mover indice de viernes -> lunes de esa semana
    df = pd.DataFrame(
        {"pen_usd_var_semanal": returns, "pen_usd_volatilidad_4w": vol4w}
    )
    df.index = df.index - pd.to_timedelta(4, unit="d")
    df.index.name = "monday"
    return df


def calcular_bloque_a(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)
    mondays = [_parse_monday(s) for s in semanas["semana"]]

    min_date = min(mondays) - timedelta(days=30)   # margen para calcular vol4w
    max_date = max(mondays) + timedelta(days=8)

    print(f"Descargando {TICKER} {min_date} -> {max_date} ...")
    series = fetch_pen_usd(min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d"))
    print(f"  Semanas descargadas: {len(series)} | NaN retorno: {series['pen_usd_var_semanal'].isna().sum()}")

    results = []
    for semana_str, monday in zip(semanas["semana"], mondays):
        monday_ts = pd.Timestamp(monday)

        # Buscar exacto primero, luego nearest +-3 dias
        if monday_ts in series.index:
            row = series.loc[monday_ts]
            var_val = float(row["pen_usd_var_semanal"]) if not pd.isna(row["pen_usd_var_semanal"]) else None
            vol_val = float(row["pen_usd_volatilidad_4w"]) if not pd.isna(row["pen_usd_volatilidad_4w"]) else None
        else:
            diffs = abs(series.index - monday_ts)
            idx_min = int(diffs.argmin())
            if diffs[idx_min] <= pd.Timedelta(days=3):
                row = series.iloc[idx_min]
                var_val = float(row["pen_usd_var_semanal"]) if not pd.isna(row["pen_usd_var_semanal"]) else None
                vol_val = float(row["pen_usd_volatilidad_4w"]) if not pd.isna(row["pen_usd_volatilidad_4w"]) else None
            else:
                var_val, vol_val = None, None
                print(f"  WARN: sin datos para {semana_str} (lunes {monday})")

        results.append({
            "semana": semana_str,
            "pen_usd_var_semanal": var_val,
            "pen_usd_volatilidad_4w": vol_val,
        })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")

    n_ok  = out["pen_usd_var_semanal"].notna().sum()
    n_nan = out["pen_usd_var_semanal"].isna().sum()
    print(f"OK Guardado: {output_csv}")
    print(f"   pen_usd_var_semanal:    {n_ok} ok / {n_nan} NaN")
    n_ok2  = out["pen_usd_volatilidad_4w"].notna().sum()
    n_nan2 = out["pen_usd_volatilidad_4w"].isna().sum()
    print(f"   pen_usd_volatilidad_4w: {n_ok2} ok / {n_nan2} NaN")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "primer_master_peru", "semanas_peru.csv")
    output_csv = os.path.join(base, "primer_master_peru", "bloque_a_pen_usd.csv")

    df = calcular_bloque_a(input_csv, output_csv)
    print()
    print("--- Primeras 5 filas ---")
    print(df.head(5).to_string())
    print("--- Ultimas 5 filas ---")
    print(df.tail(5).to_string())
    print()
    # Semanas con NaN (si hay)
    nan_rows = df[df["pen_usd_var_semanal"].isna()]
    if len(nan_rows):
        print("Semanas sin dato:", nan_rows["semana"].tolist())
    else:
        print("Sin NaN en pen_usd_var_semanal.")
    nan_rows2 = df[df["pen_usd_volatilidad_4w"].isna()]
    if len(nan_rows2):
        print("Semanas sin vol4w:", nan_rows2["semana"].tolist())
