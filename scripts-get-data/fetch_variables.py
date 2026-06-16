"""
fetch_variables.py — Fetcher de las 7 nuevas variables macro para Kepler v2.

Qué hace:
  1. Lee las semanas de master_consolidado_final (eje temporal)
  2. Descarga S&P500 y Brent via yfinance (automático)
  3. Descarga Google Trends "CDT" y "acciones Colombia" via pytrends (semi-auto)
  4. Intenta TES 10Y desde BanRep API → si falla, da instrucciones para carga manual
  5. Lee ICC Fedesarrollo e IPC DANE desde CSVs manuales si existen
  6. Guarda todo en experimentacion/nuevas_variables_historicas.csv

Uso:
  cd kepler-backend
  pip install yfinance pytrends requests  (si no están)
  python experimentacion/fetch_variables.py

Variables que NO están aquí (se calculan en el pipeline):
  - trm_cambio_semanal_pct  → se calcula de TRM.pct_change() en feature_engineering_v2.py
  - is_ventana_quincena     → se calcula de la fecha como mes_prima
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
BACKEND_ROOT = SCRIPT_DIR.parent
OUT_CSV      = SCRIPT_DIR / "nuevas_variables_historicas.csv"

sys.path.insert(0, str(BACKEND_ROOT))


# ── Cargar semanas del master ─────────────────────────────────────────────────

def get_master_weeks_and_trm() -> pd.DataFrame:
    """
    Lee semana + TRM del master para tener el eje temporal.
    Retorna DataFrame con columnas: semana (datetime), TRM (float)
    """
    from dotenv import load_dotenv
    load_dotenv(BACKEND_ROOT / ".env")
    from app.services.supabase_client import get_master_df

    df = get_master_df()
    date_col = "semana" if "semana" in df.columns else "fecha_inicio"
    trm_col  = "TRM"   if "TRM"   in df.columns else "trm"

    result = pd.DataFrame()
    result["semana"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if trm_col in df.columns:
        result["TRM"] = pd.to_numeric(df[trm_col], errors="coerce")
    result = result.dropna(subset=["semana"]).sort_values("semana").reset_index(drop=True)
    return result


# ── yfinance: S&P500 y Brent ──────────────────────────────────────────────────

def fetch_yfinance_weekly_pct(ticker: str, start: str, end: str, col_name: str) -> pd.Series:
    """
    Descarga datos semanales de yfinance y calcula % cambio semana a semana.
    Alinea al lunes de cada semana (como el master).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  ⚠ yfinance no instalado. Ejecuta: pip install yfinance")
        return pd.Series(dtype=float, name=col_name)

    print(f"    Descargando {ticker} desde {start}...")
    try:
        raw = yf.download(
            ticker, start=start, end=end,
            interval="1wk", auto_adjust=True, progress=False,
        )
        if raw.empty:
            print(f"  ⚠ Sin datos para {ticker}")
            return pd.Series(dtype=float, name=col_name)

        closes = raw["Close"].squeeze()
        pct    = closes.pct_change() * 100
        pct.index = pd.to_datetime(pct.index).normalize()
        return pct.rename(col_name)
    except Exception as e:
        print(f"  ⚠ Error yfinance {ticker}: {e}")
        return pd.Series(dtype=float, name=col_name)


# ── pytrends ──────────────────────────────────────────────────────────────────

def fetch_pytrends_weekly(keyword: str, geo: str, col_name: str) -> pd.Series:
    """
    Descarga Google Trends semanal. Para períodos > 5 años usa dos ventanas
    solapadas y normaliza usando el overlap para crear una serie continua.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  ⚠ pytrends no instalado. Ejecuta: pip install pytrends")
        return pd.Series(dtype=float, name=col_name)

    pytrends = TrendReq(hl="es-CO", tz=300, timeout=(10, 30))

    today     = date.today().strftime("%Y-%m-%d")
    windows   = [
        ("2021-01-01", "2023-07-01"),
        ("2023-01-01", today),
    ]

    series_list = []
    for start_w, end_w in windows:
        print(f"    '{keyword}' {start_w} → {end_w}...")
        try:
            pytrends.build_payload([keyword], geo=geo, timeframe=f"{start_w} {end_w}")
            df_w = pytrends.interest_over_time()
            if df_w.empty or keyword not in df_w.columns:
                print(f"    ⚠ Sin datos en ventana {start_w}/{end_w}")
                time.sleep(5)
                continue
            s = df_w[keyword].astype(float)
            s.index = pd.to_datetime(s.index).normalize()
            series_list.append(s)
            time.sleep(4)  # Rate limit
        except Exception as e:
            print(f"    ⚠ Error pytrends: {e}")
            time.sleep(15)

    if not series_list:
        return pd.Series(dtype=float, name=col_name)

    if len(series_list) == 1:
        return series_list[0].rename(col_name)

    s1, s2   = series_list[0], series_list[1]
    overlap  = s1.index.intersection(s2.index)

    if len(overlap) >= 4:
        mean1   = s1.loc[overlap].mean()
        mean2   = s2.loc[overlap].mean()
        ratio   = mean1 / (mean2 + 1e-9)
        s2      = s2 * ratio

    combined = pd.concat([s1, s2]).groupby(level=0).mean()
    return combined.rename(col_name)


# ── BanRep API: TES 10Y → spread ─────────────────────────────────────────────

def fetch_tes10y_banrep() -> pd.Series:
    """
    Intenta obtener TES 10Y de las APIs públicas de BanRep.
    Si falla, da instrucciones para carga manual.
    """
    import requests

    endpoints = [
        # BanRep series estadísticas — probar varios códigos
        "https://totoro.banrep.gov.co/series-estadisticas/rest/serie/TES_10/datos/2020-12-01/2026-12-31",
        "https://totoro.banrep.gov.co/series-estadisticas/rest/serie/TBT_TES10/datos/2020-12-01/2026-12-31",
        "https://totoro.banrep.gov.co/series-estadisticas/rest/serie/TC_TES10/datos/2020-12-01/2026-12-31",
    ]

    for url in endpoints:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                continue
            data = resp.json()
            records = data.get("datos", data.get("data", []))
            if not records:
                continue
            df = pd.DataFrame(records)
            # BanRep devuelve {dato: float, fecha: "YYYY-MM-DD"}
            if "fecha" in df.columns and ("dato" in df.columns or len(df.columns) >= 2):
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
                val_col = "dato" if "dato" in df.columns else df.columns[1]
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                s = df.set_index("fecha")[val_col].sort_index()
                print(f"    ✓ TES 10Y obtenido desde BanRep: {len(s)} registros")
                return s
        except Exception:
            pass

    print("  ⚠ TES 10Y no disponible automáticamente desde BanRep.")
    print("    Para carga manual:")
    print("    1. Ve a: https://www.banrep.gov.co/es/estadisticas/tasas-interes-del-mercado-monetario")
    print("    2. Descarga la serie de TES a 10 años")
    print("    3. Guarda como: experimentacion/tes_10y_manual.csv")
    print("       Columnas requeridas: fecha (YYYY-MM-DD), tes_10y (porcentaje anual)")

    manual_path = SCRIPT_DIR / "tes_10y_manual.csv"
    if manual_path.exists():
        print(f"    ✓ Leyendo desde {manual_path}")
        df_m = pd.read_csv(manual_path)
        df_m["fecha"] = pd.to_datetime(df_m["fecha"], errors="coerce")
        s = df_m.set_index("fecha").iloc[:, 0].dropna().sort_index()
        return s

    return pd.Series(dtype=float, name="tes_10y")


# ── Variables mensuales (ICC, IPC) ────────────────────────────────────────────

def load_monthly_manual(col_name: str, fuente_hint: str) -> pd.Series:
    """
    Lee una variable mensual desde CSV manual y la propaga semanalmente (fill-forward).
    Si el archivo no existe, imprime instrucciones y retorna Serie vacía.

    Formato del CSV: mes (YYYY-MM-01), valor
    """
    manual_path = SCRIPT_DIR / f"{col_name}_manual.csv"

    if manual_path.exists():
        print(f"    ✓ Leyendo {manual_path.name}...")
        df_m = pd.read_csv(manual_path)
        # Detectar columna de fecha
        date_col = next((c for c in df_m.columns if "mes" in c.lower() or "fecha" in c.lower()), df_m.columns[0])
        val_col  = next((c for c in df_m.columns if c != date_col), df_m.columns[1])
        df_m[date_col] = pd.to_datetime(df_m[date_col], errors="coerce")
        s = df_m.set_index(date_col)[val_col].dropna().sort_index()
        print(f"    {len(s)} meses leídos")
        return s.rename(col_name)

    print(f"  ⚠ {col_name}: archivo no encontrado.")
    print(f"    Crea: experimentacion/{col_name}_manual.csv")
    print(f"    Columnas: mes (YYYY-MM-01), valor")
    print(f"    Fuente:   {fuente_hint}")
    return pd.Series(dtype=float, name=col_name)


def propagate_monthly_to_weekly(monthly: pd.Series, weekly_idx: pd.DatetimeIndex) -> pd.Series:
    """Propaga datos mensuales a frecuencia semanal usando fill-forward."""
    if monthly.empty:
        return pd.Series(np.nan, index=weekly_idx, name=monthly.name)
    combined_idx = monthly.index.union(weekly_idx).sort_values()
    s = monthly.reindex(combined_idx).ffill().reindex(weekly_idx)
    return s


# ── Alineación a semanas del master ──────────────────────────────────────────

def align_to_master(source: pd.Series, master_dates: pd.DatetimeIndex, tolerance_days: int = 3) -> pd.Series:
    """
    Alinea una serie temporal al índice de semanas del master.
    Usa nearest-neighbor con tolerancia de ±tolerance_days.
    """
    if source.empty:
        return pd.Series(np.nan, index=master_dates, name=source.name)

    source = source.copy()
    source.index = pd.to_datetime(source.index).normalize()

    # reindex con tolerancia
    result = source.reindex(master_dates, method="nearest", tolerance=pd.Timedelta(days=tolerance_days))
    return result


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Fetch nuevas variables — Kepler v2")
    print("=" * 62)

    # ── 1. Eje temporal desde master ──────────────────────────────────────────
    print("\n[1] Leyendo semanas de master_consolidado_final...")
    master_meta = get_master_weeks_and_trm()
    master_dates = pd.DatetimeIndex(master_meta["semana"].values)
    start_str = master_dates[0].strftime("%Y-%m-%d")
    end_str   = date.today().strftime("%Y-%m-%d")
    print(f"    {len(master_dates)} semanas: {start_str} → {master_dates[-1].strftime('%Y-%m-%d')}")

    result = pd.DataFrame(index=master_dates)
    result.index.name = "semana"

    # ── 2. S&P500 ─────────────────────────────────────────────────────────────
    print("\n[2] S&P500 (yfinance ^GSPC)...")
    sp500 = fetch_yfinance_weekly_pct("^GSPC", start_str, end_str, "sp500_cambio_semanal_pct")
    result["sp500_cambio_semanal_pct"] = align_to_master(sp500, master_dates)
    n_ok = result["sp500_cambio_semanal_pct"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.9 else '⚠️'} {n_ok}/{len(result)} semanas")

    # ── 3. Brent ──────────────────────────────────────────────────────────────
    print("\n[3] Brent (yfinance BZ=F)...")
    brent = fetch_yfinance_weekly_pct("BZ=F", start_str, end_str, "brent_cambio_semanal_pct")
    result["brent_cambio_semanal_pct"] = align_to_master(brent, master_dates)
    n_ok = result["brent_cambio_semanal_pct"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.9 else '⚠️'} {n_ok}/{len(result)} semanas")

    # ── 4. Google Trends: CDT ─────────────────────────────────────────────────
    print("\n[4] Google Trends 'CDT' Colombia...")
    trends_cdt = fetch_pytrends_weekly("CDT", "CO", "trends_cdt")
    result["trends_cdt"] = align_to_master(trends_cdt, master_dates)
    n_ok = result["trends_cdt"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.8 else '⚠️'} {n_ok}/{len(result)} semanas")

    # ── 5. Google Trends: acciones Colombia ──────────────────────────────────
    print("\n[5] Google Trends 'acciones Colombia'...")
    time.sleep(8)  # Rate limit entre keywords
    trends_acc = fetch_pytrends_weekly("acciones Colombia", "CO", "trends_acciones")
    result["trends_acciones"] = align_to_master(trends_acc, master_dates)
    n_ok = result["trends_acciones"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.8 else '⚠️'} {n_ok}/{len(result)} semanas")

    # ── 6. TES 10Y → spread vs BanRep ────────────────────────────────────────
    print("\n[6] TES 10Y y spread vs tasa BanRep...")
    tes_10y = fetch_tes10y_banrep()
    if not tes_10y.empty:
        tes_aligned = align_to_master(tes_10y, master_dates, tolerance_days=10)
        # spread = TES 10Y - Tasa_Intervencion_Mensual (ya en master)
        if "TRM" in master_meta.columns:
            # TRM no es lo mismo que tasa BanRep — usamos tes_aligned como spread directo
            # hasta que el usuario confirme el cálculo (o cargue tasa_banrep_manual.csv)
            banrep_path = SCRIPT_DIR / "tasa_banrep_manual.csv"
            if banrep_path.exists():
                df_br = pd.read_csv(banrep_path)
                df_br.iloc[:, 0] = pd.to_datetime(df_br.iloc[:, 0], errors="coerce")
                br_s = df_br.set_index(df_br.columns[0]).iloc[:, 0].dropna()
                br_aligned = align_to_master(br_s, master_dates, tolerance_days=31)
                result["spread_tes_banrep"] = tes_aligned - br_aligned
            else:
                # La tasa BanRep ya está en master como Tasa_Intervencion_Mensual
                result["spread_tes_banrep"] = tes_aligned  # sin restar aún
                print("    ⚠ Guardando TES 10Y crudo. El spread se calculará en patch_master.py")
                print("       usando Tasa_Intervencion_Mensual del master.")
        n_ok = result["spread_tes_banrep"].notna().sum()
        print(f"    {'✅' if n_ok > len(result) * 0.8 else '⚠️'} {n_ok}/{len(result)} semanas")
    else:
        result["spread_tes_banrep"] = np.nan
        print("    ❌ Sin datos — requiere carga manual (ver instrucciones arriba)")

    # ── 7. ICC Fedesarrollo (mensual) ─────────────────────────────────────────
    print("\n[7] ICC Fedesarrollo (mensual → semanal)...")
    icc_monthly = load_monthly_manual(
        "icc_fedesarrollo",
        "https://www.fedesarrollo.org.co/encuestas/consumidores → Indicador de confianza del consumidor"
    )
    result["icc_fedesarrollo"] = propagate_monthly_to_weekly(icc_monthly, master_dates)
    n_ok = result["icc_fedesarrollo"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.8 else '⚠️' if n_ok > 0 else '❌'} {n_ok}/{len(result)} semanas")

    # ── 8. IPC DANE (mensual) ─────────────────────────────────────────────────
    print("\n[8] IPC DANE (mensual → semanal)...")
    ipc_monthly = load_monthly_manual(
        "inflacion_ipc",
        "https://www.dane.gov.co → IPC → Serie histórica variación mensual"
    )
    result["inflacion_ipc"] = propagate_monthly_to_weekly(ipc_monthly, master_dates)
    n_ok = result["inflacion_ipc"].notna().sum()
    print(f"    {'✅' if n_ok > len(result) * 0.8 else '⚠️' if n_ok > 0 else '❌'} {n_ok}/{len(result)} semanas")

    # ── Guardar CSV ────────────────────────────────────────────────────────────
    result_out = result.reset_index()
    result_out["semana"] = result_out["semana"].dt.strftime("%Y-%m-%d")
    result_out.to_csv(OUT_CSV, index=False, sep=";")

    print(f"\n{'=' * 62}")
    print(f"  CSV guardado: {OUT_CSV}")
    print(f"\n  Cobertura de variables:")
    for col in [c for c in result.columns]:
        pct   = result[col].notna().mean() * 100
        icon  = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
        print(f"  {icon} {col:<38} {pct:5.1f}%")

    n_full = (result.notna().all(axis=1)).sum()
    print(f"\n  Semanas con TODAS las variables: {n_full}/{len(result)}")
    print(f"\n  Próximos pasos:")
    print(f"  1. Revisar {OUT_CSV.name} (spot-check de valores)")
    if result["icc_fedesarrollo"].isna().all():
        print(f"  2. Crear experimentacion/icc_fedesarrollo_manual.csv y re-correr")
    if result["inflacion_ipc"].isna().all():
        print(f"  3. Crear experimentacion/inflacion_ipc_manual.csv y re-correr")
    if result["spread_tes_banrep"].isna().all():
        print(f"  4. Crear experimentacion/tes_10y_manual.csv y re-correr")
    print(f"  5. Correr: python experimentacion/patch_master.py")


if __name__ == "__main__":
    main()
