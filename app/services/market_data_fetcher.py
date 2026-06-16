"""
Fetcher de variables macro automáticas para la semana de predicción.

Replica exactamente la lógica de:
  - scripts-get-data/fetch_market_data.py  → sp500, brent, colcap (yfinance)
  - scripts-get-data/fetch_trends.py       → trends_cdt, trends_acciones (pytrends)
  - scripts-get-data/calcular_ventana_quincena.py → pct_dias_quincena (cálculo puro)

Sin cambios en la lógica de obtención de datos.
"""

import logging
import time
import warnings
from datetime import date, timedelta
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("kepler.market_data_fetcher")

# ─── Constantes idénticas a los scripts ───────────────────────────────────────

# calcular_ventana_quincena.py
QUINCENA_DAYS = {1, 2, 3, 15, 16, 17, 28, 29, 30}

# fetch_market_data.py — tickers yfinance
YFINANCE_TICKERS: dict[str, str] = {
    "sp500_cambio_semanal_pct":  "^GSPC",
    "brent_cambio_semanal_pct":  "BZ=F",
    "colcap_cambio_semanal_pct": "ICOLCAP.CL",
}

# fetch_trends.py — keywords Colombia
TRENDS_KEYWORDS: dict[str, str] = {
    "trends_cdt":      "CDT",
    "trends_acciones": "bolsa de valores",
}

# Pausa entre keywords — idéntica a fetch_trends.py (line 202: time.sleep(30))
TRENDS_PAUSE_SECONDS = 30

# TradingView tickers (tradingview-scraper) — solo snapshot actual
TV_COLCAP_SYMBOL   = "BVC:ICOLCAP"    # ETF iColcap en BVC (COP), campo Perf.W = % semana
TV_TES10Y_SYMBOL   = "TVC:CO10Y"      # TES 10 años Colombia, campo close = yield %


# ─── Helpers de fecha ─────────────────────────────────────────────────────────

def _parse_monday(semana_str: str) -> date:
    """Parsea 'DD/MM/YYYY' o 'YYYY-MM-DD' → date (lunes de la semana)."""
    s = semana_str.strip()
    if "/" in s:
        parts = s.split("/")
        return date(int(parts[2]), int(parts[1]), int(parts[0]))
    return date.fromisoformat(s[:10])


# ─── 1. pct_dias_quincena ─────────────────────────────────────────────────────
# Lógica idéntica a calcular_ventana_quincena.py → calc_pct_quincena()

def calc_pct_quincena(monday: date) -> float:
    """
    Fracción de días (lun-dom) que caen en ventana post-quincena colombiana.
    Días de ventana: {1,2,3, 15,16,17, 28,29,30}
    Lógica idéntica a calcular_ventana_quincena.py.
    """
    count = sum(
        1 for i in range(7)
        if (monday + timedelta(days=i)).day in QUINCENA_DAYS
    )
    return round(count / 7, 4)


# ─── 2. Variables yfinance (S&P500, Brent, COLCAP) ───────────────────────────
# Lógica idéntica a fetch_market_data.py: safe_close → weekly_return_pct → shift_to_monday

def _safe_close(ticker: str, start: str, end: str) -> "pd.Series":
    """
    safe_close() de fetch_market_data.py:
    Descarga diario, normaliza a Series (maneja MultiIndex de yfinance >= 0.2.x).
    """
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


def _weekly_return_pct(daily_series: "pd.Series") -> "pd.Series":
    """
    weekly_return_pct() de fetch_market_data.py:
    Resamplea a W-FRI, pct_change * 100, round(2).
    Resultado: porcentaje con signo (ej. -3.24, 1.57).
    """
    s = daily_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    weekly = s.resample("W-FRI").last()
    return (weekly.pct_change() * 100).round(2)


def _shift_to_monday(s: "pd.Series") -> "pd.Series":
    """
    shift_to_monday() de fetch_market_data.py:
    Mueve el índice de viernes → lunes (Friday - 4 días).
    """
    s = s.copy()
    s.index = s.index - pd.to_timedelta(4, unit="d")
    return s


def fetch_yfinance_return(ticker: str, monday: date) -> float | None:
    """
    Calcula el % cambio semanal (viernes→viernes) para el ticker dado,
    mapeado al lunes de la semana target.

    Replica exactamente la lógica de fetch_market_data.py para una semana específica.
    """
    try:
        import yfinance  # noqa: F401
    except ImportError:
        logger.error("yfinance no instalado: pip install yfinance")
        return None

    # Ventana: ±14 días para garantizar 2 viernes completos
    start = (monday - timedelta(days=14)).strftime("%Y-%m-%d")
    end   = (monday + timedelta(days=9)).strftime("%Y-%m-%d")

    try:
        daily = _safe_close(ticker, start, end)
        if daily.empty:
            logger.warning("yfinance: sin datos para %s (%s → %s)", ticker, start, end)
            return None

        weekly_mon = _shift_to_monday(_weekly_return_pct(daily))

        # Buscar el lunes exacto
        monday_ts = pd.Timestamp(monday)
        if monday_ts in weekly_mon.index:
            val = weekly_mon.loc[monday_ts]
            return float(val) if not pd.isna(val) else None

        # Fallback: nearest dentro de 2 días (tolerancia fetch_market_data.py)
        diffs = abs(weekly_mon.index - monday_ts)
        if len(diffs) == 0:
            return None
        idx_min = int(diffs.argmin())
        if diffs[idx_min] <= pd.Timedelta(days=2):
            val = weekly_mon.iloc[idx_min]
            return float(val) if not pd.isna(val) else None

        logger.warning("yfinance %s: lunes %s no encontrado en serie", ticker, monday)
        return None

    except Exception as exc:
        logger.warning("Error yfinance %s: %s", ticker, exc)
        return None


# ─── 3. Google Trends ─────────────────────────────────────────────────────────
# Lógica idéntica a fetch_trends.py: fetch_range → alinear con tolerance 4D/8D

def fetch_trends_current(keyword: str, monday: date, retries: int = 3) -> float | None:
    """
    Obtiene el índice Google Trends (0-100) para la semana que contiene `monday`.

    Replica fetch_range() + alineación de fetch_trends.py:
      - Ventana de 3 meses (suficiente para la semana actual)
      - Reindex nearest con tolerance=4D (Google Trends indexa al domingo)
      - Fallback tolerance=8D
      - Reintenta con espera exponencial en 429 (45s, 90s)
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends no instalado: pip install pytrends")
        return None

    today_str          = date.today().strftime("%Y-%m-%d")
    three_months_ago   = (monday - timedelta(days=90)).strftime("%Y-%m-%d")
    timeframe          = f"{three_months_ago} {today_str}"

    pt = TrendReq(hl="es-CO", tz=300, timeout=(15, 45))

    for attempt in range(1, retries + 1):
        try:
            pt.build_payload([keyword], geo="CO", timeframe=timeframe)
            df = pt.interest_over_time()

            if df.empty or keyword not in df.columns:
                logger.warning("pytrends: respuesta vacía para '%s' (intento %d)", keyword, attempt)
                return None

            # Replicar fetch_range() de fetch_trends.py
            s = df[keyword].astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s[s > 0]

            if s.empty:
                return None

            monday_ts  = pd.Timestamp(monday)
            target_idx = pd.DatetimeIndex([monday_ts])

            # Alineación con tolerance=4D (fetch_trends.py línea 216)
            aligned = s.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("4D"))
            val = aligned.iloc[0] if not aligned.empty else None
            if val is not None and not pd.isna(val):
                return float(val)

            # Fallback tolerance=8D (fetch_trends.py línea 219)
            aligned2 = s.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("8D"))
            val2 = aligned2.iloc[0] if not aligned2.empty else None
            return float(val2) if val2 is not None and not pd.isna(val2) else None

        except Exception as exc:
            msg = str(exc)
            is_429 = "429" in msg
            logger.warning(
                "pytrends error (intento %d/%d) keyword='%s': %s",
                attempt, retries, keyword, msg[:200],
            )
            if is_429 and attempt < retries:
                wait = 45 * attempt  # 45s, 90s — misma lógica fetch_trends.py
                logger.info("Rate limit 429 — esperando %ds...", wait)
                time.sleep(wait)
            else:
                return None

    return None


# ─── 4. TradingView snapshot (COLCAP + TES 10Y + BanRep) ─────────────────────
# tradingview-scraper solo da snapshot actual — válido para el flujo semanal
# porque el usuario siempre consulta la semana que acaba de terminar (domingo).

def _tv_get(symbol: str, fields: list) -> dict | None:
    """
    Obtiene campos de snapshot actual desde TradingView scanner API.
    Retorna el dict data o None si falla.
    """
    try:
        from tradingview_scraper.symbols.overview import Overview
        ov = Overview()
        result = ov.get_symbol_overview(symbol, fields=fields)
        if result.get("status") == "success":
            return result["data"]
        logger.warning("TradingView %s fallo: %s", symbol, result.get("error", ""))
        return None
    except Exception as exc:
        logger.warning("TradingView error para %s: %s", symbol, exc)
        return None


def fetch_colcap_tv(monday: date) -> float | None:
    """
    Obtiene la variación semanal del COLCAP en COP usando TradingView BVC:ICOLCAP.

    Campo Perf.W = % de cambio lunes-viernes de la semana actual (o más reciente).
    Más preciso que ICOLCAP.CL (yfinance) porque está en COP puro sin FX COP/CLP.

    Nota: retorna None si el lunes consultado es anterior a la semana actual
    (TradingView solo tiene snapshot actual, no histórico).
    """
    data = _tv_get(TV_COLCAP_SYMBOL, ["Perf.W", "close", "change"])
    if data is None:
        return None
    perf_w = data.get("Perf.W")
    if perf_w is None:
        return None
    return round(float(perf_w), 4)


def fetch_spread_tes_banrep(monday: date, banrep_tasa: float | None = None) -> float | None:
    """
    Calcula el spread TES 10Y − Tasa BanRep (puntos porcentuales).

    TES 10Y : TVC:CO10Y vía TradingView (snapshot automático, campo 'close')
    BanRep  : parámetro banrep_tasa — el usuario lo ingresa antes del auto-fetch.
              BanRep publica la tasa pública en su sitio; cambia máximo 8 veces/año.

    spread = TES10Y_yield - banrep_tasa
    Ejemplo: TES=12.10%, BanRep=10.25% → spread=1.85 pp

    Retorna None si banrep_tasa no fue provisto o TradingView falla.
    """
    if banrep_tasa is None:
        logger.warning("spread_tes_banrep: banrep_tasa no provisto — no se puede calcular spread")
        return None

    tes_data = _tv_get(TV_TES10Y_SYMBOL, ["close"])
    if tes_data is None:
        logger.warning("spread_tes_banrep: no se obtuvo TES 10Y de TradingView (TVC:CO10Y)")
        return None

    tes_yield = tes_data.get("close")
    if tes_yield is None:
        return None

    spread = round(float(tes_yield) - float(banrep_tasa), 4)
    logger.info(
        "spread_tes_banrep: TES=%.4f%% - BanRep=%.4f%% = %.4f pp",
        tes_yield, banrep_tasa, spread,
    )
    return spread


# ─── 5. TRM — datos.gov.co (Superfinanciera) ─────────────────────────────────

# API REST oficial Superfinanciera vía datos.gov.co (Socrata SODA, sin auth)
TRM_API_URL = "https://www.datos.gov.co/resource/mcec-87by.json"


def fetch_trm(monday: date) -> float | None:
    """
    Obtiene la TRM (COP/USD) como promedio semanal lunes-domingo.

    Fuente: Superintendencia Financiera vía datos.gov.co (Socrata SODA).
    Lógica por día:
      - Lun-Vie: query directa (vigenciadesde <= día AND vigenciahasta >= día)
      - Sáb-Dom: si la API no tiene dato, reutiliza el valor del viernes anterior
        (BanRep no publica en fin de semana)
      - Si un día hábil no tiene dato (festivo), busca hasta 5 días atrás

    Retorna round(sum(7_valores) / 7, 2) o None si falla más de 5 días.
    """
    try:
        import httpx
    except ImportError:
        logger.error("httpx no instalado: pip install httpx")
        return None

    def _query_day(target: date) -> float | None:
        date_str = target.strftime("%Y-%m-%dT00:00:00.000")
        params = {
            "$where": f"vigenciadesde <= '{date_str}' AND vigenciahasta >= '{date_str}'",
            "$limit": "1",
        }
        try:
            resp = httpx.get(TRM_API_URL, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            if data:
                return round(float(data[0]["valor"]), 2)
        except Exception as exc:
            logger.warning("TRM API error para %s: %s", target, exc)
        return None

    def _get_day(target: date, weekday: int) -> float | None:
        """weekday: 0=lun … 6=dom"""
        val = _query_day(target)
        if val is not None:
            return val
        # Fin de semana → reutiliza el viernes
        if weekday >= 5:
            friday = target - timedelta(days=weekday - 4)
            return _query_day(friday)
        # Día hábil sin dato (festivo) → busca días anteriores
        for days_back in range(1, 6):
            val = _query_day(target - timedelta(days=days_back))
            if val is not None:
                return val
        return None

    daily_vals: list[float] = []
    for offset in range(7):
        d = monday + timedelta(days=offset)
        val = _get_day(d, d.weekday())
        if val is not None:
            daily_vals.append(val)
            logger.debug("TRM %s = %.2f", d, val)
        else:
            logger.warning("TRM: sin dato para %s", d)

    if not daily_vals:
        logger.warning("TRM: sin dato disponible para semana %s", monday)
        return None

    avg = round(sum(daily_vals) / len(daily_vals), 2)
    logger.info(
        "TRM semana %s: promedio %.2f (%d días, min=%.2f max=%.2f)",
        monday, avg, len(daily_vals), min(daily_vals), max(daily_vals),
    )
    return avg


# ─── Orquestador ──────────────────────────────────────────────────────────────

def fetch_auto_variables(semana_str: str, banrep_tasa: float | None = None) -> dict[str, Any]:
    """
    Obtiene todas las variables macro auto-obtenibles para la semana dada.

    Variables auto:
      pct_dias_quincena        → cálculo puro desde fecha (~inmediato)
      trm                      → datos.gov.co Superfinanciera (promedio 7 días lun-dom)
      colcap_cambio_semanal_pct→ TradingView BVC:ICOLCAP Perf.W (COP, lun-vie)
      sp500_cambio_semanal_pct → yfinance ^GSPC (vie-vie)
      brent_cambio_semanal_pct → yfinance BZ=F (vie-vie)
      spread_tes_banrep        → TVC:CO10Y (TradingView) − banrep_tasa (input usuario)
      trends_cdt               → pytrends 'CDT' CO (~30-60s con pausa anti-429)
      trends_acciones          → pytrends 'bolsa de valores' CO (~30-60s)

    banrep_tasa: tasa de intervención BanRep en %, ej. 10.25
                 Si no se provee, spread_tes_banrep queda como "error".

    Returns:
      {
        "semana": str,
        "values": {campo: float | None},
        "status": {campo: "ok" | "error" | "pending"},
        "errors": [str]
      }
    """
    monday = _parse_monday(semana_str)
    values: dict[str, float | None] = {}
    status: dict[str, str] = {}
    errors: list[str] = []

    logger.info("=== fetch_auto_variables: %s (lunes %s) ===", semana_str, monday)

    # 1. pct_dias_quincena — cálculo puro, nunca falla
    values["pct_dias_quincena"] = calc_pct_quincena(monday)
    status["pct_dias_quincena"] = "ok"
    logger.info("pct_dias_quincena = %.4f", values["pct_dias_quincena"])

    # 2. TRM — datos.gov.co Superfinanciera (promedio lun-dom)
    logger.info("Fetching TRM promedio semanal desde datos.gov.co...")
    trm_val = fetch_trm(monday)
    values["trm"] = trm_val
    status["trm"] = "ok" if trm_val is not None else "error"
    if trm_val is None:
        errors.append("trm: sin datos de Superfinanciera (datos.gov.co)")
    else:
        logger.info("trm = %.2f", trm_val)

    # 3. COLCAP — TradingView BVC:ICOLCAP Perf.W (COP, sin FX CLP)
    logger.info("Fetching COLCAP desde TradingView BVC:ICOLCAP...")
    colcap_val = fetch_colcap_tv(monday)
    values["colcap_cambio_semanal_pct"] = colcap_val
    status["colcap_cambio_semanal_pct"] = "ok" if colcap_val is not None else "error"
    if colcap_val is None:
        errors.append("colcap_cambio_semanal_pct: sin datos de TradingView (BVC:ICOLCAP)")
    else:
        logger.info("colcap_cambio_semanal_pct = %.4f%%", colcap_val)

    # 4. yfinance: S&P500 y Brent (COLCAP ya va por TradingView)
    yfinance_subset = {k: v for k, v in YFINANCE_TICKERS.items() if k != "colcap_cambio_semanal_pct"}
    for field, ticker in yfinance_subset.items():
        val = fetch_yfinance_return(ticker, monday)
        values[field] = val
        status[field] = "ok" if val is not None else "error"
        if val is None:
            errors.append(f"{field}: sin datos de yfinance ({ticker})")
        else:
            logger.info("%s = %.2f%%", field, val)

    # 5. Spread TES 10Y − BanRep — TVC:CO10Y (TV) − banrep_tasa (input usuario)
    if banrep_tasa is not None:
        logger.info("Fetching TES 10Y desde TradingView (BanRep=%.4f%%)...", banrep_tasa)
    else:
        logger.info("spread_tes_banrep: banrep_tasa no provisto — omitiendo")
    spread_val = fetch_spread_tes_banrep(monday, banrep_tasa=banrep_tasa)
    values["spread_tes_banrep"] = spread_val
    status["spread_tes_banrep"] = "ok" if spread_val is not None else "error"
    if spread_val is None:
        if banrep_tasa is None:
            errors.append("spread_tes_banrep: ingresá la tasa BanRep antes de hacer auto-fetch")
        else:
            errors.append("spread_tes_banrep: sin datos de TES 10Y de TradingView (TVC:CO10Y)")
    else:
        logger.info("spread_tes_banrep = %.4f pp", spread_val)

    # 6. Google Trends: CDT y bolsa de valores (pausa entre keywords como en fetch_trends.py)
    trend_items = list(TRENDS_KEYWORDS.items())
    for i, (field, keyword) in enumerate(trend_items):
        logger.info("Google Trends '%s'...", keyword)
        val = fetch_trends_current(keyword, monday)
        values[field] = val
        status[field] = "ok" if val is not None else "error"
        if val is None:
            errors.append(f"{field}: sin datos de Google Trends ('{keyword}')")
        else:
            logger.info("%s = %.1f", field, val)

        # Pausa entre keywords — idéntica a fetch_trends.py línea 202: time.sleep(30)
        if i < len(trend_items) - 1:
            logger.info("Pausa %ds entre keywords (evitar rate limit 429)...", TRENDS_PAUSE_SECONDS)
            time.sleep(TRENDS_PAUSE_SECONDS)

    n_ok  = sum(1 for v in status.values() if v == "ok")
    n_err = sum(1 for v in status.values() if v == "error")
    logger.info(
        "=== fetch_auto_variables fin: %d ok, %d error, %d pending ===",
        n_ok, n_err, sum(1 for v in status.values() if v == "pending"),
    )

    return {
        "semana":  semana_str,
        "values":  values,
        "status":  status,
        "errors":  errors,
    }
