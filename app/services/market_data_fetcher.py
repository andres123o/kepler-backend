"""
Connector engine para variables macro automáticas.

Arquitectura de 2 niveles:
  Level 1 (código): conectores genéricos — yfinance_weekly_pct, google_trends, socrata_daily_avg, etc.
  Level 2 (BD):     instancias por funnel — qué símbolo, qué URL, qué keyword, qué geo.
                    Se leen de config JSONB → "data_sources" array via FunnelClient.

Agregar un cliente nuevo = insertar filas en Supabase. Cero cambios en código.
"""

import logging
import time
import warnings
from datetime import date, timedelta
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("kepler.market_data_fetcher")

# Pausa entre llamadas consecutivas a Google Trends (rate-limit 429)
_TRENDS_PAUSE_SECONDS = 30


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE FECHA
# ══════════════════════════════════════════════════════════════════════════════

def _parse_monday(semana_str: str) -> date:
    """Parsea 'DD/MM/YYYY' o 'YYYY-MM-DD' → date (lunes de la semana)."""
    s = semana_str.strip()
    if "/" in s:
        parts = s.split("/")
        return date(int(parts[2]), int(parts[1]), int(parts[0]))
    return date.fromisoformat(s[:10])


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIONES LOW-LEVEL (reutilizadas por los conectores)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_close(ticker: str, start: str, end: str) -> "pd.Series":
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


def _weekly_return_pct(daily_series: "pd.Series") -> "pd.Series":
    s = daily_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    weekly = s.resample("W-FRI").last()
    return (weekly.pct_change() * 100).round(2)


def _shift_to_monday(s: "pd.Series") -> "pd.Series":
    s = s.copy()
    s.index = s.index - pd.to_timedelta(4, unit="d")
    return s


def fetch_yfinance_return(ticker: str, monday: date) -> float | None:
    """% cambio viernes→viernes para el ticker, mapeado al lunes de la semana."""
    try:
        import yfinance  # noqa: F401
    except ImportError:
        logger.error("yfinance no instalado: pip install yfinance")
        return None

    start = (monday - timedelta(days=14)).strftime("%Y-%m-%d")
    end   = (monday + timedelta(days=9)).strftime("%Y-%m-%d")
    try:
        daily = _safe_close(ticker, start, end)
        if daily.empty:
            logger.warning("yfinance: sin datos para %s (%s → %s)", ticker, start, end)
            return None
        weekly_mon = _shift_to_monday(_weekly_return_pct(daily))
        monday_ts  = pd.Timestamp(monday)
        if monday_ts in weekly_mon.index:
            val = weekly_mon.loc[monday_ts]
            return float(val) if not pd.isna(val) else None
        diffs   = abs(weekly_mon.index - monday_ts)
        idx_min = int(diffs.argmin())
        if len(diffs) > 0 and diffs[idx_min] <= pd.Timedelta(days=2):
            val = weekly_mon.iloc[idx_min]
            return float(val) if not pd.isna(val) else None
        logger.warning("yfinance %s: lunes %s no encontrado en serie", ticker, monday)
        return None
    except Exception as exc:
        logger.warning("Error yfinance %s: %s", ticker, exc)
        return None


def fetch_trends_value(keyword: str, monday: date, geo: str = "CO", hl: str = "es-CO", retries: int = 3) -> float | None:
    """Índice Google Trends (0-100) para la semana que contiene `monday`."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends no instalado: pip install pytrends")
        return None

    today_str        = date.today().strftime("%Y-%m-%d")
    three_months_ago = (monday - timedelta(days=90)).strftime("%Y-%m-%d")
    timeframe        = f"{three_months_ago} {today_str}"
    pt               = TrendReq(hl=hl, tz=300, timeout=(15, 45))

    for attempt in range(1, retries + 1):
        try:
            pt.build_payload([keyword], geo=geo, timeframe=timeframe)
            df = pt.interest_over_time()
            if df.empty or keyword not in df.columns:
                logger.warning("pytrends: respuesta vacía para '%s' (intento %d)", keyword, attempt)
                return None
            s          = df[keyword].astype(float)
            s.index    = pd.to_datetime(s.index).tz_localize(None).normalize()
            s          = s[s > 0]
            if s.empty:
                return None
            monday_ts  = pd.Timestamp(monday)
            target_idx = pd.DatetimeIndex([monday_ts])
            aligned    = s.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("4D"))
            val        = aligned.iloc[0] if not aligned.empty else None
            if val is not None and not pd.isna(val):
                return float(val)
            aligned2   = s.reindex(target_idx, method="nearest", tolerance=pd.Timedelta("8D"))
            val2       = aligned2.iloc[0] if not aligned2.empty else None
            return float(val2) if val2 is not None and not pd.isna(val2) else None
        except Exception as exc:
            msg    = str(exc)
            is_429 = "429" in msg
            logger.warning("pytrends error (intento %d/%d) '%s': %s", attempt, retries, keyword, msg[:200])
            if is_429 and attempt < retries:
                wait = 45 * attempt
                logger.info("Rate limit 429 — esperando %ds...", wait)
                time.sleep(wait)
            else:
                return None
    return None


def _tv_get(symbol: str, fields: list) -> dict | None:
    """Snapshot actual de TradingView para los campos dados."""
    try:
        from tradingview_scraper.symbols.overview import Overview
        ov     = Overview()
        result = ov.get_symbol_overview(symbol, fields=fields)
        if result.get("status") == "success":
            return result["data"]
        logger.warning("TradingView %s fallo: %s", symbol, result.get("error", ""))
        return None
    except Exception as exc:
        logger.warning("TradingView error para %s: %s", symbol, exc)
        return None


def fetch_socrata_daily_avg(
    url: str,
    value_field: str,
    date_from_field: str,
    date_to_field: str,
    monday: date,
) -> float | None:
    """
    Promedio semanal desde una API Socrata con rango de vigencia por día.
    Maneja fin de semana (reutiliza viernes) y festivos (busca hasta 5 días atrás).
    """
    try:
        import httpx
    except ImportError:
        logger.error("httpx no instalado: pip install httpx")
        return None

    def _query_day(target: date) -> float | None:
        date_str = target.strftime("%Y-%m-%dT00:00:00.000")
        params   = {
            "$where": f"{date_from_field} <= '{date_str}' AND {date_to_field} >= '{date_str}'",
            "$limit": "1",
        }
        try:
            resp = httpx.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            if data:
                return round(float(data[0][value_field]), 2)
        except Exception as exc:
            logger.warning("Socrata API error para %s (%s): %s", url, target, exc)
        return None

    def _get_day(target: date, weekday: int) -> float | None:
        val = _query_day(target)
        if val is not None:
            return val
        if weekday >= 5:
            friday = target - timedelta(days=weekday - 4)
            return _query_day(friday)
        for days_back in range(1, 6):
            val = _query_day(target - timedelta(days=days_back))
            if val is not None:
                return val
        return None

    daily_vals: list[float] = []
    for offset in range(7):
        d   = monday + timedelta(days=offset)
        val = _get_day(d, d.weekday())
        if val is not None:
            daily_vals.append(val)
            logger.debug("Socrata %s = %.2f", d, val)
        else:
            logger.warning("Socrata: sin dato para %s", d)

    if not daily_vals:
        logger.warning("Socrata: sin dato disponible para semana %s", monday)
        return None
    avg = round(sum(daily_vals) / len(daily_vals), 2)
    logger.info("Socrata promedio semana %s: %.2f (%d días)", monday, avg, len(daily_vals))
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTOR REGISTRY — Level 1
# Cada función recibe (monday, params, manual_params) y retorna float | None
# ══════════════════════════════════════════════════════════════════════════════

def _run_quincena_fraction(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Fracción de días lun-dom que caen en ventana post-quincena.
    params.quincena_days: lista de días del mes (ej. [1,2,3,15,16,17,28,29,30])
    """
    quincena_days = set(params.get("quincena_days", [1, 2, 3, 15, 16, 17, 28, 29, 30]))
    count         = sum(1 for i in range(7) if (monday + timedelta(days=i)).day in quincena_days)
    return round(count / 7, 4)


def _run_yfinance_weekly_pct(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    % cambio semanal viernes→viernes vía yfinance.
    params.ticker: símbolo yfinance (ej. "^GSPC", "BZ=F")
    """
    return fetch_yfinance_return(params["ticker"], monday)


def _run_tradingview_perf_w(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    % rendimiento semanal desde TradingView (campo Perf.W).
    params.symbol: símbolo TradingView (ej. "BVC:ICOLCAP")
    """
    data   = _tv_get(params["symbol"], ["Perf.W", "close", "change"])
    if data is None:
        return None
    perf_w = data.get("Perf.W")
    return round(float(perf_w), 4) if perf_w is not None else None


def _run_tradingview_minus_manual(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Spread = valor TradingView − parámetro manual ingresado por el usuario.
    params.symbol:       símbolo TradingView (ej. "TVC:CO10Y")
    params.tv_field:     campo del snapshot (ej. "close")
    params.manual_param: clave en manual_params (ej. "banrep_tasa")
    """
    manual_param = params["manual_param"]
    manual_val   = manual_params.get(manual_param)
    if manual_val is None:
        logger.warning("tradingview_minus_manual: '%s' no provisto en manual_params", manual_param)
        return None
    tv_field = params.get("tv_field", "close")
    data     = _tv_get(params["symbol"], [tv_field])
    if data is None:
        return None
    tv_val = data.get(tv_field)
    if tv_val is None:
        return None
    spread = round(float(tv_val) - float(manual_val), 4)
    logger.info("tradingview_minus_manual: TV=%.4f - manual=%.4f = %.4f", tv_val, manual_val, spread)
    return spread


def _run_socrata_daily_avg(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Promedio semanal desde API Socrata con campo de vigencia por día.
    params.url:             endpoint Socrata
    params.value_field:     campo del valor (ej. "valor")
    params.date_from_field: campo inicio vigencia (ej. "vigenciadesde")
    params.date_to_field:   campo fin vigencia (ej. "vigenciahasta")
    """
    return fetch_socrata_daily_avg(
        url             = params["url"],
        value_field     = params["value_field"],
        date_from_field = params["date_from_field"],
        date_to_field   = params["date_to_field"],
        monday          = monday,
    )


def _run_google_trends(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Índice Google Trends (0-100) para la semana dada.
    params.keyword: término de búsqueda (ej. "CDT")
    params.geo:     código de país (ej. "CO", "PE", "CL")
    params.hl:      locale (ej. "es-CO")
    """
    return fetch_trends_value(
        keyword = params["keyword"],
        monday  = monday,
        geo     = params.get("geo", "CO"),
        hl      = params.get("hl", "es-CO"),
    )


def _run_yfinance_rolling_std(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Desviación estándar de los últimos N retornos semanales de un ticker.
    Útil para medir volatilidad reciente (ej. pen_usd_volatilidad_4w).

    params.ticker: símbolo yfinance (ej. "PEN=X")
    params.weeks:  número de semanas a incluir (default 4)
    """
    ticker = params["ticker"]
    weeks  = int(params.get("weeks", 4))

    try:
        import yfinance  # noqa: F401
    except ImportError:
        logger.error("yfinance no instalado: pip install yfinance")
        return None

    # Necesitamos weeks+2 semanas de historia para tener N retornos completos
    start = (monday - timedelta(days=(weeks + 2) * 7)).strftime("%Y-%m-%d")
    end   = (monday + timedelta(days=9)).strftime("%Y-%m-%d")

    try:
        daily  = _safe_close(ticker, start, end)
        if daily.empty:
            logger.warning("yfinance_rolling_std: sin datos para %s (%s → %s)", ticker, start, end)
            return None

        weekly_mon = _shift_to_monday(_weekly_return_pct(daily))
        monday_ts  = pd.Timestamp(monday)

        # Filtrar hasta el lunes objetivo (inclusive)
        past = weekly_mon[weekly_mon.index <= monday_ts].dropna()
        if len(past) < 2:
            logger.warning("yfinance_rolling_std %s: menos de 2 retornos para std", ticker)
            return None

        last_n = past.tail(weeks)
        std    = float(last_n.std())
        logger.info("yfinance_rolling_std %s last_%dw std=%.4f", ticker, weeks, std)
        return round(std, 4)
    except Exception as exc:
        logger.warning("Error yfinance_rolling_std %s: %s", ticker, exc)
        return None


def _run_quincena_binary(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    1.0 si algún día de la semana lun-dom cae en ventana de quincena, 0.0 si no.
    Versión binaria de quincena_fraction para funnels que modelan esto como bool.

    Default [1,15,16,28,29,30,31]: dias exactos de quincena + dia siguiente.
    Alcanza 91.9% accuracy vs master Peru. El 8.1% restante son semanas donde
    el dia 1 o 15 cae en lunes (pago real fue el viernes anterior).

    params.quincena_days: lista de días del mes (override del default)
    """
    quincena_days = set(params.get("quincena_days", [1, 15, 16, 28, 29, 30, 31]))
    hit = any((monday + timedelta(days=i)).day in quincena_days for i in range(7))
    return 1.0 if hit else 0.0


def _run_calendar_window_binary(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    1.0 si algún día de la semana cae dentro de alguna ventana calendario definida.
    Ideal para eventos de nómina predecibles: CTS (may 1-15, nov 1-15),
    Gratificación (jul 1-20, dic 1-20), etc.

    params.windows: lista de {"month": int, "day_start": int, "day_end": int}
    """
    windows = params.get("windows", [])
    for i in range(7):
        d = monday + timedelta(days=i)
        for w in windows:
            if d.month == w["month"] and w["day_start"] <= d.day <= w["day_end"]:
                return 1.0
    return 0.0


def _run_working_days_count(monday: date, params: dict, manual_params: dict) -> float | None:
    """
    Cuenta días hábiles (lun-vie) en la semana descontando festivos nacionales.

    params.country: código ISO del país (ej. "PE", "CO", "CL")
    Usa la librería `holidays` (pip install holidays). Soporta 100+ países.
    """
    country = params.get("country", "CO")
    try:
        import holidays as holiday_lib
        country_holidays = holiday_lib.country_holidays(country)
    except ImportError:
        logger.error("holidays no instalado: pip install holidays")
        return None
    except Exception as exc:
        logger.warning("Error cargando holidays para %s: %s", country, exc)
        return None

    count = sum(1 for i in range(5) if (monday + timedelta(days=i)) not in country_holidays)
    return float(count)


CONNECTOR_REGISTRY: dict[str, Any] = {
    "quincena_fraction":         _run_quincena_fraction,
    "quincena_binary":           _run_quincena_binary,
    "calendar_window_binary":    _run_calendar_window_binary,
    "yfinance_weekly_pct":       _run_yfinance_weekly_pct,
    "yfinance_rolling_std":      _run_yfinance_rolling_std,
    "tradingview_perf_w":        _run_tradingview_perf_w,
    "tradingview_minus_manual":  _run_tradingview_minus_manual,
    "socrata_daily_avg":         _run_socrata_daily_avg,
    "google_trends":             _run_google_trends,
    "working_days_count":        _run_working_days_count,
}


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR — lee data_sources del config JSONB y ejecuta cada conector
# ══════════════════════════════════════════════════════════════════════════════

def fetch_auto_variables(
    semana_str: str,
    banrep_tasa: float | None = None,
    fc=None,  # FunnelClient | None — evita import circular
) -> dict[str, Any]:
    """
    Ejecuta todos los conectores definidos en data_sources del config JSONB del funnel.

    manual_params se construye a partir de los query params recibidos (ej. banrep_tasa).
    Agregar un manual_param nuevo = agregarlo al query string del endpoint + pasarlo aquí.

    Returns:
      { "semana": str, "values": {campo: float|None}, "status": {campo: str}, "errors": [str] }
    """
    from app.services.supabase_client import _default_fc
    _fc = fc or _default_fc()

    funnel_cfg   = _fc.get_funnel_config()
    data_sources = funnel_cfg.get("data_sources")
    if not data_sources:
        raise ValueError(
            f"El funnel no tiene 'data_sources' en su config JSONB. "
            f"Agrega el array al config del funnel en Supabase."
        )

    monday = _parse_monday(semana_str)
    manual_params: dict[str, float] = {}
    if banrep_tasa is not None:
        manual_params["banrep_tasa"] = banrep_tasa

    values: dict[str, float | None] = {}
    status: dict[str, str]          = {}
    errors: list[str]               = []

    logger.info("=== fetch_auto_variables: %s (lunes %s) — %d conectores ===",
                semana_str, monday, len(data_sources))

    last_connector: str = ""

    for source in data_sources:
        field          = source.get("field", "")
        connector_type = source.get("connector", "")
        params         = source.get("params", {})

        if not field or not connector_type:
            logger.warning("data_source inválido (sin field o connector): %s", source)
            continue

        runner = CONNECTOR_REGISTRY.get(connector_type)
        if runner is None:
            logger.error("Conector desconocido: '%s' para field '%s'", connector_type, field)
            errors.append(f"{field}: conector '{connector_type}' no registrado en Level 1")
            values[field] = None
            status[field] = "error"
            continue

        # Pausa entre llamadas consecutivas a Google Trends
        if connector_type == "google_trends" and last_connector == "google_trends":
            logger.info("Pausa %ds entre google_trends (anti-429)...", _TRENDS_PAUSE_SECONDS)
            time.sleep(_TRENDS_PAUSE_SECONDS)

        logger.info("Ejecutando [%s] → %s (params=%s)", connector_type, field, params)
        try:
            val           = runner(monday, params, manual_params)
            values[field] = val
            status[field] = "ok" if val is not None else "error"
            if val is None:
                errors.append(f"{field}: conector {connector_type} retornó None")
            else:
                logger.info("%s = %s", field, val)
        except Exception as exc:
            logger.error("Error en conector [%s] → %s: %s", connector_type, field, exc)
            values[field] = None
            status[field] = "error"
            errors.append(f"{field}: excepción en {connector_type} — {exc}")

        last_connector = connector_type

    n_ok  = sum(1 for v in status.values() if v == "ok")
    n_err = sum(1 for v in status.values() if v == "error")
    logger.info("=== fetch_auto_variables fin: %d ok, %d error ===", n_ok, n_err)

    return {
        "semana":  semana_str,
        "values":  values,
        "status":  status,
        "errors":  errors,
    }
