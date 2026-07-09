"""
Enrichers de features — transformaciones de dominio, OPCIONALES y declarativas.

Cada enricher es una función (df, params, target_name) -> (df, output_columns)
que agrega una o más columnas. Un funnel declara en su config CUÁLES quiere
via `ml.feature_enrichers` (lista de {type, params}) — si la lista está
vacía, el pipeline entrena directo sobre las columnas crudas del CSV
(`funnel_features` + `macro_features`), sin ninguna asunción de dominio.

Esto es lo que hace que ml_pipeline sirva para CUALQUIER dataset tabular
semanal con columna de fecha y target — no solo el funnel de depósito de
Trii. Un funnel de un dominio completamente distinto simplemente declara
`feature_enrichers: []` y usa el motor genérico (contratos de datos, poda
de multicolinealidad, Optuna+XGBoost, walk-forward CV) tal cual.

Registro — agregar un enricher nuevo es agregar una función + una entrada
en ENRICHER_REGISTRY, nunca tocar build_feature_matrix ni ningún funnel
existente.

Idempotencia: donde tiene sentido (columnas derivadas de la fecha), si el
CSV ya trae la columna con valores, solo se rellenan los NaN — nunca se
sobreescribe un dato que ya vino correcto (ej. Chile ya trae
dias_habiles_semana calculado con su propio calendario en get-data-chile/).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("kepler.ml.enrichers")


def _resolve_source(source: str, target_name: str) -> str:
    """'$target' es un placeholder que resuelve al target_name del funnel — evita
    que cada config tenga que repetir el nombre exacto del target."""
    return target_name if source == "$target" else source


def _fill_idempotent(df: pd.DataFrame, column: str, computed: pd.Series) -> pd.DataFrame:
    """Crea la columna si no existe; si existe, solo rellena los NaN. Nunca sobreescribe."""
    if column not in df.columns:
        df[column] = computed
    else:
        mask = df[column].isna()
        if mask.any():
            df.loc[mask, column] = computed[mask]
    return df


# ─── business_days ─────────────────────────────────────────────────────────────
#
# params:
#   country        REQUERIDO. Código ISO para la librería `holidays` (ej. "PE", "CL"),
#                  o "CO_EXACT" para el calendario colombiano verificado a mano
#                  (Ley Emiliani + Semana Santa, ver cicloUsuario.md) — no usar la
#                  librería genérica para Colombia, cambiaría el training data histórico.
#   date_column    default "_date_parsed"
#   horizon_days   default 7 — cuántos días adelante calcular la "proyección"
#   output_current default "dias_habiles_semana"
#   output_next    default "dias_habiles_proyeccion"

def _next_monday(d: date) -> date:
    days_ahead = (7 - d.weekday()) % 7
    return d if days_ahead == 0 else d + timedelta(days=days_ahead)


def _colombia_exact_holidays(year: int) -> set[date]:
    """Calendario colombiano exacto (Ley Emiliani + Pascua) — ver feature_engineering
    histórico. Verificado contra el histórico real de depósitos en cicloUsuario.md."""
    from dateutil.easter import easter

    holidays: set[date] = set()
    holidays.update([
        date(year, 1, 1), date(year, 5, 1), date(year, 7, 20),
        date(year, 8, 7), date(year, 12, 8), date(year, 12, 25),
    ])
    pascua = easter(year)
    holidays.add(pascua - timedelta(days=3))
    holidays.add(pascua - timedelta(days=2))
    holidays.add(_next_monday(pascua + timedelta(days=39)))
    holidays.add(_next_monday(pascua + timedelta(days=60)))
    holidays.add(_next_monday(pascua + timedelta(days=68)))
    holidays.update([
        _next_monday(date(year, 1, 6)), _next_monday(date(year, 3, 19)),
        _next_monday(date(year, 6, 29)), _next_monday(date(year, 8, 15)),
        _next_monday(date(year, 10, 12)), _next_monday(date(year, 11, 1)),
        _next_monday(date(year, 11, 11)),
    ])
    return holidays


_holidays_cache: dict[tuple[str, int], set[date]] = {}


def _holidays_for_year(country: str, year: int) -> set[date]:
    key = (country, year)
    if key not in _holidays_cache:
        if country == "CO_EXACT":
            _holidays_cache[key] = _colombia_exact_holidays(year)
        else:
            import holidays as holidays_lib
            _holidays_cache[key] = set(holidays_lib.country_holidays(country, years=year).keys())
    return _holidays_cache[key]


def _count_business_days(monday: date, country: str) -> int:
    years = {monday.year, (monday + timedelta(days=4)).year}
    all_holidays: set[date] = set()
    for y in years:
        all_holidays |= _holidays_for_year(country, y)
    return sum(1 for i in range(5) if (monday + timedelta(days=i)) not in all_holidays)


def business_days(df: pd.DataFrame, params: dict, target_name: str):
    country = params.get("country")
    if not country:
        raise ValueError("Enricher 'business_days' requiere el param 'country' (ej. 'PE', 'CL', 'CO_EXACT').")
    date_col = params.get("date_column", "_date_parsed")
    horizon_days = int(params.get("horizon_days", 7))
    out_current = params.get("output_current", "dias_habiles_semana")
    out_next = params.get("output_next", "dias_habiles_proyeccion")

    if date_col not in df.columns:
        logger.warning("business_days: columna '%s' no encontrada, saltando.", date_col)
        return df, [out_current, out_next]

    df = df.copy()
    current_vals, next_vals = [], []
    for ts in df[date_col]:
        if pd.isna(ts):
            current_vals.append(np.nan)
            next_vals.append(np.nan)
            continue
        d = ts.date()
        current_vals.append(_count_business_days(d, country))
        next_vals.append(_count_business_days(d + timedelta(days=horizon_days), country))

    df = _fill_idempotent(df, out_current, pd.Series(current_vals, index=df.index))
    df = _fill_idempotent(df, out_next, pd.Series(next_vals, index=df.index))
    logger.info("business_days: %s/%s rellenados (país=%s).", out_current, out_next, country)
    return df, [out_current, out_next]


# ─── calendar_days_pct ──────────────────────────────────────────────────────────
#
# % de días de la semana de 7 días que caen dentro de un set de días-del-mes dado.
# Generaliza pct_dias_quincena (que estaba hardcodeado a [1,2,3,15,16,17,28,29,30]).
#
# params: days (REQUERIDO, list[int]), output (REQUERIDO), date_column (default "_date_parsed")

def calendar_days_pct(df: pd.DataFrame, params: dict, target_name: str):
    days = params.get("days")
    output = params.get("output")
    if not days or not output:
        raise ValueError("Enricher 'calendar_days_pct' requiere 'days' (list[int]) y 'output'.")
    date_col = params.get("date_column", "_date_parsed")
    days_set = set(days)

    if date_col not in df.columns:
        logger.warning("calendar_days_pct: columna '%s' no encontrada, saltando.", date_col)
        return df, [output]

    df = df.copy()
    computed = pd.Series(
        [
            (sum(1 for i in range(7) if (ts.date() + timedelta(days=i)).day in days_set) / 7.0)
            if not pd.isna(ts) else np.nan
            for ts in df[date_col]
        ],
        index=df.index,
    )
    df = _fill_idempotent(df, output, computed)
    logger.info("calendar_days_pct: '%s' rellenada (days=%s).", output, sorted(days_set))
    return df, [output]


# ─── weighted_lag_pipeline ──────────────────────────────────────────────────────
#
# Comprime la historia de N semanas de una variable en un solo número ponderado.
# Uso original: full_users_aprobados -> aprobados_ponderados (ciclo aprobación→
# depósito), pero es genérico — cualquier variable + cualquier vector de pesos.
#
# params: source (REQUERIDO, o "$target"), weights (REQUERIDO, list[float]), output (REQUERIDO)

def weighted_lag_pipeline(df: pd.DataFrame, params: dict, target_name: str):
    source = _resolve_source(params.get("source", ""), target_name)
    weights = params.get("weights")
    output = params.get("output")
    if not source or not weights or not output:
        raise ValueError("Enricher 'weighted_lag_pipeline' requiere 'source', 'weights' y 'output'.")

    df = df.copy()
    if source not in df.columns:
        logger.warning("weighted_lag_pipeline: '%s' no encontrada, '%s' será NaN.", source, output)
        df[output] = np.nan
        return df, [output]

    active_lags = [(i, w) for i, w in enumerate(weights) if w > 0.0]
    weighted_sum = pd.Series(0.0, index=df.index)
    for i, w in active_lags:
        weighted_sum += df[source].shift(i).fillna(0) * w

    all_nan_mask = pd.concat([df[source].shift(i).isna() for i, _ in active_lags], axis=1).all(axis=1)
    weighted_sum[all_nan_mask] = np.nan

    df[output] = weighted_sum
    logger.info("weighted_lag_pipeline: %s -> %s.", source, output)
    return df, [output]


# ─── autoregressive_lag ─────────────────────────────────────────────────────────
#
# lag(source, n) — cualquier columna, no solo el target. Reemplaza add_lags().
#
# params: source (REQUERIDO, o "$target"), lag (default 1), output (REQUERIDO)

def autoregressive_lag(df: pd.DataFrame, params: dict, target_name: str):
    source = _resolve_source(params.get("source", ""), target_name)
    output = params.get("output")
    lag = int(params.get("lag", 1))
    if not source or not output:
        raise ValueError("Enricher 'autoregressive_lag' requiere 'source' y 'output'.")

    df = df.copy()
    if source not in df.columns:
        logger.warning("autoregressive_lag: '%s' no encontrada, '%s' será NaN.", source, output)
        df[output] = np.nan
        return df, [output]

    df[output] = df[source].shift(lag)
    logger.info("autoregressive_lag: %s(t-%d) -> %s.", source, lag, output)
    return df, [output]


# ─── trend_slope ────────────────────────────────────────────────────────────────
#
# Pendiente OLS de una ventana de N observaciones, normalizada por la media del
# período (velocidad de cambio, no solo nivel). Reemplaza add_tendencias().
#
# params: source (REQUERIDO, o "$target"), window (default 4), output (REQUERIDO),
#         ffill_source (default False — usar True para series con huecos tipo target)

def _slope_ols(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
            continue
        ventana = series.iloc[max(0, i - (window - 1)): i + 1].dropna()
        if len(ventana) < min_periods:
            slopes.append(np.nan)
            continue
        x = np.arange(len(ventana), dtype=float)
        slope = np.polyfit(x, ventana.values.astype(float), 1)[0]
        media = ventana.mean()
        slopes.append(slope / media if media != 0 else np.nan)
    return pd.Series(slopes, index=series.index)


def trend_slope(df: pd.DataFrame, params: dict, target_name: str):
    source = _resolve_source(params.get("source", ""), target_name)
    output = params.get("output")
    window = int(params.get("window", 4))
    min_periods = int(params.get("min_periods", max(3, window - 1)))
    ffill_source = bool(params.get("ffill_source", False))
    if not source or not output:
        raise ValueError("Enricher 'trend_slope' requiere 'source' y 'output'.")

    df = df.copy()
    if source not in df.columns:
        logger.warning("trend_slope: '%s' no encontrada, '%s' será NaN.", source, output)
        df[output] = np.nan
        return df, [output]

    series = df[source].ffill() if ffill_source else df[source]
    df[output] = _slope_ols(series, window, min_periods)
    logger.info("trend_slope: %s (window=%d) -> %s.", source, window, output)
    return df, [output]


ENRICHER_REGISTRY = {
    "business_days": business_days,
    "calendar_days_pct": calendar_days_pct,
    "weighted_lag_pipeline": weighted_lag_pipeline,
    "autoregressive_lag": autoregressive_lag,
    "trend_slope": trend_slope,
}
