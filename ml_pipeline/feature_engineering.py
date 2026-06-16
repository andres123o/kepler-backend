"""
Feature engineering v2 — primer_deposito.

Pipeline completo:
  1. Parse fechas (DD/MM/YYYY y YYYY-MM-DD)
  2. dias_habiles semana t y t+1 — festivos colombianos completos (Ley Emiliani + Pascua)
  3. pct_dias_quincena desde fecha (idempotente: no sobreescribe si ya viene del CSV)
  4. Target shift(-1): fila t predice semana t+1
  5. Weighted pipeline → aprobados_ponderados (ciclo aprobación→depósito)
  6. Lags → lag_1_target, full_users_aprobados_lag1
  7. Tendencias OLS 4w → tendencia_aprobados_4w, tendencia_depositos_4w
  8. Ensamble + drop warm-up

Sin EWMA. Sin semana_del_mes, mes_prima ni otras estacionales de calendario.
NaN se mantienen: XGBoost los maneja nativamente (sparsity-aware).
"""

import logging
from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd

from ml_pipeline.config import (
    ALL_CSV_FEATURES,
    COMPUTED_FEATURE_NAMES,
    CONVERSION_CYCLE_WEIGHTS,
    CSV_DELIMITER,
    CSV_ENCODING,
    TARGET_NAME,
    WEIGHTED_PIPELINE_VARIABLES,
)

logger = logging.getLogger("kepler.ml.feature_engineering")


# ─── Festivos colombianos ──────────────────────────────────────────────────────

def _next_monday(d: date) -> date:
    """Ley Emiliani: mueve el festivo al lunes siguiente si no cae en lunes."""
    days_ahead = (7 - d.weekday()) % 7
    return d if days_ahead == 0 else d + timedelta(days=days_ahead)


def _get_colombian_holidays(year: int) -> set[date]:
    """Retorna el conjunto exacto de festivos colombianos para el año dado."""
    from dateutil.easter import easter

    holidays: set[date] = set()

    holidays.update([
        date(year, 1, 1),   # Año Nuevo
        date(year, 5, 1),   # Día del Trabajo
        date(year, 7, 20),  # Día de la Independencia
        date(year, 8, 7),   # Batalla de Boyacá
        date(year, 12, 8),  # Inmaculada Concepción
        date(year, 12, 25), # Navidad
    ])

    pascua = easter(year)
    holidays.add(pascua - timedelta(days=3))                    # Jueves Santo
    holidays.add(pascua - timedelta(days=2))                    # Viernes Santo
    holidays.add(_next_monday(pascua + timedelta(days=39)))     # Ascensión del Señor
    holidays.add(_next_monday(pascua + timedelta(days=60)))     # Corpus Christi
    holidays.add(_next_monday(pascua + timedelta(days=68)))     # Sagrado Corazón de Jesús

    holidays.update([
        _next_monday(date(year, 1, 6)),    # Reyes Magos
        _next_monday(date(year, 3, 19)),   # San José
        _next_monday(date(year, 6, 29)),   # San Pedro y San Pablo
        _next_monday(date(year, 8, 15)),   # Asunción de la Virgen
        _next_monday(date(year, 10, 12)),  # Día de la Raza
        _next_monday(date(year, 11, 1)),   # Todos los Santos
        _next_monday(date(year, 11, 11)),  # Independencia de Cartagena
    ])

    return holidays


_holidays_cache: dict[int, set[date]] = {}


def _holidays_for_year(year: int) -> set[date]:
    if year not in _holidays_cache:
        _holidays_cache[year] = _get_colombian_holidays(year)
    return _holidays_cache[year]


def _count_business_days(monday: date) -> int:
    """
    Cuenta días hábiles (lunes a viernes) en la semana de 7 días que empieza en `monday`,
    descontando festivos colombianos. Rango posible: 0–5.
    """
    years_in_week = {monday.year, (monday + timedelta(days=4)).year}
    all_holidays: set[date] = set()
    for y in years_in_week:
        all_holidays |= _holidays_for_year(y)
    return sum(1 for i in range(5) if (monday + timedelta(days=i)) not in all_holidays)


# ─── pct_dias_quincena ─────────────────────────────────────────────────────────

QUINCENA_DAYS = {1, 2, 3, 15, 16, 17, 28, 29, 30}


def _pct_quincena(monday: date) -> float:
    count = sum(1 for i in range(7) if (monday + timedelta(days=i)).day in QUINCENA_DAYS)
    return count / 7.0


# ─── Date parsing ─────────────────────────────────────────────────────────────

def _parse_date(s) -> "pd.Timestamp | None":
    """
    Acepta DD/MM/YYYY (formato legacy) y YYYY-MM-DD (formato ISO/BQ).
    Retorna None si el valor está vacío o no es parseable.
    """
    if pd.isna(s) or not str(s).strip():
        return None
    s = str(s).strip()
    try:
        if "/" in s:
            return pd.Timestamp(pd.to_datetime(s, dayfirst=True))
        if "-" in s and len(s) >= 10 and s[4] == "-":
            return pd.Timestamp(pd.to_datetime(s, format="%Y-%m-%d"))
    except (ValueError, TypeError):
        pass
    return None


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """Carga CSV local con separador ; y encoding UTF-8."""
    df = pd.read_csv(path, sep=CSV_DELIMITER, encoding=CSV_ENCODING)
    logger.info("CSV cargado: %s — %d filas, %d columnas.", path, len(df), len(df.columns))
    return df


def csv_bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Parsea CSV desde bytes (para uso desde ml_runner.py con datos de Supabase)."""
    text = data.decode(CSV_ENCODING)
    return pd.read_csv(StringIO(text), sep=CSV_DELIMITER, encoding=CSV_ENCODING)


# ─── Preparación base ─────────────────────────────────────────────────────────

def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Detecta columna temporal ('semana' o 'fecha_inicio').
    2. Parsea fechas (acepta DD/MM/YYYY y YYYY-MM-DD).
    3. Ordena cronológicamente.
    4. Coerce todas las columnas restantes a numérico.
       NaN se mantienen — XGBoost los maneja nativamente, no se imputan con 0.
    """
    df = df.copy()

    if "semana" in df.columns:
        date_col = "semana"
    elif "fecha_inicio" in df.columns:
        date_col = "fecha_inicio"
    else:
        raise ValueError("CSV debe tener columna 'semana' o 'fecha_inicio'.")

    df["_date_parsed"] = df[date_col].map(_parse_date)
    df = df.dropna(subset=["_date_parsed"]).sort_values("_date_parsed").reset_index(drop=True)

    skip = {date_col, "_date_parsed"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("prepare_base_df: %d filas ordenadas por '%s'.", len(df), date_col)
    return df


# ─── dias_habiles ─────────────────────────────────────────────────────────────

def add_dias_habiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula días hábiles usando el calendario colombiano completo.

    dias_habiles_semana    : días hábiles de la semana t (la semana que el usuario ingresó).
    dias_habiles_proyeccion: días hábiles de la semana t+1 (la semana que se predice).

    Semana Santa es el caso crítico: reduce a 3 días hábiles → impacto en depósitos.
    Requiere _date_parsed apuntando al lunes de cada semana.
    """
    if "_date_parsed" not in df.columns:
        logger.warning("add_dias_habiles: _date_parsed no encontrada, saltando.")
        return df

    df = df.copy()
    hab_t, hab_t1 = [], []

    for ts in df["_date_parsed"]:
        if pd.isna(ts):
            hab_t.append(np.nan)
            hab_t1.append(np.nan)
            continue
        d = ts.date()
        hab_t.append(_count_business_days(d))
        hab_t1.append(_count_business_days(d + timedelta(days=7)))

    df["dias_habiles_semana"] = hab_t
    df["dias_habiles_proyeccion"] = hab_t1

    logger.info("dias_habiles calculados — festivos colombianos completos.")
    return df


# ─── pct_dias_quincena (idempotente) ──────────────────────────────────────────

def add_pct_quincena(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula pct_dias_quincena desde la fecha.
    - Si la columna no existe: la crea para todas las filas.
    - Si la columna existe con NaN: rellena solo los NaN (preserva valores del CSV).
    - Si la columna existe sin NaN: no toca nada (idempotente).

    Esto garantiza que la fila de predicción (ultima_semana) siempre tenga este valor
    aunque el usuario no lo ingrese manualmente.
    """
    if "_date_parsed" not in df.columns:
        logger.warning("add_pct_quincena: _date_parsed no encontrada, saltando.")
        return df

    df = df.copy()
    computed = pd.Series(
        [_pct_quincena(ts.date()) if not pd.isna(ts) else np.nan for ts in df["_date_parsed"]],
        index=df.index,
    )

    if "pct_dias_quincena" in df.columns:
        mask = df["pct_dias_quincena"].isna()
        if mask.any():
            df.loc[mask, "pct_dias_quincena"] = computed[mask]
            logger.info("pct_dias_quincena: %d NaN rellenados desde _date_parsed.", int(mask.sum()))
    else:
        df["pct_dias_quincena"] = computed
        logger.info("pct_dias_quincena calculada desde _date_parsed.")

    return df


# ─── Weighted pipeline ────────────────────────────────────────────────────────

def add_weighted_pipeline(
    df: pd.DataFrame,
    variables: list[str],
    weights: list[float],
) -> pd.DataFrame:
    """
    Comprime la historia de N semanas de una variable en un solo número ponderado.
    Input: full_users_aprobados → Output: aprobados_ponderados

    Pesos basados en el ciclo de conversión aprobación→depósito (cicloUsuario.md).
    Solo aplica lags con peso > 0. NaN donde todo el historial activo es NaN (warm-up).
    """
    df = df.copy()
    name_map = {
        "full_users_aprobados": "aprobados_ponderados",
    }

    for var in variables:
        if var not in df.columns:
            logger.warning("Weighted pipeline: '%s' no encontrada en el DataFrame, saltando.", var)
            continue

        col_name = name_map.get(var, f"{var}_ponderado")
        active_lags = [(i, w) for i, w in enumerate(weights) if w > 0.0]

        weighted_sum = pd.Series(0.0, index=df.index)
        for i, w in active_lags:
            weighted_sum += df[var].shift(i).fillna(0) * w

        all_nan_mask = pd.concat(
            [df[var].shift(i).isna() for i, _ in active_lags], axis=1
        ).all(axis=1)
        weighted_sum[all_nan_mask] = np.nan

        df[col_name] = weighted_sum
        logger.info("Weighted pipeline: %s → %s.", var, col_name)

    return df


# ─── Lags autoregresivos ───────────────────────────────────────────────────────

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    lag_1_target              : target(t-1) — componente autoregresivo del modelo.
    full_users_aprobados_lag1 : aprobados semana t-1 — pasado covariante del pipeline.
    """
    df = df.copy()
    df["lag_1_target"] = df[TARGET_NAME].shift(1)

    if "full_users_aprobados" in df.columns:
        df["full_users_aprobados_lag1"] = df["full_users_aprobados"].shift(1)
    else:
        logger.warning("full_users_aprobados no encontrada; full_users_aprobados_lag1 será NaN.")

    logger.info("Lags añadidos: lag_1_target, full_users_aprobados_lag1.")
    return df


# ─── Tendencias OLS 4 semanas ─────────────────────────────────────────────────

def _slope_4w(series: pd.Series, min_periods: int = 3) -> pd.Series:
    """
    Pendiente OLS de las últimas 4 observaciones, normalizada por la media del período.
    Captura velocidad de cambio (aceleración/desaceleración), no solo el nivel.
    NaN si hay menos de min_periods valores válidos en la ventana.
    """
    slopes = []
    for i in range(len(series)):
        if i < 3:
            slopes.append(np.nan)
            continue
        ventana = series.iloc[max(0, i - 3): i + 1].dropna()
        if len(ventana) < min_periods:
            slopes.append(np.nan)
            continue
        x = np.arange(len(ventana), dtype=float)
        slope = np.polyfit(x, ventana.values.astype(float), 1)[0]
        media = ventana.mean()
        slopes.append(slope / media if media != 0 else np.nan)
    return pd.Series(slopes, index=series.index)


def add_tendencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    tendencia_aprobados_4w : slope OLS 4w de full_users_aprobados.
    tendencia_depositos_4w : slope OLS 4w del target.
    """
    df = df.copy()

    if "full_users_aprobados" in df.columns:
        df["tendencia_aprobados_4w"] = _slope_4w(df["full_users_aprobados"])
    else:
        logger.warning("full_users_aprobados no encontrada; tendencia_aprobados_4w será NaN.")

    df["tendencia_depositos_4w"] = _slope_4w(df[TARGET_NAME].ffill())

    logger.info("Tendencias OLS 4w calculadas: tendencia_aprobados_4w, tendencia_depositos_4w.")
    return df


# ─── Pipeline completo ────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, drop_na: bool = True):
    """
    Aplica todo el pipeline de feature engineering y retorna (X, y, feature_names).

    Orden de pasos:
      1. dias_habiles (semana t y t+1) — festivos colombianos completos
      2. pct_dias_quincena — desde fecha si no viene del CSV (idempotente)
      3. Target shift(-1) — fila t predice semana t+1
      4. Weighted pipeline — aprobados_ponderados (ciclo aprobación→depósito)
      5. Lags — lag_1_target, full_users_aprobados_lag1
      6. Tendencias OLS 4w — tendencia_aprobados_4w, tendencia_depositos_4w
      7. Ensamble: ALL_CSV_FEATURES + COMPUTED_FEATURE_NAMES
      8. Drop warm-up: filas donde COMPUTED_FEATURE_NAMES tienen NaN

    Los primeros ~6 filas son warm-up y se eliminan. Con 310+ filas quedan ~304 útiles.

    Returns:
      X            : pd.DataFrame con features
      y            : pd.Series con target (usuarios_primer_cashin semana t+1)
      feature_names: list[str] — features presentes en X
    """
    df = df.copy()

    # 1. dias_habiles — requiere _date_parsed (creado en prepare_base_df)
    df = add_dias_habiles(df)

    # 2. pct_dias_quincena — idempotente
    df = add_pct_quincena(df)

    # 3. Target: shift(-1) — la fila de semana t predice semana t+1
    df["target"] = df[TARGET_NAME].shift(-1)

    # 4. Weighted pipeline
    df = add_weighted_pipeline(df, WEIGHTED_PIPELINE_VARIABLES, CONVERSION_CYCLE_WEIGHTS)

    # 5. Lags
    df = add_lags(df)

    # 6. Tendencias
    df = add_tendencias(df)

    # 7. Ensamble — solo features que efectivamente existen en el DataFrame
    feature_names = [
        f for f in (list(ALL_CSV_FEATURES) + list(COMPUTED_FEATURE_NAMES))
        if f in df.columns
    ]

    X = df[feature_names]
    y = df["target"]

    # 8. Drop warm-up: eliminar filas donde las computed features son NaN
    if drop_na:
        mask = y.notna()
        for col in COMPUTED_FEATURE_NAMES:
            if col in X.columns:
                mask = mask & X[col].notna()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        logger.info(
            "build_feature_matrix: %d filas tras warm-up drop | %d features.",
            len(X), len(feature_names),
        )

    return X, y, feature_names
