"""
Fase 2: Ingeniería de características — shift target, weighted pipeline, lags, EWMA,
y variables estacionales calculadas desde la fecha (festivos colombianos Ley Emiliani).
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
    EWMA_SPAN,
    EWMA_VARIABLES,
    SEASONAL_FEATURES,
    TARGET_NAME,
    WEIGHTED_PIPELINE_VARIABLES,
)

logger = logging.getLogger("kepler.ml.feature_engineering")


# ─────────────────────────────────────────────
# Festivos colombianos
# ─────────────────────────────────────────────

def _next_monday(d: date) -> date:
    """Ley Emiliani: si el festivo no cae en lunes, se pasa al lunes siguiente."""
    days_ahead = (7 - d.weekday()) % 7  # weekday() 0=lunes
    return d if days_ahead == 0 else d + timedelta(days=days_ahead)


def _get_colombian_holidays(year: int) -> set[date]:
    """
    Retorna el conjunto de festivos colombianos para el año dado.

    Tipos:
    - Fijos: siempre la misma fecha, no se mueven.
    - Ley Emiliani: si no caen en lunes, se mueven al lunes siguiente.
    - Móviles religiosos: calculados desde Pascua (algoritmo de Butcher).
    """
    from dateutil.easter import easter  # incluido en dateutil, dependencia de pandas

    holidays: set[date] = set()

    # --- Festivos fijos (no se mueven) ---
    holidays.update([
        date(year, 1, 1),    # Año Nuevo
        date(year, 5, 1),    # Día del Trabajo
        date(year, 7, 20),   # Día de la Independencia
        date(year, 8, 7),    # Batalla de Boyacá
        date(year, 12, 8),   # Inmaculada Concepción
        date(year, 12, 25),  # Navidad
    ])

    # --- Festivos basados en Pascua ---
    pascua = easter(year)
    holidays.add(pascua - timedelta(days=3))                    # Jueves Santo
    holidays.add(pascua - timedelta(days=2))                    # Viernes Santo
    holidays.add(_next_monday(pascua + timedelta(days=39)))     # Ascensión del Señor
    holidays.add(_next_monday(pascua + timedelta(days=60)))     # Corpus Christi
    holidays.add(_next_monday(pascua + timedelta(days=68)))     # Sagrado Corazón de Jesús

    # --- Festivos Ley Emiliani (se mueven al lunes siguiente) ---
    holidays.update([
        _next_monday(date(year, 1, 6)),   # Reyes Magos
        _next_monday(date(year, 3, 19)),  # San José
        _next_monday(date(year, 6, 29)),  # San Pedro y San Pablo
        _next_monday(date(year, 8, 15)),  # Asunción de la Virgen
        _next_monday(date(year, 10, 12)), # Día de la Raza
        _next_monday(date(year, 11, 1)),  # Todos los Santos
        _next_monday(date(year, 11, 11)), # Independencia de Cartagena
    ])

    return holidays


# Cache para no recalcular festivos del mismo año múltiples veces
_holidays_cache: dict[int, set[date]] = {}


def _holidays_for_year(year: int) -> set[date]:
    if year not in _holidays_cache:
        _holidays_cache[year] = _get_colombian_holidays(year)
    return _holidays_cache[year]


# ─────────────────────────────────────────────
# Date parsing
# ─────────────────────────────────────────────

def _parse_date(s) -> pd.Timestamp | None:
    """
    Detección automática de formato de fecha:
    - Si contiene '/' -> DD/MM/YYYY (formato legacy)
    - Si contiene '-' y empieza con 4 dígitos -> YYYY-MM-DD (nuevo formato BQ)
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


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def csv_bytes_to_dataframe(data: bytes) -> pd.DataFrame:
    """Parsea CSV (delimitador ;, UTF-8) y retorna DataFrame."""
    text = data.decode(CSV_ENCODING)
    df = pd.read_csv(StringIO(text), sep=CSV_DELIMITER, encoding=CSV_ENCODING)
    logger.info("CSV cargado: %d filas, %d columnas.", len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────
# Preparación base
# ─────────────────────────────────────────────

def prepare_base_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Busca columna temporal ('semana' o 'fecha_inicio'), parsea fechas,
    ordena cronológicamente y coerce a numérico.
    IMPORTANTE: No convierte NaN a 0. Los NaN se mantienen para XGBoost.
    """
    df = df.copy()

    if "semana" in df.columns:
        date_col = "semana"
    elif "fecha_inicio" in df.columns:
        date_col = "fecha_inicio"
    else:
        raise ValueError("CSV debe tener columna 'semana' o 'fecha_inicio'.")

    df["_date_parsed"] = df[date_col].map(_parse_date)
    df = df.dropna(subset=["_date_parsed"])
    df = df.sort_values("_date_parsed").reset_index(drop=True)
    logger.info("Ordenado por tiempo (%s): %d filas.", date_col, len(df))

    skip = {date_col, "_date_parsed"}
    for col in df.columns:
        if col in skip:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────
# Features estacionales (calculadas desde la fecha)
# ─────────────────────────────────────────────

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula variables estacionales desde _date_parsed (lunes de cada semana):

    - semana_del_mes   : 1, 2, 3 o 4 según el día del mes del lunes
    - mes_prima        : 1 si junio (6) o diciembre (12), 0 si no
    - dias_habiles_semana : días hábiles lun-vie descontando festivos colombianos (Ley Emiliani)
    """
    if "_date_parsed" not in df.columns:
        logger.warning("add_seasonal_features: _date_parsed no encontrada, saltando.")
        return df

    df = df.copy()
    semana_del_mes_vals = []
    semana_del_mes_proy_vals = []
    mes_prima_vals = []
    dias_habiles_vals = []
    dias_habiles_proy_vals = []

    for ts in df["_date_parsed"]:
        if pd.isna(ts):
            semana_del_mes_vals.append(np.nan)
            semana_del_mes_proy_vals.append(np.nan)
            mes_prima_vals.append(np.nan)
            dias_habiles_vals.append(np.nan)
            dias_habiles_proy_vals.append(np.nan)
            continue

        d = ts.date()
        d_next = d + timedelta(days=7)

        # semana_del_mes: qué semana del mes es (1-4) — semana t
        semana_del_mes_vals.append(min((d.day - 1) // 7 + 1, 4))

        # semana_del_mes_proyeccion: semana del mes de t+1 (la semana que se predice)
        semana_del_mes_proy_vals.append(min((d_next.day - 1) // 7 + 1, 4))

        # mes_prima: bono semestral (junio y diciembre) — semana t
        mes_prima_vals.append(1 if d.month in (6, 12) else 0)

        # dias_habiles_semana: lun-vie de la semana t (todos los festivos colombianos)
        years_needed = {d.year, (d + timedelta(days=4)).year}
        all_holidays: set[date] = set()
        for y in years_needed:
            all_holidays |= _holidays_for_year(y)
        dias_habiles_vals.append(
            sum(1 for i in range(5) if (d + timedelta(days=i)) not in all_holidays)
        )

        # dias_habiles_proyeccion: lun-vie de la semana t+1 (todos los festivos colombianos)
        years_needed_next = {d_next.year, (d_next + timedelta(days=4)).year}
        all_holidays_next: set[date] = set()
        for y in years_needed_next:
            all_holidays_next |= _holidays_for_year(y)
        dias_habiles_proy_vals.append(
            sum(1 for i in range(5) if (d_next + timedelta(days=i)) not in all_holidays_next)
        )

    df["semana_del_mes"] = semana_del_mes_vals
    df["semana_del_mes_proyeccion"] = semana_del_mes_proy_vals
    df["mes_prima"] = mes_prima_vals
    df["dias_habiles_semana"] = dias_habiles_vals
    df["dias_habiles_proyeccion"] = dias_habiles_proy_vals

    logger.info(
        "Variables estacionales calculadas: semana_del_mes, semana_del_mes_proyeccion, "
        "mes_prima, dias_habiles_semana (t), dias_habiles_proyeccion (t+1)."
    )
    return df


# ─────────────────────────────────────────────
# Weighted pipeline (lags comprimidos)
# ─────────────────────────────────────────────

def add_weighted_pipeline(
    df: pd.DataFrame,
    variables: list[str],
    weights: list[float],
) -> pd.DataFrame:
    """
    Crea features de 'lag comprimido' usando pesos del ciclo de conversión.
    Si TODOS los valores de la ventana son NaN (warm-up), el resultado es NaN.
    """
    df = df.copy()
    name_map = {
        "usuarios_registro_base": "registros_ponderados",
        "step_09_full_account": "aprobados_ponderados",
    }
    for var in variables:
        if var not in df.columns:
            logger.warning("Variable para weighted pipeline no encontrada: %s.", var)
            continue
        col_name = name_map.get(var, f"{var}_ponderado")

        weighted_sum = pd.Series(0.0, index=df.index)
        for i, w in enumerate(weights):
            shifted = df[var].shift(i)
            weighted_sum = weighted_sum + (shifted.fillna(0) * w)

        all_nan_mask = pd.concat(
            [df[var].shift(i).isna() for i in range(len(weights))], axis=1
        ).all(axis=1)
        weighted_sum[all_nan_mask] = np.nan

        df[col_name] = weighted_sum
        logger.info("Weighted pipeline: %s -> %s.", var, col_name)

    return df


def add_target_lag(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Añade lag_1_target = target de la semana anterior."""
    df = df.copy()
    df["lag_1_target"] = df[target_col].shift(1)
    logger.info("Lag autoregresivo añadido: lag_1_target.")
    return df


def add_ewma(df: pd.DataFrame, variables: list[str], span: int) -> pd.DataFrame:
    """EWMA con span dado (peso reciente mayor)."""
    df = df.copy()
    for var in variables:
        if var not in df.columns:
            continue
        df[f"{var}_ewma_{span}"] = df[var].ewm(span=span, adjust=False).mean()
    logger.info("EWMA añadido: %s span=%d.", variables, span)
    return df


# ─────────────────────────────────────────────
# Features de tendencia (slope)
# ─────────────────────────────────────────────

def _slope_4w(series: pd.Series, min_periods: int = 3) -> pd.Series:
    """
    Pendiente OLS de las últimas 4 observaciones, normalizada por la media
    del período (slope como fracción de la media).

    Resultado positivo = tendencia creciente.
    Resultado negativo = tendencia decreciente (caída acelerada).
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


# ─────────────────────────────────────────────
# Pipeline completo
# ─────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, drop_na: bool = True):
    """
    Pipeline completo de feature engineering:
    1. Variables estacionales desde la fecha (semana_del_mes, mes_prima, dias_habiles_semana, dias_habiles_proyeccion)
    2. Target shift(-1) para predecir próxima semana
    3. Weighted pipeline (registros_ponderados, aprobados_ponderados)
    4. Lag autoregresivo del target (lag_1_target)
    4b. Lag de full_users_aprobados (full_users_aprobados_lag1) — past covariate por date_full_user
    5. EWMA de variables de tendencia
    6. Ensamblar matriz final
    7. Eliminar filas de warm-up

    Retorna (X_df, y_series, feature_names)
    """
    df = df.copy()

    # 1. Variables estacionales (requiere _date_parsed, creado en prepare_base_df)
    df = add_seasonal_features(df)

    # 2. Target shift(-1)
    df["target"] = df[TARGET_NAME].shift(-1)

    # 3. Weighted pipeline
    df = add_weighted_pipeline(df, WEIGHTED_PIPELINE_VARIABLES, CONVERSION_CYCLE_WEIGHTS)

    # 4. Lag autoregresivo
    df = add_target_lag(df, TARGET_NAME)

    # 4b. Lag de aprobaciones reales por date_full_user (past covariate)
    # full_users_aprobados (sin lag) NO entra al modelo — solo es fuente del lag
    if "full_users_aprobados" in df.columns:
        df["full_users_aprobados_lag1"] = df["full_users_aprobados"].shift(1)
        logger.info("Lag full_users_aprobados anadido: full_users_aprobados_lag1.")
    else:
        logger.warning("full_users_aprobados no encontrada en el dataframe; full_users_aprobados_lag1 sera NaN.")

    # 4c. Features de tendencia (slope OLS últimas 4 semanas, normalizado por media)
    # Capturan velocidad de cambio del pipeline — el modelo ve si el funnel está
    # cayendo aceleradamente, no solo el nivel actual.
    df["tendencia_registros_4w"] = _slope_4w(df["usuarios_registro_base"])
    df["tendencia_depositos_4w"] = _slope_4w(df[TARGET_NAME].ffill())
    if "full_users_aprobados" in df.columns:
        df["tendencia_aprobados_4w"] = _slope_4w(df["full_users_aprobados"])
    logger.info("Features de tendencia calculadas: tendencia_registros_4w, tendencia_depositos_4w%s.",
                ", tendencia_aprobados_4w" if "full_users_aprobados" in df.columns else "")

    # 5. EWMA
    df = add_ewma(df, EWMA_VARIABLES, EWMA_SPAN)

    # 6. Ensamblar features:
    #    CSV (sin estacionales) + estacionales calculadas + features computadas + EWMA
    ewma_names = [f"{v}_ewma_{EWMA_SPAN}" for v in EWMA_VARIABLES]
    feature_names = (
        list(ALL_CSV_FEATURES)
        + list(SEASONAL_FEATURES)
        + list(COMPUTED_FEATURE_NAMES)
        + ewma_names
    )
    feature_names = [f for f in feature_names if f in df.columns]

    X = df[feature_names]
    y = df["target"]

    # 7. Warm-up: eliminar filas donde lags comprimidos o lag_1_target son NaN
    if drop_na:
        mask = y.notna()
        for col in COMPUTED_FEATURE_NAMES:
            if col in X.columns:
                mask = mask & X[col].notna()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        logger.info("Filas tras drop warm-up: %d. Features: %d.", len(X), len(feature_names))

    return X, y, feature_names
