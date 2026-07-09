"""
Feature engineering — motor genérico, sin ninguna asunción de dominio.

Pipeline:
  1. Parse fechas (DD/MM/YYYY y YYYY-MM-DD), ordena, coerce a numérico.
  2. Target shift(-N) — fila t predice t+N (N = ml_cfg.prediction_horizon_weeks).
  3. Aplica los enrichers que el funnel haya declarado en ml_cfg.feature_enrichers
     (festivos, ciclo ponderado, lags, tendencias, lo que sea — ver enrichers.py).
     Si la lista está vacía, este paso no hace nada: se entrena directo sobre
     funnel_features + macro_features tal cual vienen del CSV.
  4. Ensamble + drop warm-up (filas donde las columnas generadas por los
     enrichers quedaron NaN).

Un funnel de un dominio completamente distinto al de comunicaciones/depósitos
(otro dataset, otro problema) simplemente declara `feature_enrichers: []` en
su config y usa este mismo motor sin ninguna transformación de dominio.

NaN se mantienen: XGBoost los maneja nativamente (sparsity-aware), no se imputan con 0.
"""

import logging
from io import StringIO

import pandas as pd

from ml_pipeline.config import CSV_DELIMITER, CSV_ENCODING
from ml_pipeline.enrichers import ENRICHER_REGISTRY
from ml_pipeline.funnel_config import FunnelMLConfig

logger = logging.getLogger("kepler.ml.feature_engineering")


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


# ─── Pipeline completo ────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, ml_cfg: FunnelMLConfig, drop_na: bool = True):
    """
    Aplica el pipeline de feature engineering y retorna (X, y, feature_names).

    Orden:
      1. Target shift(-N) — N = ml_cfg.prediction_horizon_weeks (default 1)
      2. Enrichers declarados en ml_cfg.feature_enrichers, en el orden dado
         (un enricher puede consumir la salida de uno anterior)
      3. Ensamble: ml_cfg.all_csv_features + columnas generadas por los enrichers
      4. Drop warm-up: filas donde esas columnas generadas quedaron NaN

    Returns:
      X            : pd.DataFrame con features
      y            : pd.Series con target (ml_cfg.target_name, shifteado)
      feature_names: list[str] — features presentes en X
    """
    df = df.copy()

    # 1. Target: shift(-N) — la fila de semana t predice semana t+N
    df["target"] = df[ml_cfg.target_name].shift(-ml_cfg.prediction_horizon_weeks)

    # 2. Enrichers — cada uno declarado explícitamente en el config del funnel.
    # Lista vacía = sin transformaciones de dominio, se entrena sobre el CSV crudo.
    computed_names: list[str] = []
    for enricher_cfg in ml_cfg.feature_enrichers:
        etype = enricher_cfg.get("type")
        fn = ENRICHER_REGISTRY.get(etype)
        if fn is None:
            raise ValueError(
                f"Enricher desconocido: '{etype}'. Disponibles: {list(ENRICHER_REGISTRY)}"
            )
        params = dict(enricher_cfg.get("params", {}))
        df, outputs = fn(df, params, ml_cfg.target_name)
        computed_names.extend(o for o in outputs if o not in computed_names)

    # 3. Ensamble — solo features que efectivamente existen en el DataFrame
    feature_names = [
        f for f in (ml_cfg.all_csv_features + computed_names)
        if f in df.columns
    ]

    X = df[feature_names]
    y = df["target"]

    # 4. Drop warm-up: eliminar filas donde las columnas generadas por enrichers son NaN
    if drop_na:
        mask = y.notna()
        for col in computed_names:
            if col in X.columns:
                mask = mask & X[col].notna()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()
        logger.info(
            "build_feature_matrix: %d filas tras warm-up drop | %d features.",
            len(X), len(feature_names),
        )

    return X, y, feature_names
