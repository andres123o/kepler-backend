"""
Fase 1: Contratos de datos (Data Contracts).
Valida negativos, saltos anómalos y tipos antes de entrenar.
NaN en columnas de canales son legítimos — solo se validan filas no-NaN.
"""

import logging

import pandas as pd

from ml_pipeline.config import (
    ANOMALY_JUMP_PCT,
    NON_NEGATIVE_COLUMNS,
    TARGET_NAME,
)

logger = logging.getLogger("kepler.ml.data_contracts")


def validate_data_contracts(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Aplica contratos de datos. Retorna (ok, lista de mensajes de violación).
    Si ok=False, no se debe entrenar.
    """
    violations: list[str] = []

    # 1) No negativos en columnas que no lo permiten
    # Solo validar filas no-NaN (columnas de canales tienen NaN legítimos antes de su fecha de inicio)
    for col in NON_NEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        neg_count = (valid < 0).sum()
        if neg_count > 0:
            msg = f"Contrato: {col} tiene {neg_count} valores negativos."
            violations.append(msg)
            logger.warning(msg)

    # 2) Saltos anómalos semana a semana en target — ALERTA, no bloqueante
    # En datasets históricos (desde 2020) los saltos grandes son normales en fases de crecimiento.
    if TARGET_NAME in df.columns:
        diff_pct = df[TARGET_NAME].pct_change() * 100
        prev = df[TARGET_NAME].shift(1)
        mask = (prev != 0) & (prev.notna())
        jump = diff_pct.abs() > ANOMALY_JUMP_PCT
        anomaly_rows = (mask & jump).sum()
        if anomaly_rows > 0:
            logger.warning(
                "Alerta (no bloqueante): %s tiene %d saltos > %s%% semana a semana. "
                "Revisar si son datos reales o errores de ingesta.",
                TARGET_NAME, anomaly_rows, ANOMALY_JUMP_PCT,
            )

    ok = len(violations) == 0
    if ok:
        logger.info("Contratos de datos: OK, sin violaciones.")
    else:
        logger.warning(
            "Contratos de datos: %d violación(es). No entrenar hasta corregir.",
            len(violations),
        )

    return ok, violations


def clip_target_to_percentile(
    series: pd.Series, percentile: float, train_mask: pd.Series | None = None
) -> pd.Series:
    """
    Recorta la serie (target) al percentil dado. Por defecto usa toda la serie para calcular el umbral;
    si train_mask está dado, solo usa esas filas para calcular el percentil (evitar lookahead).
    """
    if train_mask is not None:
        ref = series.loc[train_mask].dropna()
    else:
        ref = series.dropna()
    if ref.empty:
        return series
    cap = ref.quantile(percentile / 100.0)
    clipped = series.clip(upper=cap)
    n_capped = (series > cap).sum()
    if n_capped > 0:
        logger.info(
            "Target clipping: %d valores por encima del p%.0f (cap=%.2f) recortados.",
            n_capped,
            percentile,
            cap,
        )
    return clipped


def prune_multicollinearity(
    df: pd.DataFrame, feature_cols: list[str], threshold: float
) -> list[str]:
    """
    Deja solo una de cada par con correlación > threshold.
    Se conserva la primera de cada par (orden de feature_cols).
    """
    if len(feature_cols) < 2:
        return feature_cols

    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 2:
        return available

    corr = df[available].corr()
    to_drop: set[str] = set()
    for i, a in enumerate(available):
        if a in to_drop:
            continue
        for b in available[i + 1:]:
            if b in to_drop:
                continue
            if abs(corr.loc[a, b]) > threshold:
                to_drop.add(b)
                logger.info(
                    "Poda multicolinealidad: eliminada %s (corr con %s > %.2f).",
                    b,
                    a,
                    threshold,
                )

    kept = [c for c in available if c not in to_drop]
    logger.info(
        "Poda multicolinealidad: %d -> %d features.", len(available), len(kept)
    )
    return kept
