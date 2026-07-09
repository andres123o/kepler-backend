"""
Config ML por funnel — nada de esto vive hardcodeado en ml_pipeline/*.py.

Fuente de verdad: funnels.config.ml (JSONB en Supabase). Se lee vía
FunnelClient — montar un funnel nuevo no requiere tocar ningún .py del
backend, solo llenar estas llaves en el config JSONB del funnel (dato, no
código). Ver CREAR_FUNNEL.md Paso 6.

`feature_enrichers` es la lista de transformaciones de dominio que el
funnel necesita (festivos, ciclo de conversión ponderado, lags, tendencias)
— ver ml_pipeline/enrichers.py por el catálogo disponible. Un funnel de un
dominio completamente distinto simplemente deja `feature_enrichers: []` y
entrena directo sobre funnel_features + macro_features, sin ninguna
transformación — el motor (contratos, poda de multicolinealidad, Optuna +
XGBoost, walk-forward CV) es agnóstico al dominio.

Para trabajo local sin Supabase (ej. probar antes de crear el funnel), se
puede pasar un dict leído de un .json temporal — load_funnel_ml_config() no
le importa el origen. Ese .json es un artefacto de trabajo, no un archivo
del proyecto: una vez el funnel existe en Supabase, la única fuente de
verdad es funnels.config.ml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ml_pipeline.config import MULTICOLLINEARITY_THRESHOLD as _DEFAULT_MULTICOLLINEARITY_THRESHOLD

_REQUIRED_KEYS = ("funnel_features", "macro_features")


@dataclass
class FunnelMLConfig:
    target_name: str
    date_column: str
    funnel_features: list[str]
    macro_features: list[str]
    feature_enrichers: list[dict] = field(default_factory=list)
    non_negative_columns: list[str] = field(default_factory=list)
    non_actionable_features: set[str] = field(default_factory=set)
    multicollinearity_threshold: float = _DEFAULT_MULTICOLLINEARITY_THRESHOLD
    prediction_horizon_weeks: int = 1

    @property
    def all_csv_features(self) -> list[str]:
        return list(self.funnel_features) + list(self.macro_features)


def load_funnel_ml_config(ml_cfg: dict) -> FunnelMLConfig:
    """
    Construye FunnelMLConfig desde el dict 'ml' del config JSONB del funnel.

    Llaves esperadas en ml_cfg:
      funnel_features (list[str])         — REQUERIDO. Columnas internas del funnel en el CSV.
      macro_features (list[str])          — REQUERIDO. Columnas externas (macro/mercado/calendario) en el CSV.
      target_name (str)                   — default 'usuarios_primer_cashin'.
      date_column (str)                   — default 'semana'.
      feature_enrichers (list[dict])      — default []. Cada uno: {"type": "...", "params": {...}}.
                                             Ver ml_pipeline/enrichers.ENRICHER_REGISTRY.
      non_negative_columns (list[str])    — default [].
      non_actionable_features (list[str]) — default [] (features que el modelo usa pero
                                             Marketing no puede accionar vía CIO — se
                                             excluyen del SHAP display).
      multicollinearity_threshold (float) — default el global de config.py (0.92).
      prediction_horizon_weeks (int)      — default 1 (predice t+1). Cuántas semanas
                                             adelante se shiftea el target.

    Lanza ValueError si faltan funnel_features o macro_features — sin esto no
    hay pipeline de features posible, mejor fallar temprano que entrenar con
    0 features silenciosamente.
    """
    missing = [k for k in _REQUIRED_KEYS if not ml_cfg.get(k)]
    if missing:
        raise ValueError(
            f"Config ML incompleto — faltan las llaves {missing} en funnels.config.ml. "
            "Un funnel nuevo necesita 'funnel_features' y 'macro_features' (listas de "
            "nombres de columna del CSV/master) antes de poder entrenar. Ver CREAR_FUNNEL.md Paso 6."
        )

    return FunnelMLConfig(
        target_name=ml_cfg.get("target_name", "usuarios_primer_cashin"),
        date_column=ml_cfg.get("date_column", "semana"),
        funnel_features=list(ml_cfg["funnel_features"]),
        macro_features=list(ml_cfg["macro_features"]),
        feature_enrichers=list(ml_cfg.get("feature_enrichers", [])),
        non_negative_columns=list(ml_cfg.get("non_negative_columns", [])),
        non_actionable_features=set(ml_cfg.get("non_actionable_features", [])),
        multicollinearity_threshold=float(
            ml_cfg.get("multicollinearity_threshold", _DEFAULT_MULTICOLLINEARITY_THRESHOLD)
        ),
        prediction_horizon_weeks=int(ml_cfg.get("prediction_horizon_weeks", 1)),
    )


def load_funnel_ml_config_from_json(path: str) -> FunnelMLConfig:
    """Solo para probar localmente antes de que el funnel exista en Supabase —
    el .json es un artefacto de trabajo temporal, no un archivo del proyecto."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return load_funnel_ml_config(json.load(f))
