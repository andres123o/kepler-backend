"""
Contexto de mercado — RSS eliminado.
El contexto de noticias/mercado entra por el campo contexto_adicional del usuario.
TRM/COLCAP/Tasa ya están en las features SHAP del modelo ML.
"""
from typing import Any


def get_market_context() -> dict[str, Any]:
    return {"trm": None, "noticias": []}
