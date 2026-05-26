from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.ml_runner import get_training_status, run_prediction, run_training
from app.services.supabase_client import get_latest_prediction, get_prediction_history

router = APIRouter()


@router.post("/predict")
def predict() -> dict[str, Any]:
    """
    Lee ultima_semana + master desde Supabase, corre el pipeline completo
    y devuelve predicción + SHAP + contexto + prescripción.
    """
    try:
        result = run_prediction()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/train")
def train() -> dict[str, Any]:
    """
    Descarga master_consolidado_final de Supabase y reentrena el modelo XGBoost.
    Puede tardar varios minutos — no usar en Vercel serverless con timeout corto.
    """
    try:
        summary = run_training()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return summary


@router.get("/training-status")
def training_status() -> dict[str, Any]:
    """Devuelve métricas del último modelo entrenado en models/."""
    try:
        return get_training_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest-prediction")
def latest_prediction() -> dict[str, Any]:
    """Devuelve la predicción más reciente. Retorna {} si no hay ninguna."""
    try:
        result = get_latest_prediction()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result or {}


@router.get("/prediction-history")
def prediction_history() -> list[dict[str, Any]]:
    """
    Devuelve todas las predicciones guardadas (más reciente primero).
    Incluye full_result para que el frontend pueda navegar el historial sin
    hacer una llamada adicional por semana.
    """
    try:
        return get_prediction_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
