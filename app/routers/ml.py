from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.services.ml_runner import get_training_status, run_prediction, run_training
from app.services.supabase_client import FunnelClient, get_funnel_client

router = APIRouter()


@router.post("/predict")
def predict(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        result = run_prediction(fc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.post("/train")
def train(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        summary = run_training(fc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return summary


@router.get("/training-status")
def training_status(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        return get_training_status(fc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest-prediction")
def latest_prediction(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    result = fc.get_latest_prediction()
    return result or {}


@router.get("/prediction-history")
def prediction_history(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    return fc.get_prediction_history()
