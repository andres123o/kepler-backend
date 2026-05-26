from typing import Any

from fastapi import APIRouter, HTTPException

from app.services.campaign_monitor import run_weekly_check

router = APIRouter()


@router.get("/check")
def weekly_check() -> dict[str, Any]:
    """
    Analiza el estado de las campañas monitoreadas.
    Lee desde el cache de Supabase — no llama a CIO.
    Para datos frescos, sincronizar antes desde /api/strategy/sync.
    """
    try:
        return run_weekly_check()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
