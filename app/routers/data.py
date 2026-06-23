from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from app.services.supabase_client import FunnelClient, get_funnel_client

router = APIRouter()


class IngestionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    semana: str


@router.get("/ultima-semana")
def read_ultima_semana(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    row = fc.get_ultima_semana_row()
    return row or {}


@router.post("/ultima-semana")
def write_ultima_semana(body: IngestionPayload, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return fc.save_ultima_semana(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ultima-semana/confirmar")
def confirmar_semana(body: IngestionPayload, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return fc.append_to_master(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/nueva-proyeccion")
def nueva_proyeccion(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    row = fc.get_ultima_semana_row()
    if not row:
        raise HTTPException(status_code=422, detail="ultima_semana está vacía. No hay datos para archivar.")
    semana_archivada = row.get("semana", "")
    try:
        fc.append_to_master({k: v for k, v in row.items() if k != "id"})
        fc.clear_ultima_semana()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "semana_archivada": semana_archivada}


@router.get("/auto-variables")
def get_auto_variables(
    semana: str = Query(...),
    banrep_tasa: float | None = Query(None),
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    from app.services.market_data_fetcher import fetch_auto_variables
    try:
        return fetch_auto_variables(semana, banrep_tasa=banrep_tasa, fc=fc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/master")
def read_master(limit: int = 20, fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        df = fc.get_master_df()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if df.empty:
        return []
    tail = df.tail(limit)
    return tail.where(tail.notna(), None).to_dict(orient="records")
