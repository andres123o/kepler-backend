from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class MeasurementRequest(BaseModel):
    campaign_name: str = ""
    test_user_ids: list[str]
    start_date: str   # YYYY-MM-DD  (date_full_user desde)
    end_date: str     # YYYY-MM-DD  (date_full_user hasta)


class SaveSnapshotPayload(BaseModel):
    semana_label: str
    inicio_semana: str | None = None  # YYYY-MM-DD
    fin_semana: str | None = None     # YYYY-MM-DD
    model_version: str = ""
    html_content: str


@router.get("/status")
def bq_status() -> dict[str, Any]:
    return {"configured": False, "message": "BigQuery no configurado (FASE 4 pendiente)"}


@router.post("/run")
def run_measurement(body: MeasurementRequest) -> dict[str, Any]:
    raise HTTPException(status_code=503, detail="BigQuery no configurado (FASE 4 pendiente)")


# ── Snapshots manuales (HTML generado por Claude desde resultados BQ) ─────────

@router.post("/save-snapshot")
def save_snapshot(body: SaveSnapshotPayload) -> dict[str, Any]:
    """
    Guarda un snapshot de medición semanal.
    Flujo: corres las queries en BQ → pegas resultados en Claude → Claude genera HTML
    → pegas el HTML aquí → queda guardado y visible en el historial.
    """
    from app.services.supabase_client import save_measurement_snapshot
    try:
        return save_measurement_snapshot(
            semana_label=body.semana_label,
            html_content=body.html_content,
            inicio_semana=body.inicio_semana,
            fin_semana=body.fin_semana,
            model_version=body.model_version,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/snapshots")
def list_snapshots() -> list[dict[str, Any]]:
    """Lista todos los snapshots guardados (solo metadatos, sin html_content)."""
    from app.services.supabase_client import get_measurement_snapshots
    try:
        return get_measurement_snapshots()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str) -> dict[str, Any]:
    """Devuelve un snapshot completo por ID (incluye html_content)."""
    from app.services.supabase_client import get_measurement_snapshot
    try:
        snap = get_measurement_snapshot(snapshot_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="Snapshot no encontrado")
        return snap
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
