from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import bigquery_client

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
    """Verifica si BigQuery está configurado."""
    return {
        "configured": bigquery_client.is_configured(),
        "message": (
            "BigQuery configurado correctamente"
            if bigquery_client.is_configured()
            else "Falta configurar BQ_CREDENTIALS_PATH o BQ_SERVICE_ACCOUNT_JSON en .env"
        ),
    }


@router.post("/run")
def run_measurement(body: MeasurementRequest) -> dict[str, Any]:
    """
    Mide el impacto de una campaña comparando grupo test vs control.
    - test_user_ids: lista de IDs de usuarios que recibieron la campaña (del reporte CIO)
    - start_date / end_date: ventana sobre date_full_user en BigQuery
    """
    if not bigquery_client.is_configured():
        raise HTTPException(
            status_code=503,
            detail="BigQuery no configurado. Agrega BQ_CREDENTIALS_PATH o BQ_SERVICE_ACCOUNT_JSON en .env",
        )
    try:
        from app.services.measurement import run_measurement as _run
        return _run(
            test_user_ids=body.test_user_ids,
            start_date=body.start_date,
            end_date=body.end_date,
            campaign_name=body.campaign_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
