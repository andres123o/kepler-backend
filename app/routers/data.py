from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.supabase_client import (
    append_to_master,
    clear_ultima_semana,
    get_master_df,
    get_ultima_semana_row,
    save_ultima_semana,
)

router = APIRouter()


class UltimaSemanaPaylod(BaseModel):
    semana: str
    usuarios_registro_base: int | None = None
    step_09_full_account: int | None = None
    tasa_basic_a_risk: float | None = None
    tasa_risk_a_fulldata: float | None = None
    tasa_fulldata_a_video: float | None = None
    tasa_video_a_review: float | None = None
    tasa_review_a_aprobado: float | None = None
    tasa_registro_a_aprobado: float | None = None
    tasa_rechazo_implicita_kyc: float | None = None
    mediana_dias_registro_a_full: float | None = None
    pct_perfil_conservador: float | None = None
    pct_perfil_arriesgado: float | None = None
    usuarios_primer_cashin: int | None = None
    full_users_aprobados: int | None = None
    push_mail_delivered_pre_deposito: int | None = None
    push_mail_converted_pre_deposito: int | None = None
    cx_friccion_kyc: int | None = None
    cx_bloqueos: int | None = None
    tasa_intervencion_mensual: float | None = None
    trm: float | None = None
    variacion_colcap: float | None = None
    intervencion_kepler: int | None = None


@router.get("/ultima-semana")
def read_ultima_semana() -> dict[str, Any]:
    """Devuelve la fila actual de ultima_semana (nombres Supabase)."""
    row = get_ultima_semana_row()
    if row is None:
        return {}
    return row


@router.post("/ultima-semana")
def write_ultima_semana(body: UltimaSemanaPaylod) -> dict[str, Any]:
    """Reemplaza ultima_semana con los datos de la semana nueva."""
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        saved = save_ultima_semana(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return saved


@router.post("/ultima-semana/confirmar")
def confirmar_semana(body: UltimaSemanaPaylod) -> dict[str, Any]:
    """
    Cierra la semana: copia ultima_semana a master_consolidado_final.
    Llamar después de que se haya corrido la predicción y los datos estén OK.
    """
    payload = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        saved = append_to_master(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return saved


@router.post("/nueva-proyeccion")
def nueva_proyeccion() -> dict[str, Any]:
    """
    Flujo del domingo:
    1. Lee ultima_semana
    2. Copia esa fila a master_consolidado_final
    3. Borra ultima_semana (deja la tabla lista para la semana nueva)
    """
    row = get_ultima_semana_row()
    if not row:
        raise HTTPException(
            status_code=422,
            detail="ultima_semana está vacía. No hay datos para archivar.",
        )
    semana_archivada = row.get("semana", "")
    try:
        append_to_master({k: v for k, v in row.items() if k != "id"})
        clear_ultima_semana()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "semana_archivada": semana_archivada}


@router.get("/master")
def read_master(limit: int = 20) -> list[dict[str, Any]]:
    """Devuelve las últimas N filas de master_consolidado_final."""
    try:
        df = get_master_df()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if df.empty:
        return []
    tail = df.tail(limit)
    return tail.where(tail.notna(), None).to_dict(orient="records")
