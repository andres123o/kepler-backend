from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.supabase_client import FunnelClient, get_funnel_client

router = APIRouter()


class AddCampaignPayload(BaseModel):
    cio_campaign_id: str


class UpdateKBPayload(BaseModel):
    activo: bool | None = None
    titulo: str | None = None
    contenido: str | None = None


class AddKBPayload(BaseModel):
    tipo: str
    titulo: str
    contenido: str


class UpdatePromptCompositePayload(BaseModel):
    content: str


@router.get("/tracked-campaigns")
def list_tracked_campaigns(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        campaigns = fc.get_campaigns_cache()
        return [
            {
                "cio_campaign_id": c["cio_campaign_id"],
                "name": c.get("name") or "",
                "status": c.get("status"),
                "funnel_step_mapped": c.get("funnel_step_mapped"),
                "trigger_event": c.get("trigger_event"),
                "delivery_rate": c.get("delivery_rate") or 0.0,
                "open_rate": c.get("open_rate") or 0.0,
                "conversion_rate": c.get("conversion_rate") or 0.0,
                "total_sent": c.get("total_sent") or 0,
                "last_synced_at": c.get("last_synced_at"),
                "unmapped_warning": (
                    not c.get("funnel_step_mapped")
                    and (c.get("total_sent") or 0) > 500
                ),
            }
            for c in campaigns
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/tracked-campaigns")
def add_campaign(body: AddCampaignPayload, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    cid = body.cio_campaign_id.strip()
    if not cid.isdigit():
        raise HTTPException(status_code=422, detail="El ID de campaña debe ser numérico")
    try:
        result = fc.add_tracked_campaign(cid)
        fc.log_audit("tracked_campaign_add", {"cio_campaign_id": cid})
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/tracked-campaigns/{campaign_id}")
def remove_campaign(campaign_id: str, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        fc.delete_tracked_campaign(campaign_id)
        fc.log_audit("tracked_campaign_delete", {"cio_campaign_id": campaign_id})
        return {"ok": True, "deleted": campaign_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/knowledge-base")
def list_knowledge_base(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        return fc.get_all_knowledge_base()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/knowledge-base")
def add_kb_entry(body: AddKBPayload, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    tipo = body.tipo.strip().upper()
    titulo = body.titulo.strip()
    contenido = body.contenido.strip()
    if not tipo or not titulo or not contenido:
        raise HTTPException(status_code=422, detail="tipo, titulo y contenido son obligatorios")
    try:
        result = fc.insert_knowledge_base_entry(tipo, titulo, contenido)
        fc.log_audit("kb_add", {"tipo": tipo, "titulo": titulo})
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/knowledge-base/{entry_id}")
def delete_kb_entry(entry_id: str, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        fc.delete_knowledge_base_entry(entry_id)
        fc.log_audit("kb_delete", {"entry_id": entry_id})
        return {"ok": True, "deleted": entry_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/knowledge-base/{entry_id}")
def update_kb_entry(entry_id: str, body: UpdateKBPayload, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No hay campos para actualizar")
    try:
        result = fc.update_knowledge_base_entry(entry_id, updates)
        fc.log_audit("kb_update", {"entry_id": entry_id, "fields": list(updates.keys())})
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/prompts/composite/{agent_type}")
def get_prompt_composite(agent_type: str, fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        return fc.get_prompt_composite(agent_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/prompts/composite/{agent_type}")
def update_prompt_composite(
    agent_type: str,
    body: UpdatePromptCompositePayload,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="El contenido no puede estar vacío")
    try:
        result = fc.save_prompt_composite(agent_type, content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    fc.log_audit("prompt_update", {"agent_type": agent_type})
    fresh = fc.get_prompt_composite(agent_type)
    return {**result, **fresh}
