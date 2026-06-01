import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.customerio_client import sync_campaigns_to_supabase
from app.services.strategy_agent import (
    execute_strategy,
    generate_structural_optimization,
    generate_weekly_strategy,
    get_funnel_health,
)
from app.services.supabase_client import (
    get_all_assignments,
    get_campaigns_cache,
    get_funnel_context,
    get_funnel_steps,
    get_knowledge_base,
    get_latest_strategy,
    get_latest_structural,
    get_strategy_history,
    get_user_campaign,
)

router = APIRouter()


class ExecuteStrategyPayload(BaseModel):
    strategy: dict[str, Any]


class UpdateNodePayload(BaseModel):
    action_id: int    # id_nodo_cio — usado para cooldown y logging
    template_id: int  # ID del template CIO donde vive el copy
    subject: str
    cuerpo: str
    preheader: str | None = None


class GeneratePayload(BaseModel):
    contexto_adicional: str | None = None


class GenerateStructuralPayload(BaseModel):
    phase2_strategy: dict[str, Any]
    contexto_adicional: str | None = None


@router.get("/safety-status")
def safety_status() -> dict[str, Any]:
    """
    Muestra el estado de los controles de seguridad de escritura a CIO.
    Revisar antes de ejecutar cualquier estrategia.
    """
    dry_run = os.getenv("CIO_DRY_RUN", "true").lower() == "true"
    max_ops = int(os.getenv("CIO_MAX_CAMPAIGNS_PER_EXECUTE", "3"))
    has_key = bool(os.getenv("CUSTOMERIO_APP_API_KEY", ""))
    return {
        "cio_dry_run": dry_run,
        "max_campaigns_per_execute": max_ops,
        "cio_key_configured": has_key,
        "escrituras_bloqueadas": dry_run,
        "mensaje": (
            "MODO SEGURO ACTIVO — ninguna escritura llegará a CIO."
            if dry_run
            else "Modo escritura activo. Las estrategias aprobadas SÍ se ejecutarán en CIO."
        ),
    }


@router.post("/sync")
def sync_campaigns() -> dict[str, Any]:
    """
    Sincroniza las campañas de Customer.io con el cache de Supabase.
    Requiere CUSTOMERIO_APP_API_KEY. Correr antes de generate.
    """
    try:
        result = sync_campaigns_to_supabase()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.get("/funnel-health")
def funnel_health() -> list[dict[str, Any]]:
    """
    Estado de salud del funnel: semáforo verde/amarillo/rojo por paso.
    Lee desde el cache de Supabase (no necesita API de CIO).
    """
    try:
        return get_funnel_health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate")
def generate(payload: GeneratePayload | None = None) -> dict[str, Any]:
    """
    Genera el preview de estrategia semanal usando la última predicción ML.
    Requiere: última predicción en Supabase + ANTHROPIC_API_KEY.
    No ejecuta nada en CIO — solo genera el preview para aprobación.
    Acepta opcionalmente contexto_adicional: noticias, eventos, contexto de negocio.
    """
    try:
        ctx = (payload.contexto_adicional or "").strip() if payload else ""

        return generate_weekly_strategy(
            contexto_adicional=ctx or None,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest-structural")
def latest_structural() -> dict[str, Any]:
    """Devuelve el resultado estructural más reciente (Fase 2B), o 404 si no hay."""
    try:
        result = get_latest_structural()
        if result is None:
            raise HTTPException(status_code=404, detail="No hay análisis estructural guardado.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate-structural")
def generate_structural(payload: GenerateStructuralPayload) -> dict[str, Any]:
    """
    Fase 2B: optimiza estructuralmente las campañas que Phase 2 no intervino.
    Inputs: detalle manual de campañas + estrategia de Phase 2 (para exclusión).
    No requiere predicción ML — análisis de salud, cadencia y copy independiente del SHAP.
    """
    try:
        ctx = (payload.contexto_adicional or "").strip() or None
        return generate_structural_optimization(
            phase2_strategy=payload.phase2_strategy,
            contexto_adicional=ctx,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/system-context")
def system_context() -> dict[str, Any]:
    """
    Retorna todos los datos de contexto activos en el sistema:
    funnel_steps, campañas en cache, eventos CIO, atributos CIO, knowledge base.
    Muestra exactamente qué datos ve Claude al generar la estrategia.
    """
    try:
        campaigns = get_campaigns_cache()
        # Excluir weekly JSON del payload (muy grande, no necesario en esta vista)
        slim_campaigns = [
            {k: v for k, v in c.items() if k != "metrics_weekly_json"}
            for c in campaigns
        ]

        context = get_funnel_context()
        events     = [c for c in context if c.get("record_type") == "event"]
        attributes = [c for c in context if c.get("record_type") == "attribute"]

        return {
            "funnel_steps":     get_funnel_steps(),
            "campaigns_cache":  slim_campaigns,
            "events":           events,
            "attributes":       attributes,
            "knowledge_base":   get_knowledge_base(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest")
def latest_strategy() -> dict[str, Any]:
    """Devuelve la estrategia más reciente guardada, o 404 si no hay ninguna."""
    try:
        result = get_latest_strategy()
        if result is None:
            raise HTTPException(status_code=404, detail="No hay estrategias guardadas.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history")
def strategy_history() -> list[dict[str, Any]]:
    """Devuelve el historial completo de estrategias en orden descendente."""
    try:
        return get_strategy_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/update-node")
def update_node(payload: UpdateNodePayload) -> dict[str, Any]:
    """
    Actualiza el copy de un nodo específico en Customer.io.
    Máximo 2 requests a CIO por llamada (GET template + PUT template).
    Cooldown de 30s por nodo para prevenir actualizaciones duplicadas.
    """
    from app.services.customerio_fly_writer import update_node_copy
    try:
        return update_node_copy(
            action_id=payload.action_id,
            template_id=payload.template_id,
            subject=payload.subject,
            body=payload.cuerpo,
            preheader=payload.preheader,
        )
    except RuntimeError as exc:
        # Cooldown activo → 429. Cualquier otro RuntimeError (config, JWT) → 503.
        status = 429 if "Cooldown activo" in str(exc) else 503
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/assignment")
def get_assignment(user_name: str) -> dict[str, Any]:
    """Devuelve la campaña asignada al usuario. Usado por el frontend para filtrar canvas."""
    campaign = get_user_campaign(user_name)
    return {"user_name": user_name, "campaign": campaign}


@router.get("/assignments")
def get_assignments() -> list[dict[str, Any]]:
    """Devuelve todas las asignaciones. Solo para admin."""
    return get_all_assignments()


@router.post("/execute")
def execute(body: ExecuteStrategyPayload) -> dict[str, Any]:
    """
    Ejecuta la estrategia aprobada en Customer.io.
    Requiere CUSTOMERIO_APP_API_KEY.
    Solo llegar acá después de que el usuario aprobó el preview en /generate.
    """
    try:
        return execute_strategy(body.strategy)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
