import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.customerio_client import sync_campaigns_to_supabase
from app.services.strategy_agent import (
    execute_strategy,
    generate_basic_strategy,
    generate_premium_strategy,
    get_funnel_health,
)
from app.services.supabase_client import (
    get_admin_status,
    get_all_assignments,
    get_campaigns_cache,
    get_funnel_context,
    get_funnel_steps,
    get_knowledge_base,
    get_latest_strategy,
    get_strategy_history,
    get_user_campaign,
)

router = APIRouter()
logger = logging.getLogger("kepler.strategy_router")


class ExecuteStrategyPayload(BaseModel):
    strategy: dict[str, Any]


class UpdateNodePayload(BaseModel):
    action_id: int    # id_nodo_cio — usado para cooldown y logging
    template_id: int  # ID del template CIO donde vive el copy
    subject: str
    cuerpo: str
    preheader: str | None = None
    user_name: str | None = None
    campaign_name: str | None = None
    semana_label: str | None = None


class ValidateNode(BaseModel):
    id_nodo_cio: int
    template_id: int
    tipo: str                    # 'email' | 'push'
    subject: str
    cuerpo: str
    preheader: str | None = None
    nombre: str | None = None
    campaign_name: str | None = None
    step_code: str | None = None  # funnel_step_mapped de la campaña


class ValidateAndSendPayload(BaseModel):
    nodes: list[ValidateNode]
    semana_label: str
    user_name: str | None = None


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


@router.post("/generate-premium")
def generate_premium() -> dict[str, Any]:
    """
    Genera la estrategia semanal usando el agente premium (campaña Primer Depósito).

    Flujo único: SHAP + Perplexity research + Journey CIO → una sola llamada Claude.
    La campaña se identifica desde Supabase (agent_tier='premium'), no está hardcodeada.
    Requiere: predicción ML en Supabase + ANTHROPIC_API_KEY + PERPLEXITY_API_KEY.
    """
    try:
        return generate_premium_strategy()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate-basic")
def generate_basic() -> dict[str, Any]:
    """
    Genera la revisión semanal del funnel básico usando el agente de orquestación Tier Básico.

    Flujo: Journeys CIO (campañas agent_tier='basic') + Compliance KB + fecha → Claude.
    Sin SHAP ni Perplexity. Deriva contexto solo del calendario colombiano.
    Requiere: ANTHROPIC_API_KEY + campañas con agent_tier='basic' en Supabase.
    """
    try:
        return generate_basic_strategy()
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


@router.get("/latest-structural")
def latest_structural() -> dict[str, Any]:
    """Devuelve el resultado básico (estructural) más reciente, o 404 si no hay ninguno."""
    from app.services.supabase_client import get_latest_structural
    result = get_latest_structural()
    if not result:
        raise HTTPException(status_code=404, detail="No hay resultado estructural guardado")
    return result


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
            user_name=payload.user_name,
            campaign_name=payload.campaign_name,
            semana_label=payload.semana_label,
        )
    except RuntimeError as exc:
        # Cooldown activo → 429. Cualquier otro RuntimeError (config, JWT) → 503.
        status = 429 if "Cooldown activo" in str(exc) else 503
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/sent-nodes")
def sent_nodes(semana_label: str, after: str | None = None) -> dict[str, Any]:
    """
    Devuelve action_ids enviados a CIO para la semana.
    after: ISO timestamp — filtra solo los enviados desde esa fecha (aísla por estrategia).
    """
    from app.services.supabase_client import get_sent_nodes
    return {"semana_label": semana_label, "sent": get_sent_nodes(semana_label, after)}


@router.get("/assignment")
def get_assignment(user_name: str) -> dict[str, Any]:
    """Devuelve la campaña asignada al usuario. Usado por el frontend para filtrar canvas."""
    campaign = get_user_campaign(user_name)
    return {"user_name": user_name, "campaign": campaign}


@router.get("/assignments")
def get_assignments() -> list[dict[str, Any]]:
    """Devuelve todas las asignaciones. Solo para admin."""
    return get_all_assignments()


@router.get("/admin-status")
def admin_status() -> list[dict[str, Any]]:
    """Panel admin: estado de cada agente — nodos actualizados y última actividad."""
    try:
        return get_admin_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/validate-and-send")
async def validate_and_send(payload: ValidateAndSendPayload) -> dict[str, Any]:
    """
    Valida y envía cada nodo a CIO de forma independiente y secuencial.

    Flujo por nodo:
      L1 (determinista, $0) → si falla → status='cambios' con error exacto
      L2 (Claude judge, ~$0.006) → si falla → status='cambios' con razón del juez
      send (update_node_copy) → status='listo' | status='error_envio'

    300ms de delay entre nodos: evita rate limits de CIO + efecto visual en canvas.
    Si el juez L2 falla con excepción (timeout/API error) → se envía igualmente (L1 ya validó).
    """
    from app.services.copy_validator import validate_node
    from app.services.anthropic_client import call_judge_agent
    from app.services.customerio_fly_writer import update_node_copy

    kb_entries = get_knowledge_base()

    # KB completo como texto plano — el judge necesita todo el contexto para
    # verificar cifras correctamente; el keyword-matching parcial perdía entradas.
    kb_full = "\n\n".join(
        f"[{e.get('tipo', '').upper()}] {e.get('titulo', '')}:\n{e.get('contenido', '')}"
        for e in kb_entries
        if e.get("contenido")
    )

    results: list[dict[str, Any]] = []

    for node_payload in payload.nodes:
        node = node_payload.model_dump()
        campaign = {
            "funnel_step_mapped": node.get("step_code") or "",
            "name": node.get("campaign_name") or "",
        }

        await asyncio.sleep(0.3)  # delay visual + rate limit CIO

        # ── L1 — determinista ──────────────────────────────────────────────────
        l1 = validate_node(node, campaign, kb_entries)
        if not l1["passed"]:
            logger.info(
                "[VALIDATE] L1 FALLO nodo %s: %s",
                node_payload.id_nodo_cio, l1["errors"],
            )
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L1",
                "errors":   l1["errors"],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

        # ── L2 — LLM judge ────────────────────────────────────────────────────
        judge_approved = True
        judge_reason: str = ""
        try:
            verdict = call_judge_agent(node, campaign, kb_full)
            judge_approved = bool(verdict.get("aprobado", True))
            judge_reason   = verdict.get("razon", "")
            logger.info(
                "[VALIDATE] L2 %s nodo %s (confianza=%.2f): %s",
                "OK" if judge_approved else "FALLO",
                node_payload.id_nodo_cio,
                verdict.get("confianza", 0.0),
                judge_reason,
            )
        except Exception as exc:
            # Si el judge falla → L1 ya pasó → dejamos pasar para no bloquear
            logger.warning(
                "[VALIDATE] L2 excepción nodo %s — enviando igualmente (L1 OK): %s",
                node_payload.id_nodo_cio, exc,
            )

        if not judge_approved:
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L2",
                "errors":   [judge_reason or "El juez rechazó el copy — revisá la alineación con el objetivo"],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

        # ── Envío a CIO ────────────────────────────────────────────────────────
        try:
            update_node_copy(
                action_id=node_payload.id_nodo_cio,
                template_id=node_payload.template_id,
                subject=node_payload.subject,
                body=node_payload.cuerpo,
                preheader=node_payload.preheader,
                user_name=payload.user_name or "kepler-auto",
                campaign_name=node_payload.campaign_name,
                semana_label=payload.semana_label,
            )
            logger.info("[VALIDATE] ENVIADO nodo %s", node_payload.id_nodo_cio)
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "listo",
                "layer":    "sent",
                "errors":   [],
                "warnings": l1["warnings"],
                "sent":     True,
            })
        except RuntimeError as exc:
            # Cooldown activo (429) o error de config/JWT (503)
            status_code = 429 if "Cooldown activo" in str(exc) else 503
            logger.warning(
                "[VALIDATE] Error envío nodo %s (%d): %s",
                node_payload.id_nodo_cio, status_code, exc,
            )
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "send_error",
                "errors":   [str(exc)],
                "warnings": l1["warnings"],
                "sent":     False,
            })
        except Exception as exc:
            logger.error("[VALIDATE] Error inesperado nodo %s: %s", node_payload.id_nodo_cio, exc)
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "send_error",
                "errors":   [f"Error al enviar: {exc}"],
                "warnings": l1["warnings"],
                "sent":     False,
            })

    total_sent  = sum(1 for r in results if r["sent"])
    total_block = sum(1 for r in results if not r["sent"])
    logger.info("[VALIDATE] Completado: %d enviados, %d bloqueados", total_sent, total_block)

    return {
        "results":      results,
        "total_sent":   total_sent,
        "total_blocked": total_block,
    }


@router.post("/validate-and-send-premium")
async def validate_and_send_premium(payload: ValidateAndSendPayload) -> dict[str, Any]:
    """
    Valida y envía nodos del agente premium a CIO — pipeline independiente del básico.

    L1 premium: SFC, voseo, Liquid, chars, KB rates. Sin restricción de market data.
    L2 premium: verifica solo tasas/montos de producto contra KB; ignora cifras de mercado.
    """
    from app.services.copy_validator import validate_node_premium
    from app.services.anthropic_client import call_judge_agent_premium
    from app.services.customerio_fly_writer import update_node_copy

    kb_entries = get_knowledge_base()
    kb_full = "\n\n".join(
        f"[{e.get('tipo', '').upper()}] {e.get('titulo', '')}:\n{e.get('contenido', '')}"
        for e in kb_entries
        if e.get("contenido")
    )

    results: list[dict[str, Any]] = []

    for node_payload in payload.nodes:
        node = node_payload.model_dump()

        await asyncio.sleep(0.3)

        # ── L1 premium ────────────────────────────────────────────────────────
        l1 = validate_node_premium(node, kb_entries)
        if not l1["passed"]:
            logger.info("[VALIDATE-PREMIUM] L1 FALLO nodo %s: %s",
                        node_payload.id_nodo_cio, l1["errors"])
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L1",
                "errors":   l1["errors"],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

        # ── L2 premium ────────────────────────────────────────────────────────
        judge_approved = True
        judge_reason: str = ""
        try:
            verdict = call_judge_agent_premium(node, kb_full)
            judge_approved = bool(verdict.get("aprobado", True))
            judge_reason   = verdict.get("razon", "")
            logger.info("[VALIDATE-PREMIUM] L2 %s nodo %s (confianza=%.2f): %s",
                        "OK" if judge_approved else "FALLO",
                        node_payload.id_nodo_cio,
                        verdict.get("confianza", 0.0),
                        judge_reason)
        except Exception as exc:
            logger.warning("[VALIDATE-PREMIUM] L2 excepción nodo %s: %s — enviando igual",
                           node_payload.id_nodo_cio, exc)

        if not judge_approved:
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L2",
                "errors":   [judge_reason],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

        # ── Envío a CIO (con reintento automático para errores de patcher HTML) ──
        send_kwargs = dict(
            action_id=node_payload.id_nodo_cio,
            template_id=node_payload.template_id,
            subject=node_payload.subject,
            body=node_payload.cuerpo,
            preheader=node_payload.preheader,
            user_name=payload.user_name or "kepler-auto",
            campaign_name=node_payload.campaign_name,
            semana_label=payload.semana_label,
        )
        sent_ok = False
        send_error: str = ""
        for attempt in range(2):
            try:
                update_node_copy(**send_kwargs)
                sent_ok = True
                logger.info("[VALIDATE-PREMIUM] ENVIADO nodo %s (intento %d)",
                            node_payload.id_nodo_cio, attempt + 1)
                break
            except Exception as exc:
                send_error = str(exc)
                logger.warning("[VALIDATE-PREMIUM] Intento %d falló nodo %s: %s",
                               attempt + 1, node_payload.id_nodo_cio, exc)
                if attempt == 0:
                    await asyncio.sleep(1.5)  # breve pausa antes del reintento

        if sent_ok:
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "listo",
                "errors":   [],
                "warnings": l1["warnings"],
                "sent":     True,
            })
        else:
            logger.error("[VALIDATE-PREMIUM] Error CIO nodo %s tras 2 intentos: %s",
                         node_payload.id_nodo_cio, send_error)
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "send_error",
                "errors":   [send_error],
                "warnings": l1["warnings"],
                "sent":     False,
            })

    total_sent  = sum(1 for r in results if r["sent"])
    total_block = sum(1 for r in results if not r["sent"])
    logger.info("[VALIDATE-PREMIUM] Completado: %d enviados, %d bloqueados", total_sent, total_block)

    return {
        "results":       results,
        "total_sent":    total_sent,
        "total_blocked": total_block,
    }


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
