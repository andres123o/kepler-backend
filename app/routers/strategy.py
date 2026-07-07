import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.services.customerio_client import sync_campaigns_to_supabase
from app.services.strategy_agent import (
    execute_strategy,
    generate_basic_strategy,
    generate_premium_strategy,
    get_funnel_health,
)
from app.services.supabase_client import FunnelClient, get_funnel_client

router = APIRouter()
logger = logging.getLogger("kepler.strategy_router")


class GeneratePremiumRequest(BaseModel):
    market_research: dict | None = None


class ExecuteStrategyPayload(BaseModel):
    strategy: dict[str, Any]


class UpdateNodePayload(BaseModel):
    action_id: int
    template_id: int
    subject: str
    cuerpo: str
    preheader: str | None = None
    user_name: str | None = None
    campaign_name: str | None = None
    semana_label: str | None = None


class ValidateNode(BaseModel):
    id_nodo_cio: int
    template_id: int
    tipo: str
    subject: str
    cuerpo: str
    preheader: str | None = None
    nombre: str | None = None
    campaign_name: str | None = None
    step_code: str | None = None


class ValidateAndSendPayload(BaseModel):
    nodes: list[ValidateNode]
    semana_label: str
    user_name: str | None = None


@router.get("/safety-status")
def safety_status() -> dict[str, Any]:
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
def sync_campaigns(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        result = sync_campaigns_to_supabase(fc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.get("/funnel-health")
def funnel_health(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        return get_funnel_health(fc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/fetch-research")
def fetch_research(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    from app.services.perplexity_client import fetch_market_research
    try:
        prediction   = fc.get_latest_prediction()
        semana_label = (prediction or {}).get("semana_label") or (prediction or {}).get("semana_datos", "")
        return fetch_market_research(semana_label, fc=fc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate-premium")
def generate_premium(
    body: GeneratePremiumRequest,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    try:
        return generate_premium_strategy(fc, market_research=body.market_research)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate-basic")
def generate_basic(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        return generate_basic_strategy(fc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/system-context")
def system_context(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        campaigns = fc.get_campaigns_cache()
        slim_campaigns = [
            {k: v for k, v in c.items() if k != "metrics_weekly_json"}
            for c in campaigns
        ]
        context    = fc.get_funnel_context()
        events     = [c for c in context if c.get("record_type") == "event"]
        attributes = [c for c in context if c.get("record_type") == "attribute"]
        return {
            "funnel_steps":    fc.get_funnel_steps(),
            "campaigns_cache": slim_campaigns,
            "events":          events,
            "attributes":      attributes,
            "knowledge_base":  fc.get_knowledge_base(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest")
def latest_strategy(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    try:
        result = fc.get_latest_strategy()
        if result is None:
            raise HTTPException(status_code=404, detail="No hay estrategias guardadas.")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history")
def strategy_history(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        return fc.get_strategy_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/latest-structural")
def latest_structural(fc: FunnelClient = Depends(get_funnel_client)) -> dict[str, Any]:
    result = fc.get_latest_structural()
    if not result:
        raise HTTPException(status_code=404, detail="No hay resultado estructural guardado")
    return result


@router.post("/update-node")
def update_node(
    payload: UpdateNodePayload,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
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
            fc=fc,
        )
    except RuntimeError as exc:
        status = 429 if "Cooldown activo" in str(exc) else 503
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/sent-nodes")
def sent_nodes(
    semana_label: str,
    after: str | None = None,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    return {"semana_label": semana_label, "sent": fc.get_sent_nodes(semana_label, after)}


@router.get("/assignment")
def get_assignment(
    user_name: str,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    return {"user_name": user_name, "campaign": fc.get_user_campaign(user_name)}


@router.get("/assignments")
def get_assignments(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    return fc.get_all_assignments()


@router.get("/admin-status")
def admin_status(fc: FunnelClient = Depends(get_funnel_client)) -> list[dict[str, Any]]:
    try:
        return fc.get_admin_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/validate-and-send")
async def validate_and_send(
    payload: ValidateAndSendPayload,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    from app.services.copy_validator import validate_node
    from app.services.anthropic_client import call_judge_agent
    from app.services.customerio_fly_writer import update_node_copy

    funnel_cfg = fc.get_funnel_config()
    company_description = funnel_cfg.get("company_description") or ""
    kb_entries = fc.get_knowledge_base()
    kb_full = "\n\n".join(
        f"[{e.get('tipo', '').upper()}] {e.get('titulo', '')}:\n{e.get('contenido', '')}"
        for e in kb_entries
        if e.get("contenido")
    )
    rules = funnel_cfg.get("validation_rules") or {}
    if not rules:
        raise HTTPException(
            status_code=422,
            detail=(
                "Este funnel no tiene 'validation_rules' configuradas. "
                "Agrega el campo al config JSONB del funnel en Supabase antes de validar y enviar."
            ),
        )

    results: list[dict[str, Any]] = []

    for node_payload in payload.nodes:
        node = node_payload.model_dump()
        campaign = {
            "funnel_step_mapped": node.get("step_code") or "",
            "name": node.get("campaign_name") or "",
        }

        await asyncio.sleep(0.3)

        l1 = validate_node(node, campaign, kb_entries, rules=rules)
        if not l1["passed"]:
            logger.info("[VALIDATE] L1 FALLO nodo %s: %s", node_payload.id_nodo_cio, l1["errors"])
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L1",
                "errors":   l1["errors"],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

        judge_approved = True
        judge_reason: str = ""
        try:
            verdict = call_judge_agent(node, campaign, kb_full, company_description)
            judge_approved = bool(verdict.get("aprobado", True))
            judge_reason   = verdict.get("razon", "")
            logger.info("[VALIDATE] L2 %s nodo %s (confianza=%.2f): %s",
                        "OK" if judge_approved else "FALLO",
                        node_payload.id_nodo_cio, verdict.get("confianza", 0.0), judge_reason)
        except Exception as exc:
            logger.warning("[VALIDATE] L2 excepción nodo %s — enviando igualmente (L1 OK): %s",
                           node_payload.id_nodo_cio, exc)

        if not judge_approved:
            results.append({
                "id_nodo_cio": node_payload.id_nodo_cio,
                "status":   "cambios",
                "layer":    "L2",
                "errors":   [judge_reason or "El juez rechazó el copy"],
                "warnings": l1["warnings"],
                "sent":     False,
            })
            continue

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
                fc=fc,
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
            status_code = 429 if "Cooldown activo" in str(exc) else 503
            logger.warning("[VALIDATE] Error envío nodo %s (%d): %s",
                           node_payload.id_nodo_cio, status_code, exc)
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
    return {"results": results, "total_sent": total_sent, "total_blocked": total_block}


@router.post("/validate-and-send-premium")
async def validate_and_send_premium(
    payload: ValidateAndSendPayload,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    from app.services.copy_validator import validate_node_premium
    from app.services.anthropic_client import call_judge_agent_premium
    from app.services.customerio_fly_writer import update_node_copy

    funnel_cfg = fc.get_funnel_config()
    company_description = funnel_cfg.get("company_description") or ""
    kb_entries = fc.get_knowledge_base()
    kb_full = "\n\n".join(
        f"[{e.get('tipo', '').upper()}] {e.get('titulo', '')}:\n{e.get('contenido', '')}"
        for e in kb_entries
        if e.get("contenido")
    )
    rules = funnel_cfg.get("validation_rules") or {}
    if not rules:
        raise HTTPException(
            status_code=422,
            detail=(
                "Este funnel no tiene 'validation_rules' configuradas. "
                "Agrega el campo al config JSONB del funnel en Supabase antes de validar y enviar."
            ),
        )

    # Cifras de mercado verificadas esta semana — grounded contra el research real
    # (ver anthropic_client.extract_market_cifras). Sin esto, L1/L2 no pueden distinguir
    # una cifra real de una alucinada.
    strategy_saved = fc.get_strategy_by_semana(payload.semana_label)
    cifras_verificadas = (strategy_saved or {}).get("research_cifras") or []
    logger.info("[VALIDATE-PREMIUM] Cifras de mercado verificadas para semana=%s: %d",
                payload.semana_label, len(cifras_verificadas))

    results: list[dict[str, Any]] = []

    for node_payload in payload.nodes:
        node = node_payload.model_dump()
        await asyncio.sleep(0.3)

        l1 = validate_node_premium(node, kb_entries, rules=rules, cifras_verificadas=cifras_verificadas)
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

        judge_approved = True
        judge_reason: str = ""
        try:
            verdict = call_judge_agent_premium(node, kb_full, company_description, cifras_verificadas)
            judge_approved = bool(verdict.get("aprobado", True))
            judge_reason   = verdict.get("razon", "")
            logger.info("[VALIDATE-PREMIUM] L2 %s nodo %s (confianza=%.2f): %s",
                        "OK" if judge_approved else "FALLO",
                        node_payload.id_nodo_cio,
                        verdict.get("confianza", 0.0), judge_reason)
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

        send_kwargs = dict(
            action_id=node_payload.id_nodo_cio,
            template_id=node_payload.template_id,
            subject=node_payload.subject,
            body=node_payload.cuerpo,
            preheader=node_payload.preheader,
            user_name=payload.user_name or "kepler-auto",
            campaign_name=node_payload.campaign_name,
            semana_label=payload.semana_label,
            fc=fc,
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
                    await asyncio.sleep(1.5)

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
    return {"results": results, "total_sent": total_sent, "total_blocked": total_block}


@router.post("/execute")
def execute(
    body: ExecuteStrategyPayload,
    fc: FunnelClient = Depends(get_funnel_client),
) -> dict[str, Any]:
    try:
        return execute_strategy(body.strategy, fc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
