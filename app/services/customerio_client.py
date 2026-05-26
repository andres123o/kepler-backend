"""
Customer.io App API client para Kepler.

Base URL: https://api.customer.io/v1
Auth:     Authorization: Bearer <CUSTOMERIO_APP_API_KEY>

SEGURIDAD:
- Kepler NUNCA llama la Track API (track.customer.io) — esa crea/modifica contactos.
- Kepler SOLO usa esta App API para leer/crear campañas (journeys).
- CIO_DRY_RUN=true en .env bloquea TODAS las escrituras (modo seguro por defecto).
- MAX_CAMPAIGNS_PER_EXECUTE=3 limita cuántas campañas se crean/modifican por llamada.
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.services.supabase_client import (
    get_campaigns_cache,
    get_funnel_steps,
    get_tracked_campaign_ids,
    upsert_campaigns_cache,
)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.customerio")

CIO_APP_API_KEY: str = os.getenv("CUSTOMERIO_APP_API_KEY", "")
CIO_BASE_URL = "https://api.customer.io/v1"

_DRY_RUN: bool = os.getenv("CIO_DRY_RUN", "true").lower() == "true"
_MAX_OPS = int(os.getenv("CIO_MAX_CAMPAIGNS_PER_EXECUTE", "3"))

# Umbral de spike: si delivered aumentó más de este número entre syncs → alerta.
# Protege contra el escenario de Juanita (loop de API duplicando base de contactos).
_SPIKE_THRESHOLD = int(os.getenv("CIO_SPIKE_THRESHOLD", "10000"))


def _headers() -> dict[str, str]:
    if not CIO_APP_API_KEY:
        raise RuntimeError("CUSTOMERIO_APP_API_KEY no configurada en .env")
    return {
        "Authorization": f"Bearer {CIO_APP_API_KEY}",
        "Content-Type": "application/json",
    }


def _guard_write(operation: str, detail: str = "") -> None:
    """Bloquea escrituras cuando CIO_DRY_RUN=true. Lanza error explícito."""
    if _DRY_RUN:
        msg = f"[DRY RUN] {operation} bloqueada{': ' + detail if detail else ''}. Pon CIO_DRY_RUN=false en .env para ejecutar."
        logger.warning(msg)
        raise RuntimeError(msg)
    logger.info("CIO WRITE: %s %s", operation, detail)


def get_campaign(campaign_id: str | int) -> dict[str, Any]:
    """Obtiene detalle de una campaña por ID."""
    resp = httpx.get(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("campaign", data)


def get_campaign_actions_full(campaign_id: str | int) -> list[dict[str, Any]]:
    """
    GET /v1/campaigns/{id}/actions — paginates through all pages to return every action
    with content (subject, body included). The endpoint uses cursor-based pagination via
    the 'next' key; without pagination we only get the first 10 actions.
    """
    all_actions: list[dict[str, Any]] = []
    params: dict[str, str] = {}
    page = 0

    while True:
        resp = httpx.get(
            f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions",
            headers=_headers(),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        all_actions.extend(data.get("actions", []))
        next_cursor = data.get("next")
        if not next_cursor:
            break
        params = {"start": next_cursor}
        page += 1
        if page > 20:  # safety: never loop forever
            logger.warning("get_campaign_actions_full: more than 20 pages for campaign %s, stopping", campaign_id)
            break

    return all_actions


def get_campaign_nodes_with_content(campaign_id: str | int) -> list[dict[str, Any]]:
    """
    Obtiene los nodos de mensaje de una campaña con su contenido completo.
    Usa GET /v1/campaigns/{id}/actions que retorna subject y body reales.
    Los tipos en este endpoint son "push" y "email" (no "push_action"/"email_action").
    El preheader de email no está disponible en este endpoint.
    """
    import re

    actions = get_campaign_actions_full(campaign_id)
    logger.info("CIO get_campaign_nodes_with_content: campaña=%s → %d actions totales",
                campaign_id, len(actions))

    nodes: list[dict[str, Any]] = []
    for a in actions:
        t = a.get("type", "")
        if t not in ("push", "email"):
            continue

        is_email = t == "email"
        subject  = (a.get("subject") or "").strip()
        raw_body = (a.get("body") or "").strip()

        # Para email el body es HTML — extraemos texto plano sin CSS/scripts
        if is_email and raw_body:
            no_style  = re.sub(r"<style[^>]*>.*?</style>", " ", raw_body, flags=re.DOTALL | re.IGNORECASE)
            no_script = re.sub(r"<script[^>]*>.*?</script>", " ", no_style, flags=re.DOTALL | re.IGNORECASE)
            no_tags   = re.sub(r"<[^>]+>", " ", no_script)
            body      = " ".join(no_tags.split())[:800]
        else:
            body = raw_body[:500]

        node: dict[str, Any] = {
            "id":      str(a.get("id", "")),
            "name":    a.get("name", ""),
            "type":    t,
            "subject": subject,
            "body":    body,
        }
        if is_email:
            node["preheader"] = ""  # no disponible en este endpoint

        nodes.append(node)

    msg_count = len(nodes)
    has_content = sum(1 for n in nodes if n.get("subject"))
    logger.info("CIO get_campaign_nodes_with_content: campaña=%s → %d nodos mensaje, %d con subject",
                campaign_id, msg_count, has_content)

    return nodes


def create_campaign(config: dict[str, Any]) -> dict[str, Any]:
    """
    Crea una nueva campaña en CIO. Bloqueado si CIO_DRY_RUN=true.
    Verifica que no exista ya una running con el mismo trigger.
    """
    _guard_write("create_campaign", config.get("name", ""))

    trigger = config.get("event_name", "") or config.get("event", "")
    if trigger:
        for cid in get_tracked_campaign_ids():
            try:
                c = get_campaign(cid)
                existing_trigger = c.get("event_name") or c.get("event") or ""
                if existing_trigger == trigger and c.get("state") == "running":
                    raise RuntimeError(
                        f"Ya existe campaña running con trigger '{trigger}': "
                        f"'{c['name']}' (ID {c['id']}). Usa OPTIMIZAR en vez de CREAR."
                    )
            except httpx.HTTPStatusError:
                pass

    resp = httpx.post(
        f"{CIO_BASE_URL}/campaigns",
        headers=_headers(),
        json={"campaign": config},
        timeout=60,
    )
    resp.raise_for_status()
    campaign = resp.json().get("campaign", resp.json())
    logger.info("CIO create_campaign: creada '%s' id=%s", campaign.get("name"), campaign.get("id"))
    return campaign


def add_action(campaign_id: str | int, action: dict[str, Any]) -> dict[str, Any]:
    """Agrega un nodo/acción a una campaña. Bloqueado si DRY_RUN."""
    _guard_write("add_action", f"campaign={campaign_id} type={action.get('type')}")
    resp = httpx.post(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions",
        headers=_headers(),
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    action_result = resp.json().get("action", resp.json())
    logger.info("CIO add_action: tipo=%s id=%s campaña=%s",
                action.get("type"), action_result.get("id"), campaign_id)
    return action_result


def add_edge(campaign_id: str | int, from_id: str, to_id: str,
             edge_type: str = "continue", index: int | None = None) -> dict[str, Any]:
    """Conecta dos nodos. Bloqueado si DRY_RUN."""
    _guard_write("add_edge", f"campaign={campaign_id} {from_id}->{to_id}")
    edge: dict[str, Any] = {"from": from_id, "to": to_id, "type": edge_type}
    if index is not None:
        edge["index"] = index
    resp = httpx.post(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/edges",
        headers=_headers(),
        json={"edge": edge},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update_action(campaign_id: str | int, action_id: str,
                  updates: dict[str, Any]) -> dict[str, Any]:
    """Actualiza copy/subject de un nodo existente. Bloqueado si DRY_RUN."""
    _guard_write("update_action", f"campaign={campaign_id} action={action_id}")
    resp = httpx.put(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions/{action_id}",
        headers=_headers(),
        json={"action": updates},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("action", {})


def activate_campaign(campaign_id: str | int) -> dict[str, Any]:
    """Activa una campaña draft → running. Bloqueado si DRY_RUN."""
    _guard_write("activate_campaign", f"campaign={campaign_id}")
    resp = httpx.put(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}",
        headers=_headers(),
        json={"campaign": {"state": "running"}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("campaign", {})


# ─── Métricas ────────────────────────────────────────────────────────────────

def get_campaign_metrics(campaign_id: str | int, weeks: int = 8) -> dict[str, Any]:
    """
    Obtiene métricas de entrega detalladas via App API metrics endpoint.
    Devuelve totales de las últimas N semanas completas + serie semanal para tendencias.
    Solo lectura — sin costo extra en CIO (GET request).

    Métricas clave:
      - delivered / total_sent → tasa de entrega
      - human_opened           → aperturas reales (excluye bots/máquina)
      - clicked                → clicks humanos
      - converted              → conversiones al goal de la campaña
      - bounced / undeliverable → calidad de audiencia
    """
    try:
        resp = httpx.get(
            f"{CIO_BASE_URL}/campaigns/{campaign_id}/metrics",
            headers=_headers(),
            params={"period": "weeks", "steps": weeks},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        series = data.get("metric", {}).get("series", {})

        def _sum_complete(key: str) -> int:
            """Suma todas las semanas incluyendo la semana en curso."""
            vals = series.get(key, [])
            return int(sum(vals))

        delivered   = _sum_complete("delivered")
        sent        = _sum_complete("sent")
        human_open  = _sum_complete("human_opened")
        opened      = _sum_complete("opened")
        clicked     = _sum_complete("human_clicked")
        converted   = _sum_complete("converted")
        bounced     = _sum_complete("bounced")
        undeliv     = _sum_complete("undeliverable")
        created     = _sum_complete("created")

        def _rate(num: int, denom: int) -> float:
            return round(num / denom, 4) if denom > 0 else 0.0

        return {
            "delivered":          delivered,
            "total_sent":         sent,
            "opened":             opened,
            "human_opened":       human_open,
            "clicked":            clicked,
            "converted":          converted,
            "bounced":            bounced,
            "undeliverable":      undeliv,
            "delivery_rate":      _rate(delivered, sent),
            "open_rate":          _rate(human_open, delivered),
            "conversion_rate":    _rate(converted, delivered),
            "undeliverable_rate": _rate(undeliv, created),
            "metrics_weekly_json": {
                "period": "weeks",
                "start": data.get("start"),
                "end":   data.get("end"),
                "series": {
                    k: series.get(k, [])
                    for k in [
                        "delivered", "sent", "human_opened", "clicked",
                        "converted", "bounced", "undeliverable", "created",
                    ]
                },
            },
            "metrics_weeks_covered": weeks,
        }
    except Exception as exc:
        logger.warning("Métricas no disponibles para campaña %s (App API): %s", campaign_id, exc)
        return {
            "delivered": 0, "total_sent": 0, "opened": 0, "human_opened": 0,
            "clicked": 0, "converted": 0, "bounced": 0, "undeliverable": 0,
            "delivery_rate": 0.0, "open_rate": 0.0, "conversion_rate": 0.0,
            "undeliverable_rate": 0.0, "metrics_weekly_json": None,
            "metrics_weeks_covered": 0,
        }


# ─── Sincronización ───────────────────────────────────────────────────────────

# Waypoints de CIO que no son entry_events del modelo pero sí se usan como triggers.
# Photo_Validation_Completed es una frontera práctica entre fotos (steps 5-6) y video+revisión
# (steps 7-8). Los eventos granulares de foto no son confiables en CIO, así que Juanita usa
# este waypoint como goal de C4 / trigger de C5.
_CIO_WAYPOINT_STEP_MAP: dict[str, str] = {
    "photo_validation_completed": "step_06_back_photo",
}

# La API de CIO no devuelve conversion_event_name en el detalle de campaña.
# Este mapa define el goal esperado de cada trigger conocido del funnel Trii como fallback.
_TRIGGER_TO_GOAL_FALLBACK: dict[str, str] = {
    "user_created":                          "basic_data_completed",
    "basic_data_completed":                  "risk_profile_completed",
    "risk_profile_completed":                "data_validation_information_completed",
    "data_validation_information_completed": "photo_validation_completed",
    "photo_validation_completed":            "befullusercreated",
    "befullusercreated":                     "becashin",
}


def _map_to_funnel_step(campaign: dict[str, Any],
                        funnel_steps: list[dict[str, Any]]) -> str | None:
    trigger = (campaign.get("event_name") or campaign.get("event") or "").strip()
    trigger_lower = trigger.lower()

    # Mapeo por trigger: el trigger define en qué paso están los usuarios al entrar.
    for step in funnel_steps:
        entry = (step.get("entry_event") or "").lower()
        if entry and entry == trigger_lower:
            return step["step_code"]

    # Waypoints CIO que no son entry_events del modelo (ej. photo_validation_completed).
    if trigger_lower in _CIO_WAYPOINT_STEP_MAP:
        return _CIO_WAYPOINT_STEP_MAP[trigger_lower]

    return None


def sync_campaigns_to_supabase() -> dict[str, Any]:
    """
    Descarga las 5 campañas del funnel, captura métricas detalladas y detecta spikes.
    Hace 2 llamadas GET por campaña: detalle (estado/trigger) + métricas (tasas/series).
    Sin escrituras a CIO.
    """
    funnel_steps = get_funnel_steps()

    campaign_ids = get_tracked_campaign_ids()
    if not campaign_ids:
        return {
            "total_synced": 0, "mapped_to_funnel": 0, "unmapped": 0,
            "spike_alerts": [], "errors": [],
            "message": "No hay campañas configuradas. Agrégalas en /app/configuracion.",
        }

    # Leer cache previo para calcular delta de delivered (detección de spike)
    prev_cache = {c["cio_campaign_id"]: c for c in get_campaigns_cache()}

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    spikes: list[str] = []

    for cid in campaign_ids:
        try:
            c = get_campaign(cid)          # estado, tipo, trigger, goal
            m = get_campaign_metrics(cid, weeks=4)  # métricas del último mes (4 semanas completas)

            cid_str = str(c.get("id", cid))
            step_mapped = _map_to_funnel_step(c, funnel_steps)

            # Extraer subjects/preheader/body de nodos mensaje — ya vienen en el objeto campaña
            actions = c.get("actions", [])
            nodes_info = []
            for a in actions:
                if a.get("type") not in ("push_action", "email_action"):
                    continue
                node: dict[str, Any] = {
                    "id":      str(a.get("id", "")),
                    "name":    a.get("name", ""),
                    "type":    a.get("type", ""),
                    "subject": a.get("subject", "") or "",
                    "body":    a.get("body", "") or "",
                }
                if a.get("type") == "email_action":
                    node["preheader"] = a.get("preheader", "") or a.get("preview_text", "") or ""
                    # Algunos templates guardan el body en un subobjeto
                    if not node["body"] and isinstance(a.get("template"), dict):
                        node["body"] = a["template"].get("body", "") or ""
                nodes_info.append(node)

            # Fetch full actions (paginated) to count message nodes
            # Note: CIO App API does NOT expose delay/wait node durations — only message actions
            try:
                full_actions = get_campaign_actions_full(cid)
                n_nodos = sum(1 for a in full_actions if a.get("type") in ("push", "email"))
            except Exception as exc:
                logger.warning("No se pudo obtener full actions para %s: %s", cid, exc)
                n_nodos = len([n for n in nodes_info if n.get("type") in ("push_action", "email_action")])

            delivered_now = m["delivered"]

            # Delta vs último sync para detectar spike (Juanita's Law).
            # Si el sync anterior tenía 0 (sin datos reales aún), no calcular delta
            # para evitar falsos positivos en la primera sincronización real.
            prev = prev_cache.get(cid_str, {})
            prev_delivered = int(prev.get("delivered") or 0)
            first_real_sync = prev_delivered == 0 and delivered_now > 0
            delta = 0 if first_real_sync else max(0, delivered_now - prev_delivered)
            spike = delta > _SPIKE_THRESHOLD

            if spike:
                msg = (
                    f"SPIKE en campaña '{c.get('name')}' (ID {cid_str}): "
                    f"+{delta:,} delivered desde último sync"
                )
                logger.error("CIO SPIKE ALERT: %s", msg)
                spikes.append(msg)

            rows.append({
                "cio_campaign_id":      cid_str,
                "name":                 c.get("name", ""),
                "campaign_type":        c.get("type"),
                "status":               c.get("state"),
                "trigger_event":        c.get("event_name") or c.get("event"),
                # CIO App API never returns conversion_event_name in campaign detail.
                # Priority: CIO value → previous sync value → hardcoded funnel fallback.
                "goal_event":           (
                    c.get("conversion_event_name") or c.get("goal_event")
                    or prev.get("goal_event")
                    or _TRIGGER_TO_GOAL_FALLBACK.get(
                        (c.get("event_name") or c.get("event") or "").lower()
                    )
                ),
                "country":              "co",
                "funnel_step_mapped":   step_mapped,
                # Métricas de entrega (App API metrics endpoint — últimas 8 semanas completas)
                "delivered":            m["delivered"],
                "total_sent":           m["total_sent"],
                "opened":               m["opened"],
                "human_opened":         m["human_opened"],
                "clicked":              m["clicked"],
                "converted":            m["converted"],
                "bounced":              m["bounced"],
                "undeliverable":        m["undeliverable"],
                "delivery_rate":        m["delivery_rate"],
                "open_rate":            m["open_rate"],
                "conversion_rate":      m["conversion_rate"],
                "undeliverable_rate":   m["undeliverable_rate"],
                "metrics_weekly_json":  m["metrics_weekly_json"],
                "metrics_weeks_covered": m["metrics_weeks_covered"],
                # Copies actuales de los nodos (para que Claude proponga diffs)
                "nodes_json": {
                    "nodes": nodes_info or [],
                    "n_nodos": n_nodos,
                } if (nodes_info or n_nodos) else None,
                # Diagnóstico
                "entries":              int(c.get("match_count") or 0),
                "delivery_delta":       delta,
                "spike_alert":          spike,
                "last_synced_at":       datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.warning("No se pudo obtener campaña %s: %s", cid, exc)
            errors.append(str(cid))

    count = upsert_campaigns_cache(rows)
    mapped = sum(1 for r in rows if r["funnel_step_mapped"])

    logger.info(
        "sync_campaigns: %d/%d obtenidas, %d mapeadas, %d spikes",
        count, len(campaign_ids), mapped, len(spikes),
    )
    return {
        "total_synced":     count,
        "mapped_to_funnel": mapped,
        "unmapped":         count - mapped,
        "spike_alerts":     spikes,
        "errors":           errors,
    }
