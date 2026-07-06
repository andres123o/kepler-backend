"""
Customer.io App API client para Kepler.

Base URL: https://api.customer.io/v1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOKENS Y SEGURIDAD — LEE ESTO ANTES DE TOCAR ESTE ARCHIVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CIO_SA_LIVE_READONLY_KEY  (token sa_live)
  ─────────────────────────────────────────
  • Token con acceso COMPLETO al backend de CIO (máximos privilegios).
  • NO es un bearer válido contra la App API pública (api.customer.io/v1) —
    solo sirve para el intercambio OAuth de fly.customer.io. Este archivo
    NO lo usa; lo usa exclusivamente customerio_fly_client.py.

  CUSTOMERIO_APP_API_KEY  (token de escritura, también válido para lectura)
  ───────────────────────────────────────────────────────────────────────
  • Token con privilegios de lectura Y escritura de campañas en la App API pública.
  • Lectura (_headers_readonly) y escritura (_headers_write) usan este mismo token
    en este archivo — la App API pública no distingue un token de "solo lectura".
  • Escritura bloqueada por defecto: CIO_DRY_RUN=true en .env impide que
    cualquier POST/PUT llegue realmente a CIO.

OTRAS SALVAGUARDAS:
- Kepler NUNCA llama la Track API (track.customer.io) — esa crea/modifica contactos.
- CIO_DRY_RUN=true bloquea TODAS las escrituras (modo seguro por defecto).
- MAX_CAMPAIGNS_PER_EXECUTE=3 limita cuántas campañas se tocan por llamada.
"""

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from app.services.supabase_client import FunnelClient, _default_fc

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.customerio")

CIO_BASE_URL = "https://api.customer.io/v1"

# ── Flags globales de operación (no son credenciales, no van a org_secrets) ──
_DRY_RUN: bool = os.getenv("CIO_DRY_RUN", "true").lower() == "true"
_MAX_OPS = int(os.getenv("CIO_MAX_CAMPAIGNS_PER_EXECUTE", "3"))
_SPIKE_THRESHOLD = int(os.getenv("CIO_SPIKE_THRESHOLD", "10000"))


def _headers_readonly(key: str) -> dict[str, str]:
    """
    Headers para GET requests (SOLO LECTURA por convención de este módulo — la App
    API pública no impone esa restricción a nivel de token).

    ⚠️  SOLO llamar desde funciones que hacen GET requests.
    ⚠️  NUNCA usar este header en POST, PUT, PATCH ni DELETE.
    key: app_api_key de CIOCredentials (viene de org_secrets via FunnelClient).
    """
    if not key:
        raise RuntimeError(
            "CIO_APP_API_KEY no disponible. "
            "Agrega la credencial en org_secrets para esta organización."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _headers_write(key: str) -> dict[str, str]:
    """
    Headers para operaciones de escritura (POST/PUT).
    Solo llamar DESPUÉS de _guard_write() — que ya bloquea si DRY_RUN=true.

    ⚠️  NUNCA pasar el sa_live_key aquí — solo el app_api_key.
    key: app_api_key de CIOCredentials (viene de org_secrets via FunnelClient).
    """
    if not key:
        raise RuntimeError(
            "CIO_APP_API_KEY no disponible. "
            "Agrega la credencial en org_secrets para esta organización."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _guard_write(operation: str, detail: str = "") -> None:
    """Bloquea escrituras cuando CIO_DRY_RUN=true. Lanza error explícito."""
    if _DRY_RUN:
        msg = f"[DRY RUN] {operation} bloqueada{': ' + detail if detail else ''}. Pon CIO_DRY_RUN=false en .env para ejecutar."
        logger.warning(msg)
        raise RuntimeError(msg)
    logger.info("CIO WRITE: %s %s", operation, detail)


def get_campaign(campaign_id: str | int, fc) -> dict[str, Any]:
    """
    Obtiene detalle de una campaña por ID. Solo lectura.
    Usa app_api_key: sa_live_key no es un bearer válido para la App API pública
    (solo sirve para el intercambio OAuth de fly.customer.io, ver customerio_fly_client.py).
    """
    creds = fc.get_cio_credentials()
    resp = httpx.get(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}",
        headers=_headers_readonly(creds.app_api_key),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("campaign", data)


def get_campaign_actions_full(campaign_id: str | int, fc) -> list[dict[str, Any]]:
    """
    GET /v1/campaigns/{id}/actions — paginates through all pages to return every action
    with content (subject, body included). The endpoint uses cursor-based pagination via
    the 'next' key; without pagination we only get the first 10 actions.
    """
    creds = fc.get_cio_credentials()
    all_actions: list[dict[str, Any]] = []
    params: dict[str, str] = {}
    page = 0

    while True:
        resp = httpx.get(
            f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions",
            headers=_headers_readonly(creds.app_api_key),
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
        if page > 20:
            logger.warning("get_campaign_actions_full: more than 20 pages for campaign %s, stopping", campaign_id)
            break

    return all_actions


def get_campaign_nodes_with_content(campaign_id: str | int, fc) -> list[dict[str, Any]]:
    """
    Obtiene los nodos de mensaje de una campaña con su contenido completo.
    Usa GET /v1/campaigns/{id}/actions que retorna subject y body reales.
    Los tipos en este endpoint son "push" y "email" (no "push_action"/"email_action").
    El preheader viene en el campo "preheader_text" (no "preheader").
    """
    import re

    actions = get_campaign_actions_full(campaign_id, fc)
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
            node["preheader"] = (a.get("preheader_text") or a.get("preheader") or "").strip()

        nodes.append(node)

    msg_count = len(nodes)
    has_content = sum(1 for n in nodes if n.get("subject"))
    logger.info("CIO get_campaign_nodes_with_content: campaña=%s → %d nodos mensaje, %d con subject",
                campaign_id, msg_count, has_content)

    return nodes


def create_campaign(config: dict[str, Any], fc) -> dict[str, Any]:
    """
    Crea una nueva campaña en CIO. Bloqueado si CIO_DRY_RUN=true.
    Verifica que no exista ya una running con el mismo trigger.
    fc: FunnelClient — lee credenciales CIO desde org_secrets.
    """
    _guard_write("create_campaign", config.get("name", ""))
    creds = fc.get_cio_credentials()

    trigger = config.get("event_name", "") or config.get("event", "")
    if trigger:
        for cid in fc.get_tracked_campaign_ids():
            try:
                c = get_campaign(cid, fc)
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
        headers=_headers_write(creds.app_api_key),
        json={"campaign": config},
        timeout=60,
    )
    resp.raise_for_status()
    campaign = resp.json().get("campaign", resp.json())
    logger.info("CIO create_campaign: creada '%s' id=%s", campaign.get("name"), campaign.get("id"))
    return campaign


def add_action(campaign_id: str | int, action: dict[str, Any], fc) -> dict[str, Any]:
    """Agrega un nodo/acción a una campaña. Bloqueado si DRY_RUN."""
    _guard_write("add_action", f"campaign={campaign_id} type={action.get('type')}")
    creds = fc.get_cio_credentials()
    resp = httpx.post(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions",
        headers=_headers_write(creds.app_api_key),
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    action_result = resp.json().get("action", resp.json())
    logger.info("CIO add_action: tipo=%s id=%s campaña=%s",
                action.get("type"), action_result.get("id"), campaign_id)
    return action_result


def add_edge(campaign_id: str | int, from_id: str, to_id: str,
             fc=None, edge_type: str = "continue", index: int | None = None) -> dict[str, Any]:
    """Conecta dos nodos. Bloqueado si DRY_RUN."""
    _guard_write("add_edge", f"campaign={campaign_id} {from_id}->{to_id}")
    creds = fc.get_cio_credentials()
    edge: dict[str, Any] = {"from": from_id, "to": to_id, "type": edge_type}
    if index is not None:
        edge["index"] = index
    resp = httpx.post(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/edges",
        headers=_headers_write(creds.app_api_key),
        json={"edge": edge},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def update_action(campaign_id: str | int, action_id: str,
                  updates: dict[str, Any], fc=None) -> dict[str, Any]:
    """Actualiza copy/subject de un nodo existente. Bloqueado si DRY_RUN."""
    _guard_write("update_action", f"campaign={campaign_id} action={action_id}")
    creds = fc.get_cio_credentials()
    resp = httpx.put(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}/actions/{action_id}",
        headers=_headers_write(creds.app_api_key),
        json={"action": updates},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("action", {})


def activate_campaign(campaign_id: str | int, fc=None) -> dict[str, Any]:
    """Activa una campaña draft → running. Bloqueado si DRY_RUN."""
    _guard_write("activate_campaign", f"campaign={campaign_id}")
    creds = fc.get_cio_credentials()
    resp = httpx.put(
        f"{CIO_BASE_URL}/campaigns/{campaign_id}",
        headers=_headers_write(creds.app_api_key),
        json={"campaign": {"state": "running"}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("campaign", {})


# ─── Métricas ────────────────────────────────────────────────────────────────

def get_campaign_metrics(campaign_id: str | int, fc, weeks: int = 8) -> dict[str, Any]:
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
        creds = fc.get_cio_credentials()
        resp = httpx.get(
            f"{CIO_BASE_URL}/campaigns/{campaign_id}/metrics",
            headers=_headers_readonly(creds.app_api_key),
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
        detail = exc.response.text[:300] if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        logger.error("Métricas NO disponibles para campaña %s (App API) — devolviendo ceros. Causa: %s", campaign_id, detail)
        return {
            "delivered": 0, "total_sent": 0, "opened": 0, "human_opened": 0,
            "clicked": 0, "converted": 0, "bounced": 0, "undeliverable": 0,
            "delivery_rate": 0.0, "open_rate": 0.0, "conversion_rate": 0.0,
            "undeliverable_rate": 0.0, "metrics_weekly_json": None,
            "metrics_weeks_covered": 0,
        }


# ─── Sincronización ───────────────────────────────────────────────────────────

def _map_to_funnel_step(
    campaign: dict[str, Any],
    funnel_steps: list[dict[str, Any]],
    waypoint_step_map: dict[str, str] | None = None,
) -> str | None:
    trigger = (campaign.get("event_name") or campaign.get("event") or "").strip()
    trigger_lower = trigger.lower()

    # Mapeo por trigger: el trigger define en qué paso están los usuarios al entrar.
    for step in funnel_steps:
        entry = (step.get("entry_event") or "").lower()
        if entry and entry == trigger_lower:
            return step["step_code"]

    # Waypoints CIO que no son entry_events del modelo — desde config del funnel.
    wm = waypoint_step_map or {}
    if trigger_lower in wm:
        return wm[trigger_lower]

    return None


def sync_campaigns_to_supabase(fc: FunnelClient) -> dict[str, Any]:
    """
    Descarga las campañas del funnel, captura métricas detalladas y detecta spikes.
    Hace 2 llamadas GET por campaña: detalle (estado/trigger) + métricas (tasas/series).
    Sin escrituras a CIO.
    fc: FunnelClient requerido — sin fallback a tenant default.
    """
    funnel_steps = fc.get_funnel_steps()

    # Leer mapas CIO desde config del funnel
    try:
        cio_cfg = fc.get_cio_config()
        waypoint_step_map    = {k.lower(): v for k, v in (cio_cfg.get("waypoint_step_map") or {}).items()}
        trigger_to_goal_map  = {k.lower(): v for k, v in (cio_cfg.get("trigger_to_goal_map") or {}).items()}
    except ValueError:
        waypoint_step_map   = {}
        trigger_to_goal_map = {}

    campaign_ids = fc.get_tracked_campaign_ids()
    if not campaign_ids:
        return {
            "total_synced": 0, "mapped_to_funnel": 0, "unmapped": 0,
            "spike_alerts": [], "errors": [],
            "message": "No hay campañas configuradas. Agrégalas en /app/configuracion.",
        }

    # Leer cache previo para calcular delta de delivered (detección de spike)
    prev_cache = {c["cio_campaign_id"]: c for c in fc.get_campaigns_cache()}

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    spikes: list[str] = []

    for cid in campaign_ids:
        try:
            c = get_campaign(cid, fc)
            m = get_campaign_metrics(cid, fc, weeks=4)

            cid_str = str(c.get("id", cid))
            step_mapped = _map_to_funnel_step(c, funnel_steps, waypoint_step_map)

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
                full_actions = get_campaign_actions_full(cid, fc)
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
                # Priority: CIO value → previous sync value → funnel config fallback map.
                "goal_event":           (
                    c.get("conversion_event_name") or c.get("goal_event")
                    or prev.get("goal_event")
                    or trigger_to_goal_map.get(
                        (c.get("event_name") or c.get("event") or "").lower()
                    )
                ),
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

    count = fc.upsert_campaigns_cache(rows)
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
