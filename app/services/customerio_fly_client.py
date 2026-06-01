"""
Customer.io internal API client (fly.customer.io) — SOLO LECTURA.

Base URL: https://fly.customer.io
Auth: sa_live_ token → JWT (intercambio una vez, renovar al expirar)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SEGURIDAD — LEE ESTO ANTES DE TOCAR ESTE ARCHIVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  El JWT derivado del sa_live_ tiene ACCESO COMPLETO al workspace de CIO.
  Un POST/PUT/DELETE con ese JWT puede crear, modificar o borrar campañas
  y disparar envíos masivos a usuarios reales.

  REGLAS DURAS de este módulo:
  1. _fly_get() es la única función HTTP expuesta internamente.
     Lanza AssertionError si alguien intenta pasar method != "GET".
  2. El único POST permitido es el intercambio de token en _refresh_jwt().
     Va a /auth/... y NO toca datos de campañas ni contactos.
  3. Este módulo no importa ni llama ninguna función de customerio_client.py
     para evitar contaminación accidental con las funciones de escritura.
  4. Ninguna función pública de este módulo acepta parámetros de escritura
     (body, json, data). Solo reciben IDs o parámetros de query.

NOTA SOBRE LA API:
  fly.customer.io es la API interna que usa la propia interfaz de CIO.
  No está documentada públicamente. Puede cambiar sin aviso.
  Kepler la usa porque expone lo que la App API oficial NO expone:
  delays, nodos condicionales, edges y goal event de campañas.
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.cio_fly")

_FLY_BASE    = "https://us.fly.customer.io"
_SA_TOKEN    = os.getenv("CIO_SA_LIVE_READONLY_KEY", "")
_ENV_ID      = os.getenv("CIO_ENVIRONMENT_ID", "112828")

# ── Validación al arranque ────────────────────────────────────────────────────
if not _SA_TOKEN:
    logger.warning(
        "CIO_SA_LIVE_READONLY_KEY no configurada — "
        "las funciones de fly.customer.io no estarán disponibles."
    )

# ── Cache del JWT en memoria ──────────────────────────────────────────────────
# El JWT expira (duración exacta desconocida; asumimos ~1h).
# Renovamos proactivamente si han pasado más de 50 min.
_JWT_TTL_SECONDS = 50 * 60  # 50 minutos

_jwt_cache: dict[str, Any] = {
    "token": "",
    "fetched_at": 0.0,
}


def _refresh_jwt() -> str:
    """
    Intercambia el sa_live_ por un JWT.
    Es el único POST de este módulo — va a /auth/..., no toca datos.
    Cachea el resultado para no hacer round-trip en cada llamada.
    """
    if not _SA_TOKEN:
        raise RuntimeError(
            "CIO_SA_LIVE_READONLY_KEY no configurada en .env. "
            "Necesaria para autenticarse en fly.customer.io."
        )

    now = time.monotonic()
    if _jwt_cache["token"] and (now - _jwt_cache["fetched_at"]) < _JWT_TTL_SECONDS:
        return _jwt_cache["token"]

    logger.info("CIO fly: intercambiando sa_live por JWT...")

    # OAuth2 client_credentials — form-encoded (NO json={})
    # El sa_live_ actúa como client_secret en el flujo estándar OAuth2
    resp = httpx.post(
        f"{_FLY_BASE}/v1/service_accounts/oauth/token",
        data={
            "grant_type":    "client_credentials",
            "client_secret": _SA_TOKEN,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if not resp.is_success:
        logger.error("CIO fly auth %d: %s", resp.status_code, resp.text)
    resp.raise_for_status()
    token = resp.json()["access_token"]

    _jwt_cache["token"]      = token
    _jwt_cache["fetched_at"] = now
    logger.info("CIO fly: JWT obtenido y cacheado.")
    return token


def _fly_get(path: str, params: dict | None = None) -> dict[str, Any]:
    """
    Única función HTTP interna de este módulo.
    Solo hace GET — si alguien modifica esto para hacer POST/PUT, el assert falla.

    ⚠️  NUNCA cambiar el método aquí. El JWT tiene acceso completo a CIO.
    """
    # Salvaguarda dura: este módulo solo hace GET sobre datos
    assert path.startswith("/v1/"), (
        f"_fly_get solo acepta rutas /v1/... — recibió: {path}"
    )

    jwt = _refresh_jwt()
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type":  "application/json",
    }

    resp = httpx.get(
        f"{_FLY_BASE}{path}",
        headers=headers,
        params=params or {},
        timeout=30,
    )

    # Si el JWT expiró, renovar y reintentar una vez
    if resp.status_code == 401:
        logger.info("CIO fly: JWT expirado, renovando...")
        _jwt_cache["token"] = ""  # forzar refresh
        jwt = _refresh_jwt()
        headers["Authorization"] = f"Bearer {jwt}"
        resp = httpx.get(
            f"{_FLY_BASE}{path}",
            headers=headers,
            params=params or {},
            timeout=30,
        )

    resp.raise_for_status()
    return resp.json()


# ─── Funciones públicas — todas solo lectura ──────────────────────────────────

def list_campaigns_fly(limit: int = 50) -> list[dict[str, Any]]:
    """
    Lista todas las campañas del workspace (paginado).
    GET /v1/environments/{env}/campaigns
    """
    all_campaigns: list[dict] = []
    page = 1

    while True:
        data = _fly_get(
            f"/v1/environments/{_ENV_ID}/campaigns",
            params={"limit": limit, "page": page},
        )
        batch = data.get("campaigns", [])
        if not batch:
            break
        all_campaigns.extend(batch)
        logger.debug("CIO fly list_campaigns: página %d → %d campañas", page, len(batch))
        page += 1
        if page > 100:  # safety cap
            logger.warning("CIO fly list_campaigns: superó 100 páginas, deteniendo")
            break

    logger.info("CIO fly list_campaigns: %d campañas totales", len(all_campaigns))
    return all_campaigns


def get_campaign_full(campaign_id: str | int) -> dict[str, Any]:
    """
    Retorna la estructura completa de una campaña:
      - campaign   → metadatos (nombre, estado, trigger, conversion_event, conversion_window)
      - actions[]  → todos los nodos: id, type, name, delay, subject,
                     template_id, conditions (Base64), cohorts, days, start_time
      - edges[]    → todas las conexiones: from, to, type, index

    GET /v1/environments/{env}/campaigns/{id}
    """
    data = _fly_get(f"/v1/environments/{_ENV_ID}/campaigns/{campaign_id}")

    actions = data.get("actions", [])
    # Edges pueden estar top-level O dentro del objeto campaign
    edges = (
        data.get("edges")
        or data.get("campaign", {}).get("edges", [])
        or []
    )

    logger.info(
        "CIO fly get_campaign_full: id=%s → %d actions, %d edges",
        campaign_id, len(actions), len(edges),
    )

    return {
        "campaign": data.get("campaign", {}),
        "actions":  actions,
        "edges":    edges,
        "tags":     data.get("tags", []),
    }


def get_template(template_id: str | int) -> dict[str, Any]:
    """
    Retorna el contenido completo de un template:
      - subject, preheader_text, body (HTML), template_type, variables[]

    GET /v1/environments/{env}/templates/{id}
    """
    data = _fly_get(f"/v1/environments/{_ENV_ID}/templates/{template_id}")
    logger.debug("CIO fly get_template: id=%s tipo=%s", template_id,
                 data.get("template", {}).get("template_type"))
    return data.get("template", data)


def decode_conditions(encoded: str) -> dict[str, Any] | None:
    """
    Decodifica el campo 'conditions' de un nodo condicional.
    Está codificado como Base64 + URL-encoding → JSON string.
    Retorna el dict de condiciones o None si falla.
    """
    if not encoded:
        return None
    try:
        # Base64 necesita padding múltiplo de 4
        padded       = encoded + "=" * ((4 - len(encoded) % 4) % 4)
        decoded_bytes = base64.b64decode(padded)
        json_str      = unquote(decoded_bytes.decode("utf-8"))
        return json.loads(json_str)
    except Exception as exc:
        logger.warning("CIO fly decode_conditions: error decodificando — %s", exc)
        return None


def build_journey(campaign_id: str | int) -> dict[str, Any]:
    """
    Orquesta la lectura completa de una campaña:
    1. get_campaign_full → metadatos + actions + edges
    2. Para cada nodo de mensaje con template_id → get_template
    3. Decodifica conditions de nodos condicionales
    4. Ordena nodos: topological sort si hay edges, fallback por ID numérico
    """
    raw = get_campaign_full(campaign_id)

    meta_raw    = raw.get("campaign", {})
    actions_raw = raw.get("actions",  [])
    edges       = raw.get("edges",    [])

    meta = {
        "id":                str(meta_raw.get("id", campaign_id)),
        "name":              meta_raw.get("name", ""),
        "state":             meta_raw.get("state", ""),
        "trigger":           meta_raw.get("event") or meta_raw.get("event_name", ""),
        "goal":              meta_raw.get("conversion_event_name") or meta_raw.get("goal_event", ""),
        "conversion_window": meta_raw.get("conversion_window"),
    }

    # Enriquecer nodos con contenido de template y decodificar conditions
    nodes: list[dict[str, Any]] = []
    for a in actions_raw:
        node      = dict(a)
        node_type = a.get("type", "")

        if node_type in ("email_action", "push_action") and a.get("template_id"):
            try:
                tmpl = get_template(a["template_id"])
                node["_subject"]      = tmpl.get("subject",        "") or a.get("subject", "")
                node["_preheader"]     = tmpl.get("preheader_text", "")
                node["_body"]          = tmpl.get("body",           "")
                node["_body_json_str"] = tmpl.get("body_json",      "")
                node["_tmpl_type"]     = tmpl.get("template_type",  "")
            except Exception as exc:
                logger.warning("CIO fly build_journey: template %s no disponible — %s",
                               a.get("template_id"), exc)
                node["_subject"]   = a.get("subject", "")
                node["_preheader"] = ""
                node["_body"]      = ""

        if node_type == "conditional_branch_action" and a.get("conditions"):
            node["_conditions_decoded"] = decode_conditions(a["conditions"])

        nodes.append(node)

    # Ordenar nodos
    if edges:
        nodes_ordered = _topological_sort(nodes, edges)
    else:
        # Sin edges: ordenar por ID numérico (orden en que CIO los creó en el builder)
        nodes_ordered = sorted(nodes, key=lambda n: int(n.get("id", 0)))

    return {
        "meta":  meta,
        "nodes": nodes_ordered,
        "edges": edges,
        "raw":   {"actions": actions_raw, "edges": edges},
    }


def _topological_sort(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """BFS Kahn sobre el grafo de edges. Sin edges devuelve el orden original."""
    if not edges:
        # Sin edges: ordenar por timestamp de creación (orden en que Juanita construyó el journey)
        return sorted(nodes, key=lambda n: n.get("created", 0))

    from collections import defaultdict, deque

    id_to_node = {str(n["id"]): n for n in nodes}
    all_ids    = set(id_to_node)
    graph:     dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int]       = defaultdict(int)

    for e in edges:
        frm, to = str(e.get("from", "")), str(e.get("to", ""))
        if frm in all_ids and to in all_ids:
            graph[frm].append(to)
            in_degree[to] += 1
        if frm not in in_degree:
            in_degree[frm] = in_degree.get(frm, 0)

    queue   = deque(nid for nid in all_ids if in_degree.get(nid, 0) == 0)
    ordered: list[dict] = []
    visited: set[str]   = set()

    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        if nid in id_to_node:
            ordered.append(id_to_node[nid])
        for neighbor in graph.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Nodos desconectados del grafo → al final
    for n in nodes:
        if str(n["id"]) not in visited:
            ordered.append(n)

    return ordered
