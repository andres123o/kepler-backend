"""
Cliente Perplexity — dos endpoints por agente.

Premium → POST /v1/responses (Responses API, sonar-pro): research de mercado.
Basico  → POST /v1/responses (Responses API, sonar):     contexto de calendario.

Query, system prompt y parametros API se cargan desde funnel_prompts en Supabase.
fc (FunnelClient) es requerido — no hay fallback hardcodeado.

API key: PERPLEXITY_API_KEY (.env).
"""

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
logger = logging.getLogger("kepler.perplexity")

_CHAT_URL = "https://api.perplexity.ai/chat/completions"


def _headers() -> dict:
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _api_key() -> str:
    return os.getenv("PERPLEXITY_API_KEY", "")


# ─────────────────────────────────────────────────────────────
# BÁSICO → /search
# ─────────────────────────────────────────────────────────────

def fetch_calendar_context(fecha_hoy: str, fc) -> dict[str, Any]:
    """
    Basico → POST /v1/responses (Responses API, sonar)

    Query, system prompt y api_params se leen desde funnel_prompts en Supabase.
    fc es requerido — no hay fallback hardcodeado.

    Retorna:
      {"raw_text": str | None, "citations": list[str]}
    """
    key = _api_key()
    if not key:
        logger.warning("[PERPLEXITY/basic] PERPLEXITY_API_KEY no configurada — calendario omitido")
        return {"raw_text": None, "citations": []}

    system_prompt = fc.get_agent_prompt("basic", "perplexity_system")
    query         = fc.get_agent_prompt("basic", "perplexity_query")
    base_params   = fc.get_perplexity_api_params("basic")

    if not system_prompt or not query or not base_params:
        raise RuntimeError(
            "funnel_prompts no tiene filas para basic/perplexity_system, "
            "basic/perplexity_query o basic/perplexity_query(api_params). "
            "Corre seed_prompts.py para este funnel."
        )

    import re as _re
    query = _re.sub(r'Fecha de hoy:[^\n]*', f'Fecha de hoy: {fecha_hoy}.', query)

    payload = {
        **base_params,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
    }

    logger.info("[PERPLEXITY/basic] Fetching calendar context | fecha=%s", fecha_hoy)

    try:
        resp = requests.post(_CHAT_URL, json=payload, headers=_headers(), timeout=45)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.error("[PERPLEXITY/basic] Timeout — calendario omitido")
        return {"raw_text": None, "citations": []}
    except requests.exceptions.RequestException as exc:
        logger.error("[PERPLEXITY/basic] Error: %s — calendario omitido", exc)
        return {"raw_text": None, "citations": []}

    raw_text  = None
    citations = []
    choices   = data.get("choices") or []
    if choices:
        raw_text = (choices[0].get("message") or {}).get("content")
    citations = data.get("citations") or []

    if raw_text:
        logger.info("[PERPLEXITY/basic] Recibido | chars=%d | citations=%d",
                    len(raw_text), len(citations))
    else:
        logger.error("[PERPLEXITY/basic] Respuesta sin texto: %s", str(data)[:400])
    return {"raw_text": raw_text, "citations": citations}


# ─────────────────────────────────────────────────────────────
# PREMIUM → /v1/responses
# ─────────────────────────────────────────────────────────────

def fetch_market_research(
    semana_label: str,
    fecha_hoy: date | None = None,
    fc=None,
) -> dict[str, Any]:
    """
    Premium → POST /v1/responses (Responses API, sonar-pro)

    Query, system prompt y api_params se leen desde funnel_prompts en Supabase.
    fc es requerido — no hay fallback hardcodeado.

    Retorna:
      {"raw_text": str | None, "citations": list[str]}
    """
    key = _api_key()
    if not key:
        logger.warning("[PERPLEXITY/premium] PERPLEXITY_API_KEY no configurada — research omitido")
        return {"raw_text": None, "citations": []}

    if fc is None:
        raise RuntimeError(
            "fetch_market_research requiere fc (FunnelClient). "
            "Pasa el FunnelClient desde generate_premium_strategy."
        )

    system_prompt = fc.get_agent_prompt("premium", "perplexity_system")
    query         = fc.get_agent_prompt("premium", "perplexity_query")
    base_params   = fc.get_perplexity_api_params("premium")

    if not system_prompt or not query or not base_params:
        raise RuntimeError(
            "funnel_prompts no tiene filas para premium/perplexity_system, "
            "premium/perplexity_query o premium/perplexity_query(api_params). "
            "Corre seed_prompts.py para este funnel."
        )

    import re as _re
    _fecha_hoy = fecha_hoy or date.today()
    _today_str = _fecha_hoy.strftime("%Y-%m-%d")
    query = _re.sub(r'Fecha de hoy:[^\n]*', f'Fecha de hoy: {_today_str}.', query)
    if semana_label:
        query = _re.sub(r'Semana objetivo:[^\n]*', f'Semana objetivo: {semana_label}.', query)
    # Reemplazar cualquier rango DD/MM/YYYY al DD/MM/YYYY hardcodeado en el cuerpo
    # con los últimos 7 días desde hoy — nunca hardcode en Supabase ni en código
    _rango_inicio = (_fecha_hoy - timedelta(days=7)).strftime("%d/%m/%Y")
    _rango_fin    = _fecha_hoy.strftime("%d/%m/%Y")
    _ultimo_rango = f"{_rango_inicio} al {_rango_fin}"
    query = _re.sub(r'\d{2}/\d{2}/\d{4}\s+al\s+\d{2}/\d{2}/\d{4}', _ultimo_rango, query)

    payload = {
        **base_params,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query},
        ],
    }

    _model      = base_params.get("model", "?")
    _dom_filter = base_params.get("search_domain_filter")
    _recency    = base_params.get("search_recency_filter", "none")
    logger.info(
        "[PERPLEXITY/premium] REQUEST | semana=%s | fecha=%s | model=%s | recency=%s | domain_filter=%s",
        semana_label, _today_str, _model, _recency,
        f"{len(_dom_filter)} dominios" if _dom_filter else "SIN FILTRO (toda la web)",
    )
    logger.info("[PERPLEXITY/premium] QUERY (primeros 200 chars): %s", query[:200])

    try:
        resp = requests.post(_CHAT_URL, json=payload, headers=_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.error("[PERPLEXITY/premium] Timeout (60s) — research omitido")
        return {"raw_text": None, "citations": []}
    except requests.exceptions.RequestException as exc:
        logger.error("[PERPLEXITY/premium] Error HTTP %s: %s — research omitido",
                     getattr(exc.response, "status_code", "?"), exc)
        return {"raw_text": None, "citations": []}

    raw_text  = None
    citations = []
    choices   = data.get("choices") or []
    if choices:
        raw_text = (choices[0].get("message") or {}).get("content")
    citations = data.get("citations") or []

    if raw_text:
        logger.info(
            "[PERPLEXITY/premium] RESPONSE OK | chars=%d | citations=%d | primeras_urls=%s",
            len(raw_text), len(citations),
            citations[:3] if citations else "ninguna — respuesta SIN búsqueda real",
        )
        if not citations:
            logger.warning(
                "[PERPLEXITY/premium] 0 citaciones — el modelo respondió sin hacer búsqueda web. "
                "Verifica que el modelo '%s' sea un modelo online de Perplexity.", _model,
            )
    else:
        logger.error("[PERPLEXITY/premium] Respuesta sin texto reconocible: %s", str(data)[:400])

    return {"raw_text": raw_text, "citations": citations}


# ─────────────────────────────────────────────────────────────
# Formatters → texto para los agentes Claude
# ─────────────────────────────────────────────────────────────

def format_calendar_block(calendar: dict[str, Any]) -> str:
    """Convierte el output de fetch_calendar_context() al bloque del agente básico."""
    raw_text  = calendar.get("raw_text")
    citations = calendar.get("citations", [])

    if not raw_text:
        return (
            "(No hay contexto de calendario disponible esta semana — "
            "Perplexity no respondió o la API key no está configurada. "
            "Deriva el contexto desde la fecha de hoy usando tu conocimiento del calendario del país correspondiente.)"
        )

    lines = [raw_text]
    if citations:
        lines += ["", "Fuentes:"] + [f"  - {u}" for u in citations if u]
    return "\n".join(lines)


def format_research_block(research: dict[str, Any]) -> str:
    """Convierte el output de fetch_market_research() al Bloque 2 del agente premium."""
    raw_text  = research.get("raw_text")
    citations = research.get("citations", [])

    if not raw_text:
        return (
            "(No hay research de mercado disponible esta semana — "
            "Perplexity no respondió o la API key no está configurada.)"
        )

    lines = [raw_text]
    if citations:
        lines += ["", "Fuentes citadas por Perplexity:"] + [f"  - {u}" for u in citations if u]
    return "\n".join(lines)
