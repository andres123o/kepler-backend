"""
Cliente Perplexity — dos endpoints distintos por agente.

Básico   → POST /search        (Search API)    : snippets web para contexto de calendario.
Premium  → POST /v1/responses  (Responses API) : síntesis de research de mercado financiero.

API key: PERPLEXITY_API_KEY (.env).
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from app.services.prompts.premium.perplexity_query import build_market_query

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
logger = logging.getLogger("kepler.perplexity")

_SEARCH_URL    = "https://api.perplexity.ai/search"
_RESPONSES_URL = "https://api.perplexity.ai/v1/responses"


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

def _build_calendar_search_query(fecha_hoy: str) -> str:
    """Query concisa para el Search API — busca hechos de calendario colombiano."""
    return (
        f"festivos Colombia próximos 7 días desde {fecha_hoy} "
        "quincena prima legal servicios BanRep reunión política monetaria "
        "eventos electorales vencimientos tributarios DIAN"
    )


def fetch_calendar_context(fecha_hoy: str) -> dict[str, Any]:
    """
    Básico → POST /search

    Devuelve snippets de resultados web sobre el calendario colombiano
    para la semana en curso.

    Retorna:
      {"raw_text": str | None, "citations": list[str]}
    """
    key = _api_key()
    if not key:
        logger.warning("[PERPLEXITY/search] PERPLEXITY_API_KEY no configurada — calendario omitido")
        return {"raw_text": None, "citations": []}

    payload = {
        "query":              _build_calendar_search_query(fecha_hoy),
        "max_results":        5,
        "max_tokens_per_page": 512,
    }

    logger.info("[PERPLEXITY/search] Fetching calendar | fecha=%s", fecha_hoy)

    try:
        resp = requests.post(_SEARCH_URL, json=payload, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.error("[PERPLEXITY/search] Timeout (30s) — calendario omitido")
        return {"raw_text": None, "citations": []}
    except requests.exceptions.RequestException as exc:
        logger.error("[PERPLEXITY/search] Error: %s — calendario omitido", exc)
        return {"raw_text": None, "citations": []}

    results   = data.get("results", [])
    citations = [r.get("url", "") for r in results if r.get("url")]

    snippets = []
    for r in results:
        title = r.get("title", "")
        text  = r.get("text") or r.get("snippet") or r.get("content", "")
        url   = r.get("url", "")
        if text:
            snippets.append(f"[{title}]({url})\n{text.strip()}")

    raw_text = "\n\n---\n\n".join(snippets) if snippets else None

    logger.info("[PERPLEXITY/search] Resultados: %d | citations: %d", len(results), len(citations))
    return {"raw_text": raw_text, "citations": citations}


# ─────────────────────────────────────────────────────────────
# PREMIUM → /v1/responses
# ─────────────────────────────────────────────────────────────

def fetch_market_research(
    semana_label: str,
    fecha_hoy: date | None = None,  # mantenido por compatibilidad, no usado en este endpoint
) -> dict[str, Any]:
    """
    Premium → POST /v1/responses (preset: fast-search)

    Síntesis de research de mercado financiero colombiano: COLCAP, TRM,
    BanRep, noticias CO, sentimiento retail.

    Retorna:
      {"raw_text": str | None, "citations": list[str]}
    """
    key = _api_key()
    if not key:
        logger.warning("[PERPLEXITY/responses] PERPLEXITY_API_KEY no configurada — research omitido")
        return {"raw_text": None, "citations": []}

    payload = {
        "preset": "fast-search",
        "input":  build_market_query(semana_label),
    }

    logger.info("[PERPLEXITY/responses] Fetching market research | semana=%s", semana_label)

    try:
        resp = requests.post(_RESPONSES_URL, json=payload, headers=_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        logger.error("[PERPLEXITY/responses] Timeout (60s) — research omitido")
        return {"raw_text": None, "citations": []}
    except requests.exceptions.RequestException as exc:
        logger.error("[PERPLEXITY/responses] Error: %s — research omitido", exc)
        return {"raw_text": None, "citations": []}

    # Estructura real de /v1/responses:
    #   output[0] → type="search_results" → results[].url  (citations)
    #   output[1] → type="message"        → content[0].text (texto sintetizado)
    raw_text  = None
    citations = []

    for item in data.get("output", []):
        if item.get("type") == "search_results":
            for r in item.get("results", []):
                url = r.get("url", "")
                if url:
                    citations.append(url)
        elif item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    raw_text = block.get("text")
                    break

    if raw_text:
        logger.info("[PERPLEXITY/responses] Research recibido | chars=%d | citations=%d",
                    len(raw_text), len(citations))
    else:
        logger.error("[PERPLEXITY/responses] Respuesta sin texto reconocible: %s", str(data)[:400])

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
            "Deriva el contexto solo desde la fecha usando tu conocimiento del calendario colombiano.)"
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
