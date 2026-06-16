"""
Cliente Anthropic para los agentes de estrategia de Kepler.

Modelo por defecto: claude-sonnet-4-6 (~5x más barato que Opus, calidad equivalente
para generación de JSON estructurado).
Override via env: KEPLER_MODEL=claude-opus-4-7 para máxima capacidad cuando se necesite.

Agentes disponibles:
  call_premium_agent() — SHAP + Perplexity + Journey → campaña Primer Depósito
  call_basic_agent()   — Calendario colombiano → 5 campañas de onboarding

Estrategia de caching:
  - system prompt  → cacheado (reglas + schema, cambia nunca)
  - knowledge base → cacheado en primer bloque user (cambia raro)
  - datos semana   → NO cacheado (SHAP + campañas + mercado, cambia cada llamada)
"""

import json
import logging
import os
from datetime import date as _date
from pathlib import Path
from typing import Any

try:
    from json_repair import repair_json as _repair_json_lib
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

import anthropic
from dotenv import load_dotenv

from app.services.prompts.premium import PREMIUM_AGENT_SYSTEM_PROMPT
from app.services.prompts.tier_basic import BASIC_AGENT_SYSTEM_PROMPT

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.anthropic")

# Precios por millón de tokens (USD) — para log de costo estimado
_PRICING = {
    "claude-sonnet-4-6":  {"input": 3.0,  "output": 15.0, "cache_read": 0.30,  "cache_write": 3.75},
    "claude-opus-4-7":    {"input": 15.0, "output": 75.0, "cache_read": 1.50,  "cache_write": 18.75},
}

_MONTHS_ES = [
    "enero","febrero","marzo","abril","mayo","junio",
    "julio","agosto","septiembre","octubre","noviembre","diciembre",
]

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no configurada en .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _get_model() -> str:
    return os.getenv("KEPLER_MODEL", "claude-sonnet-4-6")


def _parse_json(raw: str, label: str) -> dict[str, Any]:
    """
    Intenta json.loads directo; si falla por comillas sin escapar u otros chars
    inválidos dentro de strings, usa json_repair como fallback.
    Causa más común: KB con comillas dentro del contenido que Claude copia
    literalmente en el cuerpo del email sin escaparlas.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[%s] JSON directo falló (%s) — intentando reparación", label, exc)
        if _HAS_JSON_REPAIR:
            try:
                repaired = _repair_json_lib(raw, return_objects=True)
                if isinstance(repaired, dict) and repaired:
                    logger.info("[%s] JSON reparado con json_repair", label)
                    return repaired
            except Exception as rep_exc:
                logger.error("[%s] json_repair también falló: %s", label, rep_exc)
        logger.error("[%s] JSON inválido: %s | Raw[:300]: %s", label, exc, raw[:300])
        raise ValueError(f"Respuesta del agente {label} no es JSON válido: {exc}") from exc


def _fecha_legible(fecha_iso: str) -> str:
    """Convierte '2026-06-14' → '14 de junio de 2026'. Fallback: devuelve el string original."""
    try:
        d = _date.fromisoformat(fecha_iso)
        return f"{d.day} de {_MONTHS_ES[d.month - 1]} de {d.year}"
    except ValueError:
        return fecha_iso


def call_premium_agent(
    shap_text: str,
    research_text: str,
    kb_text: str,
    journey_text: str,
    semana_label: str,
) -> dict[str, Any]:
    """
    Agente premium — una sola llamada Claude con PREMIUM_AGENT_SYSTEM_PROMPT.

    Recibe los 4 bloques definidos en agente_premium.py:
      1. PROYECCIÓN Y SHAP  (dinámico)
      2. RESEARCH DE MERCADO (Perplexity, dinámico)
      3. KNOWLEDGE BASE Y COMPLIANCE (cacheado)
      4. JOURNEY ACTUAL CIO (dinámico)
    """
    model  = _get_model()
    client = _get_client()

    # Bloque 3 — KB cacheado (cambia raro entre semanas)
    kb_block: dict[str, Any] = {
        "type": "text",
        "text": (
            "── BLOQUE 3: KNOWLEDGE BASE — PRODUCTOS DISPONIBLES EN TRII ──\n"
            "!IMPORTANT: Todo lo que aparece a continuación son PRODUCTOS PROPIOS DE TRII. "
            "Aunque la información incluye emisores, tasas y condiciones detalladas, todos estos "
            "instrumentos están disponibles y se comercializan directamente en trii. "
            "Refiérete a ellos siempre como 'en trii' o 'de trii', nunca como productos de bancos "
            "externos, competidores o del mercado genérico.\n\n"
            f"{kb_text}"
        ),
        "cache_control": {"type": "ephemeral"},
    }

    # Bloques dinámicos — sin caché (cambian cada semana)
    data_block: dict[str, Any] = {
        "type": "text",
        "text": (
            f"SEMANA: {semana_label}\n\n"
            f"── BLOQUE 1: PROYECCIÓN Y SHAP ──\n{shap_text}\n\n"
            f"── BLOQUE 2: RESEARCH DE MERCADO (Perplexity sonar-pro) ──\n{research_text}\n\n"
            f"── BLOQUE 4: JOURNEY ACTUAL CIO ──\n{journey_text}\n\n"
            "Responde SOLO JSON válido."
        ),
    }

    logger.info("[PREMIUM] Claude %s — agente premium | semana=%s", model, semana_label)

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": PREMIUM_AGENT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": [kb_block, data_block]}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()

    result = _parse_json(raw, "PREMIUM")
    _log_cost(response.usage, model)
    return result


def call_basic_agent(
    kb_text: str,
    journeys_text: str,
    fecha_hoy: str,
    calendar_text: str,
) -> dict[str, Any]:
    """
    Agente básico — una sola llamada Claude con BASIC_AGENT_SYSTEM_PROMPT.

    Recibe tres bloques:
      1. KNOWLEDGE BASE             (cacheado — cambia raro)
      2. FECHA + CONTEXTO CALENDARIO (Perplexity sonar, dinámico)
      3. JOURNEYS CIO               (dinámico — 5 campañas básicas)
    """
    model  = _get_model()
    client = _get_client()

    # Bloque cacheado — KB completo (cambia raro)
    kb_block: dict[str, Any] = {
        "type": "text",
        "text": (
            "── KNOWLEDGE BASE — PRODUCTOS DISPONIBLES EN TRII ──\n"
            "!IMPORTANT: Todo lo que aparece a continuación son PRODUCTOS PROPIOS DE TRII. "
            "Aunque incluye emisores, tasas y condiciones detalladas, todos están disponibles "
            "directamente en trii. Para copy de Revisión Backend puedes referenciarlos. "
            "Para el resto de campañas transaccionales, úsalos solo como contexto de compliance "
            "y voz de marca — no como argumento de venta.\n\n"
            f"{kb_text}"
        ),
        "cache_control": {"type": "ephemeral"},
    }

    # Bloque dinámico — fecha + calendario + journeys (cambia cada llamada)
    data_block: dict[str, Any] = {
        "type": "text",
        "text": (
            f"FECHA DE HOY: {_fecha_legible(fecha_hoy)}\n\n"
            f"── CONTEXTO DE CALENDARIO (Perplexity sonar) ──\n{calendar_text}\n\n"
            f"── JOURNEYS DE LAS CAMPAÑAS BÁSICAS ──\n{journeys_text}\n\n"
            "Responde SOLO JSON válido."
        ),
    }

    logger.info("[BASIC] Claude %s — agente básico | fecha=%s", model, fecha_hoy)

    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=[{
            "type": "text",
            "text": BASIC_AGENT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": [kb_block, data_block]}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()

    result = _parse_json(raw, "BASIC")
    _log_cost(response.usage, model)
    return result


def call_judge_agent(
    node: dict[str, Any],
    campaign: dict[str, Any],
    kb_excerpt: str,
) -> dict[str, Any]:
    """
    Layer 2 — LLM-as-Judge. Solo se llama si L1 pasó.

    Único objetivo: verificar que las cifras/tasas/montos mencionados en el copy
    existan en el Knowledge Base. Detecta alucinaciones numéricas que L1 no puede
    verificar semánticamente (ej. "CDT al 14% EA" cuando el KB dice 12.5%).

    Si no hay cifras en el copy → aprobado automáticamente.
    Si no hay KB relevante → aprobado (no podemos verificar, L1 ya hizo lo posible).

    Prompt muy corto: ~800 tokens input + 150 output → ~$0.005 por nodo.
    Si falla con excepción → el caller deja pasar (L1 ya validó lo determinista).
    """
    model_id = _get_model()
    client   = _get_client()

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject", "") or ""
    cuerpo    = node.get("cuerpo", "") or ""
    preheader = node.get("preheader", "") or ""
    campaign_name = campaign.get("name", "Sin nombre")

    preheader_line = f"Preheader: {preheader}\n" if preheader else ""

    if not kb_excerpt:
        return {"aprobado": True, "razon": "Sin KB relevante — no hay cifras que verificar", "confianza": 1.0}

    prompt = (
        f"Eres un auditor de precisión numérica para trii (fintech colombiana).\n\n"
        f"TAREA: Verificar que todas las cifras, tasas y montos mencionados en el copy "
        f"coincidan exactamente con los valores del Knowledge Base. "
        f"Solo rechazás si hay una cifra en el copy que CONTRADICE el KB.\n\n"
        f"REGLAS:\n"
        f"- Si el copy no menciona ninguna cifra/tasa/monto → aprobado\n"
        f"- Si el copy menciona cifras y todas coinciden con el KB (±0.1%) → aprobado\n"
        f"- Si hay una cifra en el copy que no aparece en el KB o difiere → rechazado\n"
        f"- NO evalúes tono, proporcionalidad, ni estructura del mensaje\n"
        f"- NO rechaces por mencionar productos — solo por cifras incorrectas\n\n"
        f"COPY A AUDITAR ({tipo} — {campaign_name}):\n"
        f"Subject: {subject}\n"
        f"{preheader_line}"
        f"Cuerpo: {cuerpo}\n\n"
        f"KNOWLEDGE BASE (fuente de verdad para cifras):\n{kb_excerpt}\n\n"
        f"Responde SOLO JSON válido: "
        f'{{ "aprobado": true/false, "razon": "...", "confianza": 0.0-1.0 }}'
    )

    logger.info("[JUDGE] Claude %s — auditando cifras nodo id=%s campaña='%s'",
                model_id, node.get("id_nodo_cio"), campaign_name)

    response = client.messages.create(
        model=model_id,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()

    result = _parse_json(raw, "JUDGE")
    _log_cost(response.usage, model_id)
    return result


def call_judge_agent_premium(
    node: dict[str, Any],
    kb_full: str,
) -> dict[str, Any]:
    """
    Layer 2 — LLM-as-Judge exclusivo para agente premium.

    El agente premium trabaja con contexto de mercado (COLCAP, TRM, S&P, Brent,
    spread TES) — esas cifras son VÁLIDAS y esperadas. El judge solo verifica
    que las tasas de PRODUCTO (CDT %, fondos, mínimos de inversión) coincidan
    con el Knowledge Base. Las cifras de mercado se ignoran en la verificación.

    Si no hay cifras de producto en el copy → aprobado automáticamente.
    """
    model_id = _get_model()
    client   = _get_client()

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject", "") or ""
    cuerpo    = node.get("cuerpo", "") or ""
    preheader = node.get("preheader", "") or ""

    if not kb_full:
        return {"aprobado": True, "razon": "Sin KB — no hay tasas de producto que verificar", "confianza": 1.0}

    preheader_line = f"Preheader: {preheader}\n" if preheader else ""

    prompt = (
        f"Eres un auditor numérico para trii. Sigue estos pasos en orden:\n\n"
        f"PASO 1 — Extrae del copy todas las cifras de PRODUCTO (tasas % EA/NMV y montos mínimos COP).\n"
        f"  Ignora: TRM, COLCAP, S&P, Brent, spread TES, BanRep, Fogafin, rentabilidades históricas.\n\n"
        f"PASO 2 — Para cada cifra de producto, búscala en el KB (tolerancia ±0.1%).\n"
        f"  - Si coincide → marca ✓\n"
        f"  - Si no coincide o no está en el KB → marca ✗ y anota cuál\n\n"
        f"PASO 3 — Decide:\n"
        f"  - Sin cifras de producto, O todas marcadas ✓ → aprobado: true\n"
        f"  - Alguna marcada ✗ → aprobado: false\n"
        f"  REGLA CRÍTICA: si en el Paso 2 todas quedaron ✓, el campo aprobado DEBE ser true.\n\n"
        f"COPY ({tipo}):\n"
        f"Subject: {subject}\n"
        f"{preheader_line}"
        f"Cuerpo: {cuerpo}\n\n"
        f"KB:\n{kb_full}\n\n"
        f"Responde SOLO JSON (razon = una frase corta con el resultado):\n"
        f'{{"aprobado": true/false, "razon": "...", "confianza": 0.0-1.0}}'
    )

    logger.info("[JUDGE-PREMIUM] Claude %s — auditando cifras nodo id=%s",
                model_id, node.get("id_nodo_cio"))

    response = client.messages.create(
        model=model_id,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()

    result = _parse_json(raw, "JUDGE-PREMIUM")
    _log_cost(response.usage, model_id)
    return result


def _log_cost(usage: Any, model: str) -> None:
    pricing = _PRICING.get(model, _PRICING["claude-sonnet-4-6"])
    cache_read   = getattr(usage, "cache_read_input_tokens", 0)
    cache_create = getattr(usage, "cache_creation_input_tokens", 0)
    regular_in   = usage.input_tokens - cache_read - cache_create

    cost = (
        regular_in   / 1_000_000 * pricing["input"]
        + cache_create / 1_000_000 * pricing["cache_write"]
        + cache_read   / 1_000_000 * pricing["cache_read"]
        + usage.output_tokens / 1_000_000 * pricing["output"]
    )

    logger.info(
        "Tokens — in:%d out:%d cache_read:%d cache_create:%d | costo estimado: $%.4f USD",
        usage.input_tokens, usage.output_tokens, cache_read, cache_create, cost,
    )
