"""
Customer.io fly API — escritura de templates de nodos.

SEGURIDAD:
  - Solo expone update_node_copy() — una función, un propósito.
  - Máximo 2 requests a CIO por invocación: GET template + PUT template.
  - Cooldown de 30s por action_id: previene doble-actualización del mismo nodo.
  - El JWT se comparte con el cliente de lectura (customerio_fly_client.py)
    para no generar tokens innecesarios.
  - Solo actualiza subject, body y preheader_text. Nunca toca estructura,
    edges, triggers ni configuración de campaña.
"""

import logging
import re
import time
from typing import Any

import httpx

from app.services.customerio_fly_client import (
    _ENV_ID,
    _FLY_BASE,
    _jwt_cache,
    _refresh_jwt,
)

logger = logging.getLogger("kepler.cio_fly_writer")

_COOLDOWN_SECS = 30
_last_update: dict[int, float] = {}  # action_id → monotonic timestamp


def _strip_wsc(s: str) -> str:
    """Elimina whitespace control de Liquid ({%- y -%}). CIO no lo soporta en ningún campo."""
    return re.sub(r"-%}", "%}", re.sub(r"\{%-", "{%", s))


def _fly_put(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """PUT autenticado via JWT. Renueva el token en 401 (una vez)."""
    jwt = _refresh_jwt()
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}

    resp = httpx.put(f"{_FLY_BASE}{path}", headers=headers, json=payload, timeout=30)

    if resp.status_code == 401:
        logger.info("CIO fly writer: JWT expirado, renovando...")
        _jwt_cache["token"] = ""
        jwt = _refresh_jwt()
        headers["Authorization"] = f"Bearer {jwt}"
        resp = httpx.put(f"{_FLY_BASE}{path}", headers=headers, json=payload, timeout=30)

    if not resp.is_success:
        logger.error("CIO fly writer: PUT falló %s — body: %s", resp.status_code, resp.text[:500])

    resp.raise_for_status()
    return resp.json()


def _fly_get(path: str) -> dict[str, Any]:
    """GET autenticado — reutiliza el JWT cacheado."""
    jwt = _refresh_jwt()
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
    resp = httpx.get(f"{_FLY_BASE}{path}", headers=headers, timeout=30)

    if resp.status_code == 401:
        _jwt_cache["token"] = ""
        jwt = _refresh_jwt()
        headers["Authorization"] = f"Bearer {jwt}"
        resp = httpx.get(f"{_FLY_BASE}{path}", headers=headers, timeout=30)

    resp.raise_for_status()
    return resp.json()


_LIQUID_RE = re.compile(r"\{%-?.*?-?%\}|\{\{.*?\}\}", re.DOTALL)


def _mask_liquid(text: str, prefix: str = "LQ") -> tuple[str, list[str]]:
    """
    Reemplaza bloques Liquid con placeholders __{prefix}_N__ para que Claude no los toque.
    Soporta: {{ var }}, {% tag %}, {%- tag -%} y variantes con whitespace control.
    """
    tokens: list[str] = []

    def replacer(m: re.Match) -> str:
        tokens.append(m.group(0))
        return f"__{prefix}_{len(tokens) - 1}__"

    return _LIQUID_RE.sub(replacer, text), tokens


def _unmask_liquid(text: str, tokens: list[str], prefix: str = "LQ") -> str:
    """Restaura los placeholders __{prefix}_N__ con el Liquid original."""
    for i, token in enumerate(tokens):
        text = text.replace(f"__{prefix}_{i}__", token)
    return text


def _patch_email_html_with_claude(full_html: str, new_text: str, action_id: int) -> str:
    """
    Agente 2 — HTML Patcher.
    Recibe el HTML completo actual del email (de CIO) y el nuevo texto propuesto.
    Claude reemplaza SOLO el texto visible del cuerpo; devuelve el HTML íntegro sin
    ningún otro cambio (estilos, links, imágenes, botones, footer, MSO conditionals).

    Usa Sonnet con temperature=0.
    Fallback: si la respuesta no es HTML válido → devuelve full_html sin cambios.
    """
    import os
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("CIO fly writer: ANTHROPIC_API_KEY no configurada (action %s)", action_id)
        return full_html

    # CIO no soporta whitespace control {%- -%} en el body — convertir a {% %}.
    new_text_clean = re.sub(r"\{%-", "{%", new_text)
    new_text_clean = re.sub(r"-%}",  "%}", new_text_clean)

    # Enmascarar Liquid en HTML y en new_text — Claude no ve Liquid crudo, no puede corromperlo.
    # HTML original: prefijo LQ   → __LQ_N__
    # new_text:      prefijo NLQ  → __NLQ_N__
    masked_html, html_tokens   = _mask_liquid(full_html,       "LQ")
    masked_text, text_tokens   = _mask_liquid(new_text_clean,  "NLQ")
    logger.info(
        "CIO fly writer: %d tokens Liquid en HTML + %d en texto enmascarados (action %s)",
        len(html_tokens), len(text_tokens), action_id,
    )

    prompt = (
        "Sos un editor quirúrgico de HTML para emails. "
        "Tu única tarea es reemplazar el texto visible del cuerpo del email con el nuevo texto que te doy.\n\n"
        "REGLAS ABSOLUTAS — cualquier violación rompe el email:\n"
        "1. Devolvé el HTML COMPLETO, exactamente igual en todo excepto el texto del cuerpo\n"
        "2. SOLO modificá el texto visible de los párrafos principales\n"
        "3. NO cambies NINGÚN atributo HTML: href, src, style, class, id, width, height, align, role, etc.\n"
        "4. NO cambies botones ni links (<a> tags)\n"
        "5. NO cambies el footer: redes sociales, copyright, disclaimer legal\n"
        "6. NO cambies imágenes ni sus atributos alt\n"
        "7. NO agregues ni elimines tags HTML\n"
        "8. Mantené EXACTAMENTE los comentarios <!--[if mso]>, <![endif]--> y tags VML\n"
        "9. CRÍTICO: los placeholders __LQ_N__ y __NLQ_N__ son código Liquid enmascarado. "
        "Devolvelos EXACTAMENTE como aparecen, sin modificar ni un carácter. "
        "Si el nuevo texto tiene un __NLQ_N__, colocalo en el HTML donde corresponda al texto nuevo.\n"
        "10. Si el nuevo texto tiene menos contenido que el original, dejá el resto como está\n\n"
        f"HTML ACTUAL DEL EMAIL:\n{masked_html}\n\n"
        f"NUEVO TEXTO DEL CUERPO:\n{masked_text}\n\n"
        "Devolvé ÚNICAMENTE el HTML completo modificado. Sin explicaciones, sin markdown, sin ```."
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        result_masked = response.content[0].text.strip()

        if response.stop_reason == "max_tokens":
            logger.error("CIO fly writer: Agent 2 truncado por max_tokens (action %s) — usando original", action_id)
            return full_html

        if not result_masked.lstrip().startswith("<"):
            logger.warning("CIO fly writer: Agent 2 no devolvió HTML (action %s) — usando original", action_id)
            return full_html

        # Restaurar: primero NLQ (Liquid del new_text que Agent 2 insertó en el HTML),
        # luego LQ (Liquid original del HTML que no fue reemplazado).
        result = _unmask_liquid(result_masked, text_tokens, "NLQ")
        result = _unmask_liquid(result,         html_tokens,  "LQ")

        restored_html  = sum(1 for t in html_tokens  if t in result)
        restored_text  = sum(1 for t in text_tokens  if t in result)
        logger.info(
            "CIO fly writer: Agent 2 OK (action %s, %d chars | LQ: %d/%d | NLQ: %d/%d restaurados)",
            action_id, len(result),
            restored_html, len(html_tokens),
            restored_text, len(text_tokens),
        )

        return result

    except Exception as exc:
        logger.error("CIO fly writer: Agent 2 falló (action %s) — %s — usando original", action_id, exc)
        return full_html


def update_node_copy(
    action_id: int,
    template_id: int,
    subject: str,
    body: str,
    preheader: str | None = None,
    user_name: str | None = None,
    campaign_name: str | None = None,
) -> dict[str, Any]:
    """
    Actualiza el copy de un nodo (push o email) en CIO.

    Flujo exacto — 2 requests a CIO:
      1. GET /templates/{template_id}  → leer todos los campos actuales
      2. PUT /templates/{template_id}  → escribir solo subject/body/preheader

    Cooldown: si el mismo action_id fue actualizado hace < 30s → RuntimeError (429).
    """
    now = time.monotonic()
    last = _last_update.get(action_id, 0.0)
    if now - last < _COOLDOWN_SECS:
        remaining = int(_COOLDOWN_SECS - (now - last))
        raise RuntimeError(
            f"Cooldown activo para nodo {action_id}: esperá {remaining}s antes de volver a actualizar."
        )

    # 1. Leer template completo (necesario para no borrar otros campos al hacer PUT)
    logger.info("CIO fly writer: GET template %s (action %s)", template_id, action_id)
    tmpl_data = _fly_get(f"/v1/environments/{_ENV_ID}/templates/{template_id}")
    tmpl = tmpl_data.get("template", tmpl_data)

    # 2. Strip whitespace control en todos los campos — CIO no acepta {%- -%} en ninguno.
    body_clean    = _strip_wsc(body)
    subject_clean = _strip_wsc(subject)
    preheader_clean = _strip_wsc(preheader) if preheader is not None else tmpl.get("preheader_text", "")

    # Para emails: Agent 2 parchea el HTML preservando estructura visual.
    # Para push: body ya limpio va directo.
    tmpl_type = tmpl.get("template_type", "")
    if tmpl_type == "email" and tmpl.get("body") and body_clean.strip():
        body_to_set = _patch_email_html_with_claude(tmpl["body"], body_clean, action_id)
    else:
        body_to_set = body_clean

    # Solo los campos de copy. body_json NO se incluye:
    # CIO valida que body y body_json sean consistentes — si enviamos body_json
    # desactualizado junto con un body nuevo → 422 Unprocessable Entity.
    tmpl_updated = {
        "subject":        subject_clean,
        "body":           body_to_set,
        "preheader_text": preheader_clean,
    }

    logger.info("CIO fly writer: PUT template %s (action %s)", template_id, action_id)
    _fly_put(
        f"/v1/environments/{_ENV_ID}/templates/{template_id}",
        {"template": tmpl_updated},
    )

    # Registrar timestamp del update exitoso para el cooldown
    _last_update[action_id] = time.monotonic()

    # Log de auditoría: quién actualizó este nodo
    if user_name:
        from app.services.supabase_client import log_node_update
        log_node_update(user_name, campaign_name, action_id)

    logger.info("CIO fly writer: nodo %s actualizado correctamente (user=%s)", action_id, user_name or "anon")
    return {
        "ok": True,
        "action_id": action_id,
        "template_id": template_id,
    }
