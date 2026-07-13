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
    _FLY_BASE,
    _refresh_jwt,
)

logger = logging.getLogger("kepler.cio_fly_writer")

_COOLDOWN_SECS = 30
_last_update: dict[int, float] = {}  # action_id → monotonic timestamp


def _strip_wsc(s: str) -> str:
    """Elimina whitespace control de Liquid ({%- y -%}). CIO no lo soporta en ningún campo."""
    return re.sub(r"-%}", "%}", re.sub(r"\{%-", "{%", s))


def _fly_put(path: str, payload: dict[str, Any], sa_token: str = "") -> dict[str, Any]:
    """PUT autenticado via JWT. Renueva el token en 401 (una vez)."""
    jwt = _refresh_jwt(sa_token)
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}

    resp = httpx.put(f"{_FLY_BASE}{path}", headers=headers, json=payload, timeout=30)

    if resp.status_code == 401:
        logger.info("CIO fly writer: JWT expirado, renovando...")
        jwt = _refresh_jwt(sa_token, force=True)
        headers["Authorization"] = f"Bearer {jwt}"
        resp = httpx.put(f"{_FLY_BASE}{path}", headers=headers, json=payload, timeout=30)

    if not resp.is_success:
        logger.error("CIO fly writer: PUT falló %s — body: %s", resp.status_code, resp.text[:500])
        # Extraer mensaje legible del error de CIO antes de lanzar
        try:
            cio_errors = resp.json().get("errors", [])
            if cio_errors:
                detail = cio_errors[0].get("detail", resp.text[:200])
                raise RuntimeError(f"CIO rechazó el email: {detail}")
        except RuntimeError:
            raise
        except Exception:
            pass
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


def _check_liquid_blocks(html: str) -> str | None:
    """
    Verifica que los bloques Liquid estén correctamente cerrados.
    Retorna mensaje de error si la estructura es inválida, None si está OK.
    """
    opens  = len(re.findall(r'\{%-?\s*(?:if|unless|for|case)\b', html))
    closes = len(re.findall(r'\{%-?\s*end(?:if|unless|for|case)\b', html))
    if opens != closes:
        return f"Liquid inválido: {opens} apertura(s) vs {closes} cierre(s) de bloque"
    return None


def _check_duplicate_conditions(html: str) -> str | None:
    """
    Detecta condiciones Liquid duplicadas — la firma exacta de una edición que ANIDÓ
    el bloque de la ronda anterior en vez de reemplazarlo (el incidente de 4 semanas
    de emails vacíos en 'Primer depósito' Colombia: 6-9 copias del mismo
    '{% if Perfil_de_riesgo == ... %}' apiladas).

    _check_liquid_blocks NO detecta esto: cada copia duplicada queda balanceada por
    separado (1 apertura, 1 cierre cada una), así que el conteo total de aperturas
    vs cierres sigue cuadrando aunque el HTML esté corrompido. Este chequeo es el
    que sí lo atrapa — antes de que salga a producción, no semanas después.
    """
    conditions = re.findall(r"\{%-?\s*(?:if|elsif)\s+(.*?)-?%\}", html, re.DOTALL)
    counts: dict[str, int] = {}
    for cond in conditions:
        key = " ".join(cond.split())
        # Chequeos simples de existencia (ej. "customer.first_name", sin comparación) se
        # reusan legítimamente una vez por cada rama externa — eso es normal, no es el bug.
        # La firma real del incidente es siempre una COMPARACIÓN ("Perfil_de_riesgo == 'X'")
        # repetida — esa sí debería aparecer una única vez por nodo.
        if "==" not in key and "!=" not in key:
            continue
        counts[key] = counts.get(key, 0) + 1
    dupes = {k: v for k, v in counts.items() if v > 1}
    if dupes:
        detail = ", ".join(f"'{k}' x{v}" for k, v in dupes.items())
        return f"Condición(es) Liquid duplicada(s) — el bloque anterior no se reemplazó, se anidó: {detail}"
    return None


# ─── Marcadores KEPLER:BODY — acotan el contenido 100% editorial de Kepler ────
# dentro del HTML de un email. Todo lo que está DENTRO de los marcadores es
# reemplazable sin ambigüedad; todo lo que está AFUERA (header, footer, botón
# CTA, MSO conditionals) nunca se toca ni se le muestra al LLM en ediciones 2+.
_MARK_START = "<!--KEPLER:BODY:START-->"
_MARK_END   = "<!--KEPLER:BODY:END-->"


def _extract_marked_block(html: str) -> tuple[str, str, str] | None:
    """
    Busca el bloque delimitado por los marcadores de Kepler.
    Devuelve (antes, adentro, después) solo si hay EXACTAMENTE un START y un END,
    en ese orden. None si no hay marcadores o están mal formados — en ese caso el
    caller cae al flujo de "primera edición", que los vuelve a colocar.
    """
    n_start = html.count(_MARK_START)
    n_end   = html.count(_MARK_END)
    if n_start != 1 or n_end != 1:
        return None
    i_start = html.find(_MARK_START)
    i_end   = html.find(_MARK_END)
    if i_end <= i_start:
        return None
    before = html[:i_start]
    inner  = html[i_start + len(_MARK_START): i_end]
    after  = html[i_end + len(_MARK_END):]
    return before, inner, after


def _repair_orphaned_liquid(html: str) -> tuple[str, int]:
    """
    Elimina closers/openers Liquid huérfanos usando una pila.
    Retorna (html_reparado, cantidad_eliminada).

    Esto cubre el caso donde Agent 2 borra un __LQ_N__ de apertura
    pero conserva su __LQ_M__ de cierre correspondiente.
    """
    TAG_RE = re.compile(
        r'(\{%-?\s*(?:if|unless|for|case|elsif|else|end(?:if|unless|for|case))\b[^%]*?-?%\})',
        re.IGNORECASE | re.DOTALL,
    )
    parts   = TAG_RE.split(html)
    result: list[str] = []
    depth   = 0
    removed = 0

    for part in parts:
        if not TAG_RE.fullmatch(part):
            result.append(part)
            continue
        inner = re.sub(r'^\{%-?\s*', '', part)
        inner = re.sub(r'\s*-?%\}$', '', inner).strip()

        if re.match(r'(?:if|unless|for|case)\b', inner, re.IGNORECASE):
            depth += 1
            result.append(part)
        elif re.match(r'end(?:if|unless|for|case)\b', inner, re.IGNORECASE):
            if depth > 0:
                depth -= 1
                result.append(part)
            else:
                removed += 1   # closer sin apertura → descartar
        elif re.match(r'(?:elsif|else)\b', inner, re.IGNORECASE):
            if depth > 0:
                result.append(part)
            else:
                removed += 1   # elsif/else sin if enclosing → descartar
        else:
            result.append(part)

    return ''.join(result), removed


def _patch_marked_snippet_with_claude(inner_html: str, new_text: str, action_id: int) -> str:
    """
    Reemplaza el snippet delimitado por los marcadores KEPLER:BODY — contenido 100%
    editorial de Kepler, sin footer/botones/header (esos viven afuera del snippet y
    ni siquiera se le muestran a Claude acá). A diferencia de _patch_email_html_with_claude,
    no hay ninguna ambigüedad sobre "qué preservar": TODO el snippet es reemplazable.
    Esto es lo que rompe el ciclo de acumulación — el LLM nunca ve ni puede arrastrar
    el resto del documento ni el Liquid de rondas anteriores.
    """
    import os
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("CIO fly writer: ANTHROPIC_API_KEY no configurada (action %s)", action_id)
        raise RuntimeError(f"No se pudo actualizar el nodo {action_id}: falta ANTHROPIC_API_KEY.")

    new_text_clean = re.sub(r"\{%-", "{%", new_text)
    new_text_clean = re.sub(r"-%}", "%}", new_text_clean)
    # Convertir saltos de línea de texto plano a <br> ANTES de mandarlo a Claude — si se lo
    # dejamos en texto plano, no hay garantía de que lo convierta bien (el bug real: el
    # auto-sanado determinista no lo convertía y el email salía sin párrafos, todo corrido).
    new_text_clean = _plain_text_to_html(new_text_clean)

    # CRÍTICO: enmascarar TAMBIÉN el Liquid del snippet viejo (como __OLDLQ_N__, no __LQ_N__
    # — prefijo distinto a propósito para no confundirlo con el de _patch_email_html_with_claude).
    # Si Claude ve el Liquid viejo crudo (ej. "{% if Perfil_de_riesgo == 'Conservador' %}"),
    # tiende a reusar ese texto literal como "patrón de formato" y envolver el Liquid nuevo
    # adentro — produciendo exactamente "{% if {% if ... %} %}" (probado empíricamente: sin
    # este enmascarado, el patch acotado duplicaba el wrapper igual que el bug original).
    # Enmascarando el viejo Liquid, Claude nunca ve su sintaxis real — no tiene nada que reusar.
    masked_old,  old_tokens  = _mask_liquid(inner_html,      "OLDLQ")
    masked_text, text_tokens = _mask_liquid(new_text_clean,  "NLQ")

    prompt = (
        "Sos un editor de HTML de emails. Te doy un snippet que es 100% contenido "
        "editorial reemplazable — no contiene footer, botones, header ni nada "
        "estructural, eso vive fuera de este snippet y no lo ves.\n\n"
        "TAREA: reemplazá TODO el contenido del snippet (texto Y Liquid) por el texto "
        "nuevo de abajo.\n\n"
        "REGLAS:\n"
        "1. Devolvé SOLO el snippet HTML reemplazado, sin agregar los comentarios "
        "marcadores — esos los agrega el código después, no vos\n"
        "2. Usá los mismos tags/estilos inline que ya tenía el snippet viejo para los párrafos "
        "(mirá dónde están los __OLDLQ_N__ solo para copiar el estilo visual alrededor, nada más)\n"
        "2b. CRÍTICO — este snippet YA está adentro de un <p> del template (con su propio "
        "font-size/color/line-height definidos ahí afuera, vos no lo ves). NUNCA envuelvas tu "
        "output en un <p>, <div> ni ningún otro tag de bloque propio — un <p> anidado dentro "
        "de otro <p> es HTML inválido: el navegador cierra el <p> externo automáticamente "
        "apenas encuentra el tuyo, y el contenido pierde TODO el estilo (se ve como texto "
        "plano sin formato). Tu output debe ser SOLO texto + <br> + Liquid, sin ningún wrapper.\n"
        "3. CRÍTICO — los placeholders __OLDLQ_N__ del snippet viejo son Liquid que se DESCARTA "
        "por completo — es contenido de una edición anterior que ya no aplica. Tu output NO debe "
        "contener NINGÚN __OLDLQ_N__ — ni uno solo, bajo ninguna circunstancia.\n"
        "4. CRÍTICO — los placeholders __NLQ_N__ son el Liquid del texto NUEVO — esos SÍ van en "
        "tu output, exactamente donde aparecen en el texto nuevo, cada uno UNA sola vez\n"
        "5. No agregues ningún Liquid que no sea uno de los __NLQ_N__ de abajo\n"
        "6. El texto nuevo ya trae <br><br> entre párrafos — dejalos tal cual, son la "
        "separación visual entre párrafos dentro del <p> que envuelve el snippet\n\n"
        f"SNIPPET VIEJO (referencia de formato visual únicamente — su Liquid __OLDLQ_N__ se descarta):\n{masked_old}\n\n"
        f"TEXTO NUEVO DEL CUERPO:\n{masked_text}\n\n"
        "Devolvé únicamente el HTML del snippet nuevo. Sin explicaciones, sin markdown, sin ```."
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        result_masked = response.content[0].text.strip()

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"El nuevo contenido del nodo {action_id} no entró en el snippet — simplificalo y reintentá."
            )

        # Guardarraíl específico de este patch: si algún __OLDLQ_N__ sobrevivió, Claude no
        # descartó el Liquid viejo como se le pidió — cortar acá, no intentar "arreglarlo".
        leaked_old = [f"__OLDLQ_{i}__" for i in range(len(old_tokens)) if f"__OLDLQ_{i}__" in result_masked]
        if leaked_old:
            raise RuntimeError(
                f"El patch del nodo {action_id} conservó Liquid viejo que debía descartarse "
                f"({len(leaked_old)} token(s)) — no se envió nada a CIO."
            )

        return _strip_redundant_wrapper(_unmask_liquid(result_masked, text_tokens, "NLQ"))

    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("CIO fly writer: patch de snippet falló (action %s) — %s", action_id, exc)
        raise RuntimeError(f"No se pudo actualizar el contenido del nodo {action_id}: {exc}") from exc


def _patch_email_html_with_claude(
    full_html: str, new_text: str, action_id: int, add_markers: bool = False,
) -> str:
    """
    Agente 2 — HTML Patcher.
    Recibe el HTML completo actual del email (de CIO) y el nuevo texto propuesto.
    Claude reemplaza SOLO el texto visible del cuerpo; devuelve el HTML íntegro sin
    ningún otro cambio (estilos, links, imágenes, botones, footer, MSO conditionals).

    add_markers: si True, le pide a Claude que envuelva el bloque editado con los
    comentarios KEPLER:BODY — así la PRÓXIMA edición de este nodo entra al camino
    acotado de _patch_marked_snippet_with_claude en vez de volver a parchear el
    documento completo. Se verifica el resultado; si Claude no los coloca bien,
    se descartan (no rompen nada) y se reintenta en la próxima edición.

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

    lq_count = len(html_tokens)
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
        f"9. CRÍTICO — PLACEHOLDERS __LQ_N__ (hay {lq_count} en total, de __LQ_0__ a __LQ_{lq_count-1}__): "
        "son código Liquid del email original. TODOS deben aparecer en tu output sin excepción. "
        "Si el cuerpo original tenía __LQ_N__ dentro del texto, conservalos en la zona donde pusiste el texto nuevo — "
        "NO los elimines aunque estén rodeados de texto viejo. "
        f"Verificá antes de responder: tu output debe tener exactamente {lq_count} tokens __LQ_N__.\n"
        "10. CRÍTICO — PLACEHOLDERS __NLQ_N__: son el Liquid del nuevo texto. "
        "Colocalos en el HTML exactamente donde aparecen en el NUEVO TEXTO, respetando su posición relativa.\n"
        "11. Si el nuevo texto tiene menos contenido que el original, dejá el resto del HTML como está\n"
        + (
            f"12. CRÍTICO — envolvé el bloque de contenido personalizado que acabás de escribir "
            f"(todo el texto+Liquid nuevo que insertaste, de punta a punta) con estos dos "
            f"comentarios HTML EXACTOS, sin espacios ni modificaciones: "
            f"{_MARK_START} inmediatamente ANTES del bloque, {_MARK_END} inmediatamente "
            f"DESPUÉS. NO envuelvas el footer, el botón CTA, el header, ni nada que no sea "
            f"el contenido personalizado que vos mismo escribiste.\n\n"
            if add_markers else "\n"
        )
        + f"HTML ACTUAL DEL EMAIL:\n{masked_html}\n\n"
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

        # Validar estructura Liquid antes de enviar a CIO
        liquid_err = _check_liquid_blocks(result)
        if liquid_err:
            logger.warning("CIO fly writer: %s (action %s) — intentando auto-reparar", liquid_err, action_id)
            repaired, n_removed = _repair_orphaned_liquid(result)
            repair_err = _check_liquid_blocks(repaired)
            if repair_err is None and n_removed > 0:
                logger.info(
                    "CIO fly writer: auto-repair OK — %d token(s) huérfano(s) eliminado(s) (action %s)",
                    n_removed, action_id,
                )
                return _finalize_markers(repaired, action_id, add_markers)
            # No se pudo reparar → error claro, no culpar al texto del usuario
            raise RuntimeError(
                "Error interno al insertar el texto en el email — la estructura Liquid resultó inválida "
                "después del procesamiento. Reintentá. Si el error persiste, simplificá los bloques "
                "{% if %}/{% endif %} del texto."
            )

        return _finalize_markers(result, action_id, add_markers)

    except RuntimeError:
        raise  # propagar errores de validación al caller
    except Exception as exc:
        logger.error("CIO fly writer: Agent 2 falló (action %s) — %s — usando original", action_id, exc)
        return full_html


def _finalize_markers(html: str, action_id: int, add_markers: bool) -> str:
    """
    Si se le pidió a Claude que agregue los marcadores KEPLER:BODY (add_markers=True),
    verifica que los haya colocado bien — exactamente 1 START antes de 1 END. Si no,
    los descarta (no rompen nada) y esta edición queda sin marcadores; la próxima
    edición de este nodo simplemente vuelve a intentar el patch de documento completo
    y a colocarlos — degradación segura, nunca peor que el comportamiento sin marcadores.
    """
    if not add_markers:
        return html
    n_start = html.count(_MARK_START)
    n_end   = html.count(_MARK_END)
    if n_start == 1 and n_end == 1 and html.find(_MARK_START) < html.find(_MARK_END):
        logger.info("CIO fly writer: marcadores KEPLER:BODY colocados correctamente (action %s)", action_id)
        return html
    logger.warning(
        "CIO fly writer: Agent 2 no colocó los marcadores correctamente "
        "(action %s, start=%d, end=%d) — se sigue sin marcadores por ahora",
        action_id, n_start, n_end,
    )
    return html.replace(_MARK_START, "").replace(_MARK_END, "")


def _plain_text_to_html(text: str) -> str:
    """
    Convierte texto plano con saltos de línea (el formato de 'cuerpo' que usa el resto
    del sistema — validador, push, etc.) a HTML apto para insertar dentro de un <p>
    ya existente: '\\n\\n' (separación de párrafo) -> '<br><br>', '\\n' suelto -> '<br>'.
    Sin esto, el texto se inserta crudo y el email se ve corrido, sin separación visual
    entre párrafos — el <p> que lo envuelve preserva el estilo, pero el contenido no
    respeta los saltos de línea de HTML por sí solo.
    """
    return text.replace("\n\n", "<br><br>").replace("\n", "<br>")


_WRAPPER_RE = re.compile(r"^\s*<(p|div)\b[^>]*>(.*)</\1>\s*$", re.IGNORECASE | re.DOTALL)


def _strip_redundant_wrapper(html: str) -> str:
    """
    Quita un <p>/<div> que envuelva TODO el snippet devuelto por Claude.

    El snippet vive DENTRO de un <p> del template estático (con su propio
    font-size/color/line-height, definidos afuera del snippet) — si Claude agrega su
    propio wrapper <p> o <div>, queda un <p> anidado dentro de otro <p>, que es HTML
    inválido: el navegador cierra el <p> externo (el que tenía el estilo real) apenas
    encuentra el interno, y el contenido se ve como texto plano sin formato (bug real
    observado: Claude agregó '<p style="margin:0;padding:0;">...</p>' por su cuenta).

    Esto es una defensa en profundidad — la regla 2b del prompt ya se lo prohíbe
    explícitamente, pero no hay que confiar solo en que un LLM siga una instrucción.
    """
    m = _WRAPPER_RE.match(html)
    if m:
        logger.warning(
            "CIO fly writer: se detectó un <%s> envolviendo todo el snippet — se elimina "
            "para no anidarlo dentro del <p> del template",
            m.group(1),
        )
        return m.group(2)
    return html


def _self_heal_corrupted_body(full_html: str, new_text: str, action_id: int) -> str:
    """
    Auto-sanación para nodos corrompidos ANTES de que existiera el sistema de marcadores
    (el incidente de 4 semanas de emails vacíos). En vez de pasarle este HTML ya duplicado
    a Agent 2 — que por su regla de "preservar todo el Liquid existente" reproduciría o
    empeoraría la corrupción — se hace un empalme determinista, sin LLM:

    Por construcción del bug original (cada edición envolvía a la anterior en vez de
    reemplazarla), TODA la corrupción queda contenida entre el PRIMER '{% if %}' del
    documento y el ÚLTIMO '{% endif %}' — el cierre más externo es siempre el último,
    porque cada capa nueva envolvía por fuera. Todo lo anterior al primer '{% if %}' y
    todo lo posterior al último '{% endif %}' es contenido estructural real (header,
    footer, botón CTA) que nunca se tocó — se preserva intacto.
    """
    if "{% if" not in full_html:
        raise RuntimeError(
            f"Nodo {action_id}: se detectó Liquid duplicado pero no se encontró un '{{% if %}}' "
            "de referencia para acotar la zona corrupta — revisar manualmente, no se envió nada a CIO."
        )
    i_first = full_html.find("{% if")

    # Si hay un {% assign %} pegado justo antes del primer {% if %} (solo espacio en
    # blanco entre medio, ej. "{% assign perfil = customer.Perfil_de_riesgo %}"), es una
    # variable auxiliar del MISMO bloque corrupto, no contenido estático real — se
    # extiende el límite hacia atrás para incluirla también (visto en producción: PE
    # tenía una edición vieja que usaba este patrón).
    assign_matches = list(re.finditer(r"\{%-?\s*assign\s+.*?-?%\}", full_html[:i_first]))
    if assign_matches:
        last_assign = assign_matches[-1]
        if full_html[last_assign.end():i_first].strip() == "":
            i_first = last_assign.start()

    endif_matches = list(re.finditer(r"\{%-?\s*endif\s*-?%\}", full_html))
    if not endif_matches:
        raise RuntimeError(
            f"Nodo {action_id}: se detectó Liquid duplicado pero no se encontró ningún "
            "'{% endif %}' para acotar la zona corrupta — revisar manualmente, no se envió nada a CIO."
        )
    i_last_end = endif_matches[-1].end()

    new_text_clean = _plain_text_to_html(_strip_wsc(new_text))
    healed = full_html[:i_first] + _MARK_START + new_text_clean + _MARK_END + full_html[i_last_end:]
    logger.warning(
        "CIO fly writer: nodo %s auto-sanado — Liquid duplicado preexistente reemplazado "
        "por empalme determinista (%d chars -> %d chars)",
        action_id, len(full_html), len(healed),
    )
    return healed


def _patch_email_html(full_html: str, new_text: str, action_id: int) -> str:
    """
    Punto de entrada único para parchear el body de un email — reemplaza la llamada
    directa a _patch_email_html_with_claude en update_node_copy.

    - Si el HTML ya tiene los marcadores KEPLER:BODY (ediciones 2+ de este nodo, ya sano):
      usa el patch ACOTADO (_patch_marked_snippet_with_claude), que solo ve y
      reemplaza el snippet entre marcadores. El resto del documento (header,
      footer, CTA) ni se le muestra al LLM — estructuralmente no puede duplicar
      ni anidar nada del resto del email.
    - Si el HTML YA tiene condiciones duplicadas (nodo corrompido por ediciones de
      ANTES de este fix): auto-sanación determinista (_self_heal_corrupted_body),
      sin pasarle el desastre a un LLM. Esto es lo que permite arreglar los nodos
      viejos con solo hacer clic en 'Validar y enviar' desde la interfaz — no hace
      falta ningún script manual.
    - Si no tiene ni marcadores ni corrupción (primera vez real que Kepler toca este
      nodo, documento limpio): patch de documento completo de siempre, pidiéndole a
      Claude que agregue los marcadores alrededor de lo que edita.

    Guardarraíl final: bloquea (RuntimeError, no se envía a CIO) si el resultado
    tiene Liquid desbalanceado o condiciones duplicadas — la firma exacta del
    incidente que causó 4 semanas de emails vacíos en 'Primer depósito' Colombia.
    """
    marked = _extract_marked_block(full_html)
    if marked is not None:
        before, inner, after = marked
        logger.info("CIO fly writer: marcadores KEPLER:BODY encontrados (action %s) — patch acotado", action_id)
        new_inner = _patch_marked_snippet_with_claude(inner, new_text, action_id)
        result = before + _MARK_START + new_inner + _MARK_END + after
    elif _check_duplicate_conditions(full_html):
        logger.warning(
            "CIO fly writer: Liquid duplicado preexistente en el body actual (action %s) — "
            "auto-sanando en vez de patch de documento completo",
            action_id,
        )
        result = _self_heal_corrupted_body(full_html, new_text, action_id)
    else:
        logger.info(
            "CIO fly writer: sin marcadores ni corrupción (action %s) — primera edición, patch de documento completo",
            action_id,
        )
        result = _patch_email_html_with_claude(full_html, new_text, action_id, add_markers=True)

    dup_err = _check_duplicate_conditions(result)
    if dup_err:
        raise RuntimeError(
            f"Guardarraíl bloqueó el envío del nodo {action_id} — {dup_err}. "
            "No se envió nada a CIO. Esto no debería pasar con el sistema de marcadores; revisar manualmente."
        )
    balance_err = _check_liquid_blocks(result)
    if balance_err:
        raise RuntimeError(f"Guardarraíl bloqueó el envío del nodo {action_id} — {balance_err}. No se envió nada a CIO.")

    return result


def update_node_copy(
    action_id: int,
    template_id: int,
    subject: str,
    body: str,
    preheader: str | None = None,
    user_name: str | None = None,
    campaign_name: str | None = None,
    semana_label: str | None = None,
    fc=None,
) -> dict[str, Any]:
    """
    Actualiza el copy de un nodo (push o email) en CIO.

    Flujo exacto — 2 requests a CIO:
      1. GET /templates/{template_id}  → leer todos los campos actuales
      2. PUT /templates/{template_id}  → escribir solo subject/body/preheader

    Cooldown: si el mismo action_id fue actualizado hace < 30s → RuntimeError (429).
    fc: FunnelClient para el audit log en node_update_log. Requerido para multi-tenant.
    """
    now = time.monotonic()
    last = _last_update.get(action_id, 0.0)
    if now - last < _COOLDOWN_SECS:
        remaining = int(_COOLDOWN_SECS - (now - last))
        raise RuntimeError(
            f"Cooldown activo para nodo {action_id}: esperá {remaining}s antes de volver a actualizar."
        )

    # 1. Leer template completo (necesario para no borrar otros campos al hacer PUT)
    if fc is None:
        raise RuntimeError(
            "update_node_copy requiere fc (FunnelClient) para obtener credenciales CIO."
        )
    creds = fc.get_cio_credentials()
    from app.services.customerio_fly_client import _fly_get
    logger.info("CIO fly writer: GET template %s (action %s)", template_id, action_id)
    tmpl_data = _fly_get(
        f"/v1/environments/{creds.environment_id}/templates/{template_id}",
        sa_token=creds.sa_live_key,
    )
    tmpl = tmpl_data.get("template", tmpl_data)

    # 2. Strip whitespace control en todos los campos — CIO no acepta {%- -%} en ninguno.
    body_clean    = _strip_wsc(body)
    subject_clean = _strip_wsc(subject)
    preheader_clean = _strip_wsc(preheader) if preheader is not None else tmpl.get("preheader_text", "")

    # Para emails: Agent 2 parchea el HTML preservando estructura visual.
    # Para push: body ya limpio va directo.
    tmpl_type = tmpl.get("template_type", "")
    if tmpl_type == "email" and tmpl.get("body") and body_clean.strip():
        body_to_set = _patch_email_html(tmpl["body"], body_clean, action_id)
    else:
        body_to_set = body_clean

    # Guardarraíl final, para AMBOS tipos de nodo (push también puede traer Liquid
    # mal formado directo del generador de estrategia) — bloquea antes del PUT, nunca
    # después. _patch_email_html ya corre este mismo chequeo para email, pero repetirlo
    # acá es gratis y cubre push, que no pasa por ahí.
    guard_err = _check_duplicate_conditions(body_to_set) or _check_liquid_blocks(body_to_set)
    if guard_err:
        raise RuntimeError(f"Guardarraíl bloqueó el envío del nodo {action_id} — {guard_err}. No se envió nada a CIO.")

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
        f"/v1/environments/{creds.environment_id}/templates/{template_id}",
        {"template": tmpl_updated},
        sa_token=creds.sa_live_key,
    )

    # Registrar timestamp del update exitoso para el cooldown
    _last_update[action_id] = time.monotonic()

    # Log de auditoría: quién actualizó este nodo (scoped al tenant via fc)
    if user_name and fc is not None:
        fc.log_node_update(user_name, campaign_name, action_id, semana_label)

    logger.info("CIO fly writer: nodo %s actualizado correctamente (user=%s)", action_id, user_name or "anon")
    return {
        "ok": True,
        "action_id": action_id,
        "template_id": template_id,
    }
