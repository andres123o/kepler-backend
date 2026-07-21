"""
Validador de copy Capa 1 — determinista, sin API calls, costo $0.

Reglas de validación para copy de nodos CIO antes del envío automático.
Solo bloquea lo que se puede verificar mecánicamente; la proporcionalidad
y alineación copy↔objetivo queda para el juez L2 (call_judge_agent).

Las reglas vienen siempre del config JSONB del funnel (validation_rules).
No hay defaults hardcodeados — si faltan campos en BD, las funciones lanzan ValueError.
"""
import re
from typing import Any

# ─── Parser de rules — todo viene de la BD ───────────────────────────────────

def _resolve_rules(rules: dict) -> dict:
    """
    Normaliza el dict de validation_rules del config JSONB del funnel.
    Lanza ValueError si falta alguna clave obligatoria.
    """
    missing = []
    for key in ("forbidden_sfc_terms", "market_patterns", "char_limits",
                "revision_backend_steps", "brand_fallback_name"):
        if key not in rules:
            missing.append(key)
    if missing:
        raise ValueError(
            f"validation_rules en el config JSONB del funnel faltan campos: {missing}. "
            "Agrega 'validation_rules' al config en Supabase."
        )
    return {
        "forbidden_terms":        rules["forbidden_sfc_terms"],
        "market_patterns":        rules["market_patterns"],
        "voseo_enabled":          rules.get("voseo_check", {}).get("enabled", False),
        "voseo_pattern":          rules.get("voseo_check", {}).get("pattern", ""),
        "char_limits":            rules["char_limits"],
        "revision_backend_steps": set(rules["revision_backend_steps"]),
        "brand_fallback_name":    rules["brand_fallback_name"],
    }


# ─── Constantes estructurales (invariantes entre funnels) ────────────────────

# Tasas y porcentajes en el copy.
# El \b final va DENTRO del grupo opcional del sufijo (EA/anual/...) — si va afuera,
# nunca hace match cuando el "%" no tiene sufijo pegado (ej. "12%," o "12% y"), porque
# "%" y el caracter siguiente (espacio, coma, punto) son ambos no-palabra y \b nunca
# encuentra el borde. Esto dejaba pasar sin detectar cualquier cifra de mercado escrita
# como "BanRep al 12%" en vez de "12% anual" — invisible para las reglas 6 y 7.
_RATE_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?)\s*%(?:\s*(?:EA|NMV|NTV|E\.A\.|anual|mensual|efectiv[ao])\b)?",
    re.IGNORECASE,
)

# Palabras que indican que una cifra de mercado trae su período pegado.
# Sin una de estas cerca, la cifra es ambigua (ver validate_node_premium regla 6).
_PERIOD_WORDS_RE = re.compile(
    r"\b(esta semana|este mes|hoy|diari[oa]|semanal|mensual|anual|acumulad[oa]|"
    r"año corrido|ano corrido|semestre|en enero|en febrero|en marzo|en abril|en mayo|en junio|"
    r"en julio|en agosto|en septiembre|en octubre|en noviembre|en diciembre|en lo que va)\b",
    re.IGNORECASE,
)

# Montos en pesos/dólares (ej. "$200.000", "$55.000") — se cuentan junto con las tasas
# para el tope de "una sola cifra por nodo" (ver validate_node_premium regla 6/7). No se
# verifican contra el KB (sería un enriquecimiento aparte); solo cuentan como "una idea
# cuantificada más" para detectar apilamiento de datos en un mismo mensaje.
_MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?")

# Liquid — aperturas y cierres de bloque
_LIQUID_OPEN  = re.compile(r"\{%-?\s*(?:if|unless|for|case)\b")
_LIQUID_CLOSE = re.compile(r"\{%-?\s*end(?:if|unless|for|case)\b")


def _strip_liquid(text: str) -> str:
    """Quita bloques Liquid para que las reglas de texto no falsen al leer variables."""
    text = re.sub(r"\{%.*?%\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
    return text


# Tags de control Liquid que marcan fronteras de rama
_LIQUID_BRANCH_TAG = re.compile(
    r"\{%-?\s*(?:if|elsif|else|endif|unless|endunless|for|endfor|case|when|endcase)[^%]*-?%\}",
    re.IGNORECASE | re.DOTALL,
)
# Variables Liquid {{ ... }}
_LIQUID_VAR = re.compile(r"\{\{[^}]*\}\}")

# Longitud típica de una variable renderizada (nombre de usuario, atributo, etc.)
_VAR_PLACEHOLDER_LEN = 15


def _max_rendered_length(text: str) -> int:
    """
    Estima el largo MÁXIMO posible cuando el campo tiene ramas Liquid.

    Estrategia:
    1. Reemplaza {{ variables }} por un placeholder de _VAR_PLACEHOLDER_LEN chars.
    2. Divide el texto por los tags de control Liquid (if/elsif/else/endif/etc.)
       — lo que queda son fragmentos del texto visible de cada rama.
    3. Devuelve el largo del fragmento más largo (la rama que más chars ocupa).

    Si el texto no tiene Liquid, devuelve len(text) directamente.

    Ejemplo:
      '{% if X %}💚 Hola {{ first_name }}{% else %}triier{% endif %}'
      → ramas: ['💚 Hola ···············', 'triier']
      → max: len('💚 Hola ···············') = 8 + 15 = 23 chars
    """
    if "{%" not in text:
        return len(text)

    # Reemplazar variables por placeholder de longitud fija
    text_no_vars = _LIQUID_VAR.sub("·" * _VAR_PLACEHOLDER_LEN, text)

    # Dividir por tags de control — las ramas quedan como fragmentos
    fragments = _LIQUID_BRANCH_TAG.split(text_no_vars)

    # Tomar el fragmento más largo (ignorando espacios/saltos de línea sobrantes)
    return max((len(f.strip()) for f in fragments), default=0)


def _is_revision_backend(campaign: dict[str, Any], revision_steps: set) -> bool:
    step = (campaign.get("funnel_step_mapped") or "").lower()
    return any(code in step for code in revision_steps)


def _kb_rates(kb_entries: list[dict[str, Any]]) -> list[float]:
    """Extrae todas las tasas numéricas mencionadas en el KB."""
    rates: list[float] = []
    for entry in kb_entries:
        for m in _RATE_RE.finditer(entry.get("contenido", "") or ""):
            try:
                rates.append(float(m.group(1).replace(",", ".")))
            except ValueError:
                pass
    return rates


def _rate_in_kb(rate: float, kb_rates: list[float], tol: float = 0.1) -> bool:
    """True si la tasa está dentro de ±tol de al menos una tasa del KB."""
    if not kb_rates:
        return True  # sin KB no podemos verificar — no bloqueamos
    return any(abs(rate - kr) <= tol for kr in kb_rates)


def _cifra_numbers(cifra_str: str) -> list[float]:
    """Extrae todos los números dentro de un string de cifra (soporta rangos: '6-10%' → [6, 10])."""
    out = []
    for m in re.finditer(r"\d+(?:[.,]\d+)?", cifra_str or ""):
        try:
            out.append(float(m.group(0).replace(",", ".")))
        except ValueError:
            pass
    return out


def _rate_in_cifras(rate: float, cifras: list[dict[str, Any]], tol: float = 0.5) -> bool:
    """True si `rate` coincide (±tol) con algún número de alguna cifra verificada del research."""
    for c in cifras:
        if any(abs(rate - n) <= tol for n in _cifra_numbers(c.get("cifra", ""))):
            return True
    return False



def validate_node(
    node: dict[str, Any],
    campaign: dict[str, Any],
    kb_entries: list[dict[str, Any]],
    rules: dict,
) -> dict[str, Any]:
    """
    Valida un nodo de copy — Layer 1 determinista.

    rules: dict de validation_rules desde el config JSONB del funnel (obligatorio).

    Returns: {passed: bool, errors: list[str], warnings: list[str]}
    """
    r         = _resolve_rules(rules)
    errors:   list[str] = []
    warnings: list[str] = []

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject",   "") or ""
    cuerpo    = node.get("cuerpo",    "") or ""
    preheader = node.get("preheader", "") or ""
    fallback  = r["brand_fallback_name"]

    all_text   = f"{subject} {preheader} {cuerpo}"
    plain_text = _strip_liquid(all_text)
    limits     = r["char_limits"].get(tipo, {})

    # ── 1. Límites de caracteres ────────────────────────────────────────────────
    subj_len = _max_rendered_length(subject)
    if limits.get("subject") and subj_len > limits["subject"]:
        errors.append(
            f"Subject demasiado largo: ~{subj_len} chars renderizados (máximo {limits['subject']})"
        )
    if limits.get("preheader") and preheader:
        pre_len = _max_rendered_length(preheader)
        if pre_len > limits["preheader"]:
            errors.append(
                f"Preheader demasiado largo: ~{pre_len} chars renderizados (máximo {limits['preheader']})"
            )
    if tipo == "push" and limits.get("cuerpo"):
        cuerpo_len = _max_rendered_length(cuerpo)
        if cuerpo_len > limits["cuerpo"]:
            errors.append(
                f"Cuerpo push demasiado largo: ~{cuerpo_len} chars renderizados (máximo {limits['cuerpo']})"
            )

    # ── 2. Términos prohibidos (regulador) ─────────────────────────────────────
    plain_lower = plain_text.lower()
    for term in r["forbidden_terms"]:
        if term.lower() in plain_lower:
            errors.append(f"Término regulatorio prohibido detectado: '{term}'")

    # ── 3. Liquid desbalanceado ─────────────────────────────────────────────────
    opens  = len(_LIQUID_OPEN.findall(all_text))
    closes = len(_LIQUID_CLOSE.findall(all_text))
    if opens != closes:
        errors.append(
            f"Liquid desbalanceado: {opens} apertura(s) vs {closes} cierre(s) — revisá los bloques if/endif"
        )

    # ── 4. Voseo / idioma ──────────────────────────────────────────────────────
    if r["voseo_enabled"]:
        voseo_re = re.compile(r["voseo_pattern"], re.IGNORECASE)
        m_voseo  = voseo_re.search(plain_text)
        if m_voseo:
            errors.append(
                f"Conjugación no permitida: '{m_voseo.group(0)}' — usar tuteo, no voseo"
            )

    # ── 5. first_name sin wrapper (cualquier canal — push también soporta Liquid) ──
    if "{{customer.first_name}}" in all_text:
        if "{% if customer.first_name %}" not in all_text:
            errors.append(
                f"{{{{customer.first_name}}}} sin wrapper — puede quedar vacío en usuarios sin nombre. "
                f"Envolver con: {{% if customer.first_name %}}{{{{ customer.first_name }}}}{{% else %}}{fallback}{{% endif %}}"
            )

    # ── 6. Market data en campaña transaccional ────────────────────────────────
    if not _is_revision_backend(campaign, r["revision_backend_steps"]):
        market_re = re.compile("|".join(r["market_patterns"]), re.IGNORECASE)
        m_market  = market_re.search(plain_text)
        if m_market:
            errors.append(
                f"Dato de mercado '{m_market.group(0)}' en campaña de proceso — "
                "solo permitido en el paso de revisión backend"
            )

    # ── 7. Tasas no en Knowledge Base ──────────────────────────────────────────
    kb_rate_list = _kb_rates(kb_entries)
    for m_rate in _RATE_RE.finditer(plain_text):
        try:
            rate_num = float(m_rate.group(1).replace(",", "."))
        except ValueError:
            continue
        if not _rate_in_kb(rate_num, kb_rate_list):
            errors.append(
                f"Tasa '{m_rate.group(0).strip()}' no encontrada en Knowledge Base — "
                "verificá que sea correcta o eliminá la cifra específica"
            )

    # ── 8. Warnings (no bloquean) ──────────────────────────────────────────────
    if tipo == "email" and not preheader.strip():
        warnings.append("Sin preheader — se recomienda para mejorar el open rate")

    return {
        "passed":   len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


def validate_node_premium(
    node: dict[str, Any],
    kb_entries: list[dict[str, Any]],
    rules: dict,
    cifras_verificadas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Valida un nodo del agente premium — Layer 1 determinista.

    Igual que validate_node, EXCEPTO la regla de menciones de mercado (COLCAP, TRM, etc.
    por nombre es válido, es el contexto que el agente SHAP + Perplexity inyecta a propósito).

    Las CIFRAS numéricas de mercado sí se validan (a diferencia de antes, cuando se
    ignoraban por completo): deben coincidir con `cifras_verificadas` (ver
    anthropic_client.extract_market_cifras), no pueden ir en subject/preheader, deben
    traer período explícito, y máximo 1 por nodo. Esto es lo que evita que el agente
    mezcle o invente cifras de mercado en el copy final.

    rules: dict de validation_rules desde config JSONB del funnel (obligatorio).
    cifras_verificadas: lista de cifras grounded contra el research de esta semana.
    """
    r         = _resolve_rules(rules)
    errors:   list[str] = []
    warnings: list[str] = []

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject",   "") or ""
    cuerpo    = node.get("cuerpo",    "") or ""
    preheader = node.get("preheader", "") or ""
    fallback  = r["brand_fallback_name"]

    all_text   = f"{subject} {preheader} {cuerpo}"
    plain_text = _strip_liquid(all_text)
    limits     = r["char_limits"].get(tipo, {})

    # ── 1. Límites de caracteres ────────────────────────────────────────────────
    subj_len = _max_rendered_length(subject)
    if limits.get("subject") and subj_len > limits["subject"]:
        errors.append(
            f"Subject demasiado largo: ~{subj_len} chars renderizados (máximo {limits['subject']})"
        )
    if limits.get("preheader") and preheader:
        pre_len = _max_rendered_length(preheader)
        if pre_len > limits["preheader"]:
            errors.append(
                f"Preheader demasiado largo: ~{pre_len} chars renderizados (máximo {limits['preheader']})"
            )
    if tipo == "push" and limits.get("cuerpo"):
        cuerpo_len = _max_rendered_length(cuerpo)
        if cuerpo_len > limits["cuerpo"]:
            errors.append(
                f"Cuerpo push demasiado largo: ~{cuerpo_len} chars renderizados (máximo {limits['cuerpo']})"
            )

    # ── 2. Términos prohibidos (regulador) ─────────────────────────────────────
    plain_lower = plain_text.lower()
    for term in r["forbidden_terms"]:
        if term.lower() in plain_lower:
            errors.append(f"Término regulatorio prohibido detectado: '{term}'")

    # ── 3. Liquid desbalanceado ─────────────────────────────────────────────────
    opens  = len(_LIQUID_OPEN.findall(all_text))
    closes = len(_LIQUID_CLOSE.findall(all_text))
    if opens != closes:
        errors.append(
            f"Liquid desbalanceado: {opens} apertura(s) vs {closes} cierre(s) — revisá los bloques if/endif"
        )

    # ── 4. Voseo / idioma ──────────────────────────────────────────────────────
    if r["voseo_enabled"]:
        voseo_re = re.compile(r["voseo_pattern"], re.IGNORECASE)
        m_voseo  = voseo_re.search(plain_text)
        if m_voseo:
            errors.append(
                f"Conjugación no permitida: '{m_voseo.group(0)}' — usar tuteo, no voseo"
            )

    # ── 5. first_name sin wrapper (cualquier canal — push también soporta Liquid) ──
    if "{{customer.first_name}}" in all_text:
        if "{% if customer.first_name %}" not in all_text:
            errors.append(
                f"{{{{customer.first_name}}}} sin wrapper — puede quedar vacío en usuarios sin nombre. "
                f"Envolver con: {{% if customer.first_name %}}{{{{ customer.first_name }}}}{{% else %}}{fallback}{{% endif %}}"
            )

    # ── 6/7. Cifras en subject/preheader/cuerpo — producto contra KB, mercado contra research ───
    # Sin restricción de ubicación — el agente decide dónde va mejor cada cifra (2026-07-09,
    # decisión de producto: antes se prohibía cifra de mercado en subject/preheader a secas,
    # pero eso bloqueaba también cifras reales y verificadas; ahora la única barrera es que
    # la cifra esté grounded, no dónde aparece).
    # Cifras de producto (CDT %, fondos): deben coincidir con el KB (±0.1%).
    # Cifras de mercado (COLCAP, TRM, tasas BanRep, S&P, Brent, etc.): deben coincidir con
    # cifras_verificadas (grounded contra el research real) y traer período explícito en
    # algún lugar del nodo. Antes esta regla se saltaba cualquier número ≥100 sin chequear
    # nada — eso dejaba pasar cifras como "160%" sin ninguna verificación; ahora todo número
    # se valida contra alguna fuente.
    # Tope de "una sola idea cuantificada por nodo" (2026-07-21): antes solo se limitaban las
    # cifras de MERCADO a 1 por nodo, y las de producto no tenían tope ni se contaban los
    # montos ($) — eso dejaba pasar nodos con CDT% + tasa BanRep% + monto mínimo todos juntos,
    # exactamente el "volcado de datos" que un push/email no debe tener (una idea, un dato).
    # Ahora el tope de 1 aplica al TOTAL combinado (producto + mercado + montos).
    subject_plain   = _strip_liquid(subject)
    preheader_plain = _strip_liquid(preheader)
    cuerpo_plain    = _strip_liquid(cuerpo)
    node_plain      = f"{subject_plain} {preheader_plain} {cuerpo_plain}"
    kb_rate_list    = _kb_rates(kb_entries)
    cifras_verificadas = cifras_verificadas or []
    product_matches: list[str] = []
    market_matches: list[str] = []
    for field_name, field_text in (("Subject", subject_plain), ("Preheader", preheader_plain), ("Cuerpo", cuerpo_plain)):
        for m_rate in _RATE_RE.finditer(field_text):
            try:
                rate_num = float(m_rate.group(1).replace(",", "."))
            except ValueError:
                continue
            if _rate_in_kb(rate_num, kb_rate_list):
                product_matches.append(m_rate.group(0).strip())
                continue
            if _rate_in_cifras(rate_num, cifras_verificadas):
                market_matches.append(m_rate.group(0).strip())
            else:
                errors.append(
                    f"{field_name}: cifra '{m_rate.group(0).strip()}' no está en el Knowledge Base ni en "
                    "las cifras de mercado verificadas del research de esta semana — posible "
                    "alucinación, verificala o eliminala"
                )

    money_matches = [m.group(0).strip() for m in _MONEY_RE.finditer(node_plain)]
    total_cifras  = product_matches + market_matches + money_matches

    if len(total_cifras) > 1:
        errors.append(
            f"El nodo menciona {len(total_cifras)} cifras distintas ({', '.join(total_cifras)}) — "
            "máximo 1 cifra en total por nodo (de producto, de mercado o monto, sin importar el "
            "tipo) — un mensaje debe tener una sola idea central, no varios datos apilados"
        )
    elif market_matches and not _PERIOD_WORDS_RE.search(node_plain):
        errors.append(
            f"La cifra de mercado '{market_matches[0]}' no tiene un período explícito "
            "en el nodo (ej. 'esta semana', 'en julio', 'acumulado del año') — "
            "sin período es ambigua"
        )

    # ── 8. Warnings ───────────────────────────────────────────────────────────
    if tipo == "email" and not preheader.strip():
        warnings.append("Sin preheader — se recomienda para mejorar el open rate")

    return {
        "passed":   len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }
