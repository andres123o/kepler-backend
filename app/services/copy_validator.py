"""
Validador de copy Capa 1 — determinista, sin API calls, costo $0.

Reglas de validación para copy de nodos CIO antes del envío automático.
Solo bloquea lo que se puede verificar mecánicamente; la proporcionalidad
y alineación copy↔objetivo queda para el juez L2 (call_judge_agent).
"""
import re
from typing import Any


# Términos prohibidos SFC — listado conservador, solo lo que está en la ley
_SFC_TERMS = [
    "garantizado",
    "sin riesgo",
    "capital protegido",
    "libre de riesgo",
    "sin pérdida",
    "ganancias aseguradas",
    "rentabilidad garantizada",
    "sin perder",
    "retorno garantizado",
]

# Datos de mercado macro — prohibidos en campañas transaccionales de proceso
_MARKET_PATTERNS = [
    r"\bCOLCAP\b",
    r"\bTRM\b",
    r"\bS&P[\s\-]?500\b",
    r"\bS&P\b",
    r"\bBrent\b",
    r"\bspread\s+TES\b",
    r"\bBanRep\b",
    r"\btasa\s+interbancaria\b",
    r"\bíndice\s+bursátil\b",
]
_MARKET_RE = re.compile("|".join(_MARKET_PATTERNS), re.IGNORECASE)

# Voseo colombiano — usar tuteo (tú/usted)
_VOSEO_RE = re.compile(
    r"\b(podés|tenés|invertís|abrís|empezás|hacés|querés|sabés|venís|"
    r"entrás|completás|subís|mirás|buscás|encontrás|traés|traigás)\b",
    re.IGNORECASE,
)

# Tasas y porcentajes en el copy
_RATE_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d+)?)\s*%(?:\s*(?:EA|NMV|NTV|E\.A\.|anual|mensual|efectiv[ao]))?\b",
    re.IGNORECASE,
)

# Liquid — aperturas y cierres de bloque
_LIQUID_OPEN  = re.compile(r"\{%-?\s*(?:if|unless|for|case)\b")
_LIQUID_CLOSE = re.compile(r"\{%-?\s*end(?:if|unless|for|case)\b")

# Campaigns where product/rate mentions are OK (step includes 'befullusercreated')
_REVISION_BACKEND_STEPS = {"befullusercreated", "photo_validation_completed"}

# Límites de caracteres por tipo
_CHAR_LIMITS = {
    "email": {"subject": 50, "preheader": 85},
    "push":  {"subject": 60, "cuerpo": 180},
}


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


def _is_revision_backend(campaign: dict[str, Any]) -> bool:
    step = (campaign.get("funnel_step_mapped") or "").lower()
    return any(code in step for code in _REVISION_BACKEND_STEPS)


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


def extract_relevant_kb(node: dict[str, Any], kb_entries: list[dict[str, Any]]) -> str:
    """
    Extrae entradas de KB relevantes para este nodo (keyword match contra el copy).
    Usado por el juez L2 — solo pasa lo que el copy menciona, no el KB completo.
    """
    copy_text = f"{node.get('subject', '')} {node.get('cuerpo', '')}".lower()

    keyword_groups = {
        "cdt": ["cdt", "certificado de depósito", "depósito a término"],
        "fondo": ["fondo", "accival", "fondos de inversión", "fondo vista"],
        "acciones": ["acciones", "bolsa", "acciones colombianas", "acciones internacionales"],
        "cripto": ["cripto", "bitcoin", "ethereum"],
    }

    relevant: list[str] = []
    seen: set[str] = set()
    for entry in kb_entries:
        titulo_lower = (entry.get("titulo") or "").lower()
        entry_key = entry.get("titulo", "")
        if entry_key in seen:
            continue
        for _, patterns in keyword_groups.items():
            if any(p in copy_text for p in patterns) and any(p in titulo_lower for p in patterns):
                # Recorta contenido a 300 chars para mantener el prompt del juez pequeño
                contenido = (entry.get("contenido") or "")[:300]
                relevant.append(f"[{entry.get('tipo', '').upper()}] {entry.get('titulo')}:\n{contenido}")
                seen.add(entry_key)
                break

    return "\n\n".join(relevant)


def validate_node(
    node: dict[str, Any],
    campaign: dict[str, Any],
    kb_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Valida un nodo de copy — Layer 1 determinista.

    Args:
        node: dict con {tipo, subject, cuerpo, preheader}
        campaign: dict con {funnel_step_mapped, name}
        kb_entries: lista de entradas activas del knowledge base

    Returns:
        {passed: bool, errors: list[str], warnings: list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject", "") or ""
    cuerpo    = node.get("cuerpo", "")   or ""
    preheader = node.get("preheader", "") or ""

    all_text    = f"{subject} {preheader} {cuerpo}"
    plain_text  = _strip_liquid(all_text)  # sin variables Liquid para reglas de texto

    limits = _CHAR_LIMITS.get(tipo, {})

    # ── 1. Límites de caracteres ────────────────────────────────────────────────
    # Usar _max_rendered_length: mide la rama más larga del Liquid, no el template crudo.
    # Un subject con {% if perfil %}...{% elsif %}...{% endif %} se mide por su rama más larga.
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

    # ── 2. Términos SFC prohibidos ──────────────────────────────────────────────
    plain_lower = plain_text.lower()
    for term in _SFC_TERMS:
        if term in plain_lower:
            errors.append(f"Término SFC prohibido detectado: '{term}'")

    # ── 3. Liquid desbalanceado ─────────────────────────────────────────────────
    opens  = len(_LIQUID_OPEN.findall(all_text))
    closes = len(_LIQUID_CLOSE.findall(all_text))
    if opens != closes:
        errors.append(
            f"Liquid desbalanceado: {opens} apertura(s) vs {closes} cierre(s) — revisá los bloques if/endif"
        )

    # ── 4. Voseo ───────────────────────────────────────────────────────────────
    m_voseo = _VOSEO_RE.search(plain_text)
    if m_voseo:
        errors.append(
            f"Voseo detectado: '{m_voseo.group(0)}' — trii usa tuteo (tú/usted), no voseo"
        )

    # ── 5. first_name sin wrapper (email) ─────────────────────────────────────
    if tipo == "email" and "{{customer.first_name}}" in all_text:
        if "{% if customer.first_name %}" not in all_text:
            errors.append(
                "{{customer.first_name}} sin wrapper — puede quedar vacío en usuarios sin nombre. "
                "Envolver con: {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}"
            )

    # ── 6. Market data en campaña transaccional ────────────────────────────────
    if not _is_revision_backend(campaign):
        m_market = _MARKET_RE.search(plain_text)
        if m_market:
            errors.append(
                f"Dato de mercado '{m_market.group(0)}' en campaña de proceso — "
                "solo permitido en Revisión Backend"
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
) -> dict[str, Any]:
    """
    Valida un nodo del agente premium — Layer 1 determinista.

    Igual que validate_node EXCEPTO que el dato de mercado (COLCAP, TRM, S&P,
    Brent, spread TES) es VÁLIDO y esperado en los nodos premium — es el contexto
    que el agente SHAP + Perplexity inyecta intencionalmente en el copy.

    Reglas activas: chars, SFC, Liquid balance, voseo, first_name wrapper, KB rates.
    Regla desactivada: market data (no aplica para premium).
    """
    errors: list[str] = []
    warnings: list[str] = []

    tipo      = node.get("tipo", "push")
    subject   = node.get("subject", "") or ""
    cuerpo    = node.get("cuerpo", "")   or ""
    preheader = node.get("preheader", "") or ""

    all_text   = f"{subject} {preheader} {cuerpo}"
    plain_text = _strip_liquid(all_text)

    # ── 1. Términos SFC prohibidos ──────────────────────────────────────────────
    plain_lower = plain_text.lower()
    for term in _SFC_TERMS:
        if term in plain_lower:
            errors.append(f"Término SFC prohibido detectado: '{term}'")

    # ── 3. Liquid desbalanceado ─────────────────────────────────────────────────
    opens  = len(_LIQUID_OPEN.findall(all_text))
    closes = len(_LIQUID_CLOSE.findall(all_text))
    if opens != closes:
        errors.append(
            f"Liquid desbalanceado: {opens} apertura(s) vs {closes} cierre(s) — revisá los bloques if/endif"
        )

    # ── 4. Voseo ───────────────────────────────────────────────────────────────
    m_voseo = _VOSEO_RE.search(plain_text)
    if m_voseo:
        errors.append(
            f"Voseo detectado: '{m_voseo.group(0)}' — trii usa tuteo (tú/usted), no voseo"
        )

    # ── 5. first_name sin wrapper (email) ─────────────────────────────────────
    if tipo == "email" and "{{customer.first_name}}" in all_text:
        if "{% if customer.first_name %}" not in all_text:
            errors.append(
                "{{customer.first_name}} sin wrapper — puede quedar vacío en usuarios sin nombre. "
                "Envolver con: {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}"
            )

    # ── 6. Market data — NO SE VALIDA en premium (es el contexto esperado) ──────

    # ── 7. Tasas de producto no en Knowledge Base ──────────────────────────────
    # Solo verifica tasas de producto (CDT %, fondos). Las cifras de mercado
    # (TRM, spread TES, variación COLCAP) no se buscan en el KB.
    kb_rate_list = _kb_rates(kb_entries)
    for m_rate in _RATE_RE.finditer(plain_text):
        try:
            rate_num = float(m_rate.group(1).replace(",", "."))
        except ValueError:
            continue
        # Cifras de mercado típicas (TRM ~4000, COLCAP ~1800, S&P >1000):
        # solo verificar tasas de rendimiento plausibles (< 100 = porcentaje)
        if rate_num >= 100:
            continue
        if not _rate_in_kb(rate_num, kb_rate_list):
            errors.append(
                f"Tasa '{m_rate.group(0).strip()}' no encontrada en Knowledge Base — "
                "verificá que sea correcta o eliminá la cifra específica"
            )

    # ── 8. Warnings ───────────────────────────────────────────────────────────
    if tipo == "email" and not preheader.strip():
        warnings.append("Sin preheader — se recomienda para mejorar el open rate")

    return {
        "passed":   len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }
