"""
Orquestador del agente de estrategia de Kepler.
Lee datos de Supabase + CIO + mercado → llama Claude → devuelve preview.
"""

import json
import logging
import re
from typing import Any

from app.services.anthropic_client import call_basic_agent, call_premium_agent
from app.services.customerio_fly_client import build_journey
from app.services.supabase_client import FunnelClient, _default_fc

logger = logging.getLogger("kepler.strategy_agent")


def _get_n_nodos(c: dict[str, Any]) -> int | None:
    raw = c.get("nodes_json")
    if isinstance(raw, dict):
        return raw.get("n_nodos")
    if isinstance(raw, list):
        return len(raw) or None
    return None


# ─── Formatters para el prompt ────────────────────────────────────────────────

def _format_shap_analysis(prediction: dict[str, Any], internal_vars: frozenset[str] = frozenset()) -> str:
    """
    Formatea análisis SHAP para el agente premium — solo variables externas/accionables.
    internal_vars viene del config JSONB del funnel (clave "ml_internal_features").
    """
    full = prediction.get("full_result") or prediction

    prediccion = full.get("prediccion_siguiente_semana")
    baseline   = full.get("baseline_12w")
    brecha     = full.get("brecha_vs_baseline", 0)
    semana     = full.get("semana_label") or full.get("semana_datos", "")

    contexto_raw: list[dict] = full.get("contexto_historico_top_features") or []
    # Solo variables externas — marketing no puede actuar sobre métricas internas de funnel
    contexto = [f for f in contexto_raw if f.get("feature") not in internal_vars]

    if not contexto:
        return (
            f"Predicción: {prediccion} usuarios | Baseline 12w: {baseline} | Brecha: {brecha:+}\n"
            "⚠ SHAP no disponible — corre primero /api/ml/predict."
        )

    def _z(f: dict) -> float:
        return f.get("z_score") or 0.0

    def _fmt(f: dict) -> str:
        cv   = f.get("current_value")
        m12  = f.get("trailing_12w_mean")
        shap = f.get("shap_contribution", 0)

        if cv is not None and m12 is not None and m12 != 0:
            pct = (cv - m12) / m12 * 100
            direction = f"subió {pct:.0f}%" if pct >= 0 else f"cayó {abs(pct):.0f}%"
            trend = f"{direction} vs media 12 sem (actual: {cv:.2f})"
        elif f.get("z_score") is not None:
            trend = f"z={f['z_score']:+.2f} vs media"
        else:
            trend = "sin comparación histórica"

        impact = f"+{shap:.0f}" if shap >= 0 else f"{shap:.0f}"
        return f"  {f['feature']}: {trend} → {impact} depósitos"

    pred_s  = f"{prediccion:,.0f}" if isinstance(prediccion, (int, float)) else str(prediccion)
    base_s  = f"{baseline:,.0f}"   if isinstance(baseline,   (int, float)) else str(baseline)
    brecha_s = f"{brecha:+.0f}"    if isinstance(brecha,     (int, float)) else str(brecha)

    lines = [
        f"PREDICCIÓN {semana}: {pred_s} usuarios",
        f"Baseline 12 semanas: {base_s} | Brecha vs baseline: {brecha_s}",
        "",
        "── DRIVERS PRINCIPALES — por qué el modelo predice este número ──",
    ]

    top_pos = sorted([f for f in contexto if (f.get("shap_contribution") or 0) > 0],
                     key=lambda x: x.get("shap_contribution", 0), reverse=True)[:3]
    top_neg = sorted([f for f in contexto if (f.get("shap_contribution") or 0) < 0],
                     key=lambda x: x.get("shap_contribution", 0))[:3]

    if top_pos:
        lines.append("Impulsando la predicción HACIA ARRIBA (capitalizar):")
        lines.extend(_fmt(f) for f in top_pos)
    if top_neg:
        lines.append("Presionando la predicción HACIA ABAJO (corregir):")
        lines.extend(_fmt(f) for f in top_neg)

    criticos  = [f for f in contexto if _z(f) < -1.5]
    positivos = [f for f in contexto if _z(f) >= 0.5]
    leves_neg = [f for f in contexto if -1.5 <= _z(f) < -0.5]
    estables  = [f for f in contexto if abs(_z(f)) < 0.5]

    lines += ["", "── SEÑALES POR URGENCIA ──"]

    if criticos:
        lines.append(f"🔴 BAJO PRESIÓN — {len(criticos)} variable(s), actuar esta semana:")
        lines.extend(_fmt(f) for f in criticos)

    if positivos:
        lines.append(f"🟢 CON IMPULSO — {len(positivos)} variable(s), capitalizar:")
        lines.extend(_fmt(f) for f in positivos)

    if leves_neg:
        lines.append(f"🟡 VIGILAR — {len(leves_neg)} variable(s), monitorear:")
        lines.extend(_fmt(f) for f in leves_neg)

    if estables:
        names = ", ".join(f["feature"] for f in estables)
        lines.append(f"⚪ ESTABLES: {names}")

    return "\n".join(lines)


def _sanitize_kb_text(text: str) -> str:
    """
    Limpia texto libre del KB antes de enviarlo al agente.
    Comillas dobles → simples: el carácter más común que Claude copia literal
    al cuerpo de un email y rompe el JSON de salida.
    Barras invertidas sueltas → / : evita secuencias de escape inválidas en JSON.
    """
    # Comillas dobles — rectas y tipográficas → comilla simple
    text = text.replace('“', "'").replace('”', "'")  # " "
    text = text.replace('«', "'").replace('»', "'")  # « »
    text = text.replace('"', "'")
    # Barras invertidas no seguidas de escape válido JSON → /
    text = re.sub(r'\\(?!["\\/bfnrtu])', '/', text)
    # Chars de control: tabs → espacios, CR → nada
    text = text.replace('\t', '  ').replace('\r', '')
    return text.strip()


def _format_knowledge_base(kb_entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in kb_entries:
        titulo = _sanitize_kb_text(entry['titulo'])
        contenido = _sanitize_kb_text(entry['contenido'])
        lines.append(f"[{entry['tipo'].upper()}] {titulo}:\n{contenido}")
    return "\n\n".join(lines)



# ─── Helpers para formatear journey (fly API) ─────────────────────────────────

def _strip_html(html: str) -> str:
    """Extrae texto plano de HTML eliminando tags, scripts y CSS."""
    s = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<script[^>]*>.*?</script>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.DOTALL)
    s = re.sub(r"<[^>]+>", " ", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&lt;", "<").replace("&gt;", ">")
          .replace("&quot;", '"').replace("&#39;", "'"))
    return " ".join(s.split())


def _format_journey_for_enrichment(
    journey: dict[str, Any],
    step_code: str | None = None,
    step_name: str | None = None,
) -> str:
    """
    Convierte el output de build_journey() al bloque de texto que reciben los agentes premium y básico.
    Incluye estructura completa del journey: delays, condiciones, A/B splits y contenido
    de mensajes (subject, preheader, body completos — sin truncar).

    step_code / step_name: si se proveen (flujo básico), se añaden al encabezado para
    que el agente pueda copiar step_code y step_name sin inventarlos.
    """
    meta  = journey["meta"]
    nodes = journey["nodes"]

    header = f"## {meta['name']} | ID: {meta['id']}"
    if step_code:
        header += f" | Step: {step_code}"
    if step_name:
        header += f" | Step Name: {step_name}"

    lines: list[str] = [
        header,
        f"Trigger: {meta['trigger']}  |  Goal: {meta['goal']}  |  Estado: {meta['state']}",
        "",
    ]

    msg_count = 0
    for node in nodes:
        tipo  = node.get("type", "")
        nombre = node.get("name") or ""

        if tipo in ("delay_action", "delay_seconds_action"):
            secs = node.get("delay")
            if secs is not None:
                secs  = int(secs)
                horas = secs // 3600
                mins  = (secs % 3600) // 60
                if horas >= 24:
                    dias   = horas // 24
                    resto  = horas % 24
                    legible = f"{dias} dia(s)" + (f" {resto}h" if resto else "")
                elif horas > 0:
                    legible = f"{horas}h" + (f" {mins}min" if mins else "")
                else:
                    legible = f"{mins} minuto(s)"
                lines.append(f"  [Delay: {legible}]")

        elif tipo == "delay_time_window_action":
            start = node.get("start_time", "?")
            end   = node.get("end_time",   "?")
            zone  = node.get("zone", "")
            lines.append(f"  [Ventana: {start} -> {end} ({zone})]")

        elif tipo == "conditional_branch_action":
            conds = node.get("_conditions_decoded")
            cond_str = json.dumps(conds, ensure_ascii=False) if conds else "(condicion no disponible)"
            lines.append(f"  [Condicion: {cond_str}]")

        elif tipo == "random_cohort_branch_action":
            cohorts = node.get("cohorts", [])
            names   = node.get("cohort_names", [])
            ramas   = []
            for idx, pct in enumerate(cohorts):
                n = names[idx] if idx < len(names) and names[idx] else f"Rama {idx + 1}"
                ramas.append(f"{n} {pct / 10:.0f}%")
            lines.append(f"  [A/B Split: {' / '.join(ramas)}]")

        elif tipo == "exit_action":
            lines.append("  [Salida del journey]")

        elif tipo in ("email_action", "push_action"):
            msg_count += 1
            is_email  = tipo == "email_action"
            node_type = "Email" if is_email else "Push"
            subject   = (node.get("_subject")   or node.get("subject",        "")).strip()
            preheader = (node.get("_preheader")  or node.get("preheader_text", "")).strip()
            body_raw  = (node.get("_body")       or node.get("body",           "")).strip()

            body = _strip_html(body_raw) if is_email and body_raw else body_raw

            lines.append("")
            lines.append(f"  [{node_type} #{msg_count}] ID_CIO: {node.get('id', '?')} | NOMBRE: \"{nombre}\"")
            lines.append(f"    subject: \"{subject}\"")
            if is_email and preheader:
                lines.append(f"    preheader: \"{preheader}\"")
            if body:
                lines.append(f"    body: \"{body}\"")
            elif is_email:
                lines.append("    body: (plantilla visual — no disponible)")

        # attribute_update, webhook → omitir (no relevante para optimizacion de copy)

    return "\n".join(lines)


def _cuerpo_for_display(propuesta: dict | None, node: dict[str, Any], is_email: bool) -> str:
    """
    Devuelve el texto a mostrar en canvas para el campo cuerpo.
    - Email con propuesta: Claude devolvió HTML → strip para mostrar texto legible
    - Email sin propuesta: strip del body actual de CIO
    - Push: texto plano tal cual
    """
    raw_cuerpo = propuesta.get("cuerpo", "") if propuesta else ""
    if not is_email:
        return raw_cuerpo or (node.get("_body") or node.get("body") or "")
    from app.services.email_html_patcher import extract_editable_text
    if raw_cuerpo:
        # Claude devolvió algo: puede ser HTML o texto plano
        return extract_editable_text(raw_cuerpo) if raw_cuerpo.lstrip().startswith("<") else raw_cuerpo
    return _email_body_for_display(node, is_email)


def _email_body_for_display(node: dict[str, Any], is_email: bool) -> str:
    """
    Para nodos email sin propuesta: extrae texto legible del HTML para el canvas.
    Para push: devuelve el body tal cual.
    """
    raw = node.get("_body") or node.get("body") or ""
    if not is_email or not raw:
        return raw
    from app.services.email_html_patcher import extract_editable_text
    return extract_editable_text(raw)


def _build_nodos_completos(
    journey: dict[str, Any],
    propuesta_nodos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Lista completa de nodos de mensaje del journey (email/push) para el canvas.
    Nodos con nombre en propuesta_nodos → modificado=True, usan el copy propuesto.
    Nodos sin cambios → modificado=False, usan el copy actual del journey (read-only en UI).
    El delay se infiere del nodo delay_action inmediatamente anterior en la secuencia.
    """
    # Índices de lookup: ID exacto (primario) y nombre exacto (fallback)
    propuesta_by_id: dict[str, dict[str, Any]] = {
        str(n["id_nodo_cio"]): n for n in propuesta_nodos if n.get("id_nodo_cio")
    }
    propuesta_by_nombre: dict[str, dict[str, Any]] = {
        n["nombre"]: n for n in propuesta_nodos if n.get("nombre")
    }

    result: list[dict[str, Any]] = []
    msg_count     = 0
    pending_delay = 0.0

    for node in journey.get("nodes", []):
        tipo = node.get("type", "")

        if tipo in ("delay_action", "delay_seconds_action"):
            secs = node.get("delay") or 0
            pending_delay = int(secs) / 3600
        elif tipo == "delay_time_window_action":
            pass  # no modifica pending_delay; el delay real ya fue capturado por el nodo anterior

        elif tipo in ("email_action", "push_action"):
            msg_count += 1
            is_email  = tipo == "email_action"
            node_id   = str(node.get("id", ""))
            nombre    = node.get("name") or f"{'Email' if is_email else 'Push'} {msg_count}"

            # Match 1: por ID exacto de CIO (infalible)
            propuesta = propuesta_by_id.get(node_id)
            # Match 2: por nombre exacto (fallback si Claude no incluyó id_nodo_cio)
            if propuesta is None:
                propuesta = propuesta_by_nombre.get(nombre)

            raw_template_id = node.get("template_id")
            entry: dict[str, Any] = {
                "orden":                     msg_count,
                "id_nodo_cio":               int(node_id) if node_id.isdigit() else None,
                "template_id":               int(raw_template_id) if raw_template_id else None,
                "nombre":                    nombre,
                "tipo":                      "email" if is_email else "push",
                "delay_desde_anterior_horas": pending_delay,
                "modificado":                propuesta is not None,
                "subject": propuesta.get("subject", "") if propuesta else (node.get("_subject") or node.get("subject") or ""),
                "cuerpo":  _cuerpo_for_display(propuesta, node, is_email),
            }
            if is_email:
                entry["preheader"] = (
                    propuesta.get("preheader") if propuesta
                    else (node.get("_preheader") or "")
                )
            result.append(entry)
            pending_delay = 0.0

    return result


# ─── Funciones principales ────────────────────────────────────────────────────

def get_funnel_health(fc: FunnelClient) -> list[dict[str, Any]]:
    """
    Devuelve diagnóstico detallado del funnel basado en el cache de campañas.
    Semáforo: verde/amarillo/rojo/spike — considera tasas reales, no solo estado.
    NO requiere CIO API key — solo lee el cache de Supabase (poblado por /sync).
    """
    funnel_steps = fc.get_funnel_steps()
    campaigns = fc.get_campaigns_cache()

    step_map: dict[str, list[dict]] = {s["step_code"]: [] for s in funnel_steps}
    for c in campaigns:
        code = c.get("funnel_step_mapped")
        if code and code in step_map:
            step_map[code].append(c)

    result: list[dict[str, Any]] = []
    for step in funnel_steps:
        code = step["step_code"]
        step_campaigns = step_map.get(code, [])

        # ── Métricas agregadas del paso ──────────────────────────────────────
        total_delivered   = sum(c.get("delivered") or 0 for c in step_campaigns)
        total_sent        = sum(c.get("total_sent") or 0 for c in step_campaigns)
        total_converted   = sum(c.get("converted") or 0 for c in step_campaigns)
        total_human_open  = sum(c.get("human_opened") or 0 for c in step_campaigns)
        total_undeliv     = sum(c.get("undeliverable") or 0 for c in step_campaigns)
        total_delta       = sum(c.get("delivery_delta") or 0 for c in step_campaigns)
        total_created     = total_sent + total_undeliv  # aproximación

        def _rate(n: int, d: int) -> float:
            return round(n / d, 4) if d > 0 else 0.0

        delivery_rate   = _rate(total_delivered, total_sent)
        conversion_rate = _rate(total_converted, total_delivered)
        open_rate       = _rate(total_human_open, total_delivered)
        undeliv_rate    = _rate(total_undeliv, total_created)

        # ── Semáforo (basado en métricas agregadas del paso) ─────────────────
        has_spike   = any(c.get("spike_alert") for c in step_campaigns)
        has_running = any(c.get("status") == "running" for c in step_campaigns)

        n_camp = len(step_campaigns)
        camp_str = "campaña" if n_camp == 1 else "campañas"

        if has_spike:
            health = "spike"
            label  = "Pico de tráfico inusual esta semana — revisá antes de ejecutar"
        elif not step_campaigns:
            health = "rojo"
            label  = "Los usuarios en este paso no reciben ningún mensaje"
        elif not has_running:
            health = "amarillo"
            label  = f"{n_camp} {camp_str} sin envíos activos esta semana"
        elif total_sent > 100 and delivery_rate < 0.50:
            health = "amarillo"
            label  = f"Entrega muy baja en este paso ({delivery_rate:.0%})"
        elif total_delivered > 50 and conversion_rate < 0.02:
            health = "amarillo"
            label  = f"Conversión por debajo del mínimo esperado ({conversion_rate:.1%})"
        else:
            health = "verde"
            cr_str = f" — {conversion_rate:.1%} convierte al objetivo" if total_delivered > 50 else ""
            label  = f"{n_camp} {camp_str} activa{'s' if n_camp != 1 else ''}{cr_str}"

        # ── Warnings por campaña (cada una con sus métricas propias) ─────────
        def _camp_warnings(c: dict) -> list[str]:
            w: list[str] = []
            c_sent      = c.get("total_sent") or 0
            c_delivered = c.get("delivered") or 0
            c_converted = c.get("converted") or 0
            c_undeliv   = c.get("undeliverable") or 0
            c_created   = c_sent + c_undeliv

            c_dr = c_delivered / c_sent    if c_sent > 0      else 0.0
            c_cr = c_converted / c_delivered if c_delivered > 0 else 0.0
            c_ur = c_undeliv   / c_created  if c_created > 0  else 0.0

            if c_sent > 100 and c_dr < 0.50:
                w.append(
                    f"Solo el {c_dr:.0%} de los mensajes llega al destino — "
                    f"revisá si los usuarios tienen notificaciones activas."
                )
            if c_delivered > 50 and c_cr < 0.02:
                w.append(
                    f"Conversión del {c_cr:.1%} — el mínimo esperado es 2%. "
                    f"El copy o el momento de envío pueden estar fallando."
                )
            if c_created > 100 and c_ur > 0.25:
                w.append(
                    f"{c_ur:.0%} de los mensajes no llegan — "
                    f"los usuarios pueden tener notificaciones desactivadas o correos no válidos."
                )
            return w

        def _campaign_dict(c: dict, step_name: str | None, step_code: str | None, exit_event: str | None) -> dict:
            return {
                "cio_campaign_id":     c["cio_campaign_id"],
                "name":                c["name"],
                "status":              c.get("status"),
                "funnel_step_name":    step_name,
                "funnel_step_code":    step_code,
                "goal_event":          c.get("goal_event") or exit_event,
                "delivery_rate":       c.get("delivery_rate") or 0.0,
                "open_rate":           c.get("open_rate") or 0.0,
                "conversion_rate":     c.get("conversion_rate") or 0.0,
                "delivered":           c.get("delivered") or 0,
                "total_sent":          c.get("total_sent") or 0,
                "converted":           c.get("converted") or 0,
                "undeliverable":       c.get("undeliverable") or 0,
                "human_opened":        c.get("human_opened") or 0,
                "clicked":             c.get("clicked") or 0,
                "spike_alert":         bool(c.get("spike_alert")),
                "last_synced_at":      c.get("last_synced_at"),
                "metrics_weekly_json": c.get("metrics_weekly_json"),
                "n_nodos":             _get_n_nodos(c),
                "node_list":           [
                    {"id": n.get("id"), "type": n.get("type"), "name": n.get("name") or ""}
                    for n in ((c.get("nodes_json") or {}).get("nodes") or [])
                    if n.get("type") in ("email_action", "push_notification_action")
                ],
                "warnings":            _camp_warnings(c),
            }

        result.append({
            "step_order":    step["step_order"],
            "step_code":     code,
            "step_name":     step["step_name"],
            "health":        health,
            "label":         label,
            "warnings":      [],  # warnings ahora van por campaña, no por paso
            "metrics": {
                "delivered":       total_delivered,
                "total_sent":      total_sent,
                "converted":       total_converted,
                "human_opened":    total_human_open,
                "undeliverable":   total_undeliv,
                "delivery_rate":   delivery_rate,
                "open_rate":       open_rate,
                "conversion_rate": conversion_rate,
                "delivery_delta":  total_delta,
            },
            "campaigns": [
                _campaign_dict(c, step_name=step["step_name"], step_code=code, exit_event=step.get("exit_event"))
                for c in step_campaigns
            ],
            "entry_event": step.get("entry_event"),
            "exit_event":  step.get("exit_event"),
        })

    # ── Campañas activas sin funnel_step_mapped ──────────────────────────────
    # No se descartan: sin esto, una campaña real y corriendo desaparece por
    # completo de esta vista solo porque el mapeo trigger→step no calzó.
    mapped_ids = {c["cio_campaign_id"] for step_campaigns in step_map.values() for c in step_campaigns}
    unmapped = [c for c in campaigns if c["cio_campaign_id"] not in mapped_ids]
    if unmapped:
        n_camp = len(unmapped)
        camp_str = "campaña" if n_camp == 1 else "campañas"
        result.append({
            "step_order":    len(funnel_steps) + 1,
            "step_code":     None,
            "step_name":     "Sin paso mapeado",
            "health":        "amarillo",
            "label":         f"{n_camp} {camp_str} activa{'s' if n_camp != 1 else ''} sin paso de funnel asociado — revisá el mapeo trigger→step",
            "warnings":      [],
            "metrics": {
                "delivered":       sum(c.get("delivered") or 0 for c in unmapped),
                "total_sent":      sum(c.get("total_sent") or 0 for c in unmapped),
                "converted":       sum(c.get("converted") or 0 for c in unmapped),
                "human_opened":    sum(c.get("human_opened") or 0 for c in unmapped),
                "undeliverable":   sum(c.get("undeliverable") or 0 for c in unmapped),
                "delivery_rate":   0.0,
                "open_rate":       0.0,
                "conversion_rate": 0.0,
                "delivery_delta":  sum(c.get("delivery_delta") or 0 for c in unmapped),
            },
            "campaigns": [
                _campaign_dict(c, step_name=None, step_code=c.get("funnel_step_mapped"), exit_event=None)
                for c in unmapped
            ],
            "entry_event": None,
            "exit_event":  None,
        })

    return result


def generate_premium_strategy(fc: FunnelClient, market_research: dict | None = None) -> dict[str, Any]:
    """
    Flujo único del agente premium (campaña marcada como agent_tier='premium' en Supabase).

    Pasos:
      1. SHAP + proyección desde Supabase
      2. Campaña premium leída desde cio_campaigns_cache (NO hardcodeada)
      3. Journey completo desde CIO fly API
      4. Knowledge Base
      5. Research Perplexity (sonar-pro) — se salta si market_research ya viene provisto
      6. Una sola llamada Claude con PREMIUM_AGENT_SYSTEM_PROMPT
      7. Guardar + retornar

    Args:
        fc: FunnelClient del funnel activo.
        market_research: resultado pre-fetched de Perplexity ({"raw_text": ..., "citations": [...]}).
                         Si es None, se llama a Perplexity internamente.
    """
    from app.services.perplexity_client import fetch_market_research, format_research_block

    logger.info("══════════════════════════════════════════════════")
    logger.info("[PREMIUM] Iniciando flujo agente premium")

    # 1. SHAP + proyección
    prediction = fc.get_latest_prediction()
    if not prediction:
        raise ValueError("No hay predicción guardada. Corre primero /api/ml/predict.")

    semana_label = prediction.get("semana_label") or prediction.get("semana_datos", "")
    logger.info("[PREMIUM] Semana: %s", semana_label)
    internal_vars = frozenset(fc.get_funnel_config().get("ml_internal_features") or [])
    shap_text = _format_shap_analysis(prediction, internal_vars)

    # 2. Campaña premium desde Supabase (agent_tier='premium' — NO hardcodeado)
    campaigns = fc.get_campaigns_cache()
    premium_campaign = next(
        (c for c in campaigns if c.get("agent_tier") == "premium"),
        None,
    )
    if not premium_campaign:
        raise ValueError(
            "No hay campaña con agent_tier='premium' en cio_campaigns_cache. "
            "Corre: UPDATE cio_campaigns_cache SET agent_tier='premium' "
            "WHERE funnel_step_mapped='step_09_full_account';"
        )

    cid   = str(premium_campaign["cio_campaign_id"])
    cname = premium_campaign.get("name", cid)
    logger.info("[PREMIUM] Campaña premium: '%s' (ID %s)", cname, cid)

    # 3. Journey completo desde CIO fly API
    journey    = build_journey(cid, fc)
    journey_text = _format_journey_for_enrichment(journey)
    msg_nodes  = [n for n in journey["nodes"] if n.get("type") in ("email_action", "push_action")]
    logger.info("[PREMIUM] Journey: %d nodos totales | %d mensajes", len(journey["nodes"]), len(msg_nodes))

    # 4. Knowledge Base
    kb_entries = fc.get_knowledge_base()
    kb_text    = _format_knowledge_base(kb_entries)

    # 5. Research Perplexity — usa el provisto externamente o fetcha uno nuevo
    if market_research is not None:
        research      = market_research
        research_text = format_research_block(research)
        logger.info("[PREMIUM] Research provisto externamente: %d chars | %d citations",
                    len(research_text), len(research.get("citations", [])))
    else:
        logger.info("[PREMIUM] Fetching research Perplexity (sonar-pro)...")
        research      = fetch_market_research(semana_label, fc=fc)
        research_text = format_research_block(research)
        logger.info("[PREMIUM] Research: %d chars | %d citations",
                    len(research_text), len(research.get("citations", [])))

    # 6. Prompts desde BD del funnel — sin fallback, error explícito si faltan
    system_prompt = fc.get_agent_prompt("premium", "system")
    kb_preamble   = fc.get_agent_prompt("premium", "kb_preamble")
    user_template = fc.get_agent_prompt("premium", "user_template")
    if not system_prompt or not kb_preamble or not user_template:
        raise RuntimeError(
            "funnel_prompts le faltan filas a premium: system, kb_preamble o user_template. "
            "Corre seed_prompts.py para este funnel."
        )
    logger.info("[PREMIUM] Llamando agente premium Claude...")
    strategy = call_premium_agent(
        shap_text=shap_text,
        research_text=research_text,
        kb_text=kb_text,
        journey_text=journey_text,
        semana_label=semana_label,
        system_prompt=system_prompt,
        kb_preamble=kb_preamble,
        user_template=user_template,
    )
    strategy["semana_label"] = semana_label

    # Attachar nodos_completos para el canvas del frontend
    acciones = strategy.get("acciones") or []
    if acciones:
        propuesta_nodos = (acciones[0].get("propuesta") or {}).get("nodos") or []
        if propuesta_nodos:
            acciones[0]["nodos_completos"] = _build_nodos_completos(journey, propuesta_nodos)

    # 7. Guardar resultado
    try:
        fc.save_strategy_result(strategy)
        logger.info("[PREMIUM] Resultado guardado en strategy_results")
    except Exception as exc:
        logger.error("[PREMIUM] ❌ FALLO AL GUARDAR strategy_results: %s", exc, exc_info=True)

    logger.info("[PREMIUM] Flujo completado ✓")
    logger.info("══════════════════════════════════════════════════")
    return strategy


def generate_basic_strategy(fc: FunnelClient) -> dict[str, Any]:
    """
    Flujo del agente básico (campañas con agent_tier='basic' en Supabase).

    Pasos:
      1. Campañas básicas desde cio_campaigns_cache (agent_tier='basic')
      2. Journey de cada campaña desde CIO fly API
      3. Compliance KB (tipo='compliance') — sin productos
      4. Una sola llamada Claude con BASIC_AGENT_SYSTEM_PROMPT + fecha_hoy
      5. Guardar + retornar
    """
    from datetime import date

    logger.info("══════════════════════════════════════════════════")
    logger.info("[BASIC] Iniciando flujo agente básico")

    # 1. Campañas básicas (agent_tier='basic' en Supabase)
    campaigns = fc.get_campaigns_cache()
    basic_campaigns = [c for c in campaigns if c.get("agent_tier") == "basic"]

    if not basic_campaigns:
        raise ValueError(
            "No hay campañas con agent_tier='basic' en cio_campaigns_cache. "
            "Por defecto todas las campañas son 'basic' — verifica que la columna exista."
        )
    logger.info("[BASIC] Campañas básicas encontradas: %d", len(basic_campaigns))

    # 2. Journey de cada campaña básica — guardar por cid para nodos_completos después
    journey_blocks: list[str] = []
    journeys_by_cid: dict[str, dict[str, Any]] = {}
    for c in basic_campaigns:
        cid        = str(c["cio_campaign_id"])
        cname      = c.get("name", cid)
        step_code  = c.get("funnel_step_mapped")
        step_name  = c.get("funnel_step_name")
        try:
            journey = build_journey(cid, fc)
            journeys_by_cid[cid] = journey
            msg_count = sum(
                1 for n in journey["nodes"]
                if n.get("type") in ("email_action", "push_action")
            )
            logger.info("[BASIC] Journey '%s' (ID %s): %d mensajes", cname, cid, msg_count)
            journey_blocks.append(
                _format_journey_for_enrichment(journey, step_code=step_code, step_name=step_name)
            )
        except Exception as exc:
            logger.warning("[BASIC] No se pudo cargar journey '%s' (ID %s): %s", cname, cid, exc)
            journey_blocks.append(f"## {cname} | ID: {cid}\n⚠ Journey no disponible: {exc}")

    journeys_text = "\n\n".join(journey_blocks)

    # 3. Knowledge Base completo
    kb_entries = fc.get_knowledge_base()
    kb_text    = _format_knowledge_base(kb_entries)
    logger.info("[BASIC] KB: %d entries activos", len(kb_entries))

    # 4. Contexto de calendario — query y params desde BD del funnel
    from app.services.perplexity_client import fetch_calendar_context, format_calendar_block

    fecha_hoy = date.today().isoformat()
    logger.info("[BASIC] Fetching calendar context (Perplexity sonar) | fecha=%s", fecha_hoy)
    calendar      = fetch_calendar_context(fecha_hoy, fc=fc)
    calendar_text = format_calendar_block(calendar)
    logger.info("[BASIC] Calendario: festivos=%d | citations=%d",
                len((calendar.get("raw_json") or {}).get("festivos_semana", [])),
                len(calendar.get("citations", [])))

    # 5. Prompts desde BD del funnel — sin fallback, error explícito si faltan
    system_prompt = fc.get_agent_prompt("basic", "system")
    kb_preamble   = fc.get_agent_prompt("basic", "kb_preamble")
    user_template = fc.get_agent_prompt("basic", "user_template")
    if not system_prompt or not kb_preamble or not user_template:
        raise RuntimeError(
            "funnel_prompts le faltan filas a basic: system, kb_preamble o user_template. "
            "Corre seed_prompts.py para este funnel."
        )
    logger.info("[BASIC] Llamando agente basico Claude | fecha=%s", fecha_hoy)
    strategy = call_basic_agent(
        kb_text=kb_text,
        journeys_text=journeys_text,
        fecha_hoy=fecha_hoy,
        calendar_text=calendar_text,
        system_prompt=system_prompt,
        kb_preamble=kb_preamble,
        user_template=user_template,
    )
    # Usar semana_label de la predicción (ej. "15 al 21 de junio 2026") para que el
    # frontend pueda compararlo contra predLabel y no lo marque como estrategia vieja.
    # Fallback a fecha_hoy si no hay predicción guardada.
    prediction = fc.get_latest_prediction()
    strategy["semana_label"] = (prediction.get("semana_label") if prediction else None) or fecha_hoy

    # Attachar nodos_completos para el canvas: una por acción, matcheando su journey
    for accion in strategy.get("acciones") or []:
        cid = str(accion.get("campaña_existente_id") or "")
        journey = journeys_by_cid.get(cid)
        if not journey:
            logger.warning("[BASIC] Journey no encontrado para campaña_id=%s — sin nodos_completos", cid)
            continue
        propuesta_nodos = (accion.get("propuesta") or {}).get("nodos") or []
        if propuesta_nodos:
            accion["nodos_completos"] = _build_nodos_completos(journey, propuesta_nodos)
            logger.info("[BASIC] nodos_completos: %d nodos para campaña_id=%s",
                        len(accion["nodos_completos"]), cid)

    # 6. Guardar resultado (estructural — distingue de premium en la DB)
    try:
        fc.save_structural_result(strategy)
        logger.info("[BASIC] Resultado guardado en strategy_results (_tipo=estructural)")
    except Exception as exc:
        logger.error("[BASIC] ❌ FALLO AL GUARDAR strategy_results: %s", exc, exc_info=True)

    logger.info("[BASIC] Flujo completado ✓")
    logger.info("══════════════════════════════════════════════════")
    return strategy



def execute_strategy(strategy: dict[str, Any], fc: FunnelClient) -> dict[str, Any]:
    """
    Ejecuta las acciones aprobadas de una estrategia en CIO.
    Solo ejecuta acciones con prioridad != 'sin_accion'.

    SALVAGUARDAS:
    - CIO_DRY_RUN=true bloquea todo (activo por defecto en .env)
    - Máximo CIO_MAX_CAMPAIGNS_PER_EXECUTE operaciones por llamada (default 3)
    - Anti-duplicado en create_campaign (ver customerio_client.py)
    - Nunca toca la Track API — solo Journeys App API
    fc: FunnelClient requerido — lee trigger_event, conversion_event, cashin_attribute
        desde config JSONB del funnel (sección 'cio').
    """
    # ⚠️  WRITE — solo se ejecuta si CIO_DRY_RUN=false en .env (bloqueado por defecto)
    from app.services.customerio_client import _MAX_OPS

    executed: list[dict] = []
    errors: list[dict] = []
    ops_count = 0

    acciones_a_ejecutar = [
        a for a in strategy.get("acciones", [])
        if a.get("prioridad") != "sin_accion"
    ]

    if len(acciones_a_ejecutar) > _MAX_OPS:
        logger.warning(
            "execute_strategy: %d acciones solicitadas, límite es %d. Ejecutando solo las primeras.",
            len(acciones_a_ejecutar), _MAX_OPS,
        )
        acciones_a_ejecutar = acciones_a_ejecutar[:_MAX_OPS]

    for accion in acciones_a_ejecutar:
        if ops_count >= _MAX_OPS:
            break

        tipo = accion.get("tipo_accion")
        propuesta = accion.get("propuesta", {})

        try:
            if tipo == "crear":
                result = _create_journey_from_propuesta(propuesta, fc)
                executed.append({
                    "step_code": accion["step_code"],
                    "tipo": "crear",
                    "campaign_id": result.get("id"),
                    "nombre": result.get("name"),
                })
                ops_count += 1

            elif tipo in ("optimizar", "reforzar"):
                campaign_id = accion.get("campaña_existente_id")
                if campaign_id:
                    result = _update_journey_copy(campaign_id, propuesta)
                    executed.append({
                        "step_code": accion["step_code"],
                        "tipo": tipo,
                        "campaign_id": campaign_id,
                        "nodos_actualizados": result.get("nodos_actualizados", 0),
                    })
                    ops_count += 1

        except Exception as exc:
            logger.error("Error ejecutando acción %s para %s: %s",
                         tipo, accion.get("step_code"), exc)
            errors.append({
                "step_code": accion.get("step_code"),
                "tipo": tipo,
                "error": str(exc),
            })

    return {
        "executed": executed,
        "errors": errors,
        "total_ejecutadas": len(executed),
        "total_errores": len(errors),
    }


def _create_journey_from_propuesta(propuesta: dict[str, Any], fc: FunnelClient) -> dict[str, Any]:
    """
    Crea un journey completo en CIO a partir de la propuesta del agente.
    Sigue el patrón de la campaña Kepler v4:
    AB split → [delay + time_window + push + check_cashin] × nodos → exit

    fc: requerido para leer trigger_event, conversion_event y cashin_attribute
        desde la sección 'cio' del config JSONB del funnel.
    """
    # ⚠️  WRITE — usa CUSTOMERIO_APP_API_KEY, NO el token sa_live.
    # Todas estas funciones llaman _guard_write() que bloquea si CIO_DRY_RUN=true.
    from app.services.customerio_client import create_campaign, add_action, add_edge, activate_campaign

    cio_cfg = fc.get_cio_config()
    default_trigger  = cio_cfg.get("trigger_event")
    default_conv     = cio_cfg.get("conversion_event")
    cashin_attribute = cio_cfg.get("cashin_attribute")

    if not default_trigger or not default_conv or not cashin_attribute:
        raise ValueError(
            "La sección 'cio' del config JSONB del funnel debe incluir "
            "'trigger_event', 'conversion_event' y 'cashin_attribute'."
        )

    config = {
        "name": propuesta.get("nombre_campaña", f"{fc.funnel_slug}_Kepler_Journey"),
        "type": "transactional",
        "event": propuesta.get("trigger_event") or default_trigger,
        "event_type": "event",
        "conversion_type": "perform_event",
        "conversion_action": "receiving",
        "conversion_event_name": propuesta.get("conversion_event") or default_conv,
        "conversion_window": 604800,
        "restart_mode": "rematch",
        "exit_on_conversion_matched": False,
        "send_to_unsubscribed": False,
        "filters": [],
        "anchors": [],
    }

    campaign = create_campaign(config, fc)
    campaign_id = campaign["id"]

    exit_action = add_action(campaign_id, {"type": "exit_action", "sub_type": "default"}, fc)
    exit_id = str(exit_action["id"])

    prev_id: str | None = None
    nodos = propuesta.get("nodos", [])

    for nodo in nodos:
        delay_secs = int(nodo.get("delay_desde_anterior_horas", 24) * 3600)
        tipo = nodo.get("tipo", "push")

        delay = add_action(campaign_id, {
            "type": "delay_seconds_action",
            "sub_type": "default",
            "delay": delay_secs,
        }, fc)
        delay_id = str(delay["id"])

        if prev_id:
            add_edge(campaign_id, prev_id, delay_id, fc=fc)

        subject_liquid  = nodo.get("subject", "")
        cuerpo_liquid   = nodo.get("cuerpo", "")
        preheader_liquid = nodo.get("preheader", "")

        if tipo == "push":
            msg_action = add_action(campaign_id, {
                "type": "push_action",
                "sub_type": "default",
                "name": f"Push {nodo['orden']}",
                "subject": subject_liquid,
                "body": cuerpo_liquid,
                "sending_state": "automatic",
                "send_to_platform": "all",
            }, fc)
        else:
            email_payload: dict[str, Any] = {
                "type": "email_action",
                "sub_type": "default",
                "name": f"Email {nodo['orden']}",
                "subject": subject_liquid,
                "sending_state": "automatic",
                "tracked": True,
            }
            if preheader_liquid:
                email_payload["preheader"] = preheader_liquid
            msg_action = add_action(campaign_id, email_payload, fc)

        msg_id = str(msg_action["id"])
        add_edge(campaign_id, delay_id, msg_id)

        check = add_action(campaign_id, {
            "type": "conditional_branch_action",
            "sub_type": "default",
            "conditions": [{"field": cashin_attribute, "operator": "exists", "type": "attribute"}],
        })
        check_id = str(check["id"])
        add_edge(campaign_id, msg_id, check_id, fc=fc)
        add_edge(campaign_id, check_id, exit_id, fc=fc, edge_type="branch", index=0)

        prev_id = check_id

    if prev_id:
        add_edge(campaign_id, prev_id, exit_id, fc=fc, edge_type="branch", index=1)

    activate_campaign(campaign_id, fc=fc)
    logger.info("Journey creado y activado: id=%s nombre=%s", campaign_id, config["name"])
    return campaign


def _update_journey_copy(campaign_id: str, propuesta: dict[str, Any], fc: FunnelClient) -> dict[str, Any]:
    """
    Actualiza el copy de los nodos email/push de una campaña existente.
    fc: FunnelClient requerido para credenciales CIO desde org_secrets.
    """
    from app.services.customerio_client import get_campaign, update_action

    campaign = get_campaign(campaign_id, fc)
    actions = campaign.get("actions", [])
    nodos_propuesta = propuesta.get("nodos", [])

    msg_actions = [
        a for a in actions
        if a.get("type") in ("email_action", "push_action")
    ]

    updated = 0
    for i, action in enumerate(msg_actions):
        if i >= len(nodos_propuesta):
            break
        nodo = nodos_propuesta[i]
        subject_liquid = nodo.get("subject", "")
        update_action(campaign_id, str(action["id"]), {"subject": subject_liquid}, fc=fc)
        updated += 1

    return {"nodos_actualizados": updated}
