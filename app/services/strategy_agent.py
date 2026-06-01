"""
Orquestador del agente de estrategia de Kepler.
Lee datos de Supabase + CIO + mercado → llama Claude → devuelve preview.
"""

import json
import logging
import re
from typing import Any

from app.services.anthropic_client import generate_strategy, generate_strategy_enriched, generate_structural_strategy
from app.services.customerio_fly_client import build_journey
from app.services.supabase_client import (
    get_campaigns_cache,
    get_funnel_context,
    get_funnel_steps,
    get_knowledge_base,
    get_latest_prediction,
    save_strategy_result,
    save_structural_result,
)

logger = logging.getLogger("kepler.strategy_agent")


# ─── Helpers para leer nodes_json ────────────────────────────────────────────

def _nodes_list(c: dict[str, Any]) -> list[dict]:
    """Extrae lista de nodos de nodes_json (soporta formato dict nuevo y lista viejo)."""
    raw = c.get("nodes_json")
    if isinstance(raw, dict):
        return raw.get("nodes", [])
    if isinstance(raw, list):
        return raw  # backward compat
    return []


def _get_n_nodos(c: dict[str, Any]) -> int | None:
    raw = c.get("nodes_json")
    if isinstance(raw, dict):
        return raw.get("n_nodos")
    if isinstance(raw, list):
        return len(raw) or None
    return None


# ─── Formatters para el prompt ────────────────────────────────────────────────

def _format_shap_analysis(prediction: dict[str, Any]) -> str:
    """
    Formatea análisis SHAP completo desde contexto_historico_top_features.
    Incluye las top 20 features con z-scores, valores actuales vs media 12w,
    y contribuciones SHAP en depósitos. Clasifica en 4 grupos de urgencia.
    """
    full = prediction.get("full_result") or prediction

    prediccion = full.get("prediccion_siguiente_semana")
    baseline   = full.get("baseline_12w")
    brecha     = full.get("brecha_vs_baseline", 0)
    semana     = full.get("semana_label") or full.get("semana_datos", "")

    contexto: list[dict] = full.get("contexto_historico_top_features") or []

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


def _format_campaigns_summary(
    campaigns: list[dict[str, Any]],
    funnel_steps: list[dict[str, Any]],
    shap_contexto: list[dict[str, Any]] | None = None,
) -> str:
    """
    Formatea campañas mapeadas al funnel cruzadas con señales SHAP por paso.
    Para cada paso muestra: estado campaña + métricas + señal del modelo.
    """
    step_map: dict[str, list[dict]] = {s["step_code"]: [] for s in funnel_steps}

    unmapped_count = 0
    for c in campaigns:
        code = c.get("funnel_step_mapped")
        if code and code in step_map:
            step_map[code].append(c)
        else:
            unmapped_count += 1

    # Mapeo SHAP por paso: match por substring del step_code en el nombre del feature
    shap_by_step: dict[str, list[dict]] = {s["step_code"]: [] for s in funnel_steps}
    if shap_contexto:
        for feat in shap_contexto:
            fname = feat["feature"]
            for step in funnel_steps:
                scode = step["step_code"]
                # Extrae el número del paso (ej. "02" de "step_02_email_kyc")
                parts = scode.split("_")
                step_num = parts[1] if len(parts) >= 2 and parts[1].isdigit() else ""
                if scode in fname or (step_num and f"step_{step_num}" in fname):
                    shap_by_step[scode].append(feat)
                    break

    lines: list[str] = []
    for step in funnel_steps:
        code = step["step_code"]
        name = step["step_name"]
        step_campaigns = step_map.get(code, [])
        step_shap = shap_by_step.get(code, [])

        # Señal SHAP del paso (la feature con mayor |SHAP| entre las del paso)
        shap_tag = ""
        if step_shap:
            top  = max(step_shap, key=lambda x: abs(x.get("shap_contribution", 0)))
            z    = top.get("z_score")
            shap = top.get("shap_contribution", 0)
            cv   = top.get("current_value")
            m12  = top.get("trailing_12w_mean")

            if cv is not None and m12 is not None and m12 != 0:
                pct = (cv - m12) / m12 * 100
                direction = f"{'subió' if pct >= 0 else 'cayó'} {abs(pct):.0f}%"
            else:
                direction = f"z={z:+.2f}" if z is not None else "?"

            impact = f"+{shap:.0f}" if shap >= 0 else f"{shap:.0f}"

            if z is not None:
                if z < -1.5:
                    shap_tag = f" [🔴 {direction} vs media → {impact} dep]"
                elif z >= 0.5:
                    shap_tag = f" [🟢 {direction} vs media → {impact} dep — capitalizar]"
                elif z < -0.5:
                    shap_tag = f" [🟡 {direction} vs media → {impact} dep]"
                else:
                    shap_tag = " [⚪ estable]"

        if not step_campaigns:
            lines.append(f"{code} ({name}): GAP — sin campaña activa{shap_tag}")
        else:
            for c in step_campaigns:
                status    = c.get("status", "?")
                dr        = c.get("delivery_rate") or 0.0
                cr        = c.get("conversion_rate") or 0.0
                or_       = c.get("open_rate") or 0.0
                delivered = c.get("delivered") or 0
                weeks     = c.get("metrics_weeks_covered") or 0

                cr_flag = ""
                if delivered > 100:
                    if cr < 0.02:
                        cr_flag = " ⚠️CR MUY BAJO"
                    elif cr < 0.05:
                        cr_flag = " ↓CR bajo"
                    elif cr >= 0.07:
                        cr_flag = " ✓CR bueno"

                metrics_str = (
                    f" entrega={dr:.0%} open={or_:.0%} CR={cr:.1%}{cr_flag}"
                    f" ({delivered:,} delivered, {weeks}w)"
                ) if delivered > 0 else " (sin datos de entrega aún)"

                lines.append(
                    f"{code} ({name}): ID={c['cio_campaign_id']} '{c['name']}'"
                    f" [{status}]{metrics_str}{shap_tag}"
                )

                # Copies actuales de los nodos — para que Claude proponga diffs específicos
                msg_nodes = [n for n in _nodes_list(c) if n.get("type") in ("push_action", "email_action")]
                for n in msg_nodes:
                    is_email  = n.get("type") == "email_action"
                    node_type = "Email" if is_email else "Push"
                    label = n.get("name") or node_type
                    subj  = (n.get("subject")    or "").strip()
                    body  = (n.get("body")        or "").strip()
                    pre   = (n.get("preheader")   or "").strip()

                    def _trunc(s: str, n: int) -> str:
                        return s[:n] + "…" if len(s) > n else s

                    lines.append(f"    {label} ({node_type}):")
                    lines.append(f"      subject:   \"{_trunc(subj, 120)}\"")
                    if is_email and pre:
                        lines.append(f"      preheader: \"{_trunc(pre, 120)}\"")
                    if body:
                        lines.append(f"      body:      \"{_trunc(body, 300)}\"")
                    elif is_email:
                        lines.append(f"      body:      (plantilla visual — no disponible en API)")

    if unmapped_count:
        lines.append(f"(+{unmapped_count} campañas sin mapear al funnel — ignoradas)")

    return "\n".join(lines)


def _format_knowledge_base(kb_entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for entry in kb_entries:
        lines.append(f"[{entry['tipo'].upper()}] {entry['titulo']}:\n{entry['contenido']}")
    return "\n\n".join(lines)


def _format_funnel_context(context: list[dict[str, Any]]) -> str:
    """
    Formatea eventos y atributos CIO para el prompt del agente.
    El agente usa esto para saber qué trigger_event usar en cada paso
    y qué atributos están disponibles para personalización y segmentación.
    """
    events = [c for c in context if c["record_type"] == "event"]
    attributes = [c for c in context if c["record_type"] == "attribute"]

    lines: list[str] = ["EVENTOS CIO POR PASO (usa estos como trigger_event y conversion_event):"]
    by_step: dict[str, list[dict]] = {}
    for e in events:
        step = e.get("funnel_step_code") or "sin_paso"
        by_step.setdefault(step, []).append(e)

    for step, evs in by_step.items():
        lines.append(f"\n  {step}:")
        for e in evs:
            role = e.get("event_role", "")
            lines.append(f"    [{role}] {e['name']} — {e['description']}")

    lines.append("\nATRIBUTOS CIO DISPONIBLES (para segmentación y personalización de copy):")
    for a in attributes:
        vals = f" | valores: {a['possible_values']}" if a.get("possible_values") else ""
        lines.append(f"  {a['name']}: {a['description']}{vals}")

    return "\n".join(lines)


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


def _format_journey_for_enrichment(journey: dict[str, Any]) -> str:
    """
    Convierte el output de build_journey() al texto que recibe generate_strategy_enriched().
    Incluye estructura completa del journey: delays, condiciones, A/B splits y contenido
    de mensajes (subject, preheader, body completos — sin truncar).
    """
    meta  = journey["meta"]
    nodes = journey["nodes"]

    lines: list[str] = [
        f"## {meta['name']} | ID: {meta['id']}",
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

def get_funnel_health() -> list[dict[str, Any]]:
    """
    Devuelve diagnóstico detallado del funnel basado en el cache de campañas.
    Semáforo: verde/amarillo/rojo/spike — considera tasas reales, no solo estado.
    NO requiere CIO API key — solo lee el cache de Supabase (poblado por /sync).
    """
    funnel_steps = get_funnel_steps()
    campaigns = get_campaigns_cache()

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
                {
                    "cio_campaign_id":   c["cio_campaign_id"],
                    "name":              c["name"],
                    "status":            c.get("status"),
                    "goal_event":        c.get("goal_event") or step.get("exit_event"),
                    "delivery_rate":     c.get("delivery_rate") or 0.0,
                    "open_rate":         c.get("open_rate") or 0.0,
                    "conversion_rate":   c.get("conversion_rate") or 0.0,
                    "delivered":         c.get("delivered") or 0,
                    "total_sent":        c.get("total_sent") or 0,
                    "converted":         c.get("converted") or 0,
                    "undeliverable":     c.get("undeliverable") or 0,
                    "metrics_weekly_json": c.get("metrics_weekly_json"),
                    "n_nodos":           _get_n_nodos(c),
                    "warnings":          _camp_warnings(c),
                }
                for c in step_campaigns
            ],
            "entry_event": step.get("entry_event"),
            "exit_event":  step.get("exit_event"),
        })

    return result


def generate_weekly_strategy(
    contexto_adicional: str | None = None,
) -> dict[str, Any]:
    """
    Genera el preview de estrategia para la semana actual.

    FASE 1 — datos resumidos → Claude identifica qué campañas necesitan trabajo
    FASE 2 — para esas campañas: fetch template content real → Claude produce diffs específicos
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("[ESTRATEGIA] Iniciando generación semanal")

    # ── Carga de datos ─────────────────────────────────────────────────────────
    prediction = get_latest_prediction()
    if not prediction:
        raise ValueError("No hay predicción guardada. Corre primero /api/ml/predict.")

    funnel_steps = get_funnel_steps()
    campaigns    = get_campaigns_cache()
    kb_entries   = get_knowledge_base()
    funnel_ctx   = get_funnel_context()

    semana_label = prediction.get("semana_label") or prediction.get("semana_datos", "")
    logger.info("[ESTRATEGIA] Semana: %s | Campañas en cache: %d | KB entries: %d",
                semana_label, len(campaigns), len(kb_entries))

    if not campaigns:
        logger.warning("[ESTRATEGIA] Cache de campañas vacío — diagnóstico tendrá gaps. "
                       "Corre /api/strategy/sync primero.")

    full = prediction.get("full_result") or prediction
    shap_contexto: list[dict] = full.get("contexto_historico_top_features") or []

    shap_text       = _format_shap_analysis(prediction)
    campaigns_text  = _format_campaigns_summary(campaigns, funnel_steps, shap_contexto)
    kb_text         = _format_knowledge_base(kb_entries)
    funnel_ctx_text = _format_funnel_context(funnel_ctx)

    # ── FASE 1: Claude con datos resumidos ─────────────────────────────────────
    logger.info("──────────────────────────────────────────────────")
    logger.info("[FASE 1] Enviando a Claude datos resumidos de %d campaña(s)...", len(campaigns))
    logger.info("[FASE 1] SHAP features: %d | Funnel steps: %d | Contexto adicional: %s",
                len(shap_contexto), len(funnel_steps),
                "sí" if contexto_adicional else "no")

    strategy = generate_strategy(
        shap_analysis=shap_text,
        campaigns_summary=campaigns_text,
        knowledge_base_text=kb_text,
        funnel_context_text=funnel_ctx_text,
        semana_label=semana_label,
        contexto_adicional=contexto_adicional,
    )
    strategy["semana_label"] = semana_label

    acciones = strategy.get("acciones", [])
    logger.info("[FASE 1] Claude devolvió: %d acción(es) | estado_funnel=%s",
                len(acciones), strategy.get("estado_funnel", "?"))
    for a in acciones:
        logger.info("[FASE 1]   → %s | %s | campaña_id=%s | tipo=%s",
                    a.get("step_code"), a.get("prioridad"),
                    a.get("campaña_existente_id", "nueva"), a.get("tipo_accion"))

    # ── FASE 2: Auto-fetch del journey completo para campañas seleccionadas ────
    a_enriquecer = [
        a for a in acciones
        if a.get("tipo_accion") in ("optimizar", "reforzar")
        and a.get("campaña_existente_id")
    ][:2]  # Máximo 2 — nunca más, prohibido

    logger.info("──────────────────────────────────────────────────")
    if not a_enriquecer:
        logger.info("[FASE 2] No hay campañas para enriquecer — saltando")
    else:
        logger.info("[FASE 2] %d campaña(s) a leer via fly API: %s",
                    len(a_enriquecer),
                    [a["campaña_existente_id"] for a in a_enriquecer])

        enriched_parts:   list[str]        = []
        journeys_by_cid:  dict[str, dict]  = {}

        for accion in a_enriquecer:
            cid   = str(accion["campaña_existente_id"])
            cname = accion.get("campaña_existente_nombre", cid)
            logger.info("[FASE 2] Leyendo journey campaña '%s' (ID %s)...", cname, cid)

            try:
                journey = build_journey(cid)
                journeys_by_cid[cid] = journey
                enriched_parts.append(_format_journey_for_enrichment(journey))
                msg_nodes = [n for n in journey["nodes"]
                             if n.get("type") in ("email_action", "push_action")]
                logger.info("[FASE 2]   → %d nodos totales | %d mensajes leídos",
                            len(journey["nodes"]), len(msg_nodes))
            except Exception as exc:
                logger.error("[FASE 2]   → Error leyendo campaña %s: %s", cid, exc)

        if enriched_parts:
            enriched_text = "\n\n".join(enriched_parts)
            strategy = generate_strategy_enriched(
                phase1_strategy=strategy,
                enriched_campaigns_text=enriched_text,
                semana_label=semana_label,
            )
            strategy["semana_label"] = semana_label

            # Attachar nodos_completos a cada acción enriquecida para el canvas
            for accion in strategy.get("acciones", []):
                cid = str(accion.get("campaña_existente_id") or "")
                if cid in journeys_by_cid:
                    propuesta_nodos = (accion.get("propuesta") or {}).get("nodos") or []
                    accion["nodos_completos"] = _build_nodos_completos(
                        journeys_by_cid[cid], propuesta_nodos
                    )

            logger.info("[FASE 2] Estrategia enriquecida con journeys reales de %d campaña(s)",
                        len(enriched_parts))
        else:
            logger.warning("[FASE 2] Ninguna campaña pudo leerse — usando resultado de Fase 1")

    # ── Guardar resultado ──────────────────────────────────────────────────────
    logger.info("──────────────────────────────────────────────────")
    try:
        save_strategy_result(strategy)
        logger.info("[ESTRATEGIA] Resultado guardado en strategy_results")
    except Exception as exc:
        logger.warning("[ESTRATEGIA] No se pudo guardar en strategy_results: %s", exc)

    logger.info("[ESTRATEGIA] Generación completada ✓")
    logger.info("══════════════════════════════════════════════════")
    return strategy


def generate_structural_optimization(
    phase2_strategy: dict[str, Any],
    contexto_adicional: str | None = None,
) -> dict[str, Any]:
    """
    Fase 2B: analiza campañas que Phase 2 no tocó y propone optimizaciones estructurales.
    No usa SHAP — lee el journey completo de cada campaña restante via fly API.
    """
    logger.info("══════════════════════════════════════════════════")
    logger.info("[FASE 2B] Iniciando optimización estructural")

    kb_entries   = get_knowledge_base()
    funnel_ctx   = get_funnel_context()

    # Normalizar a str — Claude puede devolver el ID como int en JSON (sin comillas),
    # pero Supabase lo guarda como str. Sin str(), "4626" in {4626} → False y la campaña
    # no se excluye, apareciendo analizada dos veces.
    phase2_ids: set[str] = {
        str(a["campaña_existente_id"])
        for a in phase2_strategy.get("acciones", [])
        if a.get("campaña_existente_id") is not None
    }
    logger.info(
        "[FASE 2B] Phase 2 ya actuó en %d campaña(s): %s | tipos: %s",
        len(phase2_ids),
        phase2_ids,
        {type(x).__name__ for x in phase2_ids},
    )

    phase2_lines: list[str] = []
    for a in phase2_strategy.get("acciones", []):
        cid = a.get("campaña_existente_id", "nueva campaña")
        phase2_lines.append(
            f"  {a.get('step_code', '?')}: {a.get('tipo_accion', '?')} "
            f"'{a.get('campaña_existente_nombre', cid)}' — "
            f"{a.get('razon', '')[:120]}"
        )
    phase2_summary = (
        "\n".join(phase2_lines) if phase2_lines
        else "  (Phase 2 no tuvo acciones esta semana)"
    )

    health = get_funnel_health()
    health_lines: list[str] = []
    campanas_restantes: list[dict[str, Any]] = []  # para el auto-fetch

    for step in health:
        for c in step["campaigns"]:
            if str(c["cio_campaign_id"]) in phase2_ids:
                logger.debug(
                    "[FASE 2B] Excluyendo campaña '%s' (ID %s) — ya intervenida en Modo 1",
                    c["name"], c["cio_campaign_id"],
                )
                continue
            warns = " | ".join(c.get("warnings", []))
            n_nodos = c.get("n_nodos")
            health_lines.append(
                f"{step['step_code']} ({step['step_name']}): "
                f"ID={c['cio_campaign_id']} '{c['name']}' "
                f"estado={c.get('status', '?')} "
                f"entrega={c['delivery_rate']:.0%} open={c['open_rate']:.0%} "
                f"CR={c['conversion_rate']:.1%} "
                f"({c['delivered']:,} entregados · {c['undeliverable']:,} no entregados)"
                + (f" · {n_nodos} nodos" if n_nodos else "")
                + (f" | ⚠ {warns}" if warns else "")
            )
            campanas_restantes.append(c)

    health_text = (
        "\n".join(health_lines) if health_lines
        else "(no hay campañas fuera del alcance de Phase 2)"
    )

    # ── Auto-fetch: journey completo de cada campaña restante via fly API ────────
    logger.info("[FASE 2B] Leyendo journeys de %d campaña(s) restantes via fly API...",
                len(campanas_restantes))
    journey_parts: list[str] = []
    journeys_by_cid: dict[str, dict[str, Any]] = {}
    for c in campanas_restantes:
        cid   = str(c["cio_campaign_id"])
        cname = c.get("name", cid)
        try:
            journey = build_journey(cid)
            journey_parts.append(_format_journey_for_enrichment(journey))
            journeys_by_cid[cid] = journey
            msg_nodes = [n for n in journey["nodes"]
                         if n.get("type") in ("email_action", "push_action")]
            logger.info("[FASE 2B]   → '%s' (ID %s): %d nodos | %d mensajes",
                        cname, cid, len(journey["nodes"]), len(msg_nodes))
        except Exception as exc:
            logger.error("[FASE 2B]   → Error leyendo campaña '%s' (ID %s): %s", cname, cid, exc)

    detalle_campanas = "\n\n".join(journey_parts) if journey_parts else "(no se pudo leer ningún journey)"

    kb_text         = _format_knowledge_base(kb_entries)
    funnel_ctx_text = _format_funnel_context(funnel_ctx)

    semana_label = phase2_strategy.get("semana_label") or phase2_strategy.get("semana_datos", "")
    logger.info("[FASE 2B] Enviando %d campaña(s) para análisis estructural | semana=%s",
                len(campanas_restantes), semana_label)

    result = generate_structural_strategy(
        funnel_health_text=health_text,
        phase2_acciones_summary=phase2_summary,
        knowledge_base_text=kb_text,
        funnel_context_text=funnel_ctx_text,
        detalle_campanas=detalle_campanas,
        contexto_adicional=contexto_adicional,
    )
    result["semana_label"] = semana_label

    # Attachar nodos_completos (con id_nodo_cio y template_id reales) para el canvas del frontend
    for accion in result.get("acciones", []):
        cid = str(accion.get("campaña_existente_id") or "")
        if cid in journeys_by_cid:
            propuesta_nodos = (accion.get("propuesta") or {}).get("nodos") or []
            accion["nodos_completos"] = _build_nodos_completos(
                journeys_by_cid[cid], propuesta_nodos
            )

    # Fase 2B no tiene señal SHAP — nullear para que el badge no aparezca en UI
    for a in result.get("acciones", []):
        a["shap_z"] = 0.0
        a["shap_contribucion"] = None

    try:
        save_structural_result(result)
        logger.info("[FASE 2B] Resultado guardado en strategy_results (_tipo=estructural)")
    except Exception as exc:
        logger.warning("[FASE 2B] No se pudo guardar en Supabase: %s", exc)

    logger.info("[FASE 2B] Completado — %d acción(es) estructurales", len(result.get("acciones", [])))
    logger.info("══════════════════════════════════════════════════")
    return result


def execute_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    """
    Ejecuta las acciones aprobadas de una estrategia en CIO.
    Solo ejecuta acciones con prioridad != 'sin_accion'.

    SALVAGUARDAS:
    - CIO_DRY_RUN=true bloquea todo (activo por defecto en .env)
    - Máximo CIO_MAX_CAMPAIGNS_PER_EXECUTE operaciones por llamada (default 3)
    - Anti-duplicado en create_campaign (ver customerio_client.py)
    - Nunca toca la Track API — solo Journeys App API
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
                result = _create_journey_from_propuesta(propuesta)
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


def _create_journey_from_propuesta(propuesta: dict[str, Any]) -> dict[str, Any]:
    """
    Crea un journey completo en CIO a partir de la propuesta del agente.
    Sigue el patrón de la campaña Kepler v4:
    AB split → [delay + time_window + push + check_cashin] × nodos → exit
    """
    # ⚠️  WRITE — usa CUSTOMERIO_APP_API_KEY, NO el token sa_live.
    # Todas estas funciones llaman _guard_write() que bloquea si CIO_DRY_RUN=true.
    from app.services.customerio_client import create_campaign, add_action, add_edge, activate_campaign

    config = {
        "name": propuesta.get("nombre_campaña", "CO_Kepler_Journey"),
        "type": "transactional",
        "event": propuesta.get("trigger_event", "BeFullUserCreated"),
        "event_type": "event",
        "conversion_type": "perform_event",
        "conversion_action": "receiving",
        "conversion_event_name": propuesta.get("conversion_event", "BeCashIn"),
        "conversion_window": 604800,
        "restart_mode": "rematch",
        "exit_on_conversion_matched": False,
        "send_to_unsubscribed": False,
        "filters": [],
        "anchors": [],
    }

    campaign = create_campaign(config)
    campaign_id = campaign["id"]

    exit_action = add_action(campaign_id, {"type": "exit_action", "sub_type": "default"})
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
        })
        delay_id = str(delay["id"])

        if prev_id:
            add_edge(campaign_id, prev_id, delay_id)

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
            })
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
            msg_action = add_action(campaign_id, email_payload)

        msg_id = str(msg_action["id"])
        add_edge(campaign_id, delay_id, msg_id)

        check = add_action(campaign_id, {
            "type": "conditional_branch_action",
            "sub_type": "default",
            "conditions": [{"field": "date_first_cashin", "operator": "exists", "type": "attribute"}],
        })
        check_id = str(check["id"])
        add_edge(campaign_id, msg_id, check_id)
        add_edge(campaign_id, check_id, exit_id, edge_type="branch", index=0)

        prev_id = check_id

    if prev_id:
        add_edge(campaign_id, prev_id, exit_id, edge_type="branch", index=1)

    activate_campaign(campaign_id)
    logger.info("Journey creado y activado: id=%s nombre=%s", campaign_id, config["name"])
    return campaign


def _update_journey_copy(campaign_id: str, propuesta: dict[str, Any]) -> dict[str, Any]:
    """
    Actualiza el copy de los nodos email/push de una campaña existente.
    """
    # get_campaign → READONLY (sa_live) | update_action → WRITE (guarded)
    from app.services.customerio_client import get_campaign, update_action

    campaign = get_campaign(campaign_id)
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
        update_action(campaign_id, str(action["id"]), {"subject": subject_liquid})
        updated += 1

    return {"nodos_actualizados": updated}
