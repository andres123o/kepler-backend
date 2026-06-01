"""
Monitor semanal de campañas — Fase 3.
Analiza las campañas monitoreadas en cio_campaigns_cache.
No llama a CIO directamente — usa el cache ya populado por /api/strategy/sync.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.services.supabase_client import get_campaigns_cache

logger = logging.getLogger("kepler.campaign_monitor")

# Umbrales de alerta
_MIN_SENT_FOR_ANALYSIS = 50
_DR_WARN_THRESHOLD     = 0.50
_CR_WARN_THRESHOLD     = 0.01
_WOW_DROP_THRESHOLD    = -0.30

# Orden canónico del funnel de activación
FUNNEL_STEPS_ORDER = [
    "Datos básicos",
    "Fotos KYC",
    "Perfil de riesgo",
    "Datos completos",
    "Revisión backend",
    "Primer depósito",
]


def _health_rank(h: str) -> int:
    return {"ok": 0, "sin_datos": 0, "alerta": 1, "critico": 2}.get(h, 0)


def _worst(a: str, b: str) -> str:
    return a if _health_rank(a) >= _health_rank(b) else b


def _week_over_week(series: list[float | int]) -> dict[str, Any]:
    """Compara última semana completa vs promedio de las 4 anteriores."""
    complete = series[:-1] if len(series) > 1 else series
    if len(complete) < 2:
        return {}
    last  = complete[-1]
    prior = complete[-5:-1] if len(complete) >= 5 else complete[:-1]
    avg   = sum(prior) / len(prior) if prior else 0
    change = (last - avg) / avg if avg > 0 else 0.0
    return {
        "last_week": int(last),
        "prior_avg": round(avg, 1),
        "change_pct": round(change * 100, 1),
    }


def _compute_insights(c: dict[str, Any], trends: dict[str, Any]) -> list[str]:
    """Genera recomendaciones accionables específicas por campaña."""
    insights: list[str] = []

    cr         = c.get("conversion_rate") or 0.0
    total_sent = c.get("total_sent") or 0
    converted  = c.get("converted") or 0
    delivered  = c.get("delivered") or 0
    t_conv     = trends.get("converted", {})
    t_open     = trends.get("human_opened", {})
    t_del      = trends.get("delivered", {})

    # Señal de saturación: aperturas y conversiones caen juntas
    if (t_open.get("change_pct", 0) < -20 and t_conv.get("change_pct", 0) < -15):
        insights.append(
            "Saturación de audiencia: aperturas y conversiones caen en paralelo. "
            "Ampliar el segmento o rotar el creative antes del próximo envío."
        )

    # Alta presión, bajo retorno absoluto
    if total_sent > 500 and converted < 10:
        insights.append(
            f"Alto volumen ({total_sent:,} enviados) con muy bajo retorno ({converted} convertidos). "
            "Revisar relevancia del segmento o claridad del call-to-action."
        )

    # Buena eficiencia pero audiencia pequeña → oportunidad de escala
    if cr > 0.05 and total_sent < 200 and converted > 0:
        insights.append(
            f"Tasa de conversión alta ({cr:.0%}) pero audiencia pequeña ({total_sent} enviados). "
            "Oportunidad clara: ampliar el segmento manteniendo los criterios actuales."
        )

    # Fatiga de mensaje: conversiones bajan con entrega estable
    if (t_conv.get("change_pct", 0) < -25
            and t_del.get("change_pct", 0) > -10
            and t_conv.get("last_week", 0) >= 0):
        insights.append(
            f"Fatiga de mensaje: conversiones cayeron {abs(t_conv['change_pct']):.0f}% "
            "con entrega estable. El copy probablemente perdió relevancia — actualizar asunto y primer párrafo."
        )

    # Campaña activa sin ninguna conversión
    if total_sent > 100 and converted == 0 and c.get("status") == "running":
        insights.append(
            "Campaña activa sin ninguna conversión registrada. "
            "Revisar si el evento de goal está correctamente configurado en CIO o pausar para rediseñar."
        )

    # Spike de entrega pero conversión cero
    if c.get("spike_alert") and cr < 0.01:
        insights.append(
            "Spike de entrega detectado pero conversión prácticamente nula. "
            "El volumen no está generando activación — revisar si el segmento es el correcto."
        )

    # Caída fuerte en entregas (posible problema técnico)
    if t_del.get("change_pct", 0) < -40 and t_del.get("last_week", 0) > 10:
        insights.append(
            f"Entregas cayeron {abs(t_del['change_pct']):.0f}% vs semanas anteriores. "
            "Puede ser un problema de supresión, lista desactualizada o trigger roto — verificar en CIO."
        )

    return insights


def _funnel_coverage(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    """Mapea qué pasos del funnel tienen cobertura de campaña activa."""
    covered: dict[str, list[str]] = {}
    for c in campaigns:
        step = c.get("funnel_step_mapped")
        if step and c.get("status") == "running":
            covered.setdefault(step, []).append(c.get("name") or "")

    gaps = [s for s in FUNNEL_STEPS_ORDER if s not in covered]
    coverage_pct = round(len(covered) / len(FUNNEL_STEPS_ORDER) * 100) if FUNNEL_STEPS_ORDER else 0

    return {
        "steps": [
            {
                "step": s,
                "covered": s in covered,
                "campaigns": covered.get(s, []),
            }
            for s in FUNNEL_STEPS_ORDER
        ],
        "gap_steps": gaps,
        "coverage_pct": coverage_pct,
    }


def _efficiency_ranking(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top campañas por conversiones absolutas — el indicador de impacto real."""
    with_data = [r for r in results if r.get("metrics", {}).get("converted", 0) > 0]
    ranked = sorted(with_data, key=lambda r: r["metrics"]["converted"], reverse=True)
    total = sum(r["metrics"]["converted"] for r in results) or 1
    return [
        {
            "name": r["name"],
            "converted": r["metrics"]["converted"],
            "total_sent": r["metrics"]["total_sent"],
            "conversion_rate": round(r["metrics"]["conversion_rate"] * 100, 1),
            "share_of_total_pct": round(r["metrics"]["converted"] / total * 100, 1),
        }
        for r in ranked[:6]
    ]


def _executive_summary(results: list[dict[str, Any]], funnel: dict[str, Any]) -> list[str]:
    """3-5 puntos de alto nivel para el equipo de growth."""
    lines: list[str] = []

    total_converted = sum(r["metrics"]["converted"] for r in results)
    active = [r for r in results if r.get("status_campaign") == "running"]
    criticas = [r for r in results if r["health"] == "critico"]
    alertas  = [r for r in results if r["health"] == "alerta"]
    dead = [
        r for r in active
        if r["metrics"]["total_sent"] > 100 and r["metrics"]["converted"] == 0
    ]

    lines.append(
        f"{total_converted:,} usuarios activados en total a través de {len(active)} campañas activas."
    )

    top = max(results, key=lambda r: r["metrics"]["converted"], default=None)
    if top and top["metrics"]["converted"] > 0:
        share = top["metrics"]["converted"] / total_converted * 100 if total_converted else 0
        lines.append(
            f"Mayor impacto: '{top['name']}' con {top['metrics']['converted']:,} conversiones "
            f"({share:.0f}% del total). Es la campaña más crítica de mantener optimizada."
        )

    if funnel["gap_steps"]:
        lines.append(
            f"Pasos del funnel sin campaña activa: {', '.join(funnel['gap_steps'])}. "
            "Usuarios en esos pasos no reciben comunicación — posible punto de fuga."
        )

    if criticas:
        names = ", ".join(f"'{r['name']}'" for r in criticas[:2])
        lines.append(
            f"{len(criticas)} campaña(s) en estado crítico requieren acción inmediata: {names}."
        )
    elif alertas:
        names = ", ".join(f"'{r['name']}'" for r in alertas[:2])
        lines.append(
            f"{len(alertas)} alerta(s) activas: {names}. Revisar copy o segmento antes del próximo ciclo."
        )

    if dead:
        names = ", ".join(f"'{r['name']}'" for r in dead[:2])
        lines.append(
            f"{len(dead)} campaña(s) activa(s) sin ninguna conversión registrada: {names}. "
            "Pausar o rediseñar para no quemar audiencia."
        )

    return lines


def _analyze_campaign(c: dict[str, Any]) -> dict[str, Any]:
    cid        = c["cio_campaign_id"]
    name       = c.get("name") or ""
    total_sent = c.get("total_sent") or 0
    dr         = c.get("delivery_rate") or 0.0
    cr         = c.get("conversion_rate") or 0.0
    or_        = c.get("open_rate") or 0.0
    weekly     = c.get("metrics_weekly_json") or {}
    series     = weekly.get("series", {})

    base = {
        "cio_campaign_id":    cid,
        "name":               name,
        "status_campaign":    c.get("status"),
        "funnel_step_mapped": c.get("funnel_step_mapped"),
        "metrics": {
            "delivery_rate":   dr,
            "open_rate":       or_,
            "conversion_rate": cr,
            "delivered":       c.get("delivered") or 0,
            "total_sent":      total_sent,
            "converted":       c.get("converted") or 0,
        },
    }

    if total_sent < _MIN_SENT_FOR_ANALYSIS or not series:
        return {**base, "health": "sin_datos", "label": "Sin datos suficientes", "issues": [], "insights": [], "trends": {}}

    trends = {
        "delivered":    _week_over_week(series.get("delivered", [])),
        "converted":    _week_over_week(series.get("converted", [])),
        "human_opened": _week_over_week(series.get("human_opened", [])),
    }

    issues:   list[str] = []
    health = "ok"

    if c.get("spike_alert"):
        issues.append("SPIKE en entrega — revisar inmediatamente")
        health = _worst(health, "critico")

    if total_sent > 100 and dr < _DR_WARN_THRESHOLD:
        issues.append(f"Entrega baja: {dr:.0%} (umbral ≥50%)")
        health = _worst(health, "alerta")

    if (c.get("delivered") or 0) > 50 and cr < _CR_WARN_THRESHOLD:
        issues.append(f"Conversión muy baja: {cr:.1%} (umbral ≥1%)")
        health = _worst(health, "alerta")

    t_conv = trends["converted"]
    if t_conv and t_conv.get("last_week", 0) > 5 and t_conv.get("change_pct", 0) < _WOW_DROP_THRESHOLD * 100:
        issues.append(
            f"Conversiones cayeron {abs(t_conv['change_pct']):.0f}% "
            f"vs promedio prev ({t_conv['prior_avg']:.0f} → {t_conv['last_week']})"
        )
        health = _worst(health, "alerta")

    t_del = trends["delivered"]
    if t_del and t_del.get("change_pct", 0) < -40 and t_del.get("last_week", 0) > 10:
        issues.append(
            f"Entregas cayeron {abs(t_del['change_pct']):.0f}% "
            f"vs promedio prev ({t_del['prior_avg']:.0f} → {t_del['last_week']})"
        )
        health = _worst(health, "alerta")

    label    = "Funcionando bien" if health == "ok" else f"{len(issues)} problema(s) detectado(s)"
    insights = _compute_insights(c, trends)

    return {**base, "health": health, "label": label, "issues": issues, "insights": insights, "trends": trends}


def run_weekly_check() -> dict[str, Any]:
    campaigns = get_campaigns_cache()
    results   = [_analyze_campaign(c) for c in campaigns]

    criticas = [r for r in results if r["health"] == "critico"]
    alertas  = [r for r in results if r["health"] == "alerta"]
    ok       = [r for r in results if r["health"] == "ok"]
    sin_data = [r for r in results if r["health"] == "sin_datos"]

    overall  = "critico" if criticas else ("alerta" if alertas else "ok")
    funnel   = _funnel_coverage(campaigns)
    ranking  = _efficiency_ranking(results)
    summary  = _executive_summary(results, funnel)

    logger.info(
        "weekly_check: %d campañas — %d ok, %d alertas, %d críticas, %d sin datos",
        len(results), len(ok), len(alertas), len(criticas), len(sin_data),
    )

    return {
        "checked_at":        datetime.now(timezone.utc).isoformat(),
        "overall_health":    overall,
        "total_campaigns":   len(results),
        "ok":                len(ok),
        "alertas":           len(alertas),
        "criticas":          len(criticas),
        "sin_datos":         len(sin_data),
        "executive_summary": summary,
        "funnel_coverage":   funnel,
        "efficiency_ranking": ranking,
        "campaigns":         results,
    }
