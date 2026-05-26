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
_MIN_SENT_FOR_ANALYSIS = 50       # ignorar campañas con muy pocos envíos
_DR_WARN_THRESHOLD     = 0.50     # delivery_rate < 50% → alerta
_CR_WARN_THRESHOLD     = 0.01     # conversion_rate < 1% → alerta
_WOW_DROP_THRESHOLD    = -0.30    # caída >30% semana a semana → alerta


def _health_rank(h: str) -> int:
    return {"ok": 0, "sin_datos": 0, "alerta": 1, "critico": 2}.get(h, 0)


def _worst(a: str, b: str) -> str:
    return a if _health_rank(a) >= _health_rank(b) else b


def _week_over_week(series: list[float | int]) -> dict[str, Any]:
    """
    Compara la última semana completa contra el promedio de las 4 anteriores.
    La última posición del array es la semana actual (parcial) — se excluye.
    """
    complete = series[:-1] if len(series) > 1 else series
    if len(complete) < 2:
        return {}
    last    = complete[-1]
    prior   = complete[-5:-1] if len(complete) >= 5 else complete[:-1]
    avg     = sum(prior) / len(prior) if prior else 0
    change  = (last - avg) / avg if avg > 0 else 0.0
    return {
        "last_week": int(last),
        "prior_avg": round(avg, 1),
        "change_pct": round(change * 100, 1),
    }


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
        "cio_campaign_id":   cid,
        "name":              name,
        "status_campaign":   c.get("status"),
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
        return {**base, "health": "sin_datos", "label": "Sin datos suficientes", "issues": [], "trends": {}}

    trends = {
        "delivered":    _week_over_week(series.get("delivered", [])),
        "converted":    _week_over_week(series.get("converted", [])),
        "human_opened": _week_over_week(series.get("human_opened", [])),
    }

    issues: list[str] = []
    health = "ok"

    # Spike
    if c.get("spike_alert"):
        issues.append("SPIKE en entrega — revisar inmediatamente")
        health = _worst(health, "critico")

    # Tasa de entrega baja
    if total_sent > 100 and dr < _DR_WARN_THRESHOLD:
        issues.append(f"Entrega baja: {dr:.0%} (umbral ≥50%)")
        health = _worst(health, "alerta")

    # Conversión baja
    if (c.get("delivered") or 0) > 50 and cr < _CR_WARN_THRESHOLD:
        issues.append(f"Conversión muy baja: {cr:.1%} (umbral ≥1%)")
        health = _worst(health, "alerta")

    # Caída semana a semana en conversiones
    t_conv = trends["converted"]
    if t_conv and t_conv.get("last_week", 0) > 5 and t_conv.get("change_pct", 0) < _WOW_DROP_THRESHOLD * 100:
        issues.append(
            f"Conversiones cayeron {abs(t_conv['change_pct']):.0f}% "
            f"vs promedio prev ({t_conv['prior_avg']:.0f} → {t_conv['last_week']})"
        )
        health = _worst(health, "alerta")

    # Caída en entregas
    t_del = trends["delivered"]
    if t_del and t_del.get("change_pct", 0) < -40 and t_del.get("last_week", 0) > 10:
        issues.append(
            f"Entregas cayeron {abs(t_del['change_pct']):.0f}% "
            f"vs promedio prev ({t_del['prior_avg']:.0f} → {t_del['last_week']})"
        )
        health = _worst(health, "alerta")

    label = "Funcionando bien" if health == "ok" else f"{len(issues)} problema(s) detectado(s)"

    return {**base, "health": health, "label": label, "issues": issues, "trends": trends}


def run_weekly_check() -> dict[str, Any]:
    """
    Corre el check de todas las campañas monitoreadas.
    Lee desde cio_campaigns_cache — no llama a CIO.
    Para datos frescos, sincronizar primero desde /api/strategy/sync.
    """
    campaigns = get_campaigns_cache()
    results   = [_analyze_campaign(c) for c in campaigns]

    criticas = [r for r in results if r["health"] == "critico"]
    alertas  = [r for r in results if r["health"] == "alerta"]
    ok       = [r for r in results if r["health"] == "ok"]
    sin_data = [r for r in results if r["health"] == "sin_datos"]

    overall = "critico" if criticas else ("alerta" if alertas else "ok")

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
        "campaigns":         results,
    }
