"""
Script de prueba desechable — NO forma parte del pipeline productivo.

Corre el flujo encadenado (chained_v2) para la campaña Primer Depósito Colombia,
usando datos REALES (SHAP, research de mercado, KB) pero un journey RECORTADO a
3 nodos reales (2 push + 1 email, con sus IDs/nombres/delays/copy reales tomados
de la campaña 4596 en CIO) para revisar la calidad del razonamiento sin correr
la campaña completa en el primer intento.

No llama a NINGUNA función de escritura de CIO — solo build de texto + llamadas
a Claude. No toca funnels.config (pipeline sigue en 'single_call' en producción).
"""
import io
import json

from app.services.supabase_client import FunnelClient
from app.services.strategy_agent import (
    _format_shap_analysis,
    _format_knowledge_base,
    _format_cifras_block,
    _format_journey_for_enrichment,
    _run_premium_chained,
)
from app.services.anthropic_client import extract_market_cifras
from app.services.perplexity_client import fetch_market_research, format_research_block

fc = FunnelClient("trii", "activacion_co")

# ── 1. SHAP real ───────────────────────────────────────────────────────────
prediction = fc.get_latest_prediction()
if not prediction:
    raise SystemExit("No hay predicción guardada — corre /api/ml/predict primero.")
semana_label = prediction.get("semana_label") or prediction.get("semana_datos", "")
internal_vars = frozenset(fc.get_funnel_config().get("ml_internal_features") or [])
shap_text = _format_shap_analysis(prediction, internal_vars)
print(f"[1/5] SHAP OK — semana={semana_label}")

# ── 2. KB real ──────────────────────────────────────────────────────────────
kb_entries = fc.get_knowledge_base()
kb_text = _format_knowledge_base(kb_entries)
print(f"[2/5] KB OK — {len(kb_entries)} entradas")

# ── 3. Research real (Perplexity) ────────────────────────────────────────────
research = fetch_market_research(semana_label, fc=fc)
research_text = format_research_block(research)
cifras = extract_market_cifras(research.get("raw_text") or "")
research_text = research_text + "\n\n" + _format_cifras_block(cifras)
print(f"[3/5] Research OK — {len(cifras)} cifras verificadas")

# ── 4. Journey RECORTADO — 2 push + 1 email reales de la campaña 4596 ────────
# IDs, nombres, delays y copy tomados directo de CIO (GET, sin escritura).
push1_subject = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}\U0001F4B0 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, el peso sube: tu CDT en trii no espera{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}\U0001F4C8 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, peso fuerte y bolsa activa: es tu momento{% else %}\U0001F4B0 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, peso en máximos: tu cuenta trii está lista{% endif %}"
push1_body = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}El dólar cayó a mínimos desde 2019. Quedarse en pesos con tasa fija es la jugada. Tu CDT en trii hasta 12.80% EA desde $200.000. Abre hoy.{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}Peso fortalecido y COLCAP al alza. El mercado colombiano favorece estar adentro. Empieza en trii desde $55.000, tú eliges el nivel de riesgo.{% else %}Dólar en mínimos históricos. CDT hasta 12.80% EA o fondos desde $55.000. Tu dinero en pesos puede rendir más. Tú decides desde $1.000 en trii.{% endif %}"

push2_subject = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}\U0001F4B0 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, pesos rinden más que dólares ahora{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}\U0001F4C8 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, Colombia en ciclo positivo: ¿ya entraste?{% else %}\U0001F4B0 {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, tu cuenta trii lista: pesos al máximo{% endif %}"
push2_body = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}Dólar en mínimos y BanRep al 12%. CDT en trii hasta 12.80% EA con tasa fija. Abre desde $200.000.{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}Peso fortalecido, COLCAP con alzas acumuladas y BanRep al 12%. Dos caminos en trii desde $55.000. Tú eliges el riesgo.{% else %}Dólar en mínimos y BanRep al 12%. CDT hasta 12.80% EA o fondos desde $55.000. Todo desde tu celular en minutos.{% endif %}"

email1_subject = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}El peso sube: tu CDT en trii captura el momento{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}Peso fuerte y bolsa activa: el momento es hoy{% else %}Pesos que rinden: dos caminos claros en trii{% endif %}"
email1_preheader = "{% if customer.Perfil_de_riesgo == '1. Conservador' %}Dólar en mínimos desde 2019. El carry en pesos nunca fue tan claro.{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}Peso fortalecido y COLCAP al alza. Tu cuenta en trii está lista para este ciclo.{% else %}BanRep al 12% y dólar en mínimos. Dos caminos desde trii, desde $1.000.{% endif %}"
email1_body = "<p>{% if customer.Perfil_de_riesgo == '1. Conservador' %}Hola {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, el dólar llegó a mínimos no vistos desde 2019. La TRM bajó de 3.350 a 3.248 COP. Con el BanRep al 12%, quedarse en pesos con tasa fija es la jugada. Tu CDT en trii hasta 12.80% EA desde $200.000. Abre tu CDT en trii hoy.{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}Hola {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, el dólar cayó a mínimos desde 2019, el BanRep sostiene el 12% y el COLCAP viene al alza. Tu cuenta ya está aprobada, el mercado colombiano favorece estar adentro. Empieza a invertir en trii.{% else %}Hola {% if customer.first_name %}{{ customer.first_name }}{% else %}triier{% endif %}, dólar en mínimos y BanRep al 12%. CDT hasta 12.80% EA desde $200.000 o fondos desde $55.000. Activa tu primera inversión en trii.{% endif %}</p>"

trimmed_journey = {
    "meta": {"name": "Primer depósito | Colombia (TEST recortado)", "id": "4596", "trigger": "BeFullUserCreated", "goal": "BeCashIn", "state": "running"},
    "nodes": [
        {"type": "delay_seconds_action", "delay": 10800},
        {"type": "push_action", "id": "36707", "name": "Push 1 Primer depósito kepler", "subject": push1_subject, "body": push1_body},
        {"type": "delay_seconds_action", "delay": 28800},
        {"type": "push_action", "id": "38017", "name": "Push 2 con kepler", "subject": push2_subject, "body": push2_body},
        {"type": "delay_seconds_action", "delay": 86400},
        {"type": "email_action", "id": "36711", "name": "Email Primer recordatorio kepler", "subject": email1_subject, "preheader_text": email1_preheader, "body": email1_body},
    ],
}
journey_text = _format_journey_for_enrichment(trimmed_journey)
print("[4/5] Journey recortado construido — 2 push + 1 email reales")

# ── 5. Correr el pipeline encadenado (sin tocar funnels.config) ─────────────
print("[5/5] Corriendo pipeline chained_v2 (3 llamadas Claude)...")
strategy = _run_premium_chained(
    fc=fc,
    shap_text=shap_text,
    research_text=research_text,
    kb_text=kb_text,
    kb_entries=kb_entries,
    journey_text=journey_text,
    semana_label=semana_label,
)

out_path = r"C:\Users\DELL\AppData\Local\Temp\claude\C--Users-DELL-Documents-Mi-primera-SaaS\8ef6fb94-f602-421c-8e3f-0ce15cd7ec3d\scratchpad\chained_test_result.json"
with io.open(out_path, "w", encoding="utf-8") as f:
    json.dump(strategy, f, ensure_ascii=False, indent=2)
print(f"\nOK — resultado completo guardado en: {out_path}")
