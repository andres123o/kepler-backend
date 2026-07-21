"""
Script de prueba reutilizable — journey COMPLETO real (no recortado), para cualquier
funnel/país (activacion_co, activacion_pe, activacion_cl).

Uso:
    python test_chained_pipeline_full.py <funnel_slug> <campaign_id> [out_filename]

Ejemplos:
    python test_chained_pipeline_full.py activacion_co 4596
    python test_chained_pipeline_full.py activacion_pe 4681

Usa build_journey() real vía la fly API de CIO — solo lectura (GET). No llama a ninguna
función de escritura, no toca funnels.config (pipeline sigue en 'single_call').
"""
import io
import json
import sys

from app.services.supabase_client import FunnelClient
from app.services.strategy_agent import (
    _format_shap_analysis,
    _format_knowledge_base,
    _format_cifras_block,
    _format_journey_for_enrichment,
    _run_premium_chained,
)
from app.services.customerio_fly_client import build_journey
from app.services.anthropic_client import extract_market_cifras
from app.services.perplexity_client import fetch_market_research, format_research_block

funnel_slug = sys.argv[1] if len(sys.argv) > 1 else "activacion_co"
campaign_id = sys.argv[2] if len(sys.argv) > 2 else "4596"
out_name    = sys.argv[3] if len(sys.argv) > 3 else f"chained_test_full_result_{funnel_slug}.json"

fc = FunnelClient("trii", funnel_slug)

prediction = fc.get_latest_prediction()
semana_label = prediction.get("semana_label") or prediction.get("semana_datos", "")
internal_vars = frozenset(fc.get_funnel_config().get("ml_internal_features") or [])
shap_text = _format_shap_analysis(prediction, internal_vars)
print(f"[1/5] SHAP OK — funnel={funnel_slug} semana={semana_label}")

kb_entries = fc.get_knowledge_base()
kb_text = _format_knowledge_base(kb_entries)
print(f"[2/5] KB OK — {len(kb_entries)} entradas")

research = fetch_market_research(semana_label, fc=fc)
research_text = format_research_block(research)
cifras = extract_market_cifras(research.get("raw_text") or "")
research_text = research_text + "\n\n" + _format_cifras_block(cifras)
print(f"[3/5] Research OK — {len(cifras)} cifras verificadas")

journey = build_journey(campaign_id, fc)
msg_nodes = [n for n in journey["nodes"] if n.get("type") in ("email_action", "push_action")]
journey_text = _format_journey_for_enrichment(journey)
print(f"[4/5] Journey REAL completo (campaign_id={campaign_id}) — {len(journey['nodes'])} nodos totales | {len(msg_nodes)} mensajes")

print("[5/5] Corriendo pipeline chained_v2 (journey completo, 4 llamadas Claude)...")
strategy = _run_premium_chained(
    fc=fc,
    shap_text=shap_text,
    research_text=research_text,
    kb_text=kb_text,
    kb_entries=kb_entries,
    journey_text=journey_text,
    semana_label=semana_label,
)

out_path = r"C:\Users\DELL\AppData\Local\Temp\claude\C--Users-DELL-Documents-Mi-primera-SaaS\8ef6fb94-f602-421c-8e3f-0ce15cd7ec3d\scratchpad\\" + out_name
with io.open(out_path, "w", encoding="utf-8") as f:
    json.dump(strategy, f, ensure_ascii=False, indent=2)
print(f"\nOK — resultado completo guardado en: {out_path}")
