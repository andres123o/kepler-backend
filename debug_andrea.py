import json, os
from dotenv import load_dotenv
load_dotenv(".env")
from supabase import create_client

sb = create_client(os.getenv("NEXT_PUBLIC_SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))

# Todo lo que tiene Andrea en el log
print("=== node_update_log para Andrea ===")
rows = sb.table("node_update_log").select("*").ilike("user_name", "andrea").order("updated_at").execute().data
for r in rows:
    print(r)

# Nodos totales de su campaña en la estrategia activa
print("\n=== Nodos de 'Fotos KYC' en la estrategia activa ===")
strat = sb.table("strategy_results").select("full_result, semana_label").order("created_at", desc=True).limit(1).execute().data[0]
full = strat["full_result"] if isinstance(strat["full_result"], dict) else json.loads(strat["full_result"])
for i, a in enumerate(full.get("acciones", [])):
    name = a.get("campana_existente_nombre") or a.get("campaña_existente_nombre") or a.get("step_name") or "?"
    if "fotos" in name.lower() or "kyc" in name.lower():
        nodos = a.get("nodos_completos") or []
        print(f"  [{i}] {name}")
        for n in nodos:
            print(f"    ord={n['orden']} tipo={n['tipo']} mod={n.get('modificado')} id={n.get('id_nodo_cio')} nombre=\"{n.get('nombre')}\"")
        if not nodos:
            print("    (sin nodos_completos)")
        break
else:
    print("  No encontrada en la estrategia activa")
