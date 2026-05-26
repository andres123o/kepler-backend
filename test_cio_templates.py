"""
Output en el mismo orden que Claude.ai MCP:
1. Estructura del journey (metadatos + nodos)
2. Contenido de cada nodo de comunicación
Corre con:  python test_cio_templates.py
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding="utf-8")
import httpx
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

API_KEY  = os.getenv("CUSTOMERIO_APP_API_KEY", "")
HEADERS  = {"Authorization": f"Bearer {API_KEY}"}
BASE     = "https://api.customer.io/v1"
CAMP_ID  = "4596"


def strip_html(html: str) -> str:
    no_style  = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL|re.IGNORECASE)
    no_script = re.sub(r"<script[^>]*>.*?</script>", " ", no_style, flags=re.DOTALL|re.IGNORECASE)
    no_tags   = re.sub(r"<[^>]+>", " ", no_script)
    return " ".join(no_tags.split())


# ══════════════════════════════════════════════════════════════
# PARTE 1 — ESTRUCTURA DEL JOURNEY
# ══════════════════════════════════════════════════════════════
r = httpx.get(f"{BASE}/campaigns/{CAMP_ID}", headers=HEADERS, timeout=30)
camp = r.json().get("campaign", r.json())

print("══════════════════════════════════════════════════════════════")
print("ESTRUCTURA DEL JOURNEY")
print("══════════════════════════════════════════════════════════════")
print(f"Nombre:      {camp.get('name')}")
print(f"Estado:      {camp.get('state')}")
print(f"Tipo:        {camp.get('type')}")
print(f"Trigger:     {camp.get('event_name') or camp.get('event_type')}")
print(f"Conversión:  {camp.get('conversion_event_name', '(no disponible con esta key)')}")

# Acciones desde el endpoint de campaign (solo id + type)
actions_simple = camp.get("actions", [])
print(f"\nNodos totales: {len(actions_simple)}")
print(f"⚠ Edges/delays/condiciones: NO disponibles con esta API key")
print(f"  (requiere GET /v1/environments/112828/campaigns/{CAMP_ID})\n")

print("Nodos detectados (sin detalle de delays):")
for a in actions_simple:
    print(f"  id={a['id']}  tipo={a['type']}")

# ══════════════════════════════════════════════════════════════
# PARTE 2 — CONTENIDO DE CADA NODO DE COMUNICACIÓN
# ══════════════════════════════════════════════════════════════
r2 = httpx.get(f"{BASE}/campaigns/{CAMP_ID}/actions", headers=HEADERS, timeout=30)
all_actions = r2.json().get("actions", [])
msg_nodes   = [a for a in all_actions if a.get("type") in ("push", "email")]

print(f"\n══════════════════════════════════════════════════════════════")
print(f"CONTENIDO DE NODOS DE COMUNICACIÓN ({len(msg_nodes)} nodos)")
print(f"══════════════════════════════════════════════════════════════")

for a in msg_nodes:
    t       = a.get("type", "?").upper()
    name    = a.get("name", "sin nombre")
    subj    = (a.get("subject") or "").strip()
    raw_b   = (a.get("body") or "").strip()
    dedup   = a.get("deduplicate_id", "")
    tmpl_id = dedup.split(":")[0] if dedup else "?"

    body = strip_html(raw_b) if t == "EMAIL" else raw_b

    print(f"\n{'─'*60}")
    print(f"[{t}] '{name}'  (action_id={a.get('id')}  template_id={tmpl_id})")
    print(f"  subject:   {subj if subj else '(vacío)'}")
    print(f"  preheader: ⚠ NO DISPONIBLE (requiere GET /v1/environments/112828/templates/{tmpl_id})")
    if body:
        print(f"  body:\n    {body[:600]}")
    else:
        print(f"  body:      (vacío)")

print(f"\n{'═'*60}")
print("RESUMEN DE BRECHAS vs Claude.ai MCP:")
print(f"  ✅ Subject:          disponible")
print(f"  ✅ Body push:        disponible")
print(f"  ✅ Body email texto: disponible (extraído de HTML)")
print(f"  ❌ Preheader:        requiere /environments/.../templates/{{id}}")
print(f"  ❌ Edges/delays:     requiere /environments/.../campaigns/{{id}}")
print(f"  ❌ Conditions:       requiere /environments/.../campaigns/{{id}}")
print(f"\n  → Solución: API key con acceso a endpoints /environments/")
print(f"    Pedir a Juanita key del workspace con permisos completos.")
