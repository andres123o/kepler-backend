"""
seed_prompts.py — Herramienta para insertar/actualizar prompts en funnel_prompts.

USO:
    python seed_prompts.py --org <org_slug> --funnel <funnel_slug> --dir <directorio_con_txts>

EJEMPLO (Perú):
    python seed_prompts.py --org trii --funnel activacion_pe --dir prompts_pe/

ESTRUCTURA del directorio de prompts (--dir):
    premium_system.txt
    premium_kb_preamble.txt
    premium_user_template.txt
    premium_perplexity_system.txt
    premium_perplexity_query.txt          ← content del query
    premium_perplexity_api_params.json    ← {"model": "...", ...}

    basic_system.txt
    basic_kb_preamble.txt
    basic_user_template.txt
    basic_perplexity_system.txt
    basic_perplexity_query.txt
    basic_perplexity_api_params.json

Los archivos que no existan en el directorio se omiten sin error.
Los que sí existen hacen upsert (INSERT o UPDATE si ya hay fila).

Prerequisito SQL (correr una sola vez):
    CREATE TABLE IF NOT EXISTS funnel_prompts (
        id          uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
        funnel_id   uuid        NOT NULL REFERENCES funnels(id) ON DELETE CASCADE,
        agent_type  text        NOT NULL,
        prompt_type text        NOT NULL,
        content     text        NOT NULL,
        api_params  jsonb,
        created_at  timestamptz DEFAULT now(),
        updated_at  timestamptz DEFAULT now(),
        UNIQUE(funnel_id, agent_type, prompt_type)
    );

NOTA: Colombia (trii/activacion_co) ya fue seedeado — no necesita correr de nuevo
a menos que quieras actualizar algún prompt.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from supabase import create_client, Client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed funnel_prompts en Supabase")
    parser.add_argument("--org",    required=True, help="org_slug (ej: trii)")
    parser.add_argument("--funnel", required=True, help="funnel_slug (ej: activacion_pe)")
    parser.add_argument("--dir",    required=True, help="Directorio con archivos .txt/.json")
    args = parser.parse_args()

    prompt_dir = Path(args.dir)
    if not prompt_dir.is_dir():
        print(f"ERROR: directorio '{prompt_dir}' no existe.")
        sys.exit(1)

    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("ERROR: SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY requeridos en .env")
        sys.exit(1)

    sb: Client = create_client(url, key)

    # Resolver funnel_id
    org_res = sb.table("organizations").select("id").eq("slug", args.org).execute()
    if not org_res.data:
        print(f"ERROR: organización '{args.org}' no encontrada.")
        sys.exit(1)
    org_id = org_res.data[0]["id"]

    funnel_res = (
        sb.table("funnels")
        .select("id")
        .eq("org_id", org_id)
        .eq("slug", args.funnel)
        .execute()
    )
    if not funnel_res.data:
        print(f"ERROR: funnel '{args.funnel}' no encontrado para org '{args.org}'.")
        sys.exit(1)
    funnel_id = funnel_res.data[0]["id"]
    print(f"funnel_id = {funnel_id}  ({args.org}/{args.funnel})\n")

    # Mapa de archivo → (agent_type, prompt_type)
    FILE_MAP = {
        "premium_system.txt":               ("premium", "system"),
        "premium_kb_preamble.txt":          ("premium", "kb_preamble"),
        "premium_user_template.txt":        ("premium", "user_template"),
        "premium_perplexity_system.txt":    ("premium", "perplexity_system"),
        "premium_perplexity_query.txt":     ("premium", "perplexity_query"),
        "basic_system.txt":                 ("basic",   "system"),
        "basic_kb_preamble.txt":            ("basic",   "kb_preamble"),
        "basic_user_template.txt":          ("basic",   "user_template"),
        "basic_perplexity_system.txt":      ("basic",   "perplexity_system"),
        "basic_perplexity_query.txt":       ("basic",   "perplexity_query"),
    }
    PARAMS_MAP = {
        "premium_perplexity_api_params.json": ("premium", "perplexity_query"),
        "basic_perplexity_api_params.json":   ("basic",   "perplexity_query"),
    }

    rows: dict[tuple, dict] = {}

    # Leer archivos de contenido
    for filename, (agent_type, prompt_type) in FILE_MAP.items():
        path = prompt_dir / filename
        if not path.exists():
            print(f"  omitido (no existe): {filename}")
            continue
        content = path.read_text(encoding="utf-8").strip()
        key_tuple = (agent_type, prompt_type)
        rows[key_tuple] = {
            "funnel_id":   funnel_id,
            "agent_type":  agent_type,
            "prompt_type": prompt_type,
            "content":     content,
            "api_params":  None,
        }

    # Leer api_params
    for filename, (agent_type, prompt_type) in PARAMS_MAP.items():
        path = prompt_dir / filename
        if not path.exists():
            continue
        params = json.loads(path.read_text(encoding="utf-8"))
        key_tuple = (agent_type, prompt_type)
        if key_tuple in rows:
            rows[key_tuple]["api_params"] = params
        else:
            print(f"  advertencia: {filename} existe pero no hay .txt de query asociado — omitido")

    if not rows:
        print("No se encontró ningún archivo de prompt en el directorio.")
        sys.exit(1)

    # Upsert
    print()
    for (agent_type, prompt_type), row in rows.items():
        label = f"{agent_type}/{prompt_type}"
        try:
            sb.table("funnel_prompts") \
              .upsert(row, on_conflict="funnel_id,agent_type,prompt_type") \
              .execute()
            params_note = f" | api_params: sí" if row.get("api_params") else ""
            print(f"OK  {label}  ({len(row['content'])} chars{params_note})")
        except Exception as exc:
            print(f"ERROR  {label}: {exc}")

    print("\nSeed completado.")
    print("Verifica: SELECT agent_type, prompt_type, length(content) FROM funnel_prompts WHERE funnel_id = '" + funnel_id + "';")


if __name__ == "__main__":
    main()
