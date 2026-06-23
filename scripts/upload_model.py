"""
Sube los archivos del modelo entrenado a Supabase Storage.

Uso:
    python scripts/upload_model.py --org trii --funnel activacion_pe --version 2 \
        --dir primer_master_peru/models_peru/v2

El bucket de destino es siempre "models".
Path dentro del bucket: {org}/{funnel}/v{version}/model.json (y meta + summary).

Después de correr este script, actualiza el config del funnel en Supabase:
    UPDATE funnels
    SET config = jsonb_set(
      jsonb_set(config, '{ml,model_storage}', '"models"'),
      '{ml,model_version}', '2'
    )
    WHERE slug = 'activacion_pe' AND org_id = (SELECT id FROM organizations WHERE slug = 'trii');
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.services.supabase_client import _get_client

BUCKET = "models"
FILES  = ("model.json", "model_meta.json", "training_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org",     required=True, help="org slug (ej. trii)")
    parser.add_argument("--funnel",  required=True, help="funnel slug (ej. activacion_pe)")
    parser.add_argument("--version", required=True, type=int, help="número de versión (ej. 2)")
    parser.add_argument("--dir",     required=True, type=Path, help="carpeta local con model.json etc.")
    args = parser.parse_args()

    model_dir = ROOT / args.dir if not args.dir.is_absolute() else args.dir
    if not model_dir.exists():
        print(f"ERROR: directorio no encontrado: {model_dir}")
        sys.exit(1)

    client = _get_client()
    prefix = f"{args.org}/{args.funnel}/v{args.version}"

    print(f"Subiendo a bucket '{BUCKET}' → {prefix}/")
    for fname in FILES:
        fpath = model_dir / fname
        if not fpath.exists():
            print(f"  SKIP  {fname} (no existe localmente)")
            continue

        path_in_bucket = f"{prefix}/{fname}"
        # Borrar si ya existe (upsert manual)
        try:
            client.storage.from_(BUCKET).remove([path_in_bucket])
        except Exception:
            pass

        client.storage.from_(BUCKET).upload(path_in_bucket, fpath.read_bytes())
        size_kb = fpath.stat().st_size / 1024
        print(f"  OK    {fname}  ({size_kb:.1f} KB)")

    print(f"\nListo. Ahora corre en Supabase:")
    print(f"""
UPDATE funnels
SET config = jsonb_set(
  jsonb_set(config, '{{ml,model_storage}}', '"models"'),
  '{{ml,model_version}}', '{args.version}'
)
WHERE slug = '{args.funnel}'
  AND org_id = (SELECT id FROM organizations WHERE slug = '{args.org}');
""")


if __name__ == "__main__":
    main()
