"""
seed_org_secrets.py — Migra las credenciales CIO de .env → tabla org_secrets en Supabase.

INSTRUCCIONES:
  1. Asegúrate de que el .env tiene las vars CIO antes de correr este script.
  2. Activa el venv: .venv\\Scripts\\activate
  3. Corre: python seed_org_secrets.py
  4. Verifica que imprime "OK" para las 3 keys.
  5. RECIÉN ENTONCES puedes borrar las vars CIO del .env y de Railway.

El script usa UPSERT (INSERT ... ON CONFLICT DO UPDATE) — es seguro correrlo varias veces.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Cargar .env local
load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY no encontradas en .env")
    sys.exit(1)

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Lee las keys del .env ─────────────────────────────────────────────────────
sa_live      = os.getenv("CIO_SA_LIVE_READONLY_KEY", "").strip()
app_api_key  = os.getenv("CIO_APP_API_KEY", "").strip() or os.getenv("CUSTOMERIO_APP_API_KEY", "").strip()
env_id       = os.getenv("CIO_ENVIRONMENT_ID", "").strip()

missing = []
if not sa_live:     missing.append("CIO_SA_LIVE_READONLY_KEY")
if not app_api_key: missing.append("CIO_APP_API_KEY (o CUSTOMERIO_APP_API_KEY)")
if not env_id:      missing.append("CIO_ENVIRONMENT_ID")

if missing:
    print(f"ERROR: faltan estas vars en .env: {', '.join(missing)}")
    sys.exit(1)

if sa_live == app_api_key:
    print("ERROR: CIO_SA_LIVE_READONLY_KEY y CIO_APP_API_KEY no pueden ser iguales.")
    sys.exit(1)

# ── Organización objetivo ─────────────────────────────────────────────────────
ORG_SLUG = os.getenv("KEPLER_DEFAULT_ORG", "trii")

secrets = [
    ("CIO_SA_LIVE_KEY",    sa_live),
    ("CIO_APP_API_KEY",    app_api_key),
    ("CIO_ENVIRONMENT_ID", env_id),
]

print(f"\nMigrando credenciales CIO para org '{ORG_SLUG}'...\n")

# Forzar reload del schema cache de PostgREST antes de insertar
# (necesario cuando la tabla fue creada en la misma sesión / cache no refrescado)
try:
    sb.rpc("pg_notify", {"channel": "pgrst", "payload": "reload schema"}).execute()
    import time; time.sleep(1)
    print("  Schema cache recargado.")
except Exception:
    pass  # pg_notify puede fallar si no hay permisos; el insert lo reintentará de todas formas

for key_name, key_value in secrets:
    try:
        sb.table("org_secrets").upsert(
            {"org_slug": ORG_SLUG, "key_name": key_name, "key_value": key_value},
            on_conflict="org_slug,key_name",
        ).execute()
        masked = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
        print(f"  OK  {key_name} = {masked}")
    except Exception as exc:
        print(f"  ERROR  {key_name}: {exc}")
        sys.exit(1)

print("\nMigración completa. Ya puedes borrar estas vars del .env y de Railway:")
print("  - CIO_SA_LIVE_READONLY_KEY")
print("  - CIO_APP_API_KEY  (y CUSTOMERIO_APP_API_KEY si existe)")
print("  - CIO_ENVIRONMENT_ID")
