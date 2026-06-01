"""
Inspección cruda del nodo #33186 — Fotos KYC | Colombia (email)
Imprime el JSON crudo de cada endpoint sin procesar nada.
"""
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

ACTION_ID    = 36711
CAMPAIGN_ID  = 4596
ENV_ID       = os.getenv("CIO_ENVIRONMENT_ID", "112828")
CIO_APP_BASE = "https://api.customer.io/v1"
CIO_FLY_BASE = "https://us.fly.customer.io"
WRITE_KEY    = os.getenv("CUSTOMERIO_APP_API_KEY", "")
SA_LIVE      = os.getenv("CIO_SA_LIVE_READONLY_KEY", "")


def get_jwt() -> str:
    resp = httpx.post(
        f"{CIO_FLY_BASE}/v1/service_accounts/oauth/token",
        data={"grant_type": "client_credentials", "client_secret": SA_LIVE},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def main() -> None:
    # ── 1. App API — action crudo ──────────────────────────────────────────
    url = f"{CIO_APP_BASE}/campaigns/{CAMPAIGN_ID}/actions/{ACTION_ID}"
    print(f"\n{'='*60}")
    print(f"GET {url}")
    print(f"{'='*60}")
    resp = httpx.get(url, headers={"Authorization": f"Bearer {WRITE_KEY}"}, timeout=30)
    print(f"status: {resp.status_code}")
    print(resp.text)

    action    = resp.json().get("action", resp.json())
    dedup     = action.get("deduplicate_id", "")
    tmpl_id   = int(dedup.split(":")[0]) if ":" in dedup else None
    print(f"\n→ template_id extraído de deduplicate_id: {tmpl_id}")

    if not tmpl_id:
        print("No se pudo extraer template_id. Fin.")
        sys.exit(1)

    # ── 2. fly API — template crudo ────────────────────────────────────────
    jwt = get_jwt()
    url = f"{CIO_FLY_BASE}/v1/environments/{ENV_ID}/templates/{tmpl_id}"
    print(f"\n{'='*60}")
    print(f"GET {url}")
    print(f"{'='*60}")
    resp = httpx.get(url, headers={"Authorization": f"Bearer {jwt}"}, timeout=30)
    print(f"status: {resp.status_code}")
    print(resp.text)


if __name__ == "__main__":
    main()
