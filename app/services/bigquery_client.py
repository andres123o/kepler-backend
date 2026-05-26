"""
Cliente BigQuery de solo lectura para Kepler.

Autenticación (una de las dos):
  BQ_CREDENTIALS_PATH     → ruta a un JSON de service account
  BQ_SERVICE_ACCOUNT_JSON → contenido JSON del service account (útil en Vercel)

El proyecto BigQuery es siempre trii-bi.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.bigquery")

BQ_PROJECT = "trii-bi"
_SCOPES = ["https://www.googleapis.com/auth/bigquery.readonly"]


def is_configured() -> bool:
    return bool(
        os.getenv("BQ_CREDENTIALS_PATH", "").strip()
        or os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
    )


def _get_client():
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ImportError:
        raise RuntimeError(
            "google-cloud-bigquery no instalado. Corre: pip install google-cloud-bigquery"
        )

    json_content = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
    creds_path   = os.getenv("BQ_CREDENTIALS_PATH", "").strip()

    if json_content:
        info  = json.loads(json_content)
        creds = service_account.Credentials.from_service_account_info(info, scopes=_SCOPES)
        return bigquery.Client(project=BQ_PROJECT, credentials=creds)

    if creds_path:
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=_SCOPES)
        return bigquery.Client(project=BQ_PROJECT, credentials=creds)

    raise RuntimeError(
        "BigQuery no configurado. Agrega BQ_CREDENTIALS_PATH o BQ_SERVICE_ACCOUNT_JSON en .env"
    )


def run_query(sql: str, params: list | None = None) -> list[dict[str, Any]]:
    """Ejecuta SQL en BigQuery (solo lectura) y retorna filas como lista de dicts."""
    from google.cloud import bigquery

    client     = _get_client()
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    job        = client.query(sql, job_config=job_config)
    rows       = job.result()
    return [dict(row) for row in rows]
