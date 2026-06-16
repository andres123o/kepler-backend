"""
Sube master_consolidado_final_v2.csv a Supabase.

Uso (desde kepler-backend/):
  python seed_master_v2.py

Qué hace:
  1. Lee master_consolidado_final_v2.csv de la raíz del proyecto
  2. Renombra TRM → trm (Supabase usa lowercase)
  3. Convierte NaN → None (JSON serializable)
  4. Borra todas las filas existentes en master_consolidado_final
  5. Inserta todo en batches de 100 filas

Ejecutar UNA SOLA VEZ después de correr el ALTER TABLE en Supabase.
"""

import math
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

CSV_PATH   = _HERE / "master_consolidado_final_v2.csv"
TABLE_NAME = "master_consolidado_final"
BATCH_SIZE = 100


def main():
    # ── Validaciones previas ──────────────────────────────────────────────────
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY no están en .env")
        sys.exit(1)

    if not CSV_PATH.is_file():
        print(f"ERROR: CSV no encontrado: {CSV_PATH}")
        sys.exit(1)

    # ── Leer CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    print(f"CSV cargado: {len(df)} filas, {len(df.columns)} columnas.")
    print(f"Columnas: {list(df.columns)}")

    # ── Renombrar TRM → trm (Supabase es lowercase) ───────────────────────────
    if "TRM" in df.columns:
        df = df.rename(columns={"TRM": "trm"})
        print("Renombrado: TRM → trm")

    # ── Convertir a lista de dicts y limpiar tipos ───────────────────────────
    # Supabase tiene estas columnas como integer (las demás son float o text)
    INT_COLS = {"step_09_full_account", "full_users_aprobados",
                "usuarios_primer_cashin", "es_exogeno"}

    raw_rows = df.to_dict(orient="records")
    rows = []
    for r in raw_rows:
        clean: dict = {}
        for k, v in r.items():
            # NaN → None
            if isinstance(v, float) and math.isnan(v):
                clean[k] = None
            # Columnas integer: float 3088.0 → int 3088
            elif k in INT_COLS and isinstance(v, (int, float)):
                clean[k] = int(v)
            else:
                clean[k] = v
        rows.append(clean)
    print(f"Ejemplo fila 0: {rows[0]}")
    print(f"Total filas a insertar: {len(rows)}")

    # ── Conectar a Supabase ───────────────────────────────────────────────────
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"\nConectado a Supabase.")

    # ── Borrar datos existentes ───────────────────────────────────────────────
    print(f"Borrando filas existentes en {TABLE_NAME}...")
    client.table(TABLE_NAME).delete().neq("semana", "___never___").execute()
    print("Tabla limpia.")

    # ── Insertar en batches ───────────────────────────────────────────────────
    n_batches = math.ceil(len(rows) / BATCH_SIZE)
    print(f"\nInsertando {len(rows)} filas en {n_batches} batches de {BATCH_SIZE}...\n")

    inserted = 0
    for i in range(n_batches):
        batch = rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        res = client.table(TABLE_NAME).insert(batch).execute()
        inserted += len(res.data or batch)
        pct = round(inserted / len(rows) * 100)
        print(f"  Batch {i+1}/{n_batches} — {inserted}/{len(rows)} filas ({pct}%)")

    print(f"\n✓ Listo. {inserted} filas insertadas en {TABLE_NAME}.")


if __name__ == "__main__":
    main()
