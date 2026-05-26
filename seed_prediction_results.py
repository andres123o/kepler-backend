"""
Seed histórico de predicciones en Supabase.

Parsea los nombres de archivo para obtener el rango real de la semana:
  proyeccion-6-12-202603.md   → 6 al 12 de marzo 2026
  proyeccion-27-03-202603.md  → 27 de marzo al 3 de abril 2026
  proyeccion-04-10-202604.md  → 4 al 10 de abril 2026

Uso:
    python seed_prediction_results.py
"""

import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("ERROR: SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY no están en .env")

client = create_client(SUPABASE_URL, SUPABASE_KEY)
LOGS_DIR = Path(__file__).parent / "logs-historico-predicciones"

MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}


def parse_filename(name: str) -> tuple[date, date, str]:
    """
    Extrae inicio, fin y etiqueta legible del nombre del archivo.

    Formato: proyeccion-DD1-DD2-YYYYMM.md
      DD1   = día inicio
      DD2   = día fin
      YYYY  = año
      MM    = mes del día inicio

    Si DD2 < DD1 el fin cae en el mes siguiente (ej. 27-03 → 27 mar / 3 abr).
    """
    m = re.match(r"proyeccion-(\d{1,2})-(\d{1,2})-(\d{4})(\d{2})\.md", name)
    if not m:
        raise ValueError(f"Nombre de archivo no reconocido: {name}")

    d1, d2 = int(m.group(1)), int(m.group(2))
    year, month = int(m.group(3)), int(m.group(4))

    inicio = date(year, month, d1)

    if d2 < d1:                      # cruza al mes siguiente
        mes_fin = month + 1 if month < 12 else 1
        anio_fin = year if month < 12 else year + 1
        fin = date(anio_fin, mes_fin, d2)
    else:
        fin = date(year, month, d2)

    if inicio.month == fin.month:
        label = f"{d1} al {d2} de {MESES_ES[inicio.month]} {year}"
    else:
        label = (
            f"{d1} de {MESES_ES[inicio.month]}"
            f" al {d2} de {MESES_ES[fin.month]} {year}"
        )

    return inicio, fin, label


def extract_json(filepath: Path) -> dict | None:
    """Extrae el bloque JSON del final del log .md."""
    text = filepath.read_text(encoding="utf-8", errors="ignore").strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError as e:
        print(f"  ERROR JSON en {filepath.name}: {e}")
        return None


def main() -> None:
    print(f"Directorio: {LOGS_DIR}")
    print(f"Supabase:   {SUPABASE_URL[:40]}...\n")

    files = sorted(LOGS_DIR.glob("proyeccion-*.md"))
    inserted = skipped = 0

    for filepath in files:
        # ── Parsear nombre ──────────────────────────────────────────────
        try:
            inicio, fin, label = parse_filename(filepath.name)
        except ValueError as e:
            print(f"  SKIP  {filepath.name} — {e}")
            skipped += 1
            continue

        # ── Extraer JSON ────────────────────────────────────────────────
        result = extract_json(filepath)
        if result is None:
            print(f"  SKIP  {filepath.name} — sin JSON válido (archivo vacío)")
            skipped += 1
            continue

        # ── Corregir semana_datos con la fecha real del filename ────────
        semana_datos = inicio.isoformat()          # "2026-03-06"
        result["semana_datos"] = semana_datos      # sobreescribe el valor del log
        result["semana_label"] = label             # añade etiqueta

        # created_at = domingo al inicio de la semana (para orden correcto)
        created_at = f"{inicio.isoformat()}T08:00:00+00:00"

        row = {
            "semana_datos":   semana_datos,
            "semana_label":   label,
            "prediccion":     result["prediccion_siguiente_semana"],
            "baseline_12w":   result["baseline_12w"],
            "brecha":         result["brecha_vs_baseline"],
            "mae_modelo":     result.get("mae_modelo"),
            "modelo_version": result.get("modelo_version"),
            "full_result":    result,
            "created_at":     created_at,
        }

        res = client.table("prediction_results").insert(row).execute()
        if res.data:
            print(
                f"  OK    {filepath.name}"
                f"  →  {label}"
                f"  |  predicción={row['prediccion']:.0f}"
                f"  |  {row['modelo_version']}"
            )
            inserted += 1
        else:
            print(f"  ERROR {filepath.name} — Supabase no devolvió datos")

    print(f"\nListo.  Insertadas: {inserted}  |  Saltadas: {skipped}")


if __name__ == "__main__":
    main()
