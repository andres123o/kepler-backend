"""
Valida los conectores auto-fetch de Perú contra el master consolidado.

Compara el valor que calcula cada conector hoy contra el valor histórico
guardado en el Excel. Si el conector es correcto, la correlación debe ser
alta y el MAE bajo.

NOTA: Google Trends se omite — los índices históricos son relativos al
      rango de la consulta y no son reproducibles retroactivamente.

Corre desde kepler-backend/:
    python scripts-get-data/validate_pe_connectors.py

Opciones:
    --rows N    Número de semanas a validar para conectores yfinance (default 20)
    --all       Valida TODAS las semanas vía yfinance (lento, ~5 min)
    --calendar  Solo valida conectores de calendario (instantáneo, sin API)
"""

import argparse
import math
import sys
import time
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.market_data_fetcher import (
    CONNECTOR_REGISTRY,
    _parse_monday,
)

EXCEL_PATH = ROOT / "primer_master_peru" / "master_consolidado_peru_full_v2.xlsx"

# ─── Definición de qué validar ────────────────────────────────────────────────

YFINANCE_CHECKS = [
    {
        "field":     "pen_usd_var_semanal",
        "connector": "yfinance_weekly_pct",
        "params":    {"ticker": "PEN=X"},
    },
    {
        "field":     "bvl_var_semanal",
        "connector": "yfinance_weekly_pct",
        "params":    {"ticker": "EPU"},
    },
    {
        "field":     "cobre_var_semanal",
        "connector": "yfinance_weekly_pct",
        "params":    {"ticker": "HG=F"},
    },
    {
        "field":     "sp500_var_semanal",
        "connector": "yfinance_weekly_pct",
        "params":    {"ticker": "^GSPC"},
    },
    {
        "field":     "pen_usd_volatilidad_4w",
        "connector": "yfinance_rolling_std",
        "params":    {"ticker": "PEN=X", "weeks": 4},
    },
]

CALENDAR_CHECKS = [
    {
        "field":     "is_ventana_quincena",
        "connector": "quincena_binary",
        "params":    {"quincena_days": [1, 15, 16, 28, 29, 30, 31]},
    },
    {
        "field":     "is_ventana_cts",
        "connector": "calendar_window_binary",
        "params":    {"windows": [
            {"month": 5,  "day_start": 1, "day_end": 15},
            {"month": 11, "day_start": 1, "day_end": 15},
        ]},
    },
    {
        "field":     "is_ventana_gratificacion",
        "connector": "calendar_window_binary",
        "params":    {"windows": [
            {"month": 7,  "day_start": 1, "day_end": 20},
            {"month": 12, "day_start": 1, "day_end": 20},
        ]},
    },
    {
        "field":     "dias_habiles_semana",
        "connector": "working_days_count",
        "params":    {"country": "PE"},
    },
]


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _mae(pairs: list) -> float:
    errs = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    return round(sum(errs) / len(errs), 4) if errs else float("nan")


def _corr(pairs: list) -> float:
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(valid) < 3:
        return float("nan")
    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    n  = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
    return round(num / denom, 4) if denom else float("nan")


def _accuracy(pairs: list) -> float:
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not valid:
        return float("nan")
    hits = sum(1 for a, b in valid if round(a) == round(b))
    return round(hits / len(valid) * 100, 1)


def _safe_master_float(val) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_check(check: dict, df: pd.DataFrame, show_rows: int = 5) -> dict:
    """
    Corre un conector contra todas las filas del df y retorna métricas.
    """
    field     = check["field"]
    connector = check["connector"]
    params    = check["params"]
    runner    = CONNECTOR_REGISTRY[connector]

    pairs      = []
    rows_shown = 0
    examples   = []

    for _, row in df.iterrows():
        semana_str = row["semana"]
        monday     = _parse_monday(semana_str)
        master_val = _safe_master_float(row.get(field))

        try:
            computed = runner(monday, params, {})
        except Exception as exc:
            computed = None
            print(f"  [ERROR] {semana_str}: {exc}")

        pairs.append((computed, master_val))

        if rows_shown < show_rows and master_val is not None and computed is not None:
            diff = computed - master_val
            examples.append({
                "semana": semana_str, "master": master_val,
                "conector": computed, "diff": round(diff, 4),
            })
            rows_shown += 1

    return {"field": field, "connector": connector, "pairs": pairs, "examples": examples}


def print_results(result: dict, is_binary: bool = False) -> None:
    field     = result["field"]
    connector = result["connector"]
    pairs     = result["pairs"]
    examples  = result["examples"]
    n_valid   = sum(1 for a, b in pairs if a is not None and b is not None)

    print(f"\n{'-'*64}")
    print(f"  {field}  <-  [{connector}]")
    print(f"  Filas válidas: {n_valid} / {len(pairs)}")

    if is_binary:
        acc = _accuracy(pairs)
        print(f"  Accuracy: {acc}%")
    else:
        mae_val  = _mae(pairs)
        corr_val = _corr(pairs)
        print(f"  MAE         : {mae_val}")
        print(f"  Correlacion : {corr_val}")

    print(f"\n  {'semana':<13} {'master':>10} {'conector':>10} {'diff':>10}")
    for e in examples:
        flag = " OK" if abs(e["diff"]) < 0.2 else " ~" if abs(e["diff"]) < 1.0 else " X"
        print(f"  {e['semana']:<13} {e['master']:>10.4f} {e['conector']:>10.4f} {e['diff']:>10.4f}{flag}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",     type=int, default=20, help="Semanas a validar via yfinance (default 20)")
    parser.add_argument("--all",      action="store_true",  help="Valida todas las semanas via yfinance")
    parser.add_argument("--calendar", action="store_true",  help="Solo valida conectores de calendario")
    args = parser.parse_args()

    if not EXCEL_PATH.exists():
        print(f"ERROR: {EXCEL_PATH} no encontrado")
        sys.exit(1)

    df_raw = pd.read_excel(EXCEL_PATH)
    df_raw["_fecha"] = pd.to_datetime(df_raw["semana"], format="%d/%m/%Y", dayfirst=True, errors="coerce")
    df_raw = df_raw.sort_values("_fecha").reset_index(drop=True)

    print(f"Master Peru: {len(df_raw)} filas  |  {df_raw['semana'].iloc[0]} -> {df_raw['semana'].iloc[-1]}")

    # --- Conectores de calendario (sin API, instantaneo) ---
    print("\n" + "="*64)
    print("  CONECTORES DE CALENDARIO (sin API)")
    print("="*64)

    df_cal = df_raw.copy()
    for check in CALENDAR_CHECKS:
        is_binary = check["field"] != "dias_habiles_semana"
        result = run_check(check, df_cal, show_rows=6)
        print_results(result, is_binary=is_binary)

    if args.calendar:
        print("\n[--calendar] Listo. Omitiendo yfinance.")
        return

    # --- Conectores yfinance (con API) ---
    n_rows = len(df_raw) if args.all else args.rows
    df_fin = df_raw.tail(n_rows).copy()

    print(f"\n{'='*64}")
    print(f"  CONECTORES YFINANCE ({n_rows} semanas mas recientes)")
    print("="*64)
    print("  [yfinance puede tardar 30-60s por conector]\n")

    for check in YFINANCE_CHECKS:
        t0     = time.time()
        result = run_check(check, df_fin, show_rows=5)
        elapsed = time.time() - t0
        print_results(result, is_binary=False)
        print(f"  [{elapsed:.1f}s]")

    print(f"\n{'='*64}")
    print("  Validacion completa.")
    print("  OK diff < 0.20  |  ~ diff < 1.00  |  X diff >= 1.00")
    print("  Correlacion > 0.95 = excelente  |  > 0.85 = aceptable  |  < 0.85 = revisar")


if __name__ == "__main__":
    main()
