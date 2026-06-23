"""
Bloque B — Tasa de referencia BCRP (mensual → semanal).

Fuente   : Notas Informativas del Directorio del BCRP (decisiones oficiales)
           https://www.bcrp.gob.pe/politica-monetaria/tasa-de-interes-de-referencia.html
Input    : primer_master_peru/semanas_peru.csv
Output   : primer_master_peru/bloque_b_tasa_bcrp.csv

Logica   : La tasa cambia en la fecha de la reunion del Directorio. Todas las semanas
           del mismo mes-año reciben el valor vigente al final de ese mes.
           Si la reunion ocurre a mitad de mes, la tasa nueva aplica a partir de esa semana.

IMPORTANTE: Verificar y actualizar los valores en TASA_BCRP contra el historico oficial
            en https://www.bcrp.gob.pe antes de entrenar el modelo.
            Para nuevas decisiones: agregar (year, month) -> tasa al diccionario.
"""

from datetime import date, timedelta
import pandas as pd

# ─── Tasa de referencia BCRP por mes ──────────────────────────────────────────
# Fuente: Notas Informativas del Directorio BCRP
# Formato: (year, month) -> tasa en %
# La tasa aplica desde la semana de la decision hasta el siguiente cambio.
# Si no hay cambio en un mes, se hereda el valor del mes anterior.

TASA_BCRP: dict[tuple[int, int], float] = {
    # 2022 — ciclo de alzas agresivo
    (2022,  2): 3.50,   # +50bps  Feb 10, 2022
    (2022,  3): 4.00,   # +50bps  Mar 10, 2022
    (2022,  4): 4.50,   # +50bps  Abr  7, 2022
    (2022,  5): 5.00,   # +50bps  May 12, 2022
    (2022,  6): 5.50,   # +50bps  Jun  9, 2022
    (2022,  7): 6.00,   # +50bps  Jul  7, 2022
    (2022,  8): 6.50,   # +50bps  Ago 11, 2022
    (2022,  9): 7.00,   # +50bps  Sep  8, 2022
    (2022, 10): 7.25,   # +25bps  Oct  6, 2022  (desaceleracion)
    (2022, 11): 7.25,   # Pausa   Nov 10, 2022
    (2022, 12): 7.25,   # Pausa   Dic  8, 2022

    # 2023 — pico y pausa prolongada, luego cortes
    (2023,  1): 7.75,   # +50bps  Ene 12, 2023  (ultimo alza)
    (2023,  2): 7.75,   # Pausa
    (2023,  3): 7.75,   # Pausa
    (2023,  4): 7.75,   # Pausa
    (2023,  5): 7.75,   # Pausa
    (2023,  6): 7.75,   # Pausa
    (2023,  7): 7.75,   # Pausa
    (2023,  8): 7.75,   # Pausa
    (2023,  9): 7.50,   # -25bps  Sep 14, 2023  (inicio ciclo de cortes)
    (2023, 10): 7.25,   # -25bps  Oct 12, 2023
    (2023, 11): 7.00,   # -25bps  Nov  9, 2023
    (2023, 12): 6.75,   # -25bps  Dic  7, 2023

    # 2024 — ciclo de cortes continuo
    (2024,  1): 6.50,   # -25bps  Ene 11, 2024
    (2024,  2): 6.25,   # -25bps  Feb  8, 2024
    (2024,  3): 6.25,   # Pausa   Mar 14, 2024
    (2024,  4): 6.00,   # -25bps  Abr 11, 2024
    (2024,  5): 5.75,   # -25bps  May  9, 2024
    (2024,  6): 5.75,   # Pausa   Jun 13, 2024
    (2024,  7): 5.75,   # Pausa   Jul 11, 2024
    (2024,  8): 5.50,   # -25bps  Ago  8, 2024
    (2024,  9): 5.25,   # -25bps  Sep 12, 2024
    (2024, 10): 5.00,   # -25bps  Oct 10, 2024
    (2024, 11): 5.00,   # Pausa   Nov 14, 2024
    (2024, 12): 4.75,   # -25bps  Dic 12, 2024

    # 2025 — pausa prolongada en 4.50%, luego un unico corte en sep
    (2025,  1): 4.75,   # Pausa   Ene  9, 2025
    (2025,  2): 4.50,   # -25bps  Feb 13, 2025
    (2025,  3): 4.50,   # Pausa   Mar 2025
    (2025,  4): 4.50,   # Pausa   Abr 2025
    (2025,  5): 4.50,   # Pausa   May 2025
    (2025,  6): 4.50,   # Pausa   Jun 2025
    (2025,  7): 4.50,   # Pausa   Jul 2025
    (2025,  8): 4.50,   # Pausa   Ago 2025
    (2025,  9): 4.25,   # -25bps  Sep 11, 2025 (nota-informativa-2025-09-11)
    (2025, 10): 4.25,   # Pausa   Oct  9, 2025 (nota-informativa-2025-10-09)
    (2025, 11): 4.25,   # Pausa   Nov 2025
    (2025, 12): 4.25,   # Pausa   Dic 2025

    # 2026 — pausa prolongada en 4.25% confirmada hasta jun 2026
    (2026,  1): 4.25,   # Pausa   Ene 2026
    (2026,  2): 4.25,   # Pausa   Feb 2026
    (2026,  3): 4.25,   # Pausa   Mar 2026
    (2026,  4): 4.25,   # Pausa   Abr  7, 2026 (nota-informativa-2026-04-07)
    (2026,  5): 4.25,   # Pausa   May 2026 (noveno mes consecutivo en 4.25%)
    (2026,  6): 4.25,   # Pausa   Jun 2026 (confirmado)
}

# Tasa base para meses fuera del dict (carry-forward hacia adelante)
_TASA_BASE_INICIO = 3.00  # Enero 2022 (antes del primer cambio registrado)


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def get_tasa_for_month(year: int, month: int) -> float:
    """
    Retorna la tasa de referencia BCRP vigente para un mes dado.
    Si el mes no está en el dict, hace carry-forward desde el ultimo valor conocido.
    """
    if (year, month) in TASA_BCRP:
        return TASA_BCRP[(year, month)]

    # Carry-forward: busca el mes previo mas reciente que tenga dato
    for m in range(month - 1, 0, -1):
        if (year, m) in TASA_BCRP:
            return TASA_BCRP[(year, m)]
    for y in range(year - 1, 2021, -1):
        for m in range(12, 0, -1):
            if (y, m) in TASA_BCRP:
                return TASA_BCRP[(y, m)]
    return _TASA_BASE_INICIO


def calcular_bloque_b(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)

    results = []
    for semana_str in semanas["semana"]:
        monday = _parse_monday(str(semana_str))
        # Usar el mes del lunes (la semana pertenece al mes donde cae su lunes)
        tasa = get_tasa_for_month(monday.year, monday.month)
        results.append({"semana": semana_str, "tasa_bcrp": tasa})

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")
    print(f"OK Guardado: {output_csv} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "primer_master_peru", "semanas_peru.csv")
    output_csv = os.path.join(base, "primer_master_peru", "bloque_b_tasa_bcrp.csv")

    df = calcular_bloque_b(input_csv, output_csv)

    print()
    print("--- Primeras 6 filas ---")
    print(df.head(6).to_string())
    print("--- Ultimas 6 filas ---")
    print(df.tail(6).to_string())
    print()
    print("Valores unicos de tasa_bcrp:", sorted(df["tasa_bcrp"].unique().tolist()))
    print("NaN:", df["tasa_bcrp"].isna().sum())
