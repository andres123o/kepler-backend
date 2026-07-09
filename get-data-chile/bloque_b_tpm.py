"""
Bloque B — Tasa de Politica Monetaria (TPM) del Banco Central de Chile.

Fuente : comunicados oficiales de cada Reunion de Politica Monetaria (RPM) en
         bcentral.cl. mindicador.cl resulto NO viable para descarga masiva
         (fallo persistente: SSL EOF / server disconnected, probado desde 2
         redes distintas en 2 corridas separadas) — se reemplazo por las 36
         decisiones reales verificadas del Consejo, con fecha exacta de cada
         reunion, enero 2022 a junio 2026. CERO valores estimados/interpolados:
         cada fila = un comunicado real de bcentral.cl.
Input  : chile/semanas_chile.csv
Output : chile/bloque_b_tpm.csv

Logica : la TPM decidida en una reunion aplica desde esa fecha hasta la
         siguiente reunion (carry-forward por fecha exacta, no por mes).
"""

from datetime import date
import pandas as pd

# ─── Decisiones reales del Consejo BCCh (fecha exacta -> TPM %) ──────────────
# Fuente: comunicado-rpm-{mes}-{anio} en bcentral.cl para cada fila.
DECISIONES_TPM: list[tuple[date, float]] = [
    (date(2022,  1, 26), 5.50),
    (date(2022,  3, 29), 7.00),
    (date(2022,  5,  5), 8.25),
    (date(2022,  6,  7), 9.00),
    (date(2022,  7, 13), 9.75),
    (date(2022,  9,  6), 10.75),
    (date(2022, 10, 12), 11.25),
    (date(2022, 12,  6), 11.25),
    (date(2023,  1, 26), 11.25),
    (date(2023,  4,  4), 11.25),
    (date(2023,  5, 12), 11.25),
    (date(2023,  6, 19), 11.25),
    (date(2023,  7, 28), 10.25),
    (date(2023,  9,  5), 9.50),
    (date(2023, 10, 26), 9.00),
    (date(2023, 12, 19), 8.25),
    (date(2024,  1, 31), 7.25),
    (date(2024,  4,  2), 6.50),
    (date(2024,  5, 23), 6.00),
    (date(2024,  6, 18), 5.75),
    (date(2024,  7, 31), 5.75),
    (date(2024,  9,  3), 5.50),
    (date(2024, 10, 17), 5.25),
    (date(2024, 12, 17), 5.00),
    (date(2025,  1, 28), 5.00),
    (date(2025,  3, 21), 5.00),
    (date(2025,  4, 29), 5.00),
    (date(2025,  6, 17), 5.00),
    (date(2025,  7, 29), 4.75),
    (date(2025,  9,  9), 4.75),
    (date(2025, 10, 28), 4.75),
    (date(2025, 12, 16), 4.50),
    (date(2026,  1, 27), 4.50),
    (date(2026,  3, 24), 4.50),
    (date(2026,  4, 28), 4.50),
    (date(2026,  6, 16), 4.50),
]

# Antes de la primera decision del rango (26 ene 2022) — nivel vigente previo
# segun bcentral.cl (RPM dic-2021: 4.00%). El master empieza en mar-2022, no deberia usarse.
_TASA_ANTES_RANGO = 4.00


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def get_tpm_for_monday(monday: date) -> float:
    """TPM vigente en el lunes dado: la ultima decision con fecha <= lunes."""
    vigente = _TASA_ANTES_RANGO
    for fecha_decision, tasa in DECISIONES_TPM:
        if fecha_decision <= monday:
            vigente = tasa
        else:
            break
    return vigente


def calcular_bloque_b(input_csv: str, output_csv: str) -> pd.DataFrame:
    semanas = pd.read_csv(input_csv)

    results = []
    for semana_str in semanas["semana"]:
        monday = _parse_monday(str(semana_str))
        tpm = get_tpm_for_monday(monday)
        results.append({"semana": semana_str, "tpm_bcch": tpm})

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")
    print(f"OK Guardado: {output_csv} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "chile", "semanas_chile.csv")
    output_csv = os.path.join(base, "chile", "bloque_b_tpm.csv")

    df = calcular_bloque_b(input_csv, output_csv)

    print()
    print("--- Primeras 6 filas ---")
    print(df.head(6).to_string())
    print("--- Ultimas 6 filas ---")
    print(df.tail(6).to_string())
    print()
    print("Valores unicos de tpm_bcch:", sorted(df["tpm_bcch"].unique().tolist()))
    print("NaN:", df["tpm_bcch"].isna().sum())
