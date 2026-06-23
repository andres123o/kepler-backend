"""
Bloque E — Variables de calendario para Perú.

Input : primer_master_peru/semanas_peru.csv  (columna 'semana', formato DD/MM/YYYY)
Output: primer_master_peru/bloque_e_calendario.csv

Variables calculadas:
  is_ventana_afp         — 1 si algún día de la semana cae en una ventana AFP activa
  is_ventana_cts         — 1 si algún día cae en días 1-15 de mayo o noviembre
  is_ventana_gratificacion — 1 si algún día cae en días 1-15 de julio o diciembre
  is_ventana_quincena    — 1 si algún día cae en días 14-16 o 28-30 del mes
  dias_habiles_semana    — conteo lunes-viernes que no son festivo oficial PE

Lógica AFP: para la ingesta histórica las ventanas están registradas aquí.
Para uso productivo semanal: agregar filas a AFP_WINDOWS cuando el Congreso
legisle un nuevo retiro — la función is_afp_week() detecta automáticamente.
"""

from datetime import date, timedelta
import pandas as pd

# ─── Ventanas AFP registradas ──────────────────────────────────────────────────
# Agregar una tupla (inicio, fin) cuando haya un nuevo retiro legislado.
AFP_WINDOWS: list[tuple[date, date]] = [
    (date(2020, 4, 3),  date(2020, 4, 27)),   # D.U. 034-2020
    (date(2020, 4, 20), date(2020, 6, 18)),   # D.U. 038-2020
    (date(2020, 5, 18), date(2020, 7, 17)),   # Ley 31017
    (date(2020, 12, 9), date(2021, 3, 9)),    # Ley 31068
    (date(2021, 5, 27), date(2021, 8, 24)),   # Ley 31192
    (date(2022, 6, 13), date(2022, 9, 11)),   # Ley 31478
    (date(2024, 5, 20), date(2024, 8, 18)),   # Ley 32002
    (date(2025, 10, 21), date(2026, 1, 18)),  # Ley 32445
]

# ─── Festivos fijos Perú (mes, día) ───────────────────────────────────────────
FESTIVOS_FIJOS: set[tuple[int, int]] = {
    (1, 1),   # Año Nuevo
    (5, 1),   # Día del Trabajo
    (6, 29),  # San Pedro y San Pablo
    (7, 28),  # Fiestas Patrias
    (7, 29),  # Fiestas Patrias
    (8, 30),  # Santa Rosa de Lima
    (10, 8),  # Combate de Angamos
    (11, 1),  # Todos los Santos
    (12, 8),  # Inmaculada Concepción
    (12, 25), # Navidad
}

# Batalla de Junín — vigente desde 2022 (Ley 31603)
BATALLA_JUNIN_DESDE = 2022

# Viernes Santo por año (Pascua variable) — cubre 2022-2027
VIERNES_SANTO: dict[int, date] = {
    2022: date(2022, 4, 15),
    2023: date(2023, 4, 7),
    2024: date(2024, 3, 29),
    2025: date(2025, 4, 18),
    2026: date(2026, 4, 3),
    2027: date(2027, 3, 26),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_monday(s: str) -> date:
    """Parsea DD/MM/YYYY → date."""
    parts = s.strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def _week_days(monday: date) -> list[date]:
    """Retorna los 7 días de la semana lunes-domingo."""
    return [monday + timedelta(days=i) for i in range(7)]


def _is_festivo_pe(d: date) -> bool:
    """True si la fecha es festivo oficial en Perú."""
    if (d.month, d.day) in FESTIVOS_FIJOS:
        return True
    if d.year >= BATALLA_JUNIN_DESDE and d.month == 8 and d.day == 6:
        return True
    viernes = VIERNES_SANTO.get(d.year)
    if viernes and d == viernes:
        return True
    return False


# ─── Calculadoras por semana ──────────────────────────────────────────────────

def is_afp_week(monday: date) -> int:
    """1 si algún día de la semana cae dentro de cualquier ventana AFP activa."""
    sunday = monday + timedelta(days=6)
    for start, end in AFP_WINDOWS:
        # Hay solapamiento si: inicio_ventana <= domingo Y fin_ventana >= lunes
        if start <= sunday and end >= monday:
            return 1
    return 0


def is_cts_week(monday: date) -> int:
    """1 si algún día de la semana cae en días 1-15 de mayo o noviembre."""
    for d in _week_days(monday):
        if d.month in (5, 11) and 1 <= d.day <= 15:
            return 1
    return 0


def is_gratificacion_week(monday: date) -> int:
    """1 si algún día de la semana cae en días 1-15 de julio o diciembre."""
    for d in _week_days(monday):
        if d.month in (7, 12) and 1 <= d.day <= 15:
            return 1
    return 0


def is_quincena_week(monday: date) -> int:
    """1 si algún día de la semana cae en días 14-16 o 28-30 del mes."""
    QUINCENA_DAYS = {14, 15, 16, 28, 29, 30}
    for d in _week_days(monday):
        if d.day in QUINCENA_DAYS:
            return 1
    return 0


def dias_habiles(monday: date) -> int:
    """Cuenta días lunes-viernes de la semana que no son festivo PE."""
    count = 0
    for i in range(5):  # lunes=0 a viernes=4
        d = monday + timedelta(days=i)
        if not _is_festivo_pe(d):
            count += 1
    return count


# ─── Pipeline principal ───────────────────────────────────────────────────────

def calcular_bloque_e(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    results = []
    for semana_str in df["semana"]:
        monday = _parse_monday(str(semana_str))
        results.append({
            "semana":                 semana_str,
            "is_ventana_afp":         is_afp_week(monday),
            "is_ventana_cts":         is_cts_week(monday),
            "is_ventana_gratificacion": is_gratificacion_week(monday),
            "is_ventana_quincena":    is_quincena_week(monday),
            "dias_habiles_semana":    dias_habiles(monday),
        })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")
    print(f"OK Guardado: {output_csv} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "primer_master_peru", "semanas_peru.csv")
    output_csv = os.path.join(base, "primer_master_peru", "bloque_e_calendario.csv")

    df = calcular_bloque_e(input_csv, output_csv)
    print(df.head(10).to_string())
    print(f"... {len(df)} filas total")
