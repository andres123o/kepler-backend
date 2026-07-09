"""
Bloque D — Variables de calendario y comportamiento para Chile.

Input : chile/semanas_chile.csv  (columna 'semana', formato D/M/YYYY)
Output: chile/bloque_d_calendario.csv

Variables calculadas:
  dias_habiles_semana        — conteo lunes-viernes que no son festivo oficial CL
  is_ventana_liquidez_cl     — 1 si algun dia de la semana cae en 28,29,30,1,2,3,4,5
                                (reemplaza a pct_dias_fin_mes — el ciclo real de
                                deposito de Chile confirma pico fin de mes 28-29 Y
                                pico inicio de mes 3-5, ventana mas ancha que la
                                original de 28-31+1-2; dia 15 sin efecto, no es
                                quincenal como Colombia)
  is_ventana_devolucion_renta — 1 si algun dia cae entre el 25 abr y el 31 may
                                (Operacion Renta SII — devolucion de impuestos;
                                Fintual promociona activamente invertirla)
  is_ventana_gasto_estacional — 1 si algun dia cae en ene-mar (cuesta de enero +
                                vuelta a clases + permiso de circulacion feb-mar)
  is_ventana_contribuciones  — 1 si algun dia cae en los ultimos 5 dias de abr/jun/
                                sep/nov (vencimiento cuotas contribuciones, fechas
                                fijas TGR: 30 abr, 30 jun, 30 sep, 30 nov)
"""

from datetime import date, timedelta
import pandas as pd


def _parse_monday(s: str) -> date:
    parts = str(s).strip().split("/")
    return date(int(parts[2]), int(parts[1]), int(parts[0]))


def _week_days(monday: date) -> list[date]:
    return [monday + timedelta(days=i) for i in range(7)]


def dias_habiles(monday: date) -> int:
    import holidays
    festivos_cl = holidays.country_holidays("CL")
    count = 0
    for i in range(5):
        d = monday + timedelta(days=i)
        if d not in festivos_cl:
            count += 1
    return count


def is_ventana_liquidez_cl(monday: date) -> int:
    LIQUIDEZ_DAYS = {28, 29, 30, 1, 2, 3, 4, 5}
    return int(any(d.day in LIQUIDEZ_DAYS for d in _week_days(monday)))


def is_ventana_devolucion_renta(monday: date) -> int:
    for d in _week_days(monday):
        if (d.month == 4 and d.day >= 25) or (d.month == 5 and d.day <= 31):
            return 1
    return 0


def is_ventana_gasto_estacional(monday: date) -> int:
    for d in _week_days(monday):
        if d.month in (1, 2, 3):
            return 1
    return 0


def is_ventana_contribuciones(monday: date) -> int:
    for d in _week_days(monday):
        if d.month in (4, 6, 9, 11) and d.day >= 26:
            return 1
    return 0


def calcular_bloque_d(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    results = []
    for semana_str in df["semana"]:
        monday = _parse_monday(str(semana_str))
        results.append({
            "semana":                      semana_str,
            "dias_habiles_semana":         dias_habiles(monday),
            "is_ventana_liquidez_cl":      is_ventana_liquidez_cl(monday),
            "is_ventana_devolucion_renta": is_ventana_devolucion_renta(monday),
            "is_ventana_gasto_estacional": is_ventana_gasto_estacional(monday),
            "is_ventana_contribuciones":   is_ventana_contribuciones(monday),
        })

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False, sep=",")
    print(f"OK Guardado: {output_csv} ({len(out)} filas)")
    return out


if __name__ == "__main__":
    import os

    base       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_csv  = os.path.join(base, "chile", "semanas_chile.csv")
    output_csv = os.path.join(base, "chile", "bloque_d_calendario.csv")

    df = calcular_bloque_d(input_csv, output_csv)
    print(df.head(10).to_string())
    print(f"... {len(df)} filas total")
