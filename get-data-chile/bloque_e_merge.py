"""
Bloque E — Merge final: master-chile.xlsx + bloques A/B/C/D.

Input:
  chile/master-chile.xlsx          (225 filas, funnel interno + target)
  chile/bloque_a_mercados.csv
  chile/bloque_b_tpm.csv
  chile/bloque_c_trends.csv
  chile/bloque_d_calendario.csv

Output:
  chile/master_consolidado_chile_full.xlsx

Join: left join sobre columna 'semana' (D/M/YYYY). Reporte de NaN por columna al final.
"""

import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "chile")


def _normalize_semana(s: pd.Series) -> pd.Series:
    def _fmt(val: str) -> str:
        parts = str(val).strip().split("/")
        if len(parts) != 3:
            return str(val)
        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{d}/{m}/{y}"
    return s.astype(str).apply(_fmt)


def load_csv(filename: str, key_col: str = "semana") -> pd.DataFrame:
    path = os.path.join(DATA, filename)
    df = pd.read_csv(path, sep=",")
    df[key_col] = _normalize_semana(df[key_col])
    print(f"  Cargado {filename}: {len(df)} filas, cols={list(df.columns)}")
    return df


def load_master() -> pd.DataFrame:
    path = os.path.join(DATA, "master-chile.xlsx")
    df = pd.read_excel(path)

    if "semana" not in df.columns:
        for col in df.columns:
            try:
                pd.Timestamp(df[col].dropna().iloc[0])
                df = df.rename(columns={col: "semana"})
                break
            except Exception:
                pass

    def _to_dmy(v) -> str:
        try:
            ts = pd.Timestamp(v)
            return f"{ts.day}/{ts.month}/{ts.year}"
        except Exception:
            s = str(v).strip()
            if "/" in s:
                parts = s.split("/")
                return f"{int(parts[0])}/{int(parts[1])}/{int(parts[2])}"
            return s

    df["semana"] = df["semana"].apply(_to_dmy)
    print(f"  Cargado master-chile.xlsx: {len(df)} filas, cols={list(df.columns)}")
    return df


def calcular_bloque_e() -> pd.DataFrame:
    print("Cargando archivos ...")
    master = load_master()

    bloques = [
        load_csv("bloque_a_mercados.csv"),
        load_csv("bloque_b_tpm.csv"),
        load_csv("bloque_c_trends.csv"),
        load_csv("bloque_d_calendario.csv"),
    ]

    print("\nEjecutando merge ...")
    result = master.copy()
    for bloque_df in bloques:
        cols = [c for c in bloque_df.columns if c != "semana"]
        result = result.merge(bloque_df[["semana"] + cols], on="semana", how="left")
        print(f"  Despues de merge: {len(result)} filas, {len(result.columns)} columnas")

    print("\n--- NaN por columna ---")
    for col, n in result.isna().sum().items():
        print(f"  {col}: {'OK' if n == 0 else f'!! {n} NaN'}")

    output_path = os.path.join(DATA, "master_consolidado_chile_full.xlsx")
    result.to_excel(output_path, index=False)
    print(f"\nOK Guardado: {output_path}")
    print(f"   {len(result)} filas x {len(result.columns)} columnas")
    return result


if __name__ == "__main__":
    df = calcular_bloque_e()
    print()
    print("--- Primeras 3 filas (primeras 8 columnas) ---")
    print(df.iloc[:3, :8].to_string())
    print()
    print("--- Ultimas 3 filas (ultimas 8 columnas) ---")
    print(df.iloc[-3:, -8:].to_string())
    print()
    print("Columnas finales:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
