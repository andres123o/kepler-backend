"""
Investiga volumenes y queries relacionadas en Google Trends Colombia
para encontrar los mejores keywords de intencion de inversion.
"""
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pytrends.request import TrendReq

pt = TrendReq(hl='es-CO', tz=300, timeout=(15, 45))

CANDIDATOS = [
    'tasa CDT',
    'comprar acciones',
    'CDT',
    'fondos de inversion',
    'rendimiento CDT',
    'invertir dinero',
    'bolsa de valores',
    'abrir cuenta inversion',
]

# ── 1. Volumenes relativos (grupos de 5, comparados con ancla CDT) ─────────────
print("=" * 60)
print("PASO 1 — Volumen relativo (5 anos, Colombia, geo=CO)")
print("=" * 60)
print("Nota: 100 = maximo historico del grupo\n")

ancla = 'CDT'
otros = [k for k in CANDIDATOS if k != ancla]

grupos = []
for i in range(0, len(otros), 4):
    grupos.append([ancla] + otros[i:i+4])

volumenes = {}
for grupo in grupos:
    try:
        print(f"Consultando: {grupo}")
        pt.build_payload(grupo, geo='CO', timeframe='today 5-y')
        df = pt.interest_over_time()
        if not df.empty:
            for kw in grupo:
                if kw in df.columns:
                    volumenes[kw] = round(df[kw].mean(), 1)
        time.sleep(6)
    except Exception as e:
        print(f"  Error: {e}")
        time.sleep(20)

print("\nVolumen medio relativo (ordenado):")
for kw, val in sorted(volumenes.items(), key=lambda x: -x[1]):
    bar = '█' * int(val / 3)
    print(f"  {kw:<35} {val:5.1f}  {bar}")

# ── 2. Queries relacionadas para los top candidatos ───────────────────────────
print("\n" + "=" * 60)
print("PASO 2 — Top queries relacionadas (lo que la gente busca JUNTO)")
print("=" * 60)

top_kws = ['CDT', 'fondos de inversion', 'tasa CDT', 'invertir dinero']

for kw in top_kws:
    try:
        print(f"\n--- Related queries para: '{kw}' ---")
        pt.build_payload([kw], geo='CO', timeframe='today 5-y')
        related = pt.related_queries()
        data = related.get(kw, {})

        top = data.get('top')
        rising = data.get('rising')

        if top is not None and not top.empty:
            print("  TOP (mas buscadas junto):")
            for _, row in top.head(8).iterrows():
                print(f"    {row['value']:<40} valor: {row['query_value']}")

        if rising is not None and not rising.empty:
            print("  RISING (creciendo rapido):")
            for _, row in rising.head(5).iterrows():
                print(f"    {row['value']:<40} valor: {row['query_value']}")
        time.sleep(8)
    except Exception as e:
        print(f"  Error: {e}")
        time.sleep(20)

# ── 3. Comparar variabilidad semanal ─────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 3 — Variabilidad semanal (desv. estandar)")
print("Mayor variabilidad = mejor señal para el modelo ML")
print("=" * 60)

finales = ['tasa CDT', 'comprar acciones', 'fondos de inversion', 'invertir dinero', 'CDT']
try:
    pt.build_payload(finales, geo='CO', timeframe='today 5-y')
    df = pt.interest_over_time()
    if not df.empty:
        print(f"\n{'Keyword':<35} {'Media':>8} {'StdDev':>8} {'CV%':>8}")
        print("-" * 62)
        for kw in finales:
            if kw in df.columns:
                m = df[kw].mean()
                s = df[kw].std()
                cv = (s / m * 100) if m > 0 else 0
                print(f"  {kw:<33} {m:8.1f} {s:8.1f} {cv:7.1f}%")
    time.sleep(6)
except Exception as e:
    print(f"Error: {e}")

print("\nDone.")
