"""
Ronda 2: bolsa de valores + related queries correctas + test CDT colombia
"""
import time, sys
sys.stdout.reconfigure(encoding='utf-8')
from pytrends.request import TrendReq

pt = TrendReq(hl='es-CO', tz=300, timeout=(15, 45))

# ── 1. Agregar bolsa de valores al cuadro de variabilidad ─────────────────────
print("=" * 60)
print("PASO 1 — Variabilidad: incluyendo bolsa de valores")
print("=" * 60)

grupo = ['CDT', 'bolsa de valores', 'tasa CDT', 'comprar acciones', 'invertir dinero']
try:
    pt.build_payload(grupo, geo='CO', timeframe='today 5-y')
    df = pt.interest_over_time()
    if not df.empty:
        print(f"\n{'Keyword':<30} {'Volumen':>8} {'StdDev':>8} {'CV%':>8}  Interpretacion")
        print("-" * 80)
        for kw in grupo:
            if kw in df.columns:
                m = df[kw].mean()
                s = df[kw].std()
                cv = (s / m * 100) if m > 0 else 0
                if cv > 40:    tag = "EXCELENTE señal"
                elif cv > 30:  tag = "BUENA señal"
                elif cv > 20:  tag = "señal moderada"
                else:          tag = "señal debil"
                print(f"  {kw:<28} {m:8.1f} {s:8.1f} {cv:7.1f}%  {tag}")
    time.sleep(8)
except Exception as e:
    print(f"Error: {e}")

# ── 2. Related queries (formato correcto) ────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 2 — Related queries (lo que la gente busca junto con estos terminos)")
print("=" * 60)

for kw in ['CDT', 'bolsa de valores', 'fondos de inversion']:
    try:
        print(f"\n--- '{kw}' en Colombia ---")
        pt.build_payload([kw], geo='CO', timeframe='today 5-y')
        related = pt.related_queries()
        data = related.get(kw, {})

        top = data.get('top')
        if top is not None and not top.empty:
            print("  TOP queries:")
            for _, row in top.head(10).iterrows():
                cols = list(row.index)
                val_col = [c for c in cols if c != 'query'][0] if len(cols) > 1 else cols[0]
                print(f"    [{row.get(val_col, '?'):>4}] {row['query']}")

        rising = data.get('rising')
        if rising is not None and not rising.empty:
            print("  RISING (creciendo mas rapido):")
            for _, row in rising.head(5).iterrows():
                cols = list(row.index)
                val_col = [c for c in cols if c != 'query'][0] if len(cols) > 1 else cols[0]
                print(f"    [{row.get(val_col, '?'):>4}] {row['query']}")
        time.sleep(10)
    except Exception as e:
        print(f"  Error: {e}")
        time.sleep(20)

# ── 3. Test: CDT vs terminos especificos de plataformas ──────────────────────
print("\n" + "=" * 60)
print("PASO 3 — Plataformas fintech como proxy de intencion")
print("(si alguien busca Tyba/Trii/Nu es porque quiere invertir)")
print("=" * 60)

fintechs = ['Tyba', 'Nu Colombia', 'Trii', 'Lulo Bank']
try:
    pt.build_payload(fintechs, geo='CO', timeframe='today 5-y')
    df2 = pt.interest_over_time()
    if not df2.empty:
        print(f"\n{'Keyword':<20} {'Volumen':>8} {'CV%':>8}")
        print("-" * 40)
        for kw in fintechs:
            if kw in df2.columns:
                m = df2[kw].mean()
                s = df2[kw].std()
                cv = (s / m * 100) if m > 0 else 0
                print(f"  {kw:<18} {m:8.1f} {cv:7.1f}%")
    time.sleep(8)
except Exception as e:
    print(f"Error: {e}")

# ── 4. Correlacion: CDT con macro variables ───────────────────────────────────
print("\n" + "=" * 60)
print("PASO 4 — Correlacion entre keywords (deben ser independientes entre si)")
print("=" * 60)

try:
    kws_corr = ['CDT', 'bolsa de valores', 'fondos de inversion', 'tasa CDT']
    pt.build_payload(kws_corr, geo='CO', timeframe='today 5-y')
    df3 = pt.interest_over_time()
    if not df3.empty:
        corr = df3[kws_corr].corr().round(2)
        print("\nMatriz de correlacion (ideal: baja correlacion entre variables):")
        print(corr.to_string())
    time.sleep(5)
except Exception as e:
    print(f"Error: {e}")

print("\nDone.")
