# ============================================================
# kepler v2 — pct_dias_quincena
# Ejecutar desde kepler-backend/:
#   python experimentacion/calcular_ventana_quincena.py
# Output: experimentacion/kepler_ventana_quincena.csv
#
# Definición: fracción de días de la semana (lun-dom, 7 días)
#   que caen en la ventana post-quincena colombiana.
#   Días de ventana: {1, 2, 3, 15, 16, 17, 28, 29, 30}
#   — días 28-30: anticipación/acción fin de mes
#   — días  1- 3: acción post-pago fin de mes
#   — días 15-17: acción post-quincena del 15
#
# Ejemplo:
#   Semana 28/04 al 04/05 → días 28,29,30,1,2,3 en ventana → 6/7 = 0.857
#   Semana 07/04 al 13/04 → ningún día en ventana          → 0/7 = 0.000
#
# Por qué continua y no binaria:
#   La señal de quincena opera a nivel diario (+41% diferencia entre
#   día pico y valle, fuente: cicloUsuario.md). Una variable binaria
#   semanal iguala una semana con 1 día de quincena a una con 6,
#   diluyendo completamente la señal. La fracción preserva la
#   intensidad real del efecto de liquidez en la semana.
# ============================================================

import pandas as pd
from pathlib import Path

OUT_CSV = Path(__file__).parent / 'kepler_ventana_quincena.csv'

QUINCENA_DAYS = {1, 2, 3, 15, 16, 17, 28, 29, 30}

SEMANAS_RAW = """25/05/2026
18/05/2026
11/05/2026
4/05/2026
27/04/2026
20/04/2026
13/04/2026
6/04/2026
30/03/2026
23/03/2026
16/03/2026
9/03/2026
2/03/2026
23/02/2026
16/02/2026
9/02/2026
2/02/2026
26/01/2026
19/01/2026
12/01/2026
5/01/2026
29/12/2025
22/12/2025
15/12/2025
8/12/2025
1/12/2025
24/11/2025
17/11/2025
10/11/2025
3/11/2025
27/10/2025
20/10/2025
13/10/2025
6/10/2025
29/09/2025
22/09/2025
15/09/2025
8/09/2025
1/09/2025
25/08/2025
18/08/2025
11/08/2025
4/08/2025
28/07/2025
21/07/2025
14/07/2025
7/07/2025
30/06/2025
23/06/2025
16/06/2025
9/06/2025
2/06/2025
26/05/2025
19/05/2025
12/05/2025
5/05/2025
28/04/2025
21/04/2025
14/04/2025
7/04/2025
31/03/2025
24/03/2025
17/03/2025
10/03/2025
3/03/2025
24/02/2025
17/02/2025
10/02/2025
3/02/2025
27/01/2025
20/01/2025
13/01/2025
6/01/2025
30/12/2024
23/12/2024
16/12/2024
9/12/2024
2/12/2024
25/11/2024
18/11/2024
11/11/2024
4/11/2024
28/10/2024
21/10/2024
14/10/2024
7/10/2024
30/09/2024
23/09/2024
16/09/2024
9/09/2024
2/09/2024
26/08/2024
19/08/2024
12/08/2024
5/08/2024
29/07/2024
22/07/2024
15/07/2024
8/07/2024
1/07/2024
24/06/2024
17/06/2024
10/06/2024
3/06/2024
27/05/2024
20/05/2024
13/05/2024
6/05/2024
29/04/2024
22/04/2024
15/04/2024
8/04/2024
1/04/2024
25/03/2024
18/03/2024
11/03/2024
4/03/2024
26/02/2024
19/02/2024
12/02/2024
5/02/2024
29/01/2024
22/01/2024
15/01/2024
8/01/2024
1/01/2024
25/12/2023
18/12/2023
11/12/2023
4/12/2023
27/11/2023
20/11/2023
13/11/2023
6/11/2023
30/10/2023
23/10/2023
16/10/2023
9/10/2023
2/10/2023
25/09/2023
18/09/2023
11/09/2023
4/09/2023
28/08/2023
21/08/2023
14/08/2023
7/08/2023
31/07/2023
24/07/2023
17/07/2023
10/07/2023
3/07/2023
26/06/2023
19/06/2023
12/06/2023
5/06/2023
29/05/2023
22/05/2023
15/05/2023
8/05/2023
1/05/2023
24/04/2023
17/04/2023
10/04/2023
3/04/2023
27/03/2023
20/03/2023
13/03/2023
6/03/2023
27/02/2023
20/02/2023
13/02/2023
6/02/2023
30/01/2023
23/01/2023
16/01/2023
9/01/2023
2/01/2023
26/12/2022
19/12/2022
12/12/2022
5/12/2022
28/11/2022
21/11/2022
14/11/2022
7/11/2022
31/10/2022
24/10/2022
17/10/2022
10/10/2022
3/10/2022
26/09/2022
19/09/2022
12/09/2022
5/09/2022
29/08/2022
22/08/2022
15/08/2022
8/08/2022
1/08/2022
25/07/2022
18/07/2022
11/07/2022
4/07/2022
27/06/2022
20/06/2022
13/06/2022
6/06/2022
30/05/2022
23/05/2022
16/05/2022
9/05/2022
2/05/2022
25/04/2022
18/04/2022
11/04/2022
4/04/2022
28/03/2022
21/03/2022
14/03/2022
7/03/2022
28/02/2022
21/02/2022
14/02/2022
7/02/2022
31/01/2022
24/01/2022
17/01/2022
10/01/2022
3/01/2022
27/12/2021
20/12/2021
13/12/2021
6/12/2021
29/11/2021
22/11/2021
15/11/2021
8/11/2021
1/11/2021
25/10/2021
18/10/2021
11/10/2021
4/10/2021
27/09/2021
20/09/2021
13/09/2021
6/09/2021
30/08/2021
23/08/2021
16/08/2021
9/08/2021
2/08/2021
26/07/2021
19/07/2021
12/07/2021
5/07/2021
28/06/2021
21/06/2021
14/06/2021
7/06/2021
31/05/2021
24/05/2021
17/05/2021
10/05/2021
3/05/2021
26/04/2021
19/04/2021
12/04/2021
5/04/2021
29/03/2021
22/03/2021
15/03/2021
8/03/2021
1/03/2021
22/02/2021
15/02/2021
8/02/2021
1/02/2021
25/01/2021
18/01/2021
11/01/2021
4/01/2021
28/12/2020
21/12/2020
14/12/2020
7/12/2020
30/11/2020
23/11/2020
16/11/2020
9/11/2020
2/11/2020
26/10/2020
19/10/2020
12/10/2020
5/10/2020
28/09/2020
21/09/2020
14/09/2020
7/09/2020
31/08/2020
24/08/2020
17/08/2020
10/08/2020
3/08/2020
27/07/2020
20/07/2020
6/07/2020
29/06/2020
22/06/2020
15/06/2020
8/06/2020
1/06/2020
25/05/2020
20/04/2020
13/04/2020"""


def calc_pct_quincena(monday: pd.Timestamp) -> float:
    """Fracción de días de la semana (lun-dom) que caen en ventana post-quincena."""
    count = sum(
        1 for i in range(7)
        if (monday + pd.Timedelta(days=i)).day in QUINCENA_DAYS
    )
    return round(count / 7, 4)


semanas = pd.to_datetime(
    [s.strip() for s in SEMANAS_RAW.strip().split('\n')],
    format='%d/%m/%Y'
)

rows = []
for monday in semanas:
    pct_val = calc_pct_quincena(monday)
    dias_q = [
        (monday + pd.Timedelta(days=i)).strftime('%d')
        for i in range(7)
        if (monday + pd.Timedelta(days=i)).day in QUINCENA_DAYS
    ]
    rows.append({
        'semana':            monday.strftime('%d/%m/%Y'),
        'pct_dias_quincena': pct_val,
        '_n_dias':           len(dias_q),
        '_dias':             ','.join(dias_q) if dias_q else '-',
    })

df = pd.DataFrame(rows)
df_sorted = df.sort_values(
    'semana',
    ascending=False,
    key=lambda x: pd.to_datetime(x, format='%d/%m/%Y')
).reset_index(drop=True)

# Estadísticas
print(f'Total semanas    : {len(df_sorted)}')
print(f'Media pct        : {df_sorted["pct_dias_quincena"].mean():.3f}')
print(f'Valores unicos   : {sorted(df_sorted["pct_dias_quincena"].unique())}')
print(f'Semanas con 0    : {(df_sorted["pct_dias_quincena"] == 0).sum()}')
print(f'Semanas con >0.5 : {(df_sorted["pct_dias_quincena"] > 0.5).sum()}')
print()

# Muestra de distribución
print('Ejemplos (primeras 15 semanas):')
print(f'  {"semana":<14}  {"pct":>6}  {"n_dias":>6}  dias')
for _, row in df_sorted.head(15).iterrows():
    bar = '#' * row['_n_dias']
    print(f"  {row['semana']:<14}  {row['pct_dias_quincena']:>6.4f}  {row['_n_dias']:>6}  {row['_dias']}  {bar}")

# Guardar
out = df_sorted[['semana', 'pct_dias_quincena']].copy()
out.to_csv(OUT_CSV, index=False)
print(f'\nGuardado : {OUT_CSV}')
print(f'Columnas : {list(out.columns)}')
print(f'Filas    : {len(out)}')
print(f'NaN      : {out["pct_dias_quincena"].isna().sum()}')
