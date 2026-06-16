# Comparación producción v1 vs v3 — Kepler ML

**Fecha análisis:** 2026-06-08
**Período evaluado:** 4 may – 7 jun 2026 (5 semanas)
**Modelos:** v1 (`reg:squarederror`) vs v3 (`reg:absoluteerror`)

---

## Tabla maestra

| Semana | Real | v1 pred | v1 error abs | v1 error% | v3 pred | v3 error abs | v3 error% | Ganador |
|---|---|---|---|---|---|---|---|---|
| 4–10 may | 1,394 | 1,662 | 268 | 19.2% Over | 1,578 | 184 | 13.2% Over | **v3 ✅** |
| 11–17 may ⚠️ | 2,387 | 1,659 | 728 | 30.5% Under | 1,546 | 841 | 35.2% Under | v1 |
| 18–24 may | 1,422 | 1,736 | 314 | 22.1% Over | 1,618 | 196 | 13.8% Over | **v3 ✅** |
| 25–31 may | 1,796 | 1,577 | 219 | 12.2% Under | 1,591 | 205 | 11.4% Under | **v3 ✅** |
| 1–7 jun | 1,860 | 1,619 | 241 | 12.9% Under | 1,547 | 313 | 16.8% Under | v1 |

> ⚠️ Semana 11–17 may: outlier campaña exógena, excluida del MAE limpio.

---

## Métricas resumen

### MAE semanas limpias (4 semanas: excluyendo campaña)

| Modelo | Cálculo | MAE limpio |
|---|---|---|
| v1 | (268 + 314 + 219 + 241) / 4 | **260.5** |
| v3 | (184 + 196 + 205 + 313) / 4 | **224.5** |

**v3 gana por 36 unidades → mejora del 13.8% sobre v1**

### Error% promedio semanas limpias

| Modelo | Cálculo | Error% promedio |
|---|---|---|
| v1 | (19.2 + 22.1 + 12.2 + 12.9) / 4 | **16.6%** |
| v3 | (13.2 + 13.8 + 11.4 + 16.8) / 4 | **13.8%** |

**v3 reduce el error porcentual en 2.8 puntos → mejora del 17%**

### Ratio de generalización (producción vs walk-forward)

| Modelo | MAE walk-forward | MAE producción limpia | Ratio |
|---|---|---|---|
| v1 | 166 | 260.5 | **1.57x** |
| v3 | 196 | 224.5 | **1.15x** ✅ |

> Este es el número más importante. v3 tiene peor walk-forward (196 vs 166) pero generaliza
> mucho mejor en producción real. El ratio 1.15x significa que el modelo se comporta casi igual
> en producción que en backtesting. v1 con 1.57x estaba overfitteando al período de entrenamiento.

### Veredicto por métrica

| Métrica | v1 | v3 | Ganador |
|---|---|---|---|
| MAE walk-forward | 166 | 196 | v1 |
| MAE producción limpia | 260.5 | 224.5 | **v3 ✅** |
| Error% promedio limpio | 16.6% | 13.8% | **v3 ✅** |
| Ratio generalización | 1.57x | 1.15x | **v3 ✅** |
| Semanas donde gana | 2/5 | 3/5 | **v3 ✅** |

---

## Diagnóstico por semana — v3

### 4–10 mayo — pred 1,578 · real 1,394 · error 13.2% Over
Pipeline muy deprimido: `full_users_aprobados = 2,144` (z = -2.03). El modelo predijo 1,578
porque `tasa_basic_a_risk = 94.7%` (z = +2.32) compensó parcialmente el pipeline bajo.
El real fue más bajo porque el pipeline deprimido dominó. v3 se acercó más que v1 (1,662).

### 11–17 mayo — pred 1,546 · real 2,387 · error 35.2% Under — OUTLIER CAMPAÑA
Campaña exógena confirmada. Ningún modelo puede capturar esto. Excluir del análisis de
performance. El dato entra al entrenamiento futuro como historia pero se marca como anomalía.

### 18–24 mayo — pred 1,618 · real 1,422 · error 13.8% Over
`step_09_full_account = 2,755` (z = +1.17) — el modelo vio pipeline en verde y predijo al alza.
`lag_1_target = 1,394` venía de la semana del outlier campaña (2,387 real) — el modelo vio el
real inflado como lag y se ancló alto. El efecto arrastre del outlier contamina esta semana.
Error esperado dado el contexto.

### 25–31 mayo — pred 1,591 · real 1,796 · error 11.4% Under
Primera semana donde v3 predice bajo el real de forma significativa. `COLCAP = -2.18%` esa
semana pero el real subió igual. Razón probable: efecto post-primera vuelta presidencial
empezando a filtrarse — el optimismo electoral no está en ningún feature del modelo.

### 1–7 junio — pred 1,547 · real 1,860 · error 16.8% Under
`COLCAP = +6.02%` (z = +3.19) — rally electoral confirmado post-primera vuelta. v3 le asignó
solo +17 de contribución al COLCAP pero el efecto real fue masivo. v1 predijo 1,619 (más cerca).
v1 gana esta semana por razones históricas en su feature importance, no porque sea mejor modelo.
La solución estructural es un feature `flag_evento_electoral`, no un cambio de modelo.

---

## Conclusión y siguiente paso

v3 es el modelo a mantener en producción. Generaliza mejor, tiene menor error porcentual promedio
y su comportamiento en producción es más predecible que su walk-forward.

El único riesgo identificado es el evento electoral (COLCAP +6%) que ambos modelos subestiman —
eso no se resuelve reentrenando, se resuelve con el feature `flag_evento_electoral`.

Con 4 semanas limpias comparables la evidencia apunta a v3. En 3–4 semanas más habrá suficiente
para declararlo definitivamente como el modelo de producción.
