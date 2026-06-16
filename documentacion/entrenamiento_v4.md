# Entrenamiento Modelo v4 — Kepler ML

**Fecha entrenamiento:** 2026-06-08
**Versión:** v4
**Estado:** RECHAZADO — overfitting. No desplegar.
**Modelo en producción:** v3

---

## Configuración

| Parámetro | Valor |
|---|---|
| Loss function | `reg:absoluteerror` (igual que v3) |
| Tree method | `hist` |
| Optimizador | Optuna (60 trials, timeout 600s) |
| Walk-forward inicial | 52 semanas |
| Datos de entrenamiento | `master_consolidado_final_v2.csv` |
| Muestras útiles | ~270 filas × 23 features |

---

## Resultados de entrenamiento

| Métrica | v4 | v3 (referencia) | v1 (referencia) |
|---|---|---|---|
| MAE walk-forward | **200.x** | 195.5 | 166.4 |
| MAE train | **71.x** | 72.1 | 109.8 |
| R² train | ~0.928 | 0.928 | 0.931 |
| Ratio WF/train | **2.81x** ❌ | 2.71x ⚠️ | 1.51x ✅ |
| Overfitting flag | CRÍTICO | OK (umbral viejo) | OK |

> v4 era flaggeado "OK" con los umbrales viejos (crítico si >5.0). Con el umbral correcto
> de 2.0, v4 queda clasificado como CRÍTICO y debe rechazarse.

---

## Hiperparámetros óptimos (Optuna v4)

```json
{
  "max_depth": 4,
  "min_child_weight": 5,
  "subsample": ...,
  "colsample_bytree": ...,
  "reg_alpha": ...,
  "reg_lambda": ...,
  "learning_rate": ...,
  "num_boost_round": ...
}
```

> Optuna eligió max_depth=4 y min_child_weight=5 — más agresivos que v3 (depth=5/mcw=6)
> o v1 (depth=3/mcw=14). Con MAE loss (más difícil de optimizar que MSE), depth=4 y mcw=5
> genera árboles que memorizan el training set en lugar de generalizar.

---

## Predicción en producción

| Semana | Real | v4 pred | Error abs | Error% |
|---|---|---|---|---|
| 1–7 jun 2026 | 1,860 | 1,504 | 356 | 19.1% Under |

**Para comparación:**
- v3 en esa semana: pred 1,547 · error 313 · 16.8% Under
- v4 empeoró 43 unidades vs v3 (14% más de error)

---

## Diagnóstico de falla

### Por qué v4 overfitteó siendo "similar" a v3

**v4 hiperparámetros**: max_depth=4, min_child_weight=5
**v3 hiperparámetros**: max_depth=5, min_child_weight=6

Aunque la diferencia parece pequeña, con MAE loss los efectos se amplifican:
- MAE loss tiene gradientes ±1 constantes (no escala por residual)
- Esto hace que el optimizador encuentre menor beneficio en cada split
- Para compensar, Optuna elige árboles que hacen más splits por nivel (depth más efectivo)
- Con mcw=5 y depth=4, cada árbol puede crear ramas muy específicas para poca data
- Resultado: MAE_train=71 (memorización) pero MAE_WF=200 (no generaliza)

**Regla aprendida:** Con reg:absoluteerror y ~270 muestras, el espacio óptimo está:
- max_depth ≤ 3 (no 4-5)
- min_child_weight ≥ 10 (no 5-6)

El ratio WF/train de v4 (2.81x) vs v1 (1.51x) muestra que la pérdida MAE exige
regularización más fuerte, no menos.

---

## Cambios aplicados para v5

### config.py

| Parámetro | Antes (v4) | Después (v5) |
|---|---|---|
| `OPTUNA_MAX_DEPTH_RANGE` | `(3, 5)` | `(2, 3)` |
| `OPTUNA_MIN_CHILD_WEIGHT_RANGE` | `(5, 20)` | `(10, 20)` |
| `OPTUNA_REG_ALPHA_RANGE` | `(3.0, 10.0)` | `(1.0, 15.0)` |
| `OPTUNA_REG_LAMBDA_RANGE` | `(3.0, 10.0)` | `(1.0, 10.0)` |

### train.py — umbrales de overfitting

| Ratio | Antes | Después |
|---|---|---|
| > 5.0 | crítico | — |
| > 3.0 | warning | — |
| > 2.0 | ok (pasaba sin aviso) | **crítico** |
| 1.5–2.0 | ok | warning |
| < 1.5 | ok | **ok (objetivo v5)** |

---

## Cuándo reentrenar como v5

**Condición:** acumular 3–4 semanas más sin outliers exógenos confirmados.

**Semanas pendientes de actuals:**
- 8–14 jun 2026 (dato real pendiente)
- 15–21 jun 2026 (dato real pendiente)
- 22–28 jun 2026 (dato real pendiente)

**Criterio de mejora vs v3:**
- Ratio WF/train < 2.0 (obligatorio)
- MAE producción limpia < 224.5 (MAE limpio de v3)
- Si error en semana 1–7 jun baja de 313 → el reentrenamiento con datos frescos funciona

**Nota:** No reentrenar antes de tener los 3–4 actuals confirmados.
El dataset más grande (270 vs 266 filas) combinado con el search space más restringido
debería dar Optuna espacio para encontrar la combinación correcta sin memorizar.
