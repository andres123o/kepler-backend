# Entrenamiento Modelo v3 — Kepler ML

**Fecha entrenamiento:** 2026-06-07
**Versión:** v3
**Ruta modelo:** `experimentacion/entrenamiento/models/v3/`

---

## Configuración

| Parámetro | Valor |
|---|---|
| Loss function | `reg:absoluteerror` (MAE puro) |
| Tree method | `hist` |
| Optimizador | Optuna (60 trials, timeout 600s) |
| Walk-forward inicial | 52 semanas |
| Datos de entrenamiento | `master_consolidado_final_v2.csv` |
| Muestras útiles | 266 filas × 23 features |

### Por qué `reg:absoluteerror` (no squarederror ni pseudo-Huber)

- **squarederror (v1):** penaliza errores grandes cuadráticamente → semanas de campaña (error ~728)
  distorsionan los parámetros del modelo y sesgan las predicciones en semanas normales.
- **pseudohubererror (intentado):** escala los hessianos por ~1/132 con residuales típicos de ~1,500
  y slope=300, lo que exige 667+ muestras mínimas por hoja. Con 266 filas es imposible →
  `feature_importance = {}`, R² = -14.5. Descartado.
- **absoluteerror (v3):** robusto a outliers (gradiente ±1 constante, independiente de la magnitud),
  usa hessianos unitarios → todos los rangos de hiperparámetros mantienen su escala. Correcto.

---

## Resultados de entrenamiento

| Métrica | v3 | v1 (referencia) |
|---|---|---|
| MAE walk-forward | **195.5** | 166.4 |
| MAE train | 72.1 | 109.8 |
| RMSE train | 158.0 | — |
| R² train | 0.928 | 0.931 |
| Ratio WF/train | 2.71 | 1.51 |
| Overfitting flag | OK | OK |

---

## Hiperparámetros óptimos (Optuna)

```json
{
  "max_depth": 5,
  "min_child_weight": 6,
  "subsample": 0.8219,
  "colsample_bytree": 0.7511,
  "reg_alpha": 3.7499,
  "reg_lambda": 4.3857,
  "learning_rate": 0.0266,
  "num_boost_round": 270
}
```

> Nota: v1 eligió max_depth=3 (mínimo del rango). v3 eligió max_depth=5 porque MAE loss
> es más difícil de optimizar — necesita árboles más profundos para alcanzar el mismo fit.

---

## Feature importance (gain) — top a bottom

| Rank | Feature | Gain | Grupo |
|---|---|---|---|
| 1 | aprobados_ponderados | 13.35 | pipeline_autoreg |
| 2 | lag_1_target | 9.87 | pipeline_autoreg |
| 3 | tasa_risk_a_fulldata | 6.81 | funnel_informativo |
| 4 | step_09_full_account | 4.16 | funnel_informativo |
| 5 | trends_cdt | 3.75 | accionable_campana |
| 6 | dias_habiles_proyeccion | 3.09 | accionable_campana |
| 7 | full_users_aprobados | 2.93 | funnel_informativo |
| 8 | spread_tes_banrep | 2.91 | accionable_campana |
| 9 | pct_perfil_conservador | 2.83 | accionable_campana |
| 10 | dias_habiles_semana | 2.59 | accionable_campana |
| 11 | tasa_basic_a_risk | 2.58 | funnel_informativo |
| 12 | trends_acciones | 2.55 | accionable_campana |
| 13 | pct_perfil_arriesgado | 2.37 | accionable_campana |
| 14 | tasa_fulldata_a_video | 2.28 | funnel_informativo |
| 15 | sp500_cambio_semanal_pct | 2.26 | accionable_campana |
| 16 | brent_cambio_semanal_pct | 2.24 | accionable_campana |
| 17 | full_users_aprobados_lag1 | 2.23 | pipeline_autoreg |
| 18 | TRM | 2.19 | accionable_campana |
| 19 | tendencia_aprobados_4w | 2.10 | pipeline_autoreg |
| 20 | tasa_video_a_review | 2.08 | funnel_informativo |
| 21 | colcap_cambio_semanal_pct | 2.07 | accionable_campana |
| 22 | pct_dias_quincena | 1.91 | accionable_campana |
| 23 | tendencia_depositos_4w | 1.84 | accionable_campana |

**Diferencia clave vs v1:** en v3 los 23 features tienen gain balanceado (13.35 máx).
En v1, `aprobados_ponderados` dominaba con ~50% del gain total. El modelo v3 escucha más
a los features accionables (macro, trends, perfiles, timing).

---

## Features activas (23 total)

### pipeline_autoreg (informativo)
`lag_1_target`, `aprobados_ponderados`, `full_users_aprobados_lag1`, `tendencia_aprobados_4w`

### funnel_informativo (informativo)
`step_09_full_account`, `tasa_basic_a_risk`, `tasa_risk_a_fulldata`, `tasa_fulldata_a_video`,
`tasa_video_a_review`, `full_users_aprobados`

### accionable_campana (levers CIO)
`pct_perfil_conservador`, `pct_perfil_arriesgado`, `TRM`, `sp500_cambio_semanal_pct`,
`brent_cambio_semanal_pct`, `colcap_cambio_semanal_pct`, `spread_tes_banrep`, `trends_cdt`,
`trends_acciones`, `pct_dias_quincena`, `dias_habiles_semana`, `dias_habiles_proyeccion`,
`tendencia_depositos_4w`

---

## Comportamiento en producción vs walk-forward

| | v1 | v3 |
|---|---|---|
| MAE walk-forward | 166 | 196 |
| MAE producción limpia (4 semanas) | 260.5 | 224.5 |
| Ratio producción/WF | 1.57x | **1.15x** |

El ratio 1.15x de v3 es la métrica más importante: el modelo se comporta casi igual
en producción que en backtesting. v1 con 1.57x estaba sobre-ajustado al período histórico.

---

## Limitaciones conocidas

- **Compresión de predicciones:** MAE loss predice hacia la mediana del target histórico.
  Semanas con real > 1,800 tienden a ser sub-predichas. Esto no es un bug, es el comportamiento
  esperado de L1 loss.
- **Eventos electorales:** COLCAP +6% (rally post-primera vuelta) genera efecto real masivo
  que el modelo subestima. Solución futura: feature `flag_evento_electoral`.
- **Campañas exógenas:** semanas donde Trii lanza una campaña interna generan spikes no
  predecibles (variable no existe en features). Se marca como outlier en evaluación.

---

## Siguiente entrenamiento sugerido

Agregar al `master_consolidado_final_v2.csv` las semanas con actuals confirmados:
`27/04, 04/05, 11/05, 18/05` — luego reentrenar y comparar el error en 1–7 jun (real: 1,860)
usando `25/05` como `ultima_semana.xlsx`.

Criterio de mejora: si el error en esa semana baja de 313 → el reentrenamiento con datos
frescos es el lever correcto. Si no baja significativamente → explorar predicción por cuantiles.
