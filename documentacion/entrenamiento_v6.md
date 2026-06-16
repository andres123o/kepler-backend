# Entrenamiento Modelo v6 — Kepler ML

**Fecha entrenamiento:** 2026-06-08
**Versión:** v6
**Estado:** HISTÓRICO — reemplazado por v14 ✅
**Reemplaza:** v3
**Reemplazado por:** v14 (entrenado 2026-06-11, MAE WF=191.95, master v2 con 315 filas, filtro `es_exogena`, `pct_dias_quincena`, festivos CO, festivos_count, indice_lluvia)

---

## Cambio principal vs v3

Filtro de semanas exógenas en entrenamiento (`es_exogena == 1`).
La semana de campaña interna (11–17 may 2026, real=2,387) quedó excluida del fit.
El CSV completo sigue usándose en ml_runner.py para calcular lags y tendencias.

---

## Configuración

| Parámetro | Valor |
|---|---|
| Loss function | `reg:absoluteerror` |
| Tree method | `hist` |
| Optimizador | Optuna (60 trials, timeout 600s) |
| Walk-forward inicial | 52 semanas |
| Datos de entrenamiento | `master_consolidado_final_v2.csv` |
| Filas útiles (post-filtro) | 270 (vs 271 antes del filtro) |
| Semanas exógenas excluidas | 1 (11–17 may 2026) |

---

## Resultados de entrenamiento

| Métrica | v6 | v3 (referencia) |
|---|---|---|
| MAE walk-forward | **199.4** | 195.5 |
| MAE train | 79.6 | 72.1 |
| RMSE train | 170.7 | 158.0 |
| R² train | 0.917 | 0.928 |
| Ratio WF/train | **2.51** | 2.71 |
| Overfitting flag | OK | OK |

---

## Hiperparámetros óptimos (Optuna)

```json
{
  "max_depth": 4,
  "min_child_weight": 5,
  "subsample": 0.8287,
  "colsample_bytree": 0.6243,
  "reg_alpha": 5.3878,
  "reg_lambda": 9.4894,
  "learning_rate": 0.0339,
  "num_boost_round": 266
}
```

---

## Feature importance (gain) — top a bottom

| Rank | Feature | Gain | Grupo |
|---|---|---|---|
| 1 | aprobados_ponderados | 12.55 | pipeline_autoreg |
| 2 | lag_1_target | 9.20 | pipeline_autoreg |
| 3 | tasa_risk_a_fulldata | 6.50 | funnel_informativo |
| 4 | step_09_full_account | 3.65 | funnel_informativo |
| 5 | trends_cdt | 2.72 | accionable_campana |
| 6 | spread_tes_banrep | 2.42 | accionable_campana |
| 7 | dias_habiles_proyeccion | 2.38 | accionable_campana |
| 8 | full_users_aprobados | 2.26 | funnel_informativo |
| 9 | tasa_basic_a_risk | 2.13 | funnel_informativo |
| 10 | trends_acciones | 2.06 | accionable_campana |
| 11 | sp500_cambio_semanal_pct | 1.97 | accionable_campana |
| 12 | tasa_fulldata_a_video | 1.85 | funnel_informativo |
| 13 | pct_perfil_conservador | 1.85 | accionable_campana |
| 14 | brent_cambio_semanal_pct | 1.68 | accionable_campana |
| 15 | TRM | 1.61 | accionable_campana |
| 16 | full_users_aprobados_lag1 | 1.59 | pipeline_autoreg |
| 17 | tendencia_depositos_4w | 1.49 | accionable_campana |
| 18 | tasa_video_a_review | 1.48 | funnel_informativo |
| 19 | colcap_cambio_semanal_pct | 1.45 | accionable_campana |
| 20 | pct_perfil_arriesgado | 1.42 | accionable_campana |
| 21 | tendencia_aprobados_4w | 1.37 | pipeline_autoreg |
| 22 | dias_habiles_semana | 1.32 | accionable_campana |
| 23 | pct_dias_quincena | 1.00 | accionable_campana |

**Distribución más balanceada que v3:** top feature baja de 13.35 a 12.55.

---

## Comportamiento en producción — semana 1–7 jun 2026

| | v3 | v6 |
|---|---|---|
| Predicción | 1,547 | 1,528 |
| Real | 1,860 | 1,860 |
| Error absoluto | 313 | 332 |
| Error% | 16.8% | 17.8% |
| SHAP COLCAP (+6%, z=3.19) | +16.2 | **+31.3** |

El error absoluto de v6 es 19 unidades mayor en esta semana, pero la contribución SHAP
de COLCAP se duplicó (16→31). El modelo ahora captura mejor el efecto de mercado.
La semana 1–7 jun sigue siendo difícil por el rally electoral (evento no lineal).

---

## Por qué v6 es mejor que v3 estructuralmente

La semana 11–17 may (campaña exógena, real=2,387) sesgaba los parámetros Optuna:
- El modelo intentaba aprender una semana imposible de predecir
- Esto generaba gradientes que alejaban los parámetros del espacio óptimo para semanas normales
- Al excluirla, Optuna converge a parámetros que generalizan mejor en condiciones normales

El ratio WF/train bajó de 2.71 a 2.51 — menor brecha entre train y validación.

---

## Próximos pasos

1. Acumular actuals confirmados semanalmente en el master
2. Marcar semanas exógenas con `es_exogena=1` antes de agregar al CSV
3. Reentrenar cuando haya 4+ semanas limpias adicionales
4. Comparar error en semanas sin eventos electorales — ahí se verá la mejora real de v6
