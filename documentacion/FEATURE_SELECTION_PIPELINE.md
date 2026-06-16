# Feature Selection Pipeline — Kepler v2

## Estado final de cada modelo (actualizado junio 2026)

| Modelo | Target | Decisión | Razón |
|--------|--------|----------|-------|
| M1 | `tasa_basic_a_risk` | **No construido** | Sin varianza suficiente; campaña de recordatorio. No pasa el diagnóstico de tier. |
| M2 | `tasa_fulldata_a_video` | **Analizado, descartado** | 242/315 filas = 100% exacto. Quiebre de régimen 2025. Datos saturados. Ver diagnóstico en BRIEF.md. |
| M3 | `usuarios_primer_cashin` | **En producción ✅** | Modelo v14, MAE WF=191.95, auto-fetch activo. Único que pasa las 4 condiciones del diagnóstico. |

El pipeline de 6 etapas documentado abajo fue diseñado para los tres modelos, pero en la práctica se corrió completo solo para M3. El análisis de M2 se detuvo en el diagnóstico de varianza al confirmar saturación de datos. M1 no requirió análisis.

---

## Contexto original

Se diseñó el pipeline para evaluar **3 modelos separados**, uno por cada variable objetivo:

| Modelo | Target | Tipo | Transformación pre-análisis |
|--------|--------|------|-----------------------------|
| M1 | `tasa_basic_a_risk` | Tasa (0–1) | logit: `log(p / (1-p))` |
| M2 | `tasa_fulldata_a_video` | Tasa (0–1) | logit: `log(p / (1-p))` |
| M3 | `usuarios_primer_cashin` | Conteo | log: `log(y + 1)` |

**Features candidatos (8 variables macro):**

```python
macro_candidates = [
    'colcap_cambio_semanal_pct',
    'TRM',
    'Tasa_Intervencion_Mensual',
    'sp500_cambio_semanal_pct',
    'brent_cambio_semanal_pct',
    'trends_cdt',
    'trends_acciones',
    'is_ventana_quincena',
]
```

**Dataset:** `master_consolidado_final_v2.csv` — ~310 observaciones semanales.

---

## Por qué no basta con correlación

La correlación de Pearson tiene tres limitaciones críticas para este problema:

1. **Solo captura relaciones lineales** — una relación en U o logarítmica puede tener correlación ≈ 0 aunque la variable sea muy relevante.
2. **No detecta redundancia entre features** — si `trends_cdt` y `Tasa_Intervencion_Mensual` están correlacionadas entre sí, el modelo las cuenta dos veces sin que aporten información independiente.
3. **No es temporal** — no distingue si X predice Y o si Y predice X, ni descarta confounding por una tercera variable.

**Principio rector:** correlación ≠ causalidad. Este pipeline no prueba causalidad filosófica, pero sí construye evidencia estadística sólida y multi-dimensional antes de incluir un feature.

---

## El pipeline — 6 etapas en orden

### Etapa 1 — Diagnóstico de multicolinealidad entre features

**Objetivo:** Entender cómo se relacionan los 8 features entre sí, antes de mirar el target.

**Herramientas:**
- Matriz de correlación entre los 8 features
- **VIF (Variance Inflation Factor):** mide cuánto se infla la varianza de un coeficiente por la presencia de features correlacionados

| VIF | Interpretación |
|-----|----------------|
| < 5 | Aceptable |
| 5 – 10 | Revisar con criterio de dominio |
| > 10 | Eliminar o consolidar con otro feature |

**Output:** Lista de pares con alta colinealidad. Cuando dos features tienen VIF alto entre sí, en etapas posteriores MRMR elige cuál conservar — no se elimina manualmente aquí.

**Hipótesis a validar:** `trends_cdt` y `Tasa_Intervencion_Mensual` probablemente estén correlacionadas — ambas capturan "apetito por inversión en Colombia".

---

### Etapa 2 — PCA como diagnóstico de estructura latente

**Objetivo:** No reducir dimensiones (eso destruiría interpretabilidad), sino entender **cuántas señales independientes reales** hay en los 8 features.

**Procedimiento:**
1. Estandarizar los 8 features (media 0, std 1)
2. Calcular componentes principales
3. Graficar varianza explicada acumulada
4. Leer los loadings de cada componente

**Interpretación:**

| Componentes para 85%+ de varianza | Conclusión |
|-----------------------------------|------------|
| 2 – 3 | Los 8 features son básicamente 2–3 señales disfrazadas |
| 5 – 6 | Hay bastante independencia real, más features pueden sobrevivir |

**Output:** Mapa de qué features "cargan" juntos en cada componente. Eso revela grupos semánticos: ej. "señal de mercado global" vs "señal de apetito inversor local" vs "señal de liquidez".

**Nota:** PCA aquí es diagnóstico, no transformación. Los modelos finales siguen usando las variables originales.

---

### Etapa 3 — Mutual Information (MI) por target

**Objetivo:** Reemplazar la correlación como métrica de relevancia individual, usando una métrica que captura **cualquier tipo de dependencia estadística**, no solo lineal.

**Diferencia clave con correlación:**
- Correlación de Pearson = 0 **no implica independencia**
- MI = 0 **sí implica independencia** (bajo supuestos razonables)

**Procedimiento:**
1. Aplicar la transformación correspondiente al target (logit o log)
2. Calcular `mutual_info_regression(X, y_transformado)` para cada feature
3. Repetir para los 3 modelos por separado

**Output por modelo:** Score de MI para cada uno de los 8 features. Features con MI ≈ 0 en los 3 modelos son candidatos a eliminación.

---

### Etapa 4 — MRMR (Maximum Relevance, Minimum Redundancy)

**Origen:** Desarrollado originalmente en bioinformática, adoptado y popularizado por **Uber para su plataforma de ML de marketing** — exactamente el mismo caso de uso (predecir comportamiento de usuarios para campañas de adquisición).

**Lógica:** Selecciona features iterativamente maximizando:

```
score(feature_i) = MI(feature_i, target) − mean(MI(feature_i, features_ya_seleccionados))
```

Es decir: **relevancia con el target** menos **redundancia con lo que ya entró**. Esto resuelve el problema de que features correlacionados entre sí se "roben" el crédito mutuamente.

**Procedimiento:**
1. Correr MRMR para M1, M2, M3 por separado
2. Obtener ranking ordenado de features para cada modelo
3. Determinar el punto de corte: cuántos features incluir (ver Etapa 6)

**Output:** Ranking MRMR por modelo. Comparar con ranking de correlación simple — las diferencias revelan features que parecían relevantes pero son redundantes, o features que parecían irrelevantes pero aportan información independiente.

---

### Etapa 5 — Granger Causality

**Qué testea:** ¿Saber X en la semana `t-k` mejora la predicción de Y en la semana `t`, **por encima** de lo que predice el propio historial de Y?

**Por qué es más robusto que correlación:**
- No basta con que X e Y se muevan juntos
- X debe **preceder temporalmente** a Y
- Controla el historial del propio Y como baseline

**Procedimiento:**
1. Testear para lags k = 1, 2, 3 semanas
2. Correr para cada par (feature, target)
3. Registrar p-value del test de Granger

| p-value | Interpretación |
|---------|----------------|
| < 0.05 | Evidencia de relación temporal (Granger-causa) |
| 0.05 – 0.10 | Zona gris — pesar con criterio de dominio |
| > 0.10 | Sin evidencia temporal |

**Output:** Matriz de p-values por (feature × target × lag). Features que no Granger-causan ningún target a ningún lag son candidatos a eliminación.

**Limitación reconocida:** Granger causality sigue siendo predictiva, no causal en sentido estricto. Pero es órdenes de magnitud más robusto que correlación contemporánea.

---

### Etapa 6 — Estabilidad en walk-forward CV

**Objetivo:** Validar que la relevancia de un feature no es un artefacto de un período de tiempo específico.

**Procedimiento:**
1. Definir los folds del walk-forward CV (mismo esquema que el modelo de producción)
2. En cada fold, correr MI + MRMR
3. Registrar qué features fueron seleccionados en cada fold

**Criterio de retención:**
> Un feature entra al modelo final si fue seleccionado en **≥ 70% de los folds**.

**Por qué:** Un feature relevante solo en ciertos períodos señala una relación inestable que probablemente se rompa en producción (overfitting temporal).

**Output:** Tabla de estabilidad: `feature × fold → seleccionado/no`. Features con alta estabilidad cruzada son los más confiables.

---

## Decisión final por modelo

Al terminar las 6 etapas, para cada modelo se completa esta tabla:

| Feature | VIF | PCA loading | MI score | MRMR rank | Granger p | Estabilidad CV | Decisión |
|---------|-----|-------------|----------|-----------|-----------|----------------|---------|
| `colcap_cambio_semanal_pct` | | | | | | | |
| `TRM` | | | | | | | |
| `Tasa_Intervencion_Mensual` | | | | | | | |
| `sp500_cambio_semanal_pct` | | | | | | | |
| `brent_cambio_semanal_pct` | | | | | | | |
| `trends_cdt` | | | | | | | |
| `trends_acciones` | | | | | | | |
| `is_ventana_quincena` | | | | | | | |

**Criterio de inclusión:** un feature entra si pasa **al menos 4 de 5 criterios cuantitativos** (VIF OK + MI relevante + MRMR top + Granger significativo + estabilidad ≥ 70%), o si falla 1–2 criterios pero tiene justificación causal sólida de dominio.

---

## Hipótesis pre-análisis

Antes de correr el pipeline, estas son las expectativas basadas en razonamiento de dominio:

| Feature | Hipótesis |
|---------|-----------|
| `trends_cdt` | Pasa todos los criterios en M3 (cashin). Puede redundar con `Tasa_Intervencion_Mensual`. |
| `Tasa_Intervencion_Mensual` | Relevante pero con lag (efecto no inmediato). Granger con k=2–3 más que k=1. |
| `trends_acciones` | Efecto inverso a `trends_cdt`. MRMR probablemente elige uno de los dos, no ambos. |
| `is_ventana_quincena` | MI bajo por correlación lineal, pero mecanismo causal muy claro (día de pago). Granger debería rescatarlo. |
| `TRM`, `sp500`, `brent` | Correlación ≈ 0 ya era señal de alarma. Probablemente no pasan Granger. |
| `colcap_cambio_semanal_pct` | Señal débil, puede sobrevivir en M3 pero no en M1/M2. |

---

## Implementación

Cada etapa tiene un script separado en esta carpeta:

```
experimentacion/
  feature_selection/
    01_vif_multicollinearity.py
    02_pca_structure.py
    03_mutual_information.py
    04_mrmr_ranking.py
    05_granger_causality.py
    06_cv_stability.py
    results/
      m1_tasa_basic_a_risk/
      m2_tasa_fulldata_a_video/
      m3_usuarios_primer_cashin/
      summary_decision_table.csv
```

---

## Referencias

- Uber: [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing ML Platform](https://arxiv.org/pdf/1908.05376)
- [Using causal discovery for feature selection in multivariate time series — Springer ML](https://link.springer.com/article/10.1007/s10994-014-5460-1)
- [MI-VIF combined feature selection — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1386142521012294)
- [MRMR — Feature Engine docs](https://feature-engine.trainindata.com/en/1.8.x/user_guide/selection/MRMR.html)
- [Granger Causality in Statistical Inference](https://www.numberanalytics.com/blog/granger-causality-statistical-inference)
