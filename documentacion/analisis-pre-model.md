# Diagnóstico completo — Feature Selection Pipeline, Campaña Primer Depósito

## Lo que el PCA reveló primero: la estructura real

El scree plot cuenta algo importante antes de hablar de features individuales. **PC1 explica el 28% de la varianza, PC2 el 17% — después de eso, la curva se aplana y necesitás 10 componentes para llegar al 85%.** Eso significa que **los 21 features NO son tan redundantes como se esperaba** — no hay 2-3 mega-señales que lo controlen todo. Hay al menos 6-8 dimensiones genuinamente distintas en el espacio de features.

Eso es buena noticia estructuralmente, pero también significa que el problema no se resuelve solo eliminando. Hay que elegir bien.

Mirando los loadings del PCA, los 21 features se organizan en **5 factores latentes claros**:

| Factor latente | Features que lo componen | Interpretación |
|---|---|---|
| **F1 — Apetito inversor** | `trends_cdt` (+), `pct_perfil_conservador` (+), `Tasa_Intervencion_Mensual` (+), `tasa_basic_a_risk` (+) vs `tasa_fulldata_a_video` (−) | Condiciones de mercado que favorecen inversión conservadora |
| **F2 — Volumen del pipeline** | `step_09_full_account` (+), `usuarios_registro_base` (+), `full_users_aprobados` (+), `spread_tes_banrep` (+) | Cuántos usuarios están activos en el funnel |
| **F3 — Mercados globales** | `colcap_cambio_semanal_pct` (+), `brent` (+), `sp500` (+) vs `TRM` (−) | Señal de mercados internacionales — completamente independiente de todo lo demás |
| **F4 — Conversión avanzada** | `tasa_registro_a_aprobado` (+) | Eficiencia total del funnel KYC end-to-end |
| **F5 — Timing de liquidez** | `is_ventana_quincena` — **loading 0.701 en PC6, prácticamente ortogonal a todo** | Variable 100% independiente estructuralmente |

---

## El hallazgo más crítico: la trampa de simultaneidad

Los 3 features con MI más alto son `step_09_full_account` (0.907), `full_users_aprobados` (0.886) y `usuarios_registro_base` (0.875). Parecen los mejores predictores del mundo. **No lo son.**

Granger los elimina a todos:

- `step_09_full_account`: Granger p = 0.086
- `usuarios_registro_base`: Granger p = 0.112
- `full_users_aprobados`: Granger p = **0.414**

Estos features tienen correlación altísima con el target porque miden la misma semana. Son **indicadores contemporáneos, no predictores**. Si tenés 5,000 usuarios en el pipeline esta semana, claro que vas a tener más depósitos esta semana — pero eso no te dice nada antes de que ocurra.

En términos de Palantir, esto es una **violación del principio de integridad temporal**: un feature predictor debe conocerse ANTES que el target. En producción, cuando el modelo corre el domingo para predecir la semana que empieza, estas métricas todavía no existen.

**Lo que esto implica para el modelo actual**: si el modelo v1 usa estas features en tiempo t (misma semana que el target), tiene R² inflado artificialmente. No es un error de implementación — es un error de diseño. La solución es usarlas con lag t-1 (valores de la semana pasada como proxy de "momentum"), no en tiempo t.

---

## Los 4 features que realmente sobreviven el escrutinio completo

Cruzando las 5 etapas, solo estos tienen evidencia limpia:

**`pct_perfil_conservador` — 5/5 criterios. El mejor feature del análisis.**
- VIF = 7.63 (aceptable), MI = 0.51, MRMR rank 7, Granger p = 0.0037, estabilidad 100%.
- Granger significativo a lag 1: el perfil de riesgo de los usuarios aprobados esta semana predice los depósitos de la semana siguiente.
- Mecanismo causal claro: cuando entra más proporción de usuarios conservadores (que buscan CDTs, rendimientos seguros), el depósito sube. Cuando entra más proporción de arriesgados (que quieren acciones), el depósito es más errático.
- **Este es el feature más limpio de todo el dataset.**

**`tasa_fulldata_a_video` — 4/5 criterios. El mejor predictor de conversión.**
- VIF = 70.8 (problema grave — ver abajo), pero MI = 0.708, MRMR rank 1, Granger p ≈ 0.000, estabilidad 100%.
- Granger **a lag 1 con p ≈ 0** — es el predictor temporal más fuerte del dataset.
- El mecanismo es estructural: el paso de fulldata a video es el cuello de botella más discriminatorio del funnel. Cuando esa tasa sube esta semana, los usuarios que ya pasaron ese filtro depositan más la semana siguiente.
- **VIF de 70.8 es una señal de alerta, no de descarte**. El VIF alto viene de su colinealidad con las features de volumen — si se eliminan las redundantes de F2, el VIF de esta baja considerablemente.

**`trends_cdt` — 4/5 criterios. El mejor macro-predictor.**
- MI = 0.41, MRMR rank 5, Granger p = 0.0036 a lag 1, VIF = 7.9.
- El único macro variable que pasa Granger con fuerza. Estabilidad solo 33% — eso es una señal de que su relación con el target cambia según el período, probablemente correlacionada con ciclos de tasas.
- Mecanismo causal: cuando la gente busca CDTs en Google, una semana después hay más depósitos en Trii. Lead time de ~1 semana validado estadísticamente.

**`full_users_aprobados` — 4/5 criterios, pero con la advertencia de simultaneidad.**
- Pasa 4 criterios pero Granger falla (p = 0.414). Aplica la misma lógica: valioso si se usa como t-1 (usuarios aprobados la semana pasada → depósitos esta semana).

---

## Los 7 en "REVISAR" — qué hacer con cada uno

**`usuarios_registro_base` y `step_09_full_account`**: misma lógica que `full_users_aprobados`. Usarlos lagueados o descartar. La ventaja de `step_09` sobre `usuarios_registro_base` es que ya filtra a quienes llegaron más lejos en el funnel — si hay que elegir uno de los dos, `step_09`.

**`Tasa_Intervencion_Mensual`**: VIF = 29.58 (inflado por colinealidad con `trends_cdt` y `spread_tes_banrep` que capturan señales similares), Granger falla a lag 1-3. **Pero esto no la descarta** — es una variable mensual. El efecto de una subida del Banrep tarda 4-8 semanas en traducirse en comportamiento de depósito. El test de Granger con max_lag=3 semanas la subestima sistemáticamente. Necesita un test con max_lag=8-12 semanas para evaluarse correctamente.

**`tasa_basic_a_risk`**: Granger p = 0.038 (significativo), pero VIF = 62.92 y estabilidad 33%. El VIF altísimo la hace estadísticamente inestable en presencia de otras features del funnel. Es candidata a quedarse solo si se elimina la mayoría de las otras variables de conversión.

**`tasa_risk_a_fulldata`**: VIF = 10.7, Granger p = 0.071 (marginal), estabilidad 100%. Aporta información sobre si los usuarios que pasan KYC básico siguen avanzando. Interesante pero redundante con `tasa_fulldata_a_video`.

**`pct_perfil_arriesgado`**: Granger p = 0.060 (borde del umbral), estabilidad 33%. Es el opuesto de `pct_perfil_conservador`. El hecho de que pase Granger a 0.060 en lugar de 0.003 sugiere que el efecto real es capturado mejor por el conservador. Se pueden explorar ambos o solo uno.

---

## Lo que definitivamente sale

**`TRM`, `sp500_cambio_semanal_pct`, `brent_cambio_semanal_pct`**: MI cercano a 0, Granger > 0.4 en todos los lags, estabilidad 0%. Sin señal en ninguna dimensión. El F3 de PCA (mercados globales) no tiene relación con primer depósito en Trii. Esto tiene sentido económico: un usuario de clase media colombiana que va a hacer su primer depósito en CDTs no monitorea el petróleo ni el S&P.

**`colcap_cambio_semanal_pct` y `trends_acciones`**: Granger p significativo (0.007 y 0.043) pero únicamente a lag 2-3 y estabilidad 0%. Señal débil y temporalmente inestable. COLCAP a lag 3 puede ser una correlación espuria — el mercado accionario colombiano sube en las mismas condiciones macro que favorecen el ahorro, pero no es un predictor directo del depósito.

**`tasa_video_a_review`**: eliminada correctamente por NZV. CV = 0.0088 — casi nadie que ve el video deja de ir a revisión. No tiene poder discriminatorio.

**`tasa_registro_a_aprobado`**: VIF = 38.64, estabilidad 33%, Granger p = 0.118. Es un feature derivado (producto de todas las tasas intermedias), captura lo mismo que `tasa_fulldata_a_video` pero con más ruido.

**`spread_tes_banrep`**: el resultado más curioso. MI = 0.559 (6to más alto), estabilidad 100%, pero VIF = 14.76 y Granger falla (p = 0.184). Es un proxy de `Tasa_Intervencion_Mensual` — ambas capturan el spread de tasas de interés en Colombia. El spread tiene VIF alto porque está definicionalmente relacionado con la Tasa Banrep. Si entra `Tasa_Intervencion_Mensual`, `spread_tes_banrep` sale.

---

## El caso de `is_ventana_quincena` — el bug más importante del análisis

Los datos de `cicloUsuario.md` muestran un efecto de **41% de diferencia** entre días pico y valle de quincena. El análisis Granger dice p = 0.763 — sin señal. Esto no es una contradicción — es un problema de granularidad.

`cicloUsuario.md` mide el efecto a nivel **diario** (día 28 vs día 11 del mes). `is_ventana_quincena` es una variable **binaria semanal** que solo dice "esta semana cae en quincena o no". Una semana del 24 al 30 puede tener 3 días de quincena y 4 días normales — la variable marca 1, pero el efecto real es diluido.

**El feature está mal construido, no es una señal débil.** La variable correcta es `pct_dias_quincena_en_semana` — qué proporción de los 7 días de esa semana caen dentro de la ventana post-quincena (día 1-3 después del 15 o del fin de mes). Eso capturaría el 41% de diferencia que existe en los datos.

Esta es la re-ingeniería más urgente del dataset.

---

## El `mediana_dias_registro_a_full` — la sorpresa del MRMR

MI = 0.147 (rank 17 de 21 — parece irrelevante). MRMR lo sube al **rank 2** — segunda variable más informativa después de eliminar redundancias. Granger p = 0.546 y estabilidad 0%.

¿Qué está pasando? MRMR dice: una vez que controlás por la señal de volumen (step_09, usuarios_registro), esta variable agrega información independiente. Es decir: no solo importa CUÁNTOS usuarios están en el pipeline, sino qué tan rápido avanzan.

Pero Granger falla — no es un predictor temporal, es una señal contemporánea. Y la estabilidad 0% dice que esa relación cambia completamente según el período.

**Diagnóstico**: esta variable captura calidad de experiencia del usuario (¿cuántos días tarda en completar su perfil?). Cuando el onboarding está fluido, los usuarios depositan más. Pero es un indicador coincidente, no adelantado. Para usarla como predictor se necesita lagearla también.

---

## Diagnóstico conectado con el goal.md

El insight de goal.md dice: *"El modelo no es un selector de campañas. Es un detector de condiciones históricas que produjo conversion."*

Este análisis valida exactamente eso, y agrega una dimensión más precisa: **las condiciones históricas que producen depósito se dividen en dos capas temporales distintas**.

**Capa 1 — Señales de la semana pasada (predictoras genuinas):**
- `tasa_fulldata_a_video` de t-1: ¿el embudo estuvo fluido la semana pasada?
- `pct_perfil_conservador` de t-1: ¿qué tipo de usuario aprobó la semana pasada?
- `trends_cdt` de t-1: ¿hubo búsqueda de intención inversora?
- `full_users_aprobados` de t-1: ¿hay masa crítica lista para depositar?

**Capa 2 — Señales de fondo (contexto de mercado, cambian lento):**
- `Tasa_Intervencion_Mensual`: régimen de tasas actual (lag 4-8 semanas para capturar bien)
- `pct_dias_quincena_en_semana` (re-ingeniería pendiente): timing de liquidez

La pregunta del goal.md — *"¿qué combinación de condiciones externas e internas históricamente produjo más conversiones?"* — tiene una respuesta estadísticamente validada ahora:

> Semanas donde `tasa_fulldata_a_video` t-1 estaba por encima de su media, `pct_perfil_conservador` t-1 era alto, `trends_cdt` t-1 era alto y la semana cae en ventana post-quincena → históricamente son las semanas de mayor depósito.

Esa es la lógica que el modelo de la Fase 2 (el agente de estrategia) debería ejecutar en Customer.io.

---

## Recomendación de features por tier para el modelo v2

**Tier 1 — Incluir con alta confianza (features en tiempo t-1):**
1. `tasa_fulldata_a_video` (lagueada)
2. `pct_perfil_conservador` (lagueada)
3. `trends_cdt` (ya es t-1 por naturaleza — las búsquedas de esta semana predicen depósitos la próxima)
4. `full_users_aprobados` (lagueada)
5- `TRM`, `sp500_cambio_semanal_pct`, `brent_cambio_semanal_pct`, `colcap_cambio_semanal_pct`

**Tier 2 — Incluir con validación adicional:**
5. `step_09_full_account` (lagueada, o usar como alternativa a `full_users_aprobados` — no ambas)
6. `Tasa_Intervencion_Mensual` (requiere Granger con max_lag=8 antes de decidir)
7. `pct_perfil_arriesgado` (validar si agrega info sobre `pct_perfil_conservador`)

**Tier 3 — Re-ingeniería antes de usar:**
8. `pct_dias_quincena_en_semana` (reemplazar `is_ventana_quincena`)
9. `mediana_dias_registro_a_full` (lagueada, solo si el Granger con lag también mejora)



## Los 3 próximos pasos técnicos concretos

1. **Re-ingeniería de `is_ventana_quincena`**: calcular `pct_dias_quincena_en_semana` con los datos del ciclo de usuario. Un solo script con pandas sobre las fechas del master.

2. **Agregar el lag a las features de volumen**: en `patch_master.py` o en el pipeline, agregar columnas `_lag1` para los features de Tier 1 y 2 y re-correr el análisis de Granger con esas versiones. Es posible que `full_users_aprobados_lag1` pase Granger perfectamente.

3. **Granger extendido para `Tasa_Intervencion_Mensual`**: correr el test con `maxlag=8` para capturar el efecto mensual. Si pasa a lag 4-6, es candidata fuerte para el modelo.

---

## Estado y decisión (actualizado junio 2026)

Este análisis cubrió M3 (`usuarios_primer_cashin`) — el único modelo que terminó en producción.

Los pasos 1 y 2 de la lista de arriba **están hechos**: `pct_dias_quincena` reemplaza `is_ventana_quincena` en el master v2, y las features de volumen usan lag o weighted pipeline en el feature engineering. El modelo v14 incorpora estos fixes y está en producción.

El análisis para M2 (`tasa_fulldata_a_video`) se hizo por separado y concluyó que **no se construye modelo**: los datos tienen 242/315 filas en exactamente 100% (quiebre de régimen 2025) y la campaña KYC no tiene espacio de acción suficiente para que un modelo prescriba variaciones. La campaña recibe Tier 1 (orquestación contextual sin modelo). Ver diagnóstico completo en BRIEF.md.
