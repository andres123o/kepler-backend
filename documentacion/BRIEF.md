# Kepler v2 — El modelo que realmente acciona

## El problema que descubrí esta semana

Durante la primera semana en producción, explicando el sistema al equipo y poniéndolo a funcionar, me di cuenta de algo que no había visto antes: el modelo y el SHAP no estaban accionando nada real.

Lo que hacíamos era: el modelo proyecta → el SHAP dice qué variables importan → el agente prioriza 1-2 campañas → esas campañas reciben copies actualizados. Fin.

El problema es que "priorizar una campaña" en la práctica solo cambiaba un color en la UI y el copy de esa campaña. Las otras cuatro campañas también recibían copies con el mismo contexto de mercado. No había ninguna decisión estructural diferente. La etiqueta "ALTA PRIORIDAD" no cambiaba nada real en Customer.io.

En resumen: teníamos el activo más poderoso (un modelo ML entrenado con datos de Trii desde 2021 que entiende patrones históricos) y lo único que hacía era cambiar un color en pantalla.

---

## El descubrimiento — lo que el modelo debería hacer

El modelo no es un selector de campañas. Es un detector de condiciones históricas que produjeron depósitos.

La pregunta correcta no es "¿qué campaña priorizo esta semana?" sino:

> **"¿Qué combinación de condiciones externas e internas históricamente produjo más conversiones en el funnel — y cómo replico activamente esas condiciones a través de timing, copies y estructura de entrega en TODAS las campañas?"**

Un ejemplo concreto de cómo se vería esto:

**Antes (sistema actual):**
- Modelo dice "C6 es ALTA prioridad"
- Se actualizan copies de C6
- Fin

**Después (lo que queremos):**
- Modelo detecta: semana post-quincena + COLCAP subió 2.3%
- Históricamente esa combinación generó +18% depósitos en perfil arriesgado
- SHAP confirma: el COLCAP es el factor dominante esta semana
- Busca noticias del por qué subió el COLCAP
- Todas las campañas reciben esa señal en sus copies — pero cada una con el argumento que corresponde a SU paso del funnel (no todas el mismo)
- ADEMÁS: C6 recibe recomendación de frecuencia: 1 push → 2 pushes (martes 2pm + 8pm)
- ADEMÁS: C2 recibe el argumento específico que históricamente mueve el paso de KYC

El modelo sigue siendo semanal. Lo que cambia es la calidad y especificidad del output.

---

## Por qué importa — lo que dijo Andrea

La reunión del 3 de junio con Andrea Melo (responsable de Perú) fue la validación más estratégica del sistema hasta ahora. Cinco hallazgos que cambian la perspectiva:

**1. El moat real es la data acumulada, no el software.**
Andrea lo dijo con experiencia de evaluación de startups: cuando compiten un equipo con mejor tecnología versus un equipo con datos históricos desde 2011 con convenio de actualización continua, el de datos gana. El modelo entrenado con datos de Trii desde 2021 que se reentrena semanalmente es el activo más difícil de replicar.

**2. El TAM real no son las fintechs, son los grupos crediticios tradicionales.**
Las fintechs sirven para validar y construir casos de estudio. Los grupos crediticios tienen el ticket, los datos históricos y el presupuesto. La secuencia: fintechs primero para validar, banca tradicional después con esos casos de estudio.

**3. Time to adoption es el mayor asesino de SaaS B2B.**
Si un equipo externo no entiende y usa la herramienta con confianza en menos de dos días, el deal muere. Kepler logró 5 minutos con Camu y 1-2 días con el equipo. Ese número hay que mantenerlo o mejorarlo.

**4. La UI necesita una iteración antes del primer piloto externo.**
Referencia de diseño: Antigravity. No es rediseño, es simplificación de lo que ya existe. Menos info visual = más rápido de entender = más fácil de vender.

**5. Perú entra la semana del 9 de junio.**
Andrea se ofreció como champion de expansión. Funnel de 5 campañas en lugar de 6 (sin paso de Perfil de Riesgo). También ofreció conectar con bancos peruanos para pilotos.

---

## Lo que aprendimos de los datos — confirmaciones que sustentan el modelo nuevo

### Efecto quincena: confirmado con 41% de diferencia
El pico no es el día 15 o 30 (día de pago) — es 1-2 días después. La gente recibe la quincena y actúa al día siguiente. Diferencia entre día pico (3.89%) y día valle (2.76%) = **41% más depósitos solo por timing de liquidez**.

### Ciclo del usuario: más concentrado de lo esperado
- Día 0 (mismo día aprobación): 33.35% de todos los depósitos
- Días 0-2: 54.78% acumulado
- Días 0-7: ~72% acumulado

La mitad de todos los depósitos ocurren en las primeras 48 horas post-aprobación. Hay repuntes en días 14 y 21 — usuarios que coinciden con la siguiente quincena después de su aprobación.

### Efecto día de semana
Martes-Viernes concentran el 72% de todos los depósitos. El push enviado el lunes a las 8-9am genera el depósito del martes. Esto tiene implicación directa en el timing de C6.

---

## Estado actual — Migración completada ✅

Las Fases 0 y 1 del experimento técnico están terminadas. Resumen de lo que se hizo:

**Variables eliminadas (ruidosas / leakage):**
`cx_friccion_kyc`, `cx_bloqueos`, `predeposito_enviados`, `predeposito_convertidos`, `intervencion_kepler`

**Variables nuevas incorporadas al modelo:**
| Variable | Fuente | Qué captura |
|----------|--------|------------|
| `spread_tes_banrep` | TES 10Y − tasa BanRep | Expectativa del mercado sobre tasas futuras |
| `sp500_cambio_semanal_pct` | yfinance | Sentimiento global de mercado |
| `brent_cambio_semanal_pct` | yfinance | Colombia-específico: Ecopetrol ~25% del COLCAP |
| `trends_cdt` | pytrends | Intención de ahorro conservador |
| `trends_acciones` | pytrends | Intención de inversión en equity |
| `pct_dias_quincena` | Calculada | Proporción de días en ventana quincena (reemplaza `is_ventana_quincena`) |
| `festivos_count` | Ley Emiliani + Pascua | Días no hábiles en la semana |
| `indice_lluvia` | z-score 4 ciudades | Proxy de tiempo disponible en casa |

**Modelo en producción:** v14 — MAE WF = 191.95, 315 filas en Supabase, ratio WF/train = 3.20 (warning benigno).

**Auto-fetch activo:** El domingo, `/app/ingresar` → ingresar semana + tasa BanRep → ⚡ → el formulario se llena automáticamente con TRM (promedio 7 días), COLCAP (TradingView Perf.W), S&P500/Brent (yfinance), spread (TES 10Y − BanRep input), trends CDT/acciones (pytrends). El usuario solo revisa y guarda.

---

## Arquitectura del producto — el sistema como funnel inteligente

### El principio de producto
**No se vende un modelo por campaña. Se vende la capa de inteligencia sobre el funnel completo.**

Las 6 campañas (datos básicos → perfil de riesgo → datos completos → validación cuenta → KYC → primer depósito) son un solo funnel de registro a primer depósito. Todas cumplen un rol. La inteligencia del sistema cubre el funnel entero — la diferencia entre tiers es la profundidad del análisis, no la cobertura.

### El diagnóstico como primer paso
Antes de asignar tier a cualquier campaña, el sistema corre un diagnóstico. Una campaña merece la capa predictiva (Tier 2) solo si cumple las cuatro condiciones:

1. **Varianza recurrente del target** — fluctúa de forma repetida, no está pegada a un techo/piso ni dominada por un quiebre de régimen único.
2. **Drivers predecibles** — la varianza correlaciona con señales líderes observables, no con ruido ni con una transición única.
3. **Espacio de acción rico** — hay múltiples temas/timings/estructuras cuya elección óptima depende del driver.
4. **Volume y stakes suficientes** — el target mueve un número que justifica el loop completo.

El diagnóstico en sí es un entregable: demuestra que el sistema distingue dónde el modelo gana su lugar y dónde no.

### Los dos tiers

**Tier 1 — Diagnóstico + Orquestación (todas las campañas del funnel)**
El agente lee el contexto de la semana (festivos, quincena, señales macro, clima) y actualiza copy y timing en todas las campañas con sensibilidad al contexto real. No usa un modelo predictivo — usa el diagnóstico como mapa y el contexto como señal. Precio referencia: $1,000–1,500/mes sobre $3k de Customer.io.

**Tier 2 — + Capa Predictiva (campañas que pasan el diagnóstico)**
Sobre las campañas que califican, activa el loop completo: predicción semanal → SHAP top drivers → agente investiga por qué esa variable domina → prescribe acción específica → mide contra control. Precio referencia: $2,500–3,500/mes all-in, o base + componente ligado a resultado.

La clasificación no es permanente. Una campaña puede subir de tier cuando su régimen cambia. Por eso el diagnóstico es recurrente, no un sort de una vez.

### Estado actual del funnel (Trii, junio 2026)

| Campaña | Target | Varianza | Tier actual | Razón |
|---------|--------|----------|-------------|-------|
| Datos básicos | tasa_basic_a_risk | Baja | Tier 1 | Sin varianza suficiente ni espacio de acción rico |
| Perfil de riesgo | completar perfil | Baja | Tier 1 | Recordatorio transaccional |
| Datos completos | tasa_fulldata | Baja | Tier 1 | Recordatorio transaccional |
| Validación cuenta | tasa_validacion | Baja | Tier 1 | Sin varianza suficiente |
| KYC / Fotos | `tasa_fulldata_a_video` | **Régimen roto** | Tier 1 | Ver diagnóstico abajo |
| Primer depósito | `usuarios_primer_cashin` | Alta | **Tier 2** | Pasa las 4 condiciones. Es el partido entero. |

### Diagnóstico KYC — por qué no lleva modelo

`tasa_fulldata_a_video` tenía varianza (41–62%) hasta 2024. A partir de 2025, 242 de 315 filas son exactamente 100% — el paso se volvió automático o casi universal.

El quiebre de régimen (paso de ~55% a ~100%) corresponde a un cambio en el proceso (probablemente proveedor de verificación, nuevo flujo en la app, o requisito regulatorio). Las ~72 filas con varianza real son las anteriores al quiebre, y sus correlaciones espectaculares (r=0.93+) son confound temporal: todo cambió junto cuando se rompió el paso, no hay causalidad extraíble.

Además, la campaña KYC es un recordatorio transaccional de acción única ("completa tus fotos"). Aunque el modelo dijera perfectamente qué driver pesa, el espacio de acción sigue siendo el mismo copy con pequeñas variaciones. Falla la condición 3 del diagnóstico.

**Conclusión:** KYC recibe Tier 1. El agente adapta timing y copy según festivos, quincena y contexto, pero no se entrena un modelo separado. Si el régimen vuelve a tener varianza recurrente, el diagnóstico lo detecta y sube de tier.

---

## Lo que NO cambia

- El modelo sigue siendo semanal
- El equipo revisa el lunes (o domingo con auto-fetch)
- El flujo humano-en-el-loop se mantiene
- El split 50/50 en Customer.io no se toca
- La medición causal en BigQuery sigue igual

---

## Los tres tracks actuales

| Track | Qué hace | Estado |
|-------|----------|--------|
| **Track A — Producción** | v14 corriendo, equipo revisando lunes, auto-fetch domingo | Activo |
| **Track B — Siguiente reentrenamiento** | Acumular actuals limpios, marcar `es_exogena`, reentrenar cuando haya 4+ semanas limpias | Continuo |
| **Track C — Primer cliente externo** | Conversación de precio con heads de growth fintechs LATAM. Caso de estudio: ~1,514 depósitos adicionales vs control en 7 semanas | Prioritario |
