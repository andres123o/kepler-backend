# Arquitectura de Agentes — Kepler

**Versión:** post-diagnóstico funnel (junio 2026)
**Principio:** un agente dedicado por campaña + un agente de métricas del funnel completo.

---

## La pregunta central: ¿por qué un agente por campaña?

El sistema actual tiene un agente unificado que procesa todas las campañas en secuencia. Funciona para Trii, pero no escala a clientes externos ni captura la diferencia fundamental entre campañas: el objetivo de cada una, las palancas disponibles, y la señal que la mueve son distintos.

Una campaña de KYC no se optimiza igual que una de primer depósito. El agente de primer depósito tiene acceso al modelo + SHAP y debe razonar sobre sentimiento de mercado. El agente de KYC solo necesita adaptar timing y copy al contexto de la semana. Mezclarlos en un único prompt genera instrucciones contradictorias y alucinaciones.

**Regla fija:** un agente por campaña. La señal y las palancas de cada uno se definen por su tier.

---

## Los tres tipos de agente

### Tipo A — Agente Premium (campaña con modelo + SHAP)
**Aplica a:** las campañas que pasan el diagnóstico de tier (hoy en Trii: campaña Primer Depósito).

**Input por ciclo semanal:**
- Proyección XGBoost del próximo ciclo (número y confianza)
- SHAP top features con z-scores, valores actuales vs media 12w, y contribución en depósitos
- Estructura completa del journey en Customer.io (nodos, delays, copy actual) — via fly API
- Knowledge base de productos Trii + compliance
- Contexto de eventos CIO del paso (trigger events, goal events, atributos disponibles)
- **Research de contexto real** (ver sección "El stack de investigación" abajo)

**Output:**
- Copy específico por nodo (subject, preheader, body) con Liquid personalizado
- Recomendación de timing y cadencia
- Rationale basado en la señal dominante de la semana

**System prompt (framework invariable):**

El system prompt del agente premium es el "contrato conceptual" que nunca cambia — define cómo interpretar y operar el output del modelo. Se completa con el contexto dinámico de cada semana en el user message.

```
Eres el motor de inteligencia de comunicación para la campaña [NOMBRE_CAMPAÑA] de [CLIENTE].

# La distinción fundamental

Existen dos clases de variables. Confundirlas es el error de categoría que este sistema evita:

- Condiciones que observas pero no fijas (exógenas): COLCAP, TRM, spread, festivos, quincena,
  lluvia, mix de usuario. No las controlas. Solo puedes adaptarte a ellas.
- Acciones que controlas y sí fijas: timing del envío, copy, estructura, cadencia, canal.
  Estas son las únicas que puedes cambiar.

El verbo correcto del sistema es "adaptarse", no "replicar". El modelo no te dice "recrea las
condiciones de la semana X". Te dice "esta semana domina la condición Z; adapta tus acciones a Z".

# El pipeline mecánico (lo que recibes cada semana)

1. PROYECCIÓN — XGBoost proyecta el target del próximo ciclo. Responde: ¿cuánto?
2. SHAP — atribuye la proyección a cada feature. Responde: ¿jalado por qué?
3. RESEARCH — contexto real de por qué se movió la señal dominante esta semana.
4. TU ROL — interpretar SHAP + research y adaptar las palancas controlables del journey.

# La interpretación correcta (y la prohibida)

PROHIBIDA: "SHAP dice que COLCAP aportó +250 depósitos → COLCAP causa depósitos → replico
las condiciones de COLCAP." Doble error: COLCAP es exógeno (no lo replicas) y la atribución
SHAP es asociacional (describe el peso interno del modelo, no un efecto causal en el mundo).

CORRECTA: "SHAP dice que COLCAP es la condición dominante esta semana. Eso es un puntero,
no un veredicto. Me dice: investiga qué pasa con el COLCAP ahora, por qué se movió, qué
significa para el mercado y para la cabeza del usuario — y adapta el tema/timing/estructura
a ese contexto real."

El SHAP rutea atención, no prueba causalidad. Su trabajo es decirle al agente dónde mirar.
La causalidad la aporta el agente con el contexto del mundo real.

# Por qué la misma señal cambia de significado cada ciclo

"COLCAP +3%" puede significar cosas opuestas en dos semanas distintas según la causa real
(rebote técnico vs. noticia estructural vs. flujo externo). Un motor de replicación sería
incorrecto incluso en principio. Solo un motor de adaptación con razonamiento fresco cada
ciclo maneja esto correctamente.

# Tu objetivo en cada ciclo

Adaptar las palancas controlables (timing, copy, cadencia, canal) a la condición dominante
del próximo ciclo, para convertir más y más rápido siendo contextualmente relevante —
encontrar al usuario donde las condiciones actuales lo ponen.

Lo que NO es tu objetivo: replicar condiciones pasadas ni probar causalidad. Solo adaptación
impulsada por relevancia.

# Reglas de operación

- Nunca actúes mecánicamente sobre el ranking SHAP. Una feature con peso SHAP alto puede
  ser exógena, correlacional, o inestable a este n. El peso SHAP es el puntero, el research
  es la validación.
- Siempre ancla cada recomendación a qué acción específica del journey propones cambiar
  (qué nodo, qué delay, qué copy) y por qué la señal actual justifica ese cambio.
- Usa el knowledge base y el compliance para asegurarte de que el copy generado es legal
  y coherente con la marca.
- Si la señal dominante es exógena y no tiene correlato accionable claro, propón mantener
  el copy actual y ajustar solo el timing. No fuerces recomendaciones cuando la señal no
  sugiere ninguna acción clara.
```

---

### Tipo B — Agente Orquestación (campaña sin modelo)
**Aplica a:** las 5 campañas del funnel que no pasan el diagnóstico de tier (datos básicos, perfil de riesgo, datos completos, validación cuenta, KYC).

**Input por ciclo semanal:**
- Contexto de la semana: festivos, quincena, resumen macro (solo TRM y si hay movimiento relevante en el mercado)
- Estructura del journey en Customer.io (via fly API)
- Knowledge base de productos Trii + compliance
- Métricas del paso (delivery rate, conversion rate, open rate)

**Output:**
- Copy ajustado al contexto de la semana
- Recomendación de timing si aplica
- Sin análisis SHAP ni proyección — solo adaptación contextual

**Diferencia con el agente premium:** no tiene acceso al modelo ni al SHAP. El system prompt no incluye el framework de interpretación técnica. Es un agente de copywriting sensible al contexto, no un agente de inteligencia de mercado.

**Nota sobre KYC específicamente:** la campaña de KYC es un recordatorio transaccional de acción única. El agente adapta el tono según el día de la semana (finde = más tiempo disponible, quincena = momento de completar trámites pendientes) pero no tiene espacio de acción amplio. El output aquí es mínimo por diseño.

---

### Tipo C — Agente de Métricas del Funnel (nuevo)
**Aplica a:** todo el funnel completo, sin distinción de campaña.
**Frecuencia:** una vez por semana, después de correr todos los agentes de campaña.

**Input:**
- Output de `get_funnel_health()` — ya existe, devuelve semáforo + métricas por paso
- Proyección del modelo (para contexto de cuántos usuarios se esperan esta semana)

**Output:**
- Narrativa corta del estado del funnel (2-3 párrafos)
- Alertas si algún paso tiene semáforo rojo/amarillo
- Comparación vs semana anterior si hay datos

**Por qué es fácil de implementar:** `get_funnel_health()` ya existe en `strategy_agent.py` y devuelve toda la información estructurada. Este agente es literalmente una llamada a Claude con ese output como input. ~20 líneas de código adicionales. **Se recomienda implementar.**

---

## El stack de investigación para el agente premium

La pregunta del domingo para el agente premium siempre es la misma: el modelo dice que la señal dominante esta semana es X (ej: COLCAP subió 3% con z=3.2) — ¿qué está pasando realmente con X en el mercado colombiano esta semana?

Para responder eso, el agente necesita acceso a internet en tiempo real. Hay dos opciones:

### Opción A — Perplexity API (recomendada) + Claude

**Por qué Perplexity primero:**
- Diseñado específicamente para búsqueda web en tiempo real + síntesis
- Retorna fuentes citadas (URLs verificables) — valioso para mostrar al usuario en la UI
- Mucho más barato que Claude para tareas de búsqueda pura (~$0.001 por query)
- Maneja español perfectamente y entiende contexto financiero latinoamericano
- Modelos: `sonar` (rápido), `sonar-pro` (mejor calidad para este caso de uso)
- API formato OpenAI-compatible — trivial de integrar

**El flujo con Perplexity:**
```
1. SHAP señal dominante → construir query enfocada para Perplexity
   Ej: "¿Por qué subió el COLCAP [X]% esta semana? Noticias relevantes mercado financiero
       colombiano. BanRep, sectores, Ecopetrol, CDT, tendencias ahorro. Fuentes confiables."

2. Perplexity sonar-pro → research_summary (texto + URLs de fuentes)

3. Claude Sonnet → recibe: SHAP + projection + research_summary + KB + journey
   → genera: copy optimizado + timing + rationale

4. Kepler UI muestra: la estrategia + las fuentes citadas del research
```

**Nuevo servicio a crear:** `kepler-backend/app/services/perplexity_client.py`
```python
# ~30 líneas — llama a api.perplexity.ai/chat/completions con sonar-pro
# Input: query string (construida desde la señal SHAP dominante)
# Output: {summary: str, citations: list[str]}
```

**Nueva variable de entorno necesaria:** `PERPLEXITY_API_KEY`

**Costo estimado:** ~$0.001-0.003 por ciclo semanal (la query de research es ~1,000 tokens)

---

### Opción B — Claude con web_search tool (alternativa)

Anthropic introdujo en 2025 un tool nativo `web_search_20250305` disponible en la API:
```python
tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
```

Esto permite que Claude busque en internet directamente, sin una API externa.

**Ventajas:** un solo API, arquitectura más simple, Claude decide qué buscar.
**Desventajas:** más caro para la tarea de búsqueda pura; el tool de búsqueda nativo de Claude puede ser menos comprehensivo que Perplexity en contexto latinoamericano financiero; no retorna URLs citadas de forma estructurada.

**Veredicto:** Válida para el prototipo. Perplexity es la opción correcta para escala y calidad. Para el primer cliente, usar Claude con web_search acelera el desarrollo (un API, no dos).

---

## Flujo completo de un ciclo semanal

```
DOMINGO
─────────────────────────────────────────────────────────────────────────
1. Auto-fetch variables macro (⚡ en /app/ingresar)
   → TRM, COLCAP, S&P500, Brent, spread, trends, festivos

2. Guardar semana en Supabase → correr modelo → predicción + SHAP guardados

LUNES (al inicio del ciclo)
─────────────────────────────────────────────────────────────────────────
3. /api/strategy/sync → sincronizar campañas CIO → cache Supabase actualizado

4. [AGENTE FUNNEL] → get_funnel_health() → Claude → narrativa salud del funnel
   Output: resumen de qué pasos están en verde/amarillo/rojo esta semana

5. Para CADA CAMPAÑA en paralelo o secuencial:

   [CAMPAÑA PRIMER DEPÓSITO — Tipo A Premium]
   a. Leer SHAP + proyección desde Supabase
   b. Identificar señal dominante (top feature por z-score y SHAP contribution)
   c. Perplexity query: "¿Qué pasa con [señal] en Colombia esta semana?"
   d. Claude Sonnet: system_prompt_premium + KB + SHAP + research + journey
   e. Output: propuesta de copy + timing para este journey

   [CAMPAÑAS KYC, VALIDACIÓN, ETC — Tipo B Básico]
   a. Leer contexto de semana (festivos, quincena, día de semana)
   b. Leer journey de CIO via fly API
   c. Claude Sonnet: system_prompt_basico + contexto + journey
   d. Output: copy ajustado al contexto

6. Todo el output → canvas de revisión en /app/estrategia
   El usuario revisa y aprueba nodo por nodo (flujo actual, sin cambios)

7. /api/strategy/update-node → ejecuta los nodos aprobados en CIO
─────────────────────────────────────────────────────────────────────────
```

---

## Qué cambia vs la arquitectura actual

| | Actual | Nueva |
|---|---|---|
| Número de llamadas Claude | 2-3 (fase 1 unificada + fase 2 + fase 2B) | N campañas + 1 funnel |
| Contexto por agente | Todas las campañas mezcladas | Solo la campaña específica |
| Research de mercado | No existe — Claude recibe solo texto plano de SHAP | Perplexity query automática desde señal SHAP |
| System prompt | Genérico para todas las campañas | Diferenciado por tipo (premium / básico) |
| Agente de funnel | No existe | Claude sobre get_funnel_health() |
| Costo por ciclo | ~$0.20-0.40 Claude | ~$0.30-0.60 Claude + $0.003 Perplexity |

---

## Estado de implementación

| Componente | Estado | Notas |
|---|---|---|
| Agente unificado actual | ✅ Producción | Funciona, es la base del refactor |
| `get_funnel_health()` | ✅ Existe | Base del agente Tipo C |
| Agente Tipo A (premium per-campaign) | 🔜 Pendiente | Refactor del agente actual |
| Agente Tipo B (básico per-campaign) | 🔜 Pendiente | Simplificación del Tipo A |
| Agente Tipo C (funnel metrics) | 🔜 Pendiente | ~20 líneas sobre get_funnel_health() |
| `perplexity_client.py` | 🔜 Pendiente | ~30 líneas, nueva var de entorno |
| System prompts por tipo | 🔜 Pendiente | Documentados arriba, falta implementar |

**Prioridad de implementación:**
1. Agente Tipo C (funnel metrics) — el más fácil, máximo impacto en presentaciones a clientes
2. `perplexity_client.py` — pequeño servicio, añade la capa de research real
3. Agente Tipo A con Perplexity integrado — el upgrade más valioso del sistema
4. Agente Tipo B — refactor del agente actual aplicando system prompt correcto
