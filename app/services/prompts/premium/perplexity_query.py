"""
Capa de RESEARCH DE MERCADO (Bloque 2) → agente premium.

Principio rector: Perplexity es una capa de HECHOS + EVIDENCIA OBSERVABLE.
NO infiere el estado mental del inversor. NO elige la señal dominante.
Reporta datos sourced y señales de sentimiento OBSERVABLES.
La interpretación (momento mental, conexión con SHAP, copy) es del agente premium.

Output: JSON estricto que valida contra MARKET_RESEARCH_SCHEMA.
Debe acompañarse SIEMPRE de los parámetros de API en build_api_params().
"""

from datetime import date, timedelta

PERPLEXITY_SYSTEM_PROMPT = """\
Eres un analista de datos financieros, noticias relevantes y contexto del mercado colombiano. Tu única función es \
entregar HECHOS VERIFICABLES y EVIDENCIA OBSERVABLE de la semana, con fecha.

REGLAS DURAS:
- NO interpretas el estado psicológico del inversor. NO digas "el inversor está cauteloso/optimista". \
  En su lugar reporta SEÑALES OBSERVABLES con dato y fuente (ej.: "captaciones de CDT +3.2% sem/sem según Asobancaria", \
  "búsquedas Google de 'dólar hoy' en máximo de 30 días"). La interpretación NO es tu trabajo.
- NO priorices ni elijas una variable como "la dominante". Cubre todas con igual rigor.
- Cada dato numérico DEBE incluir: valor, variación semanal, fecha exacta del dato (no de publicación), y URL de fuente.
- Si un dato no está disponible con confianza, marca el campo como null y nivel_confianza="no_disponible". \
  PROHIBIDO estimar, redondear de memoria o inferir un número que no viste en una fuente.
- Prioriza fuentes primarias: BanRep, BVC, DANE, MinHacienda, prensa financiera (La República, Valora Analitik, \
  Bloomberg Línea, Portafolio, etc). Descarta agregadores SEO y foros.
- Distingue siempre: dato observado (con fuente) vs. proyección de analista (con autor citado).
- Responde ÚNICAMENTE con el objeto JSON que valida contra el schema. Sin prosa, sin markdown, sin preámbulo.
- Español neutro latinoamericano, sin voseo.
"""


def build_market_query(semana_label: str) -> str:
    fecha_hoy = date.today().isoformat()  # siempre la fecha real de ejecución
    return f"""\
Fecha de hoy: {fecha_hoy}.
Objetivo: Investigar información relevante para el mercado colombiano para los siguientes 7 días.

1. COLCAP
   - Nivel de cierre y variación % semanal. Mayor causa atribuida (flujos, resultados, macro, contagio) — citada.
   - Volumen negociado vs. semana anterior si está disponible.

2. TRM (COP/USD)
   - Nivel y variación semanal. Drivers citados (Fed, riesgo político local, petróleo, flujos).

3. TASA BANREP + MERCADO DE DEUDA
   - Tasa de intervención vigente y última decisión (subió/bajó/mantuvo + fecha de reunión).
   - Tasa del TES a 10 años (nivel y variación semanal). Calcula el spread TES10Y − tasa BanRep.
   - Rangos vigentes de CDT

4. CONTEXTO GLOBAL
   - S&P 500: variación % semanal + razón principal citada.
   - Brent: nivel y variación semanal + implicación fiscal directa para Colombia (exportador de crudo).

5. NOTICIAS FINANCIERAS COLOMBIA (top 3, esta ventana)
   - Titular + dato concreto. Implicación factual para mercados, sin inferir psicología.

6. EVIDENCIA OBSERVABLE DE APETITO RETAIL (NO conclusiones)
   - Datos verificables: captaciones de ahorro/CDT (Asobancaria/Superfinanciera), tendencias de búsqueda \
     ('invertir', 'CDT', 'comprar dólares', 'acciones Colombia'), reportes de fintech/banca masiva.
   - Comparación explícita vs. semana anterior cuando exista el dato.

Para cada dato: valor + variación + fecha del dato + URL. Si no hay dato confiable, null. No estimes.
"""


# Schema que fuerza estructura y trazabilidad por dato.
MARKET_RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "ventana_semana": {"type": "string"},
        "fecha_generacion": {"type": "string"},
        "contexto_electoral": {
            "type": "object",
            "properties": {
                "dias_a_balotaje": {"type": ["integer", "null"]},
                "reaccion_mercado": {"type": "string"},
                "senales_observables": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/senal"}
                },
                "fuentes": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["reaccion_mercado", "senales_observables"]
        },
        "variables": {
            "type": "object",
            "properties": {
                "colcap": {"$ref": "#/$defs/dato_mercado"},
                "trm": {"$ref": "#/$defs/dato_mercado"},
                "banrep_tasa": {"$ref": "#/$defs/dato_mercado"},
                "tes_10y": {"$ref": "#/$defs/dato_mercado"},
                "spread_tes_banrep": {"$ref": "#/$defs/dato_mercado"},
                "cdt_rangos": {"$ref": "#/$defs/dato_mercado"},
                "sp500": {"$ref": "#/$defs/dato_mercado"},
                "brent": {"$ref": "#/$defs/dato_mercado"}
            },
            "required": ["colcap", "trm", "banrep_tasa", "tes_10y",
                         "spread_tes_banrep", "sp500", "brent"]
        },
        "noticias_top3": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "titular": {"type": "string"},
                    "dato_concreto": {"type": "string"},
                    "fuente_url": {"type": "string"}
                },
                "required": ["titular", "fuente_url"]
            }
        },
        "evidencia_apetito_retail": {
            "type": "array",
            "items": {"$ref": "#/$defs/senal"}
        }
    },
    "required": ["ventana_semana", "variables", "noticias_top3", "evidencia_apetito_retail"],
    "$defs": {
        "dato_mercado": {
            "type": "object",
            "properties": {
                "valor": {"type": ["number", "string", "null"]},
                "variacion_semanal": {"type": ["string", "null"]},
                "causa_principal": {"type": ["string", "null"]},
                "fecha_dato": {"type": ["string", "null"]},
                "fuente_url": {"type": ["string", "null"]},
                "nivel_confianza": {"enum": ["alto", "medio", "bajo", "no_disponible"]}
            },
            "required": ["valor", "nivel_confianza"]
        },
        "senal": {
            "type": "object",
            "properties": {
                "observacion": {"type": "string"},
                "dato": {"type": ["string", "null"]},
                "fuente_url": {"type": ["string", "null"]}
            },
            "required": ["observacion"]
        }
    }
}


# ── Parámetros estáticos de la API ──────────────────────────────────────────
# search_after_date_filter se inyecta dinámicamente en build_api_params().
PERPLEXITY_API_PARAMS = {
    "model": "sonar-pro",          # citations, real-time search, response_format json_schema
    "search_recency_filter": "week",   # impone frescura; sin esto Sonar prioriza autoridad sobre fecha
    "search_domain_filter": [
        # Fuentes primarias CO — oficiales y prensa especializada
        "banrep.gov.co",           # TRM, tasa BanRep, TES, política monetaria
        "bvc.com.co",              # COLCAP oficial, volúmenes de negociación
        "dane.gov.co",             # estadísticas oficiales CO
        "minhacienda.gov.co",      # política fiscal y deuda pública CO
        "larepublica.co",          # principal diario financiero CO
        "valoraanalitik.com",      # análisis financiero especializado CO / LATAM
        "bloomberglinea.com",      # Bloomberg en español: global + LATAM
        "portafolio.co",           # sección financiera de El Tiempo
        "x.com",                   # Twitter / X — sentimiento y contexto de mercado en tiempo real
        # Denegar solo Facebook (ruido sin señal financiera)
        "-facebook.com",
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {"schema": MARKET_RESEARCH_SCHEMA},
    },
    "max_tokens": 6000,  # JSON estructurado con 8 variables + noticias + apetito ≈ 3k tokens.
                          # 6000 da 2× headroom; sonar-pro soporta hasta 8192 output tokens.
    "temperature": 0.1,  # research factual: minimiza variación creativa
}


def build_api_params(fecha_hoy: date) -> dict:
    """
    Retorna los parámetros completos para la API de Perplexity con
    search_after_date_filter anclado al lunes de la semana analizada.

    Lógica de fechas (el modelo siempre corre el domingo):
      Python weekday(): Mon=0 … Sun=6 → restar weekday() da siempre el lunes.
      Ej.: domingo 29 jun (weekday=6) → 29 - 6 = lunes 23 jun → "06/23/2025"
      Ej.: lunes 23 jun  (weekday=0) → 23 - 0 = lunes 23 jun → "06/23/2025"
    """
    lunes = fecha_hoy - timedelta(days=fecha_hoy.weekday())
    lunes_str = lunes.strftime("%m/%d/%Y")  # formato exigido por Perplexity

    return {
        **PERPLEXITY_API_PARAMS,
        "search_after_date_filter": lunes_str,
    }