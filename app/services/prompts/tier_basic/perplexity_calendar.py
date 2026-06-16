"""
Capa de CONTEXTO DE CALENDARIO (variables para el agente básico).

Principio rector: Perplexity resuelve hechos de calendario colombiano verificables
para la semana en curso — festivos, quincena, prima, BanRep, eventos electorales.
NO analiza mercados ni cifras económicas. Eso es competencia del agente premium.

Output: JSON estricto que valida contra CALENDAR_RESEARCH_SCHEMA.
"""

PERPLEXITY_CALENDAR_SYSTEM_PROMPT = """\
Eres un asistente especializado en el calendario colombiano y fechas de relevancia \
económica para el retail. Tu única función es reportar HECHOS VERIFICABLES sobre \
fechas y eventos del período indicado.

REGLAS DURAS:
- Solo reportas hechos de calendario — NO analizas mercados ni cifras económicas.
- Festivos por Ley Emiliani: si el festivo cae en día no lunes, se traslada al lunes \
  siguiente. Reporta la fecha del traslado real, no la fecha original.
- Quincena: período 1-15 y 16-fin de mes. "Aplica" si algún día de los próximos 7 \
  días contiene el día 15 o el último día hábil del mes.
- Prima legal: primera mitad debe pagarse antes del 30 de junio; segunda mitad antes \
  del 20 de diciembre. Solo marca como relevante si estamos en junio o diciembre.
- Si un dato no está confirmado con fuente, marca como null. NUNCA inventes fechas \
  ni nombres de festivos — Colombia tiene festivos fijos y Ley Emiliani bien documentada.
- Responde ÚNICAMENTE con el objeto JSON especificado. Sin prosa, sin markdown.
- Español neutro latinoamericano, sin voseo.
"""


def build_calendar_query(fecha_hoy: str) -> str:
    return f"""\
Fecha de hoy: {fecha_hoy}.
Analiza el período de los próximos 7 días a partir de esta fecha.

1. FESTIVOS COLOMBIANOS
   - Lista todos los festivos en los próximos 7 días.
   - Para cada uno: fecha exacta (YYYY-MM-DD), nombre oficial, si genera puente \
     por Ley Emiliani (el festivo se mueve al lunes siguiente).
   - Fuente preferida: calendario oficial del gobierno de Colombia o Mintrabajo.

2. QUINCENA
   - ¿Algún día de los próximos 7 días es el 15 del mes o el último día hábil del mes?
   - Si aplica: ¿es primera quincena (pago ~día 15) o segunda quincena (pago ~último \
     día hábil)? ¿Cuántos días faltan desde hoy?
   - Considera que los festivos desplazan el último día hábil.

3. PRIMA LEGAL DE SERVICIOS
   - ¿Estamos en junio (prima primer semestre, límite 30 jun) o diciembre \
     (prima segundo semestre, límite 20 dic)?
   - Si aplica: fecha límite exacta y días que faltan desde hoy.

4. REUNIÓN JUNTA DIRECTIVA BANREP
   - ¿Hay reunión de política monetaria del Banco de la República en los próximos 7 días?
   - Si hay: fecha exacta y URL de fuente (agenda oficial banrep.gov.co).

5. EVENTOS ELECTORALES COLOMBIA
   - ¿Hay jornada electoral, consulta o votación oficial programada en los próximos 7 días?
   - Solo eventos con fecha confirmada por la Registraduría u organismo oficial.

6. OTRAS SEÑALES DE CALENDARIO RELEVANTES PARA FINTECH RETAIL
   - Fechas económicas concretas en los próximos 7 días que puedan afectar la \
     disponibilidad de dinero o intención de inversión del retail colombiano. \
     Ejemplos: días sin IVA, cierre de mes bancario, fechas de declaración de renta, \
     vencimientos tributarios de personas naturales.
   - Solo si hay algo verificable. Si no hay nada, retorna array vacío.
"""


# Schema estricto — calendario colombiano sin datos de mercado.
CALENDAR_RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "fecha_consulta": {"type": "string"},
        "festivos_semana": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fecha": {"type": "string"},
                    "nombre": {"type": "string"},
                    "es_puente": {"type": "boolean"},
                    "fuente_url": {"type": ["string", "null"]},
                },
                "required": ["fecha", "nombre", "es_puente"],
            },
        },
        "quincena": {
            "type": "object",
            "properties": {
                "aplica_esta_semana": {"type": "boolean"},
                "tipo": {
                    "anyOf": [
                        {"enum": ["primera_quincena", "segunda_quincena"]},
                        {"type": "null"},
                    ]
                },
                "dias_al_pago": {"type": ["integer", "null"]},
                "razon": {"type": "string"},
            },
            "required": ["aplica_esta_semana", "razon"],
        },
        "prima_legal": {
            "type": "object",
            "properties": {
                "es_mes_de_prima": {"type": "boolean"},
                "semestre": {
                    "anyOf": [
                        {"enum": ["primero", "segundo"]},
                        {"type": "null"},
                    ]
                },
                "fecha_limite_pago": {"type": ["string", "null"]},
                "dias_al_limite": {"type": ["integer", "null"]},
                "razon": {"type": "string"},
            },
            "required": ["es_mes_de_prima", "razon"],
        },
        "reunion_banrep": {
            "type": "object",
            "properties": {
                "hay_reunion_esta_semana": {"type": "boolean"},
                "fecha_reunion": {"type": ["string", "null"]},
                "fuente_url": {"type": ["string", "null"]},
            },
            "required": ["hay_reunion_esta_semana"],
        },
        "eventos_electorales": {
            "type": "object",
            "properties": {
                "hay_evento_esta_semana": {"type": "boolean"},
                "descripcion": {"type": ["string", "null"]},
                "fecha": {"type": ["string", "null"]},
                "fuente_url": {"type": ["string", "null"]},
            },
            "required": ["hay_evento_esta_semana"],
        },
        "otras_senales_calendario": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "descripcion": {"type": "string"},
                    "fecha": {"type": ["string", "null"]},
                    "relevancia_para_fintech": {"type": "string"},
                    "fuente_url": {"type": ["string", "null"]},
                },
                "required": ["descripcion", "relevancia_para_fintech"],
            },
        },
    },
    "required": [
        "fecha_consulta",
        "festivos_semana",
        "quincena",
        "prima_legal",
        "reunion_banrep",
        "eventos_electorales",
        "otras_senales_calendario",
    ],
}


# Parámetros estáticos de la API.
# sonar (no sonar-pro): calendario es búsqueda simple y factual,
# no requiere razonamiento profundo ni contexto extenso.
CALENDAR_API_PARAMS = {
    "model": "sonar",
    "search_recency_filter": "month",  # festivos y eventos se publican con anticipación
    "search_domain_filter": [
        "gov.co",              # portales oficiales del gobierno colombiano
        "banrep.gov.co",       # agenda JD BanRep
        "registraduria.gov.co", # eventos electorales
        "mintrabajo.gov.co",   # prima legal y festivos
        "dian.gov.co",         # vencimientos tributarios
        "larepublica.co",      # días sin IVA, fechas fiscales
        "portafolio.co",
        "-facebook.com",
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {"schema": CALENDAR_RESEARCH_SCHEMA},
    },
    "max_tokens": 2000,   # calendario estructurado — respuesta compacta
    "temperature": 0,     # hechos de calendario: cero variación creativa
}
