"""
Cliente Supabase genérico. Todas las operaciones están encapsuladas en FunnelClient,
que recibe org_slug + funnel_slug y construye los nombres de tabla dinámicamente.

Convención de tablas: {org_slug}_{funnel_slug}_{table}
Ejemplo: trii_activacion_co_master, trii_activacion_co_ultima_semana, ...

Las funciones module-level son wrappers que usan KEPLER_DEFAULT_ORG / KEPLER_DEFAULT_FUNNEL
del .env — mantienen compatibilidad con código Fase 2 mientras se migra incrementalmente.
"""

import math
import os
import re
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import Header, HTTPException
from supabase import create_client, Client


@dataclass
class CIOCredentials:
    """Credenciales de Customer.io por organización — vienen de org_secrets en Supabase."""
    sa_live_key: str      # CIO_SA_LIVE_KEY  (token sa_live_... — solo lectura fly API)
    app_api_key: str      # CIO_APP_API_KEY  (token de escritura App API)
    environment_id: str   # CIO_ENVIRONMENT_ID

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.supabase")

SUPABASE_URL: str = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

NON_ML_COLS = {"es_exogeno"}


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    tp = type(obj).__name__
    if tp in ("float64", "float32", "float16", "float128"):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if tp in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
        return int(obj)
    if tp == "bool_":
        return bool(obj)
    if tp == "ndarray":
        return _make_json_safe(obj.tolist())
    return obj


def _get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY no están configurados en .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _to_ml(df: pd.DataFrame, col_mapping: dict[str, str] | None = None) -> pd.DataFrame:
    df = df.copy()
    for col in NON_ML_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.rename(columns=col_mapping or {})


# ─── Composite de prompts (UI de configuración) ───────────────────────────────
#
# Marketing edita "el prompt completo del agente" como un solo bloque de texto,
# pero internamente son 5 filas distintas de funnel_prompts (system/kb_preamble/
# user_template/perplexity_system/perplexity_query) que van a llamadas de API
# distintas (Claude vs. Perplexity). Los marcadores <!--@prompt:tipo--> permiten
# unirlas en un solo texto para la UI y separarlas de nuevo al guardar, sin perder
# de cuál fila viene cada parte.

_PROMPT_TYPES_ORDER = ["system", "kb_preamble", "user_template", "perplexity_system", "perplexity_query"]

_PROMPT_SECTION_TITLES = {
    "system":             "INSTRUCCIONES GENERALES DEL AGENTE",
    "kb_preamble":        "CÓMO USAR LA KNOWLEDGE BASE",
    "user_template":      "TEMPLATE DE DATOS SEMANALES — NO BORRAR LAS LLAVES { }",
    "perplexity_system":  "INSTRUCCIONES PARA LA BÚSQUEDA DE MERCADO (PERPLEXITY)",
    "perplexity_query":   "QUÉ LE PREGUNTAMOS AL BUSCADOR DE MERCADO",
}

# Variables {...} que el código sustituye en tiempo de ejecución (ver anthropic_client.py
# call_premium_agent / call_basic_agent) — si se borran del texto, el agente deja de recibir
# ese dato sin ningún error visible. Por eso se bloquea el guardado si falta alguna.
_REQUIRED_PLACEHOLDERS = {
    ("premium", "user_template"): ["{semana_label}", "{shap_text}", "{research_text}", "{journey_text}"],
    ("basic",   "user_template"): ["{fecha_hoy}", "{calendar_text}", "{journeys_text}"],
}

_PROMPT_MARKER_RE = re.compile(r"<!--@prompt:(\w+)-->[ \t]*\n")


def _parse_prompt_composite(composite: str) -> dict[str, str]:
    """Separa el texto compuesto de la UI en sus prompt_type según los marcadores."""
    matches = list(_PROMPT_MARKER_RE.finditer(composite))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        prompt_type = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(composite)
        body = composite[start:end]
        body = re.sub(r"^###[^\n]*\n", "", body, count=1)  # línea de título humano, decorativa
        sections[prompt_type] = body.strip()
    return sections


# ─── FunnelClient ─────────────────────────────────────────────────────────────

class FunnelClient:
    """
    Cliente scoped a un funnel específico (org_slug + funnel_slug).
    Todos los nombres de tabla se construyen dinámicamente: {org}_{funnel}_{table}.
    """

    def __init__(self, org_slug: str, funnel_slug: str) -> None:
        self.org_slug    = org_slug
        self.funnel_slug = funnel_slug
        self._prefix     = f"{org_slug}_{funnel_slug}"
        self._client     = _get_client()

    def _t(self, table: str) -> str:
        return f"{self._prefix}_{table}"

    # ── Config del funnel (tabla platform-level) ─────────────────────────────

    def get_funnel_config(self) -> dict[str, Any]:
        """Lee el campo config JSONB de la tabla funnels para este org/funnel."""
        org_res = (
            self._client.table("organizations")
            .select("id")
            .eq("slug", self.org_slug)
            .execute()
        )
        org_rows = org_res.data or []
        if not org_rows:
            return {}
        org_id = org_rows[0]["id"]
        funnel_res = (
            self._client.table("funnels")
            .select("config")
            .eq("org_id", org_id)
            .eq("slug", self.funnel_slug)
            .execute()
        )
        funnel_rows = funnel_res.data or []
        if not funnel_rows:
            return {}
        return funnel_rows[0].get("config") or {}

    def get_validation_rules(self) -> dict[str, Any]:
        """Lee validation_rules del config JSONB. Retorna {} si no existe."""
        return self.get_funnel_config().get("validation_rules") or {}

    def get_ml_column_mapping(self) -> dict[str, str]:
        """Lee ml_column_mapping del config JSONB. Mapea nombres de columnas Supabase → ML pipeline."""
        return self.get_funnel_config().get("ml_column_mapping") or {}

    def get_cio_config(self) -> dict[str, Any]:
        """
        Lee la sección 'cio' del config JSONB del funnel.
        Contiene: trigger_event, conversion_event, cashin_attribute,
                  trigger_to_goal_map, waypoint_step_map.
        Lanza ValueError si la sección no existe (obligatoria para funnels que usan CIO).
        """
        cfg = self.get_funnel_config().get("cio")
        if not cfg:
            raise ValueError(
                f"El funnel '{self.funnel_slug}' no tiene sección 'cio' en su config JSONB. "
                "Agrega la sección 'cio' con trigger_event, conversion_event, "
                "cashin_attribute, trigger_to_goal_map y waypoint_step_map."
            )
        return cfg

    def get_org_secret(self, key_name: str) -> str:
        """
        Lee una credencial desde org_secrets para esta organización.
        Lanza ValueError si no existe — credenciales deben estar en la BD, no en .env.
        """
        res = (
            self._client.table("org_secrets")
            .select("key_value")
            .eq("org_slug", self.org_slug)
            .eq("key_name", key_name)
            .execute()
        )
        rows = res.data or []
        if not rows:
            raise ValueError(
                f"Credencial '{key_name}' no encontrada en org_secrets para org '{self.org_slug}'. "
                "Corre seed_org_secrets.py para migrar las keys a la BD."
            )
        return rows[0]["key_value"]

    def get_cio_credentials(self) -> CIOCredentials:
        """
        Retorna las credenciales de CIO para esta organización desde org_secrets.
        Valida que sa_live_key y app_api_key sean distintas (seguridad dura).
        """
        sa_live_key    = self.get_org_secret("CIO_SA_LIVE_KEY")
        app_api_key    = self.get_org_secret("CIO_APP_API_KEY")
        environment_id = self.get_org_secret("CIO_ENVIRONMENT_ID")

        if sa_live_key == app_api_key:
            raise RuntimeError(
                f"SEGURIDAD CIO [{self.org_slug}]: CIO_SA_LIVE_KEY y CIO_APP_API_KEY "
                "tienen el mismo valor en org_secrets. El token sa_live es de SOLO LECTURA."
            )
        return CIOCredentials(
            sa_live_key=sa_live_key,
            app_api_key=app_api_key,
            environment_id=environment_id,
        )

    def get_agent_prompt(self, agent_type: str, prompt_type: str) -> str | None:
        """
        Lee el contenido de un prompt desde funnel_prompts.
        agent_type: 'premium' | 'basic'
        prompt_type: 'system' | 'perplexity_system' | 'perplexity_query'
        Retorna None si la fila no existe (el caller debe caer en el fallback hardcodeado).
        """
        org_res = (
            self._client.table("organizations")
            .select("id")
            .eq("slug", self.org_slug)
            .execute()
        )
        org_rows = org_res.data or []
        if not org_rows:
            return None
        org_id = org_rows[0]["id"]
        funnel_res = (
            self._client.table("funnels")
            .select("id")
            .eq("org_id", org_id)
            .eq("slug", self.funnel_slug)
            .execute()
        )
        funnel_rows = funnel_res.data or []
        if not funnel_rows:
            return None
        funnel_id = funnel_rows[0]["id"]
        res = (
            self._client.table("funnel_prompts")
            .select("content")
            .eq("funnel_id", funnel_id)
            .eq("agent_type", agent_type)
            .eq("prompt_type", prompt_type)
            .execute()
        )
        rows = res.data or []
        return rows[0]["content"] if rows else None

    def get_perplexity_api_params(self, agent_type: str) -> dict[str, Any]:
        """
        Lee los parámetros de la API Perplexity desde funnel_prompts.api_params.
        agent_type: 'premium' | 'basic'
        Retorna {} si no existe.
        """
        org_res = (
            self._client.table("organizations")
            .select("id")
            .eq("slug", self.org_slug)
            .execute()
        )
        org_rows = org_res.data or []
        if not org_rows:
            return {}
        org_id = org_rows[0]["id"]
        funnel_res = (
            self._client.table("funnels")
            .select("id")
            .eq("org_id", org_id)
            .eq("slug", self.funnel_slug)
            .execute()
        )
        funnel_rows = funnel_res.data or []
        if not funnel_rows:
            return {}
        funnel_id = funnel_rows[0]["id"]
        res = (
            self._client.table("funnel_prompts")
            .select("api_params")
            .eq("funnel_id", funnel_id)
            .eq("agent_type", agent_type)
            .eq("prompt_type", "perplexity_query")
            .execute()
        )
        rows = res.data or []
        return (rows[0].get("api_params") or {}) if rows else {}

    def get_all_prompts(self) -> list[dict[str, Any]]:
        """
        Todas las filas de funnel_prompts del funnel activo — para la UI de
        configuración donde se leen y editan los system prompts de los agentes
        (premium/basic) sin tocar código ni correr seed_prompts.py.
        """
        org_res = self._client.table("organizations").select("id").eq("slug", self.org_slug).execute()
        org_rows = org_res.data or []
        if not org_rows:
            return []
        org_id = org_rows[0]["id"]
        funnel_res = (
            self._client.table("funnels")
            .select("id")
            .eq("org_id", org_id)
            .eq("slug", self.funnel_slug)
            .execute()
        )
        funnel_rows = funnel_res.data or []
        if not funnel_rows:
            return []
        funnel_id = funnel_rows[0]["id"]
        res = (
            self._client.table("funnel_prompts")
            .select("id, agent_type, prompt_type, content, updated_at")
            .eq("funnel_id", funnel_id)
            .order("agent_type")
            .order("prompt_type")
            .execute()
        )
        return res.data or []

    def update_agent_prompt(self, agent_type: str, prompt_type: str, content: str) -> dict[str, Any]:
        """
        Actualiza (o crea si no existe) el contenido de un prompt en funnel_prompts.
        Usado por la UI de configuración.
        """
        org_res = self._client.table("organizations").select("id").eq("slug", self.org_slug).execute()
        org_rows = org_res.data or []
        if not org_rows:
            raise ValueError(f"Organización '{self.org_slug}' no encontrada.")
        org_id = org_rows[0]["id"]
        funnel_res = (
            self._client.table("funnels")
            .select("id")
            .eq("org_id", org_id)
            .eq("slug", self.funnel_slug)
            .execute()
        )
        funnel_rows = funnel_res.data or []
        if not funnel_rows:
            raise ValueError(f"Funnel '{self.funnel_slug}' no encontrado para org '{self.org_slug}'.")
        funnel_id = funnel_rows[0]["id"]
        res = (
            self._client.table("funnel_prompts")
            .upsert(
                {
                    "funnel_id": funnel_id,
                    "agent_type": agent_type,
                    "prompt_type": prompt_type,
                    "content": content,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                on_conflict="funnel_id,agent_type,prompt_type",
            )
            .execute()
        )
        if not res.data:
            raise RuntimeError(f"No se pudo guardar el prompt {agent_type}/{prompt_type}.")
        return res.data[0]

    def get_prompt_composite(self, agent_type: str) -> dict[str, Any]:
        """
        Une las 5 filas de funnel_prompts de un agent_type en un solo texto con
        marcadores <!--@prompt:tipo--> — lo que ve/edita la UI de configuración.
        """
        rows = [r for r in self.get_all_prompts() if r["agent_type"] == agent_type]
        by_type = {r["prompt_type"]: r for r in rows}

        parts: list[str] = []
        for prompt_type in _PROMPT_TYPES_ORDER:
            row = by_type.get(prompt_type)
            content = (row or {}).get("content") or ""
            title = _PROMPT_SECTION_TITLES[prompt_type]
            parts.append(f"<!--@prompt:{prompt_type}-->\n### {title}\n{content}".rstrip())

        updated_ats = [r["updated_at"] for r in rows if r.get("updated_at")]
        return {
            "agent_type": agent_type,
            "composite": "\n\n\n".join(parts) + "\n",
            "updated_at": max(updated_ats) if updated_ats else None,
        }

    def save_prompt_composite(self, agent_type: str, composite: str) -> dict[str, Any]:
        """
        Recibe el texto compuesto editado en la UI, lo separa en sus prompt_type reales,
        valida que ninguna sección quede vacía y que no falte ninguna variable {...}
        obligatoria, y guarda SOLO las secciones que realmente cambiaron — así se puede
        saber qué se modificó aunque el usuario edite todo como un solo bloque.
        """
        sections = _parse_prompt_composite(composite)

        missing_sections = [pt for pt in _PROMPT_TYPES_ORDER if not sections.get(pt, "").strip()]
        if missing_sections:
            titles = ", ".join(_PROMPT_SECTION_TITLES[pt] for pt in missing_sections)
            raise ValueError(
                f"No se puede guardar — estas secciones quedaron vacías: {titles}. "
                "Ninguna sección puede quedar en blanco (si borraste una por error, "
                "cancela y vuelve a abrir para recuperar el contenido)."
            )

        problems: list[str] = []
        for (a_type, p_type), required in _REQUIRED_PLACEHOLDERS.items():
            if a_type != agent_type:
                continue
            text = sections.get(p_type, "")
            missing = [ph for ph in required if ph not in text]
            if missing:
                problems.append(f"a '{_PROMPT_SECTION_TITLES[p_type]}' le falta(n) {', '.join(missing)}")
        if problems:
            raise ValueError(
                "No se puede guardar — " + "; ".join(problems) + ". "
                "Esas variables {...} las necesita el sistema para funcionar, no se pueden borrar."
            )

        current = {r["prompt_type"]: r["content"] for r in self.get_all_prompts() if r["agent_type"] == agent_type}

        updated: list[str] = []
        for prompt_type in _PROMPT_TYPES_ORDER:
            new_content = sections[prompt_type].strip()
            if new_content != (current.get(prompt_type) or "").strip():
                self.update_agent_prompt(agent_type, prompt_type, new_content)
                updated.append(prompt_type)

        return {"agent_type": agent_type, "updated": updated}

    def update_ml_version(self, version: int) -> None:
        """Actualiza ml.model_version en funnels.config para este funnel."""
        org_res = self._client.table("organizations").select("id").eq("slug", self.org_slug).execute()
        org_id  = org_res.data[0]["id"]
        funnel_res = (
            self._client.table("funnels").select("id, config")
            .eq("org_id", org_id).eq("slug", self.funnel_slug).execute()
        )
        row     = funnel_res.data[0]
        config  = dict(row["config"] or {})
        ml      = dict(config.get("ml") or {})
        ml["model_version"] = version
        config["ml"] = ml
        self._client.table("funnels").update({"config": config}).eq("id", row["id"]).execute()
        logger.info("%s/%s: ml.model_version actualizado → v%d", self.org_slug, self.funnel_slug, version)

    # ── Fase 1: datos ML ──────────────────────────────────────────────────────

    def _resolve_master_table(self) -> str:
        """Retorna el nombre de la tabla master para este funnel.
        Prueba {prefix}_master primero; si no existe, usa {prefix}_master_consolidado_final.
        """
        primary = self._t("master")
        try:
            self._client.table(primary).select("*").limit(1).execute()
            return primary
        except Exception:
            return self._t("master_consolidado_final")

    def get_master_df(self) -> pd.DataFrame:
        master_table = self._resolve_master_table()
        all_rows: list[dict] = []
        offset = 0
        while True:
            res = self._client.table(master_table).select("*").range(offset, offset + 999).execute()
            rows = res.data or []
            all_rows.extend(rows)
            logger.info("%s: leídas %d filas (offset %d)", master_table, len(rows), offset)
            if len(rows) < 1000:
                break
            offset += 1000
        df = pd.DataFrame(all_rows)
        logger.info("%s total: %d filas, %d columnas.", master_table, len(df), len(df.columns))
        return _to_ml(df, self.get_ml_column_mapping())

    def get_ultima_semana_row(self) -> dict[str, Any] | None:
        res = self._client.table(self._t("ultima_semana")).select("*").execute()
        rows = res.data or []
        return rows[0] if rows else None

    def get_ultima_semana_df(self) -> pd.DataFrame:
        row = self.get_ultima_semana_row()
        if not row:
            return pd.DataFrame()
        return _to_ml(pd.DataFrame([row]), self.get_ml_column_mapping())

    def save_ultima_semana(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = {k: v for k, v in payload.items() if k != "id"}
        self._client.table(self._t("ultima_semana")).delete().neq("semana", "___never___").execute()
        res = self._client.table(self._t("ultima_semana")).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s actualizada: semana=%s", self._t("ultima_semana"), saved.get("semana"))
        return saved

    def append_to_master(self, payload: dict[str, Any]) -> dict[str, Any]:
        master_table = self._resolve_master_table()
        row = {k: v for k, v in payload.items() if k != "id"}
        res = self._client.table(master_table).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s: fila insertada semana=%s", master_table, saved.get("semana"))
        return saved

    def clear_ultima_semana(self) -> None:
        self._client.table(self._t("ultima_semana")).delete().neq("semana", "___never___").execute()
        logger.info("%s vaciada", self._t("ultima_semana"))

    def save_prediction_result(self, result: dict[str, Any], semana_label: str | None = None) -> dict[str, Any]:
        safe = _make_json_safe(result)
        row = {
            "semana_datos":   safe.get("semana_datos"),
            "semana_label":   semana_label or safe.get("semana_label"),
            "prediccion":     safe.get("prediccion_siguiente_semana"),
            "baseline_12w":   safe.get("baseline_12w"),
            "brecha":         safe.get("brecha_vs_baseline"),
            "mae_modelo":     safe.get("mae_modelo"),
            "modelo_version": safe.get("modelo_version"),
            "full_result":    safe,
        }
        res = self._client.table(self._t("prediction_results")).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s: guardada predicción semana=%s", self._t("prediction_results"), row["semana_datos"])
        return saved

    def get_latest_prediction(self) -> dict[str, Any] | None:
        res = (
            self._client.table(self._t("prediction_results"))
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        row = rows[0]
        full = row.get("full_result") or {}
        full["semana_label"] = row.get("semana_label")
        return full

    def get_prediction_history(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("prediction_results"))
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        history = []
        for row in res.data or []:
            full = row.get("full_result") or {}
            full["semana_label"] = row.get("semana_label")
            full["_id"] = row.get("id")
            history.append(full)
        return history

    # ── Fase 2: estrategia ────────────────────────────────────────────────────

    def get_funnel_steps(self) -> list[dict[str, Any]]:
        res = self._client.table(self._t("funnel_steps")).select("*").order("step_order").execute()
        return res.data or []

    def get_campaigns_cache(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("campaigns_cache"))
            .select("*")
            .order("last_synced_at", desc=True)
            .execute()
        )
        return res.data or []

    def upsert_campaigns_cache(self, campaigns: list[dict[str, Any]]) -> int:
        if not campaigns:
            return 0
        res = (
            self._client.table(self._t("campaigns_cache"))
            .upsert(campaigns, on_conflict="cio_campaign_id")
            .execute()
        )
        count = len(res.data or [])
        logger.info("%s: %d campañas upserted", self._t("campaigns_cache"), count)
        return count

    def get_knowledge_base(self, tipo: str | None = None) -> list[dict[str, Any]]:
        q = self._client.table(self._t("knowledge_base")).select("*").eq("activo", True)
        if tipo:
            q = q.eq("tipo", tipo)
        return q.order("tipo").execute().data or []

    def get_all_knowledge_base(self) -> list[dict[str, Any]]:
        return self._client.table(self._t("knowledge_base")).select("*").order("tipo").execute().data or []

    def update_knowledge_base_entry(self, entry_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        res = self._client.table(self._t("knowledge_base")).update(updates).eq("id", entry_id).execute()
        return res.data[0] if res.data else {}

    def insert_knowledge_base_entry(self, tipo: str, titulo: str, contenido: str) -> dict[str, Any]:
        res = self._client.table(self._t("knowledge_base")).insert(
            {"tipo": tipo, "titulo": titulo, "contenido": contenido, "activo": True}
        ).execute()
        return res.data[0] if res.data else {}

    def delete_knowledge_base_entry(self, entry_id: str) -> None:
        self._client.table(self._t("knowledge_base")).delete().eq("id", entry_id).execute()

    def get_funnel_context(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("funnel_context"))
            .select("*")
            .eq("active", True)
            .order("record_type")
            .execute()
        )
        return res.data or []

    def save_strategy_result(self, strategy: dict[str, Any]) -> dict[str, Any]:
        safe = _make_json_safe(strategy)
        row = {
            "semana_label":  safe.get("semana_label"),
            "estado_funnel": safe.get("estado_funnel"),
            "resumen":       safe.get("resumen"),
            "full_result":   safe,
        }
        res = self._client.table(self._t("strategy_results")).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s: guardada semana=%s", self._t("strategy_results"), row["semana_label"])
        return saved

    def get_latest_strategy(self) -> dict[str, Any] | None:
        res = (
            self._client.table(self._t("strategy_results"))
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        row = rows[0]
        full = row.get("full_result") or {}
        full["_id"] = row.get("id")
        full["_created_at"] = row.get("created_at")
        return full

    def get_strategy_by_semana(self, semana_label: str) -> dict[str, Any] | None:
        """Última estrategia guardada para una semana específica (incluye research_cifras/citations)."""
        res = (
            self._client.table(self._t("strategy_results"))
            .select("*")
            .eq("semana_label", semana_label)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        row = rows[0]
        full = row.get("full_result") or {}
        full["_id"] = row.get("id")
        full["_created_at"] = row.get("created_at")
        return full

    def get_strategy_history(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("strategy_results"))
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        history = []
        for row in res.data or []:
            full = row.get("full_result") or {}
            if full.get("_tipo") == "estructural":
                continue
            full["_id"] = row.get("id")
            full["_created_at"] = row.get("created_at")
            history.append(full)
        return history

    def save_structural_result(self, result: dict[str, Any]) -> dict[str, Any]:
        stamped = {**result, "_tipo": "estructural"}
        row = {
            "semana_label":  result.get("semana_label"),
            "estado_funnel": result.get("estado_funnel"),
            "resumen":       result.get("resumen"),
            "full_result":   stamped,
        }
        res = self._client.table(self._t("strategy_results")).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s: guardado estructural semana=%s", self._t("strategy_results"), row["semana_label"])
        return saved

    def get_latest_structural(self) -> dict[str, Any] | None:
        res = (
            self._client.table(self._t("strategy_results"))
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        for row in res.data or []:
            full = row.get("full_result") or {}
            if full.get("_tipo") == "estructural":
                full["_id"] = row.get("id")
                full["_created_at"] = row.get("created_at")
                return full
        return None

    def get_user_campaign(self, user_name: str) -> str | None:
        res = (
            self._client.table(self._t("user_campaign_assignments"))
            .select("campaign_name")
            .eq("user_name", user_name)
            .single()
            .execute()
        )
        return (res.data or {}).get("campaign_name")

    def get_all_assignments(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("user_campaign_assignments"))
            .select("user_name, campaign_name")
            .execute()
        )
        return res.data or []

    def log_node_update(self, user_name: str, campaign_name: str | None, action_id: int, semana_label: str | None = None) -> None:
        try:
            self._client.table(self._t("node_update_log")).insert({
                "user_name":     user_name,
                "campaign_name": campaign_name,
                "action_id":     action_id,
                "semana_label":  semana_label,
            }).execute()
        except Exception as exc:
            logger.warning("log_node_update: no se pudo registrar (user=%s): %s", user_name, exc)

    def get_sent_nodes(self, semana_label: str, after: str | None = None) -> list[int]:
        """Devuelve action_ids enviados a CIO esta semana. Cualquier usuario que envíe
        una campaña la marca como 'listo' para todos — la canvas refleja estado real en CIO."""
        try:
            q = (self._client.table(self._t("node_update_log"))
                 .select("action_id")
                 .eq("semana_label", semana_label))
            if after:
                q = q.gte("created_at", after)
            res = q.execute()
            return [r["action_id"] for r in (res.data or [])]
        except Exception as exc:
            logger.warning("get_sent_nodes: error (semana=%s): %s", semana_label, exc)
            return []

    def get_tracked_campaign_ids(self) -> list[str]:
        res = self._client.table(self._t("campaigns_cache")).select("cio_campaign_id").execute()
        return [row["cio_campaign_id"] for row in (res.data or [])]

    def add_tracked_campaign(self, campaign_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "cio_campaign_id": campaign_id,
            "name": "Pendiente de sync",
            "delivered": 0, "total_sent": 0, "opened": 0, "human_opened": 0,
            "clicked": 0, "converted": 0, "bounced": 0, "undeliverable": 0,
            "delivery_rate": 0.0, "open_rate": 0.0, "conversion_rate": 0.0,
            "undeliverable_rate": 0.0, "metrics_weeks_covered": 0,
            "entries": 0, "delivery_delta": 0, "spike_alert": False,
        }
        res = self._client.table(self._t("campaigns_cache")).upsert(row, on_conflict="cio_campaign_id").execute()
        logger.info("%s: campaña %s agregada", self._t("campaigns_cache"), campaign_id)
        return res.data[0] if res.data else row

    def delete_tracked_campaign(self, campaign_id: str) -> None:
        self._client.table(self._t("campaigns_cache")).delete().eq("cio_campaign_id", campaign_id).execute()

    def save_measurement_snapshot(self, semana_label: str, html_content: str, inicio_semana: str | None = None, fin_semana: str | None = None, model_version: str = "") -> dict[str, Any]:
        row: dict[str, Any] = {"semana_label": semana_label, "html_content": html_content, "model_version": model_version}
        if inicio_semana:
            row["inicio_semana"] = inicio_semana
        if fin_semana:
            row["fin_semana"] = fin_semana
        res = self._client.table(self._t("bq_measurement_snapshots")).insert(row).execute()
        saved = res.data[0] if res.data else row
        logger.info("%s: guardado semana=%s", self._t("bq_measurement_snapshots"), semana_label)
        return saved

    def get_measurement_snapshots(self) -> list[dict[str, Any]]:
        res = (
            self._client.table(self._t("bq_measurement_snapshots"))
            .select("id, created_at, semana_label, inicio_semana, fin_semana, model_version")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []

    def get_measurement_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        res = self._client.table(self._t("bq_measurement_snapshots")).select("*").eq("id", snapshot_id).limit(1).execute()
        rows = res.data or []
        return rows[0] if rows else None

    def get_admin_status(self) -> list[dict[str, Any]]:
        import json
        from collections import defaultdict

        assignments = self.get_all_assignments()

        semana_label: str | None = None
        campaign_node_ids: dict[str, set] = {}

        latest = (
            self._client.table(self._t("strategy_results"))
            .select("semana_label")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data
        if latest:
            semana_label = latest[0].get("semana_label")

        if semana_label:
            all_rows = (
                self._client.table(self._t("strategy_results"))
                .select("full_result")
                .eq("semana_label", semana_label)
                .execute()
            ).data or []

            for row in all_rows:
                raw = row.get("full_result") or {}
                full_result = raw if isinstance(raw, dict) else json.loads(raw)
                for accion in full_result.get("acciones", []):
                    camp_name = (
                        accion.get("campana_existente_nombre")
                        or accion.get("campaña_existente_nombre")
                        or accion.get("step_name")
                        or ""
                    ).lower()
                    if not camp_name:
                        continue
                    for nodo in (accion.get("nodos_completos") or []):
                        if not nodo.get("modificado"):
                            continue
                        nid = nodo.get("id_nodo_cio")
                        if nid is not None:
                            campaign_node_ids.setdefault(camp_name, set()).add(nid)
                        else:
                            campaign_node_ids.setdefault(camp_name, set()).add(f"nombre:{nodo.get('nombre','?')}")

        campaign_totals = {k: len(v) for k, v in campaign_node_ids.items()}

        def _total_for(campaign_name: str) -> int | None:
            c = campaign_name.lower()
            if c in campaign_totals:
                return campaign_totals[c]
            for k, v in campaign_totals.items():
                if c in k or k in c:
                    return v
            return None

        updates_q = (
            self._client.table(self._t("node_update_log"))
            .select("user_name, action_id, campaign_name, updated_at")
            .order("updated_at", desc=True)
        )
        if semana_label:
            updates_q = updates_q.eq("semana_label", semana_label)
        updates = updates_q.execute().data or []

        def _camp_eq(a: str, b: str) -> bool:
            """Fuzzy match de nombres de campaña (tolerante a sufijos como '| keplerv4')."""
            x, y = a.lower(), b.lower()
            return x == y or x in y or y in x

        # Índice por campaña: campaign_lower → list de entries (cualquier sender)
        camp_updates: dict[str, list[dict]] = defaultdict(list)
        for u in updates:
            c = (u.get("campaign_name") or "").lower()
            if c:
                camp_updates[c].append(u)

        def _camp_entries(assigned_campaign: str) -> list[dict]:
            """Entradas de node_update_log para una campaña, enviadas por cualquier usuario."""
            target = assigned_campaign.lower()
            for key, entries in camp_updates.items():
                if _camp_eq(key, target):
                    return entries
            return []

        result = []
        for a in assignments:
            uname    = a["user_name"]
            assigned = a["campaign_name"]
            # Nodos enviados para esta campaña por CUALQUIER usuario
            camp_entries = _camp_entries(assigned)
            nodes_sent    = len({e["action_id"] for e in camp_entries})
            nodes_total   = _total_for(assigned)
            nodes_pending = max(0, nodes_total - nodes_sent) if nodes_total is not None else None
            last_update   = camp_entries[0]["updated_at"] if camp_entries else None
            sent_by       = camp_entries[0]["user_name"] if camp_entries else None

            if nodes_total is not None and nodes_sent >= nodes_total:
                status = "done"
            elif nodes_sent > 0:
                status = "partial"
            else:
                status = "pending"

            result.append({
                "user_name":     uname,
                "campaign_name": assigned,
                "nodes_updated": nodes_sent,
                "nodes_total":   nodes_total,
                "nodes_pending": nodes_pending,
                "semana_label":  semana_label,
                "last_update":   last_update,
                "sent_by":       sent_by,   # quién envió el último nodo
                "status":        status,
            })

        return result


# ─── Validación org+funnel con cache TTL ─────────────────────────────────────
# Evita 2 queries extra a Supabase en cada request sin sacrificar seguridad.
# TTL=5 min: cambios en organizations/funnels se propagan en ≤5 min.

_ORG_FUNNEL_CACHE: dict[str, float] = {}
_CACHE_TTL = 300  # segundos


def _validate_org_funnel(org_slug: str, funnel_slug: str) -> None:
    key = f"{org_slug}:{funnel_slug}"
    now = time.time()
    if key in _ORG_FUNNEL_CACHE and now - _ORG_FUNNEL_CACHE[key] < _CACHE_TTL:
        return

    client = _get_client()

    org_res = client.table("organizations").select("id").eq("slug", org_slug).execute()
    if not org_res.data:
        raise HTTPException(status_code=403, detail=f"Organización '{org_slug}' no válida")

    org_id = org_res.data[0]["id"]
    funnel_res = (
        client.table("funnels")
        .select("id")
        .eq("org_id", org_id)
        .eq("slug", funnel_slug)
        .execute()
    )
    if not funnel_res.data:
        raise HTTPException(
            status_code=403,
            detail=f"Funnel '{funnel_slug}' no pertenece a la organización '{org_slug}'"
        )

    _ORG_FUNNEL_CACHE[key] = now


# ─── FastAPI dependency ───────────────────────────────────────────────────────

def get_funnel_client(
    x_org_slug: str | None = Header(default=None),
    x_funnel_slug: str | None = Header(default=None),
) -> FunnelClient:
    if not x_org_slug or not x_funnel_slug:
        raise HTTPException(
            status_code=400,
            detail="Headers X-Org-Slug y X-Funnel-Slug son requeridos"
        )
    _validate_org_funnel(x_org_slug, x_funnel_slug)
    return FunnelClient(x_org_slug, x_funnel_slug)


# ─── Wrappers de compatibilidad (Fase 2 — migrar incrementalmente) ────────────
# Leen KEPLER_DEFAULT_ORG / KEPLER_DEFAULT_FUNNEL del .env.
# Reemplazar por FunnelClient explícito en cada router cuando se migre.

def _default_fc() -> FunnelClient:
    org    = os.getenv("KEPLER_DEFAULT_ORG")
    funnel = os.getenv("KEPLER_DEFAULT_FUNNEL")
    if not org or not funnel:
        raise RuntimeError(
            "KEPLER_DEFAULT_ORG y KEPLER_DEFAULT_FUNNEL requeridos en .env "
            "(o migrar el caller a FunnelClient explícito)"
        )
    return FunnelClient(org, funnel)


def get_master_df() -> pd.DataFrame:                                          return _default_fc().get_master_df()
def get_ultima_semana_row() -> dict[str, Any] | None:                         return _default_fc().get_ultima_semana_row()
def get_ultima_semana_df() -> pd.DataFrame:                                   return _default_fc().get_ultima_semana_df()
def save_ultima_semana(p: dict) -> dict:                                      return _default_fc().save_ultima_semana(p)
def append_to_master(p: dict) -> dict:                                        return _default_fc().append_to_master(p)
def clear_ultima_semana() -> None:                                            return _default_fc().clear_ultima_semana()
def save_prediction_result(r: dict, sl: str | None = None) -> dict:           return _default_fc().save_prediction_result(r, sl)
def get_latest_prediction() -> dict | None:                                   return _default_fc().get_latest_prediction()
def get_prediction_history() -> list:                                         return _default_fc().get_prediction_history()
def get_funnel_steps() -> list:                                               return _default_fc().get_funnel_steps()
def get_campaigns_cache() -> list:                                            return _default_fc().get_campaigns_cache()
def upsert_campaigns_cache(c: list) -> int:                                   return _default_fc().upsert_campaigns_cache(c)
def get_knowledge_base(tipo: str | None = None) -> list:                      return _default_fc().get_knowledge_base(tipo)
def get_all_knowledge_base() -> list:                                         return _default_fc().get_all_knowledge_base()
def update_knowledge_base_entry(eid: str, u: dict) -> dict:                   return _default_fc().update_knowledge_base_entry(eid, u)
def insert_knowledge_base_entry(t: str, ti: str, c: str) -> dict:             return _default_fc().insert_knowledge_base_entry(t, ti, c)
def delete_knowledge_base_entry(eid: str) -> None:                            return _default_fc().delete_knowledge_base_entry(eid)
def get_funnel_context() -> list:                                             return _default_fc().get_funnel_context()
def save_strategy_result(s: dict) -> dict:                                    return _default_fc().save_strategy_result(s)
def get_latest_strategy() -> dict | None:                                     return _default_fc().get_latest_strategy()
def get_strategy_by_semana(sl: str) -> dict | None:                           return _default_fc().get_strategy_by_semana(sl)
def get_strategy_history() -> list:                                           return _default_fc().get_strategy_history()
def save_structural_result(r: dict) -> dict:                                  return _default_fc().save_structural_result(r)
def get_latest_structural() -> dict | None:                                   return _default_fc().get_latest_structural()
def get_user_campaign(u: str) -> str | None:                                  return _default_fc().get_user_campaign(u)
def get_all_assignments() -> list:                                            return _default_fc().get_all_assignments()
def log_node_update(u: str, c: str | None, a: int, sl: str | None = None):   return _default_fc().log_node_update(u, c, a, sl)
def get_sent_nodes(sl: str, after: str | None = None) -> list:                return _default_fc().get_sent_nodes(sl, after)
def get_tracked_campaign_ids() -> list:                                       return _default_fc().get_tracked_campaign_ids()
def add_tracked_campaign(cid: str) -> dict:                                   return _default_fc().add_tracked_campaign(cid)
def delete_tracked_campaign(cid: str) -> None:                                return _default_fc().delete_tracked_campaign(cid)
def get_admin_status() -> list:                                               return _default_fc().get_admin_status()
def save_measurement_snapshot(sl: str, html: str, i: str | None = None, f: str | None = None, mv: str = "") -> dict:
    return _default_fc().save_measurement_snapshot(sl, html, i, f, mv)
def get_measurement_snapshots() -> list:                                      return _default_fc().get_measurement_snapshots()
def get_measurement_snapshot(sid: str) -> dict | None:                        return _default_fc().get_measurement_snapshot(sid)
