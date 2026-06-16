"""
Conexión a Supabase y operaciones sobre las dos tablas del proyecto:
  - master_consolidado_final  (historial completo, 310+ filas)
  - ultima_semana             (fila de la semana actual)

Mapeo de columnas:
  Supabase almacena TRM como 'trm' (lowercase).
  El ML pipeline lo espera como 'TRM' (uppercase).
  _to_ml() maneja la conversión automáticamente.
"""

import math
import os
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger("kepler.supabase")

SUPABASE_URL: str = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or ""
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""

# Columnas que existen en Supabase pero NO son features del modelo ML
NON_ML_COLS = {"es_exogeno"}

# Columnas con nombres distintos entre Supabase y el pipeline ML
# Solo TRM: Supabase lo guarda lowercase, el pipeline lo espera uppercase
SUPABASE_TO_ML: dict[str, str] = {
    "trm": "TRM",
}
ML_TO_SUPABASE: dict[str, str] = {v: k for k, v in SUPABASE_TO_ML.items()}


def _make_json_safe(obj: Any) -> Any:
    """
    Convierte recursivamente tipos no-JSON-serializables antes de insertar en Supabase.

    Casos cubiertos:
    - float('nan') / float('inf') → None  (causa más frecuente: SHAP sobre inputs NaN)
    - numpy.float64/32/16 → Python float (o None si nan/inf)
    - numpy.int64/32/16/8 → Python int
    - numpy.bool_ → Python bool
    - numpy.ndarray → list
    - dict / list → aplica recursivamente
    """
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    # numpy types — por nombre para no requerir import numpy aquí
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


def _to_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas de Supabase al formato que espera el ML pipeline."""
    df = df.copy()
    for col in NON_ML_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.rename(columns=SUPABASE_TO_ML)


def get_master_df() -> pd.DataFrame:
    """
    Lee master_consolidado_final completo desde Supabase.
    Pagina de a 1000 filas para traer las 310+ sin truncar.
    Retorna DataFrame con columnas en formato ML pipeline.
    """
    client = _get_client()
    all_rows: list[dict] = []
    offset = 0
    while True:
        res = (
            client.table("master_consolidado_final")
            .select("*")
            .range(offset, offset + 999)
            .execute()
        )
        rows = res.data or []
        all_rows.extend(rows)
        logger.info("master_consolidado_final: leídas %d filas (offset %d)", len(rows), offset)
        if len(rows) < 1000:
            break
        offset += 1000

    df = pd.DataFrame(all_rows)
    logger.info("master_consolidado_final total: %d filas, %d columnas.", len(df), len(df.columns))
    return _to_ml(df)


def get_ultima_semana_row() -> dict[str, Any] | None:
    """
    Lee la fila actual de ultima_semana.
    Retorna dict con columnas en nombres Supabase (lowercase), o None si está vacía.
    """
    client = _get_client()
    res = client.table("ultima_semana").select("*").execute()
    rows = res.data or []
    return rows[0] if rows else None


def get_ultima_semana_df() -> pd.DataFrame:
    """
    Lee ultima_semana y retorna DataFrame con columnas en formato ML pipeline.
    """
    row = get_ultima_semana_row()
    if not row:
        return pd.DataFrame()
    df = pd.DataFrame([row])
    return _to_ml(df)


def save_ultima_semana(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Reemplaza la fila de ultima_semana con los datos nuevos.
    Estrategia: delete all + insert (la tabla siempre tiene una sola fila).
    payload usa nombres de columna Supabase (lowercase para las macro).
    """
    client = _get_client()

    # Limpiar campos que no van a Supabase o son None
    row = {k: v for k, v in payload.items() if k != "id"}

    # Borrar todo y reinsertar
    client.table("ultima_semana").delete().neq("semana", "___never___").execute()
    res = client.table("ultima_semana").insert(row).execute()

    saved = res.data[0] if res.data else row
    logger.info("ultima_semana actualizada: semana=%s", saved.get("semana"))
    return saved


def append_to_master(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Inserta una nueva fila en master_consolidado_final.
    payload usa nombres de columna Supabase (lowercase para las macro).
    """
    client = _get_client()
    row = {k: v for k, v in payload.items() if k != "id"}
    res = client.table("master_consolidado_final").insert(row).execute()
    saved = res.data[0] if res.data else row
    logger.info("master_consolidado_final: fila insertada semana=%s", saved.get("semana"))
    return saved


def clear_ultima_semana() -> None:
    """Borra todas las filas de ultima_semana (deja la tabla vacía para la semana nueva)."""
    client = _get_client()
    client.table("ultima_semana").delete().neq("semana", "___never___").execute()
    logger.info("ultima_semana vaciada")


def save_prediction_result(result: dict[str, Any], semana_label: str | None = None) -> dict[str, Any]:
    """
    Guarda el resultado completo de una predicción en prediction_results.
    Inserta siempre una fila nueva (historial de predicciones).
    _make_json_safe() convierte float('nan')/numpy types antes del insert para
    evitar fallos de serialización cuando algún input feature fue NaN.
    """
    client = _get_client()
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
    res = client.table("prediction_results").insert(row).execute()
    saved = res.data[0] if res.data else row
    logger.info("prediction_results: guardada predicción semana=%s", row["semana_datos"])
    return saved


def get_latest_prediction() -> dict[str, Any] | None:
    """Devuelve la predicción más reciente, o None si no hay."""
    client = _get_client()
    res = (
        client.table("prediction_results")
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


def get_prediction_history() -> list[dict[str, Any]]:
    """
    Devuelve todas las predicciones en orden cronológico descendente.
    Incluye full_result para cada una (el frontend las almacena en memoria).
    """
    client = _get_client()
    res = (
        client.table("prediction_results")
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


# ─── Fase 2: tablas de estrategia ────────────────────────────────────────────

def get_funnel_steps() -> list[dict[str, Any]]:
    """Lee los pasos del funnel ordenados por step_order."""
    client = _get_client()
    res = client.table("funnel_steps").select("*").order("step_order").execute()
    return res.data or []


def get_campaigns_cache() -> list[dict[str, Any]]:
    """Lee las campañas cacheadas de CIO."""
    client = _get_client()
    res = (
        client.table("cio_campaigns_cache")
        .select("*")
        .order("last_synced_at", desc=True)
        .execute()
    )
    return res.data or []


def upsert_campaigns_cache(campaigns: list[dict[str, Any]]) -> int:
    """Inserta o actualiza las campañas del funnel en cio_campaigns_cache."""
    client = _get_client()
    if not campaigns:
        return 0
    res = (
        client.table("cio_campaigns_cache")
        .upsert(campaigns, on_conflict="cio_campaign_id")
        .execute()
    )
    count = len(res.data or [])
    logger.info("cio_campaigns_cache: %d campañas upserted", count)
    return count


def get_knowledge_base(tipo: str | None = None) -> list[dict[str, Any]]:
    """Lee entradas activas del knowledge_base. Filtra por tipo si se especifica."""
    client = _get_client()
    q = client.table("knowledge_base").select("*").eq("activo", True)
    if tipo:
        q = q.eq("tipo", tipo)
    res = q.order("tipo").execute()
    return res.data or []


def get_funnel_context() -> list[dict[str, Any]]:
    """Lee eventos y atributos CIO del funnel de activación desde cio_funnel_context."""
    client = _get_client()
    res = (
        client.table("cio_funnel_context")
        .select("*")
        .eq("active", True)
        .order("record_type")
        .execute()
    )
    return res.data or []


def save_strategy_result(strategy: dict[str, Any]) -> dict[str, Any]:
    """Guarda el resultado de una estrategia generada en strategy_results."""
    client = _get_client()
    safe = _make_json_safe(strategy)
    row = {
        "semana_label":  safe.get("semana_label"),
        "estado_funnel": safe.get("estado_funnel"),
        "resumen":       safe.get("resumen"),
        "full_result":   safe,
    }
    res = client.table("strategy_results").insert(row).execute()
    saved = res.data[0] if res.data else row
    logger.info("strategy_results: guardada semana=%s", row["semana_label"])
    return saved


def get_latest_strategy() -> dict[str, Any] | None:
    """Devuelve la estrategia más reciente, o None si no hay."""
    client = _get_client()
    res = (
        client.table("strategy_results")
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


def get_strategy_history() -> list[dict[str, Any]]:
    """Devuelve todas las estrategias semanales (Phase 2) en orden descendente.
    Excluye resultados estructurales (_tipo=estructural) filtrando en Python
    para no depender del comportamiento de NULL en PostgREST."""
    client = _get_client()
    res = (
        client.table("strategy_results")
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


def save_structural_result(result: dict[str, Any]) -> dict[str, Any]:
    """Guarda el resultado de optimización estructural (Fase 2B) en strategy_results."""
    client = _get_client()
    stamped = {**result, "_tipo": "estructural"}
    row = {
        "semana_label":  result.get("semana_label"),
        "estado_funnel": result.get("estado_funnel"),
        "resumen":       result.get("resumen"),
        "full_result":   stamped,
    }
    res = client.table("strategy_results").insert(row).execute()
    saved = res.data[0] if res.data else row
    logger.info("strategy_results: guardado resultado estructural semana=%s", row["semana_label"])
    return saved


def get_user_campaign(user_name: str) -> str | None:
    """Devuelve el nombre de campaña asignada al usuario, o None si no tiene."""
    client = _get_client()
    res = (
        client.table("user_campaign_assignments")
        .select("campaign_name")
        .eq("user_name", user_name)
        .single()
        .execute()
    )
    return (res.data or {}).get("campaign_name")


def get_all_assignments() -> list[dict[str, Any]]:
    """Devuelve todas las asignaciones usuario → campaña."""
    client = _get_client()
    res = (
        client.table("user_campaign_assignments")
        .select("user_name, campaign_name")
        .execute()
    )
    return res.data or []


def log_node_update(
    user_name: str,
    campaign_name: str | None,
    action_id: int,
    semana_label: str | None = None,
) -> None:
    """Registra que un usuario actualizó un nodo en CIO."""
    try:
        client = _get_client()
        client.table("node_update_log").insert({
            "user_name":     user_name,
            "campaign_name": campaign_name,
            "action_id":     action_id,
            "semana_label":  semana_label,
        }).execute()
    except Exception as exc:
        logger.warning("log_node_update: no se pudo registrar (user=%s, action=%s): %s", user_name, action_id, exc)


def get_sent_nodes(semana_label: str, after: str | None = None) -> list[int]:
    """
    Devuelve los action_ids ya enviados para la semana indicada.
    after: ISO timestamp — solo cuenta nodos enviados a partir de esa fecha/hora.
    Usado para aislar el estado de una estrategia concreta vs. sesiones anteriores.
    """
    try:
        client = _get_client()
        q = client.table("node_update_log").select("action_id").eq("semana_label", semana_label)
        if after:
            q = q.gte("created_at", after)
        res = q.execute()
        return [r["action_id"] for r in (res.data or [])]
    except Exception as exc:
        logger.warning("get_sent_nodes: error consultando log (semana=%s): %s", semana_label, exc)
        return []


def get_admin_status() -> list[dict[str, Any]]:
    """Panel admin: estado de cada agente — nodos enviados, total y pendientes para la semana activa."""
    import json
    from collections import defaultdict
    client = _get_client()

    assignments = (
        client.table("user_campaign_assignments")
        .select("user_name, campaign_name")
        .execute()
    ).data or []

    # Estrategia activa: semana_label + total de nodos modificables (Modo 1 + Modo 2 combinados)
    # Ambos modos se guardan en la misma tabla; se deduplicан por id_nodo_cio para no doblar contar.
    semana_label: str | None = None
    # campaign_name_lower → set de id_nodo_cio únicos modificables
    campaign_node_ids: dict[str, set] = {}

    # Tomar el semana_label de la fila más reciente, luego cargar TODAS las de esa semana
    latest = (
        client.table("strategy_results")
        .select("semana_label")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    ).data
    if latest:
        semana_label = latest[0].get("semana_label")

    if semana_label:
        all_rows = (
            client.table("strategy_results")
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
                        # Sin id — usar nombre como clave de dedup
                        campaign_node_ids.setdefault(camp_name, set()).add(f"nombre:{nodo.get('nombre','?')}")

    campaign_totals: dict[str, int] = {k: len(v) for k, v in campaign_node_ids.items()}

    def _total_for(campaign_name: str) -> int | None:
        c = campaign_name.lower()
        if c in campaign_totals:
            return campaign_totals[c]
        for k, v in campaign_totals.items():
            if c in k or k in c:
                return v
        return None

    # Updates filtrados por la semana activa
    updates_q = (
        client.table("node_update_log")
        .select("user_name, action_id, updated_at")
        .order("updated_at", desc=True)
    )
    if semana_label:
        updates_q = updates_q.eq("semana_label", semana_label)
    updates = updates_q.execute().data or []

    user_updates: dict[str, list[dict]] = defaultdict(list)
    for u in updates:
        user_updates[u["user_name"].lower()].append(u)

    result = []
    for a in assignments:
        uname       = a["user_name"]
        uupdates    = user_updates.get(uname.lower(), [])
        nodes_sent  = len({u["action_id"] for u in uupdates})
        nodes_total = _total_for(a["campaign_name"])
        nodes_pending = max(0, nodes_total - nodes_sent) if nodes_total is not None else None

        if nodes_total is not None and nodes_sent >= nodes_total:
            status = "done"
        elif nodes_sent > 0:
            status = "partial"
        else:
            status = "pending"

        result.append({
            "user_name":     uname,
            "campaign_name": a["campaign_name"],
            "nodes_updated": nodes_sent,
            "nodes_total":   nodes_total,
            "nodes_pending": nodes_pending,
            "semana_label":  semana_label,
            "last_update":   uupdates[0]["updated_at"] if uupdates else None,
            "status":        status,
        })

    return result


def get_latest_structural() -> dict[str, Any] | None:
    """Devuelve el resultado estructural más reciente (Fase 2B), o None si no hay.
    Filtra en Python para consistencia con get_strategy_history."""
    client = _get_client()
    res = (
        client.table("strategy_results")
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


# ─── Configuración: campañas monitoreadas ────────────────────────────────────

def get_tracked_campaign_ids() -> list[str]:
    """IDs de campañas que Kepler monitorea, leídos desde cio_campaigns_cache."""
    client = _get_client()
    res = client.table("cio_campaigns_cache").select("cio_campaign_id").execute()
    return [row["cio_campaign_id"] for row in (res.data or [])]


def add_tracked_campaign(campaign_id: str) -> dict[str, Any]:
    """Inserta una campaña mínima en cio_campaigns_cache. El sync la completa."""
    client = _get_client()
    row: dict[str, Any] = {
        "cio_campaign_id": campaign_id,
        "name": "Pendiente de sync",
        "country": "co",
        "delivered": 0, "total_sent": 0, "opened": 0, "human_opened": 0,
        "clicked": 0, "converted": 0, "bounced": 0, "undeliverable": 0,
        "delivery_rate": 0.0, "open_rate": 0.0, "conversion_rate": 0.0,
        "undeliverable_rate": 0.0, "metrics_weeks_covered": 0,
        "entries": 0, "delivery_delta": 0, "spike_alert": False,
    }
    res = (
        client.table("cio_campaigns_cache")
        .upsert(row, on_conflict="cio_campaign_id")
        .execute()
    )
    logger.info("cio_campaigns_cache: campaña %s agregada", campaign_id)
    return res.data[0] if res.data else row


def delete_tracked_campaign(campaign_id: str) -> None:
    """Elimina una campaña del monitoreo de Kepler."""
    client = _get_client()
    client.table("cio_campaigns_cache").delete().eq("cio_campaign_id", campaign_id).execute()
    logger.info("cio_campaigns_cache: campaña %s eliminada", campaign_id)


# ─── Configuración: knowledge base ───────────────────────────────────────────

def get_all_knowledge_base() -> list[dict[str, Any]]:
    """Lee TODAS las entradas del knowledge_base, incluyendo inactivas."""
    client = _get_client()
    res = client.table("knowledge_base").select("*").order("tipo").execute()
    return res.data or []


def update_knowledge_base_entry(entry_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Actualiza campos de una entrada del knowledge_base (activo, titulo, contenido)."""
    client = _get_client()
    res = (
        client.table("knowledge_base")
        .update(updates)
        .eq("id", entry_id)
        .execute()
    )
    return res.data[0] if res.data else {}


def insert_knowledge_base_entry(tipo: str, titulo: str, contenido: str) -> dict[str, Any]:
    """Inserta una nueva entrada en el knowledge_base."""
    client = _get_client()
    res = (
        client.table("knowledge_base")
        .insert({"tipo": tipo, "titulo": titulo, "contenido": contenido, "activo": True})
        .execute()
    )
    return res.data[0] if res.data else {}


def delete_knowledge_base_entry(entry_id: str) -> None:
    """Elimina permanentemente una entrada del knowledge_base por su id."""
    client = _get_client()
    client.table("knowledge_base").delete().eq("id", entry_id).execute()


# ─── Fase 4: Medición — snapshots manuales ───────────────────────────────────

def save_measurement_snapshot(
    semana_label: str,
    html_content: str,
    inicio_semana: str | None = None,
    fin_semana: str | None = None,
    model_version: str = "",
) -> dict[str, Any]:
    """Guarda un snapshot de medición (HTML generado por Claude a partir de resultados BQ)."""
    client = _get_client()
    row: dict[str, Any] = {
        "semana_label":  semana_label,
        "html_content":  html_content,
        "model_version": model_version,
    }
    if inicio_semana:
        row["inicio_semana"] = inicio_semana
    if fin_semana:
        row["fin_semana"] = fin_semana
    res = client.table("bq_measurement_snapshots").insert(row).execute()
    saved = res.data[0] if res.data else row
    logger.info("bq_measurement_snapshots: guardado semana=%s", semana_label)
    return saved


def get_measurement_snapshots() -> list[dict[str, Any]]:
    """Lista todos los snapshots (solo metadatos, sin html_content)."""
    client = _get_client()
    res = (
        client.table("bq_measurement_snapshots")
        .select("id, created_at, semana_label, inicio_semana, fin_semana, model_version")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def get_measurement_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    """Devuelve un snapshot completo por ID (incluye html_content)."""
    client = _get_client()
    res = (
        client.table("bq_measurement_snapshots")
        .select("*")
        .eq("id", snapshot_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None
