"""
Medición de impacto de campañas Kepler vs grupo control.

Metodología:
  - Grupo TEST    = usuarios que recibieron ≥1 mensaje de la campaña (IDs del reporte CIO)
  - Grupo CONTROL = todos los usuarios elegibles del mismo período que NO están en el test
  - Elegible = country='CO', embudo NOT IN ('bloqueados','revisar'),
               date_full_user en la ventana especificada
  - Conversión = hizo primer depósito (date_first_cashin IS NOT NULL)
  - Uplift abs = N_test × (rate_test − rate_control)
"""

import logging
from typing import Any

logger = logging.getLogger("kepler.measurement")

# ─── SQL ──────────────────────────────────────────────────────────────────────

_SUMMARY_SQL = """
WITH test_ids AS (
  SELECT id FROM UNNEST(@test_user_ids) AS id
),
eligible AS (
  SELECT
    CAST(id AS STRING)       AS id,
    date_full_user,
    date_first_cashin,
    Perfil_de_riesgo,
    embudo,
    COALESCE(total_cashin_usd, 0) AS total_cashin_usd,
    COALESCE(total_aum_usd,   0) AS total_aum_usd
  FROM `trii-bi.scheduled_queries.user_attributes`
  WHERE
    country = 'CO'
    AND embudo NOT IN ('bloqueados', 'revisar')
    AND date_full_user BETWEEN @start_date AND @end_date
),
labeled AS (
  SELECT
    e.*,
    CASE WHEN t.id IS NOT NULL THEN 'test' ELSE 'control' END AS grupo
  FROM eligible e
  LEFT JOIN test_ids t ON e.id = t.id
)
SELECT
  grupo,
  COUNT(*)                                        AS n,
  COUNTIF(date_first_cashin IS NOT NULL)          AS conversiones,
  SAFE_DIVIDE(
    COUNTIF(date_first_cashin IS NOT NULL),
    COUNT(*)
  )                                               AS conversion_rate,
  SUM(total_cashin_usd)                           AS total_cashin_usd,
  AVG(CASE WHEN date_first_cashin IS NOT NULL
           THEN total_cashin_usd END)             AS avg_cashin_convertidos_usd,
  SUM(total_aum_usd)                              AS total_aum_usd
FROM labeled
GROUP BY grupo
"""

_PROFILE_SQL = """
WITH test_ids AS (
  SELECT id FROM UNNEST(@test_user_ids) AS id
),
eligible AS (
  SELECT
    CAST(id AS STRING) AS id,
    Perfil_de_riesgo,
    date_first_cashin
  FROM `trii-bi.scheduled_queries.user_attributes`
  WHERE
    country = 'CO'
    AND embudo NOT IN ('bloqueados', 'revisar')
    AND date_full_user BETWEEN @start_date AND @end_date
),
labeled AS (
  SELECT
    e.*,
    CASE WHEN t.id IS NOT NULL THEN 'test' ELSE 'control' END AS grupo
  FROM eligible e
  LEFT JOIN test_ids t ON e.id = t.id
)
SELECT
  grupo,
  COALESCE(Perfil_de_riesgo, 'Sin perfil')                       AS perfil,
  COUNT(*)                                                         AS n,
  COUNTIF(date_first_cashin IS NOT NULL)                          AS conversiones,
  SAFE_DIVIDE(
    COUNTIF(date_first_cashin IS NOT NULL),
    COUNT(*)
  )                                                               AS conversion_rate
FROM labeled
GROUP BY grupo, perfil
ORDER BY grupo, n DESC
"""

_WEEKLY_SQL = """
WITH test_ids AS (
  SELECT id FROM UNNEST(@test_user_ids) AS id
),
eligible AS (
  SELECT
    CAST(id AS STRING)  AS id,
    date_full_user,
    date_first_cashin
  FROM `trii-bi.scheduled_queries.user_attributes`
  WHERE
    country = 'CO'
    AND embudo NOT IN ('bloqueados', 'revisar')
    AND date_full_user BETWEEN @start_date AND @end_date
),
labeled AS (
  SELECT
    e.*,
    DATE_TRUNC(date_full_user, WEEK(MONDAY)) AS semana,
    CASE WHEN t.id IS NOT NULL THEN 'test' ELSE 'control' END AS grupo
  FROM eligible e
  LEFT JOIN test_ids t ON e.id = t.id
)
SELECT
  semana,
  grupo,
  COUNT(*) AS n,
  COUNTIF(date_first_cashin IS NOT NULL) AS conversiones,
  SAFE_DIVIDE(
    COUNTIF(date_first_cashin IS NOT NULL),
    COUNT(*)
  ) AS conversion_rate
FROM labeled
GROUP BY semana, grupo
ORDER BY semana, grupo
"""


# ─── Main function ─────────────────────────────────────────────────────────────

def run_measurement(
    test_user_ids: list[str],
    start_date: str,
    end_date: str,
    campaign_name: str = "",
) -> dict[str, Any]:
    """
    Mide el impacto de una campaña comparando grupo test vs control en BigQuery.

    Args:
        test_user_ids : IDs de usuarios que recibieron la campaña (del reporte CIO)
        start_date    : Inicio ventana date_full_user (YYYY-MM-DD)
        end_date      : Fin ventana date_full_user (YYYY-MM-DD)
        campaign_name : Nombre descriptivo para el reporte (no va a BQ)
    """
    from app.services.bigquery_client import run_query
    from google.cloud import bigquery

    if not test_user_ids:
        raise ValueError("Se requieren IDs del grupo test (entregados por CIO)")

    # Deduplicate + cast to string
    ids = list({str(i).strip() for i in test_user_ids if str(i).strip()})

    params_base = [
        bigquery.ArrayQueryParameter("test_user_ids", "STRING", ids),
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date",   "DATE", end_date),
    ]

    summary_rows = run_query(_SUMMARY_SQL,  params_base)
    profile_rows = run_query(_PROFILE_SQL,  params_base)
    weekly_rows  = run_query(_WEEKLY_SQL,   params_base)

    # ── Parse summary ──────────────────────────────────────────────────────────
    groups: dict[str, dict] = {}
    for row in summary_rows:
        groups[row["grupo"]] = row

    test = groups.get("test",    {})
    ctrl = groups.get("control", {})

    n_test = int(test.get("n") or 0)
    n_ctrl = int(ctrl.get("n") or 0)
    r_test = float(test.get("conversion_rate") or 0)
    r_ctrl = float(ctrl.get("conversion_rate") or 0)

    uplift_pct = ((r_test - r_ctrl) / r_ctrl * 100) if r_ctrl > 0 else None
    uplift_abs = int(round(n_test * (r_test - r_ctrl))) if n_test > 0 else 0

    # ── Parse profile breakdown ────────────────────────────────────────────────
    profile_by_group: dict[str, list] = {}
    for row in profile_rows:
        g = row["grupo"]
        profile_by_group.setdefault(g, []).append({
            "perfil":           row["perfil"],
            "n":                int(row["n"]),
            "conversiones":     int(row["conversiones"]),
            "conversion_rate":  round(float(row["conversion_rate"] or 0), 4),
        })

    # ── Parse weekly trend ─────────────────────────────────────────────────────
    weekly: list[dict] = []
    weekly_map: dict[str, dict] = {}
    for row in weekly_rows:
        semana = str(row["semana"])
        if semana not in weekly_map:
            weekly_map[semana] = {"semana": semana}
        weekly_map[semana][row["grupo"]] = {
            "n":               int(row["n"]),
            "conversiones":    int(row["conversiones"]),
            "conversion_rate": round(float(row["conversion_rate"] or 0), 4),
        }
    weekly = sorted(weekly_map.values(), key=lambda x: x["semana"])

    return {
        "campaign_name":         campaign_name,
        "window":                {"start": start_date, "end": end_date},
        "n_test_ids_provided":   len(test_user_ids),
        "n_test_ids_matched":    n_test,
        "groups": {
            "test": {
                "n":                          n_test,
                "conversiones":               int(test.get("conversiones") or 0),
                "conversion_rate":            round(r_test, 4),
                "total_cashin_usd":           round(float(test.get("total_cashin_usd") or 0), 2),
                "avg_cashin_convertidos_usd": round(float(test.get("avg_cashin_convertidos_usd") or 0), 2),
                "total_aum_usd":              round(float(test.get("total_aum_usd") or 0), 2),
            },
            "control": {
                "n":                          n_ctrl,
                "conversiones":               int(ctrl.get("conversiones") or 0),
                "conversion_rate":            round(r_ctrl, 4),
                "total_cashin_usd":           round(float(ctrl.get("total_cashin_usd") or 0), 2),
                "avg_cashin_convertidos_usd": round(float(ctrl.get("avg_cashin_convertidos_usd") or 0), 2),
                "total_aum_usd":              round(float(ctrl.get("total_aum_usd") or 0), 2),
            },
        },
        "uplift": {
            "absolute":    uplift_abs,
            "pct":         round(uplift_pct, 1) if uplift_pct is not None else None,
            "significant": (
                abs(uplift_abs) >= 5
                and uplift_pct is not None
                and abs(uplift_pct) >= 5
            ),
            "direction":   "positivo" if uplift_abs > 0 else "negativo" if uplift_abs < 0 else "neutro",
        },
        "profile_breakdown": profile_by_group,
        "weekly_trend":      weekly,
    }
