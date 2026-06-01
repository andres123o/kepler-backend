import re
import sys
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).parent / ".env")

CAMPAIGN_ID  = 4596
ACTION_ID    = 36707
ENV_ID       = os.getenv("CIO_ENVIRONMENT_ID", "112828")
CIO_APP_BASE = "https://api.customer.io/v1"
CIO_FLY_BASE = "https://us.fly.customer.io"
WRITE_KEY    = os.getenv("CUSTOMERIO_APP_API_KEY", "")
SA_LIVE      = os.getenv("CIO_SA_LIVE_READONLY_KEY", "")

NUEVO_SUBJECT = (
    "{% if customer.Perfil_de_riesgo == '1. Conservador' %}"
    "💰 Tu CDT al 12% EA está listo"
    "{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}"
    "📈 El mercado corrigió: es tu momento"
    "{% else %}"
    "💚 Llegaste en el momento perfecto"
    "{% endif %}"
)

NUEVO_BODY = (
    "{% if customer.Perfil_de_riesgo == '1. Conservador' %}"
    "Los CDT siguen por encima del 11% EA mientras el mercado se mueve. "
    "Capital seguro y rindiendo desde hoy. Empezá desde $200.000."
    "{% elsif customer.Perfil_de_riesgo == '2. Moderado' or customer.Perfil_de_riesgo == '3. Arriesgado' %}"
    "El COLCAP cayó más del 7% en mayo. Históricamente, las correcciones abren oportunidades. "
    "Empezá a invertir desde $55.000."
    "{% else %}"
    "Los CDT están al 12% EA y el mercado acumula señales esta semana. "
    "Tu cuenta está lista. Empezá desde $200.000."
    "{% endif %}"
)


# ── Auth ───────────────────────────────────────────────────────────────────

def get_jwt() -> str:
    if not SA_LIVE:
        raise RuntimeError("CIO_SA_LIVE_READONLY_KEY no está en .env")
    url = f"{CIO_FLY_BASE}/v1/service_accounts/oauth/token"
    print(f"    POST {url}  (sa_live → JWT)")
    resp = httpx.post(
        url,
        data={"grant_type": "client_credentials", "client_secret": SA_LIVE},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    print(f"    → status  : {resp.status_code}")
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"    → JWT obtenido (primeros 30 chars): {token[:30]}...")
    return token


def _h_app() -> dict:
    if not WRITE_KEY:
        raise RuntimeError("CUSTOMERIO_APP_API_KEY no está en .env")
    return {"Authorization": f"Bearer {WRITE_KEY}", "Content-Type": "application/json"}


def _h_fly(jwt: str) -> dict:
    return {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}


# ── Lectura ────────────────────────────────────────────────────────────────

def get_action_app() -> dict:
    """Lee el action via App API — para verificación post-update."""
    url = f"{CIO_APP_BASE}/campaigns/{CAMPAIGN_ID}/actions/{ACTION_ID}"
    print(f"    GET {url}")
    resp = httpx.get(url, headers=_h_app(), timeout=30)
    print(f"    → status  : {resp.status_code}")
    print(f"    → respuesta: {resp.text[:400]}")
    resp.raise_for_status()
    return resp.json().get("action", resp.json())


def get_template_fly(jwt: str, template_id: int) -> dict:
    """
    Lee el template completo desde fly API.
    Retorna el objeto template con todos sus campos incluyendo subject y body
    directamente en el nivel raíz — sin wrapper 'content'.
    """
    url = f"{CIO_FLY_BASE}/v1/environments/{ENV_ID}/templates/{template_id}"
    print(f"    GET {url}")
    resp = httpx.get(url, headers=_h_fly(jwt), timeout=30)
    print(f"    → status  : {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    # Response: {"layouts": [], "template": {...}} — extraer el template
    tmpl = data.get("template", data)
    print(f"    → {len(tmpl)} campos leídos: {sorted(tmpl.keys())}")
    return tmpl


def extract_template_id(action: dict) -> int | None:
    dedup = action.get("deduplicate_id", "")
    if ":" in dedup:
        try:
            return int(dedup.split(":")[0])
        except ValueError:
            pass
    return None


# ── Escritura ──────────────────────────────────────────────────────────────

def put_template_fly(jwt: str, template_id: int, subject: str, body: str) -> dict:
    """
    PUT /v1/environments/{env}/templates/{template_id}

    subject y body viven en el template, NO en el action.
    El action los muestra como campos read-only calculados desde el template.

    Patrón correcto (igual que PUT /actions):
    1. GET template completo desde fly API
    2. Modificar subject y body (están en nivel raíz, no en sub-campo 'content')
    3. PUT con wrapper {"template": {...objeto completo modificado...}}

    Intentos anteriores que fallaron:
    - {"content": {"subject": ...}}  → campo 'content' no existe, ignorado
    - PUT /actions con subject        → subject es read-only en el action
    """
    # 1. Leer template completo
    print(f"\n    [PUT] Leyendo template {template_id} completo desde fly API...")
    tmpl_actual = get_template_fly(jwt, template_id)

    cur_subject = (tmpl_actual.get("subject") or "").strip()
    cur_body    = (tmpl_actual.get("body")    or "").strip()
    print(f"    → subject actual : {cur_subject[:80]}")
    print(f"    → body actual    : {cur_body[:80]}")

    # 2. Modificar solo subject y body — el resto de campos se mantiene igual
    tmpl_actualizado = {**tmpl_actual}
    tmpl_actualizado["subject"] = subject
    tmpl_actualizado["body"]    = body

    # 3. PUT con wrapper {"template": {...}}
    url     = f"{CIO_FLY_BASE}/v1/environments/{ENV_ID}/templates/{template_id}"
    payload = {"template": tmpl_actualizado}

    print(f"\n    PUT {url}")
    print(f"    → subject nuevo  : {subject[:100]}")
    print(f"    → body nuevo     : {body[:100]}")
    print(f"    → campos en payload: {len(tmpl_actualizado)}")

    resp = httpx.put(url, headers=_h_fly(jwt), json=payload, timeout=30)
    print(f"    → status  : {resp.status_code}")
    print(f"    → respuesta: {resp.text[:800]}")
    resp.raise_for_status()
    return resp.json()


# ── Validaciones ───────────────────────────────────────────────────────────

def check_limits() -> list[str]:
    ramas = lambda s: [r.strip() for r in re.findall(r"%\}(.*?)(?:\{%|$)", s, re.DOTALL) if r.strip()]
    warns = []
    for i, r in enumerate(ramas(NUEVO_SUBJECT), 1):
        if len(r) > 60:
            warns.append(f"  ⚠ Subject rama {i}: {len(r)} chars > 60 → '{r}'")
    for i, r in enumerate(ramas(NUEVO_BODY), 1):
        if len(r) > 180:
            warns.append(f"  ⚠ Body rama {i}: {len(r)} chars > 180 → '{r[:60]}...'")
    return warns


# ── Modo --spec ────────────────────────────────────────────────────────────

def cargar_spec(jwt: str) -> dict:
    print("\n[SPEC] Descargando OpenAPI spec fresco de fly.customer.io...")
    resp = httpx.get(
        f"{CIO_FLY_BASE}/v1/openapi.json",
        headers=_h_fly(jwt),
        timeout=60,
    )
    print(f"    → status: {resp.status_code}")
    resp.raise_for_status()
    spec = resp.json()
    spec_path = Path(__file__).parent / "cio_openapi_spec.json"
    spec_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"    → guardado en: {spec_path}")
    return spec


def analizar_spec_template(spec: dict) -> None:
    schemas  = spec.get("components", {}).get("schemas", {})
    paths    = spec.get("paths", {})

    print("\n=== SCHEMAS CON 'template' EN EL NOMBRE ===")
    for k in sorted(schemas):
        if "template" in k.lower():
            s        = schemas[k]
            required = s.get("required", [])
            props    = s.get("properties", {})
            print(f"\n{'='*50}")
            print(f"SCHEMA: {k}")
            print(f"required ({len(required)}): {required[:10]}{'...' if len(required) > 10 else ''}")
            print(f"props    ({len(props)}): {sorted(props.keys())[:15]}{'...' if len(props) > 15 else ''}")

    print("\n=== PUT /templates/{template_id} ===")
    for path, methods in paths.items():
        if "template" in path and "{template_id}" in path and "put" in methods:
            put_op     = methods["put"]
            rb         = put_op.get("requestBody", {})
            schema_ref = rb.get("content", {}).get("application/json", {}).get("schema", {})
            print(f"path: {path}")
            print(f"requestBody schema: {json.dumps(schema_ref, indent=2)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:

    # ── MODO --spec ────────────────────────────────────────────────────────
    if "--spec" in sys.argv:
        jwt  = get_jwt()
        spec = cargar_spec(jwt)
        analizar_spec_template(spec)
        return

    dry_run = "--ejecutar" not in sys.argv

    print()
    print("=" * 62)
    print("  Push #1 | Primer depósito | Colombia")
    print(f"  campaign={CAMPAIGN_ID}  action={ACTION_ID}")
    print(f"  {'DRY RUN — no se escribe nada' if dry_run else 'ESCRITURA REAL EN CIO  ⚠'}")
    print("=" * 62)

    # 1. Leer action via App API → tipo y template_id
    print("\n[1] Leyendo action via App API...")
    try:
        action_app = get_action_app()
    except httpx.HTTPStatusError as e:
        print(f"    ERROR {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    tipo        = action_app.get("type", "?")
    nombre      = action_app.get("name", "?")
    template_id = extract_template_id(action_app)
    print(f"    tipo={tipo}  nombre='{nombre}'  template_id={template_id}")

    if "push" not in tipo.lower():
        print(f"\n    STOP: nodo {ACTION_ID} es '{tipo}', no push. Abortando.")
        sys.exit(1)
    if not template_id:
        print("\n    STOP: no se pudo extraer template_id. Abortando.")
        sys.exit(1)

    # 2. Verificar límites Liquid
    warns = check_limits()
    print("\n[2] Longitud por rama Liquid:")
    if warns:
        for w in warns:
            print(w)
    else:
        print("    OK — subject ≤60 chars, body ≤180 chars por rama.")

    # 3. Auth + leer template actual
    print(f"\n[3] Auth + leyendo template {template_id} via fly API...")
    try:
        jwt  = get_jwt()
        tmpl = get_template_fly(jwt, template_id)
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)

    cur_subject = (tmpl.get("subject") or "").strip()
    cur_body    = (tmpl.get("body")    or "").strip()

    # 4. Diff
    print("\n[4] Diff — estado actual vs propuesto:")
    print(f"\n  subject actual    : {cur_subject or '(vacío)'}")
    print(f"  subject propuesto : {NUEVO_SUBJECT}")
    print(f"\n  body actual       : {cur_body[:120] or '(vacío)'}")
    print(f"  body propuesto    : {NUEVO_BODY[:120]}")

    if dry_run:
        print("\n[5] DRY RUN — sin cambios.")
        print("    Para escribir: python test_actualizar_push_36707.py --ejecutar")
        return

    # 5. Actualizar template via fly API (objeto completo con wrapper {"template": {...}})
    print(f"\n[5] Actualizando template {template_id} via fly API...")
    try:
        result = put_template_fly(jwt, template_id, NUEVO_SUBJECT, NUEVO_BODY)
    except httpx.HTTPStatusError as e:
        print(f"    ERROR {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"    ERROR: {e}")
        sys.exit(1)

    # 6. Verificar — leer template de vuelta desde fly API
    print(f"\n[6] Verificando — leyendo template {template_id} de vuelta...")
    try:
        tmpl_post    = get_template_fly(jwt, template_id)
        post_subject = (tmpl_post.get("subject") or "").strip()
        post_body    = (tmpl_post.get("body")    or "").strip()
    except Exception as e:
        print(f"    ERROR al verificar: {e}")
        sys.exit(1)

    ok_s = post_subject == NUEVO_SUBJECT
    ok_b = post_body    == NUEVO_BODY

    print(f"\n    subject : {'✅ OK' if ok_s else '❌ MISMATCH'}")
    print(f"    body    : {'✅ OK' if ok_b else '❌ MISMATCH'}")
    print(f"\n    subject en CIO : {post_subject[:150]}")
    print(f"    body en CIO    : {post_body[:150]}")

    if ok_s and ok_b:
        print("\n  ✅ Actualización verificada correctamente.")
        print(f"     Ver en CIO: https://fly.customer.io/workspaces/{ENV_ID}/journeys/campaigns/{CAMPAIGN_ID}/overview/workflow/actions?actionId={ACTION_ID}")
    else:
        print("\n  ⚠  No coincide. Verifica manualmente en CIO:")
        print(f"     https://fly.customer.io/workspaces/{ENV_ID}/journeys/campaigns/{CAMPAIGN_ID}/overview/workflow/actions?actionId={ACTION_ID}")


if __name__ == "__main__":
    main()