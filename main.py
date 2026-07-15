import logging
import os
import secrets
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routers import config, data, measure, ml, monitor, strategy

_BACKEND_SECRET = os.getenv("KEPLER_BACKEND_SECRET", "")


def verify_backend_secret(authorization: str | None = Header(default=None)) -> None:
    """Exige que cada request traiga el secreto compartido con el proxy de Next.js.
    Sin esto, cualquiera que descubra la URL del backend podía llamar cualquier
    endpoint (leer/borrar KB, editar prompts, ejecutar estrategias en CIO) sin login."""
    expected = f"Bearer {_BACKEND_SECRET}"
    if not _BACKEND_SECRET or not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="No autorizado")


app = FastAPI(title="Kepler Backend", version="2.0.0", dependencies=[Depends(verify_backend_secret)])

_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
_origins = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Org-Slug", "X-Funnel-Slug"],
)

app.include_router(data.router,     prefix="/api/data",     tags=["data"])
app.include_router(ml.router,       prefix="/api/ml",       tags=["ml"])
app.include_router(strategy.router, prefix="/api/strategy", tags=["strategy"])
app.include_router(config.router,   prefix="/api/config",   tags=["config"])
app.include_router(monitor.router,  prefix="/api/monitor",  tags=["monitor"])
app.include_router(measure.router,  prefix="/api/measure",  tags=["measure"])


@app.get("/")
def health():
    return {"status": "ok", "service": "kepler-backend"}
