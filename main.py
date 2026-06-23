import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import config, data, measure, ml, monitor, strategy

app = FastAPI(title="Kepler Backend", version="2.0.0")

import os

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
