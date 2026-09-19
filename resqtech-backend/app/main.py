"""ResQTech API entrypoint.

    uvicorn app.main:app --reload

Interactive docs at /docs, schema at /openapi.json.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.database.session import init_db
from app.ml.registry import get_registry

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("resqtech")

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    reg = get_registry()
    log.info("Risk model: %s", "trained " + reg.version if reg.loaded
             else "NONE — serving the logistic surrogate")
    log.info("Data mode: %s", settings.data_mode)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    version=settings.version,
    description=("Disaster intelligence API for ResQTech (SIH 2026, PS-206). "
                 "In demo mode every figure is seeded simulation data and is labelled "
                 "as such in each response — it is not an official advisory source."),
)

# CORS is restricted to the configured frontend origins. Never use "*" with
# credentials in a deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(api_router)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    # Never leak a stack trace or SQL to the client.
    return JSONResponse(status_code=500,
                        content={"detail": "Something went wrong on our side. "
                                           "The error has been logged."})


@app.get("/health", tags=["system"])
def health():
    reg = get_registry()
    return {"status": "ok", "version": settings.version,
            "data_mode": settings.data_mode,
            "risk_model": reg.version if reg.loaded else "surrogate"}
