from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from prometheus_fastapi_instrumentator import Instrumentator


# Routers
from api.security_api import router as security_router
from api.real_time_pipeline_api import router as pipeline_router
from api.dashboard_api import router as dashboard_router
from api.leads_api import router as leads_router
from api.routing_api import router as routing_router
from api.webhooks_api import router as webhooks_router
from api.auth_api import router as auth_router

from api.documents_api import router as documents_router
from api.docuseal_webhooks import router as docuseal_webhooks_router

import os
from database.connection import init_db

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Lead-Gen AI – API",
    version="0.1.0",
    openapi_url="/v1/openapi.json",
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
)

# CORS (adjust allowed origins via env in future)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /v1
app.include_router(security_router, prefix="/v1")
app.include_router(auth_router, prefix="/v1")
app.include_router(pipeline_router, prefix="/v1")
app.include_router(dashboard_router, prefix="/v1")


# Routers for docs are mounted earlier; ensure /v1/documents is available in OpenAPI

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/v1/metrics", include_in_schema=False)


@app.on_event("startup")
async def on_startup():
    # Initialize DB only when explicitly enabled
    if os.getenv("USE_DB", "false").lower() == "true":
        try:
            await init_db()
            logger.info("Database initialization check completed.")
        except Exception as e:
            logger.warning(f"Database init failed (continuing with in-memory): {e}")

app.include_router(leads_router, prefix="/v1")
app.include_router(documents_router, prefix="/v1")

app.include_router(routing_router, prefix="/v1")
app.include_router(webhooks_router, prefix="/v1")
app.include_router(docuseal_webhooks_router, prefix="/v1")


@app.get("/v1/health")
async def health():
    return {"status": "ok"}



