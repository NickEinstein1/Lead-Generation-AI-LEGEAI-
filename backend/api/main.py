from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from prometheus_fastapi_instrumentator import Instrumentator


# Routers
from backend.api.security_api import router as security_router
from backend.api.real_time_pipeline_api import router as pipeline_router
from backend.api.dashboard_api import router as dashboard_router
from backend.api.leads_api import router as leads_router
from backend.api.routing_api import router as routing_router
from backend.api.webhooks_api import router as webhooks_router
from backend.api.auth_api import router as auth_router
from backend.api.life_insurance_scoring_api import router as life_insurance_router
from backend.api.auto_insurance_scoring_api import router as auto_insurance_router
from backend.api.home_insurance_scoring_api import router as home_insurance_router
from backend.api.health_insurance_scoring_api import router as health_insurance_router

from backend.api.documents_api import router as documents_router
from backend.api.docuseal_webhooks import router as docuseal_webhooks_router
from backend.api.file_documents_api import router as file_documents_router

import os
from backend.database.connection import init_db
from backend.security.authentication import auth_manager, UserRole

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
app.include_router(life_insurance_router, prefix="/v1")
app.include_router(auto_insurance_router, prefix="/v1")
app.include_router(home_insurance_router, prefix="/v1")
app.include_router(health_insurance_router, prefix="/v1")
app.include_router(file_documents_router, prefix="/v1/file-management")  # New file document management


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
    else:
        # Initialize default users for in-memory mode
        logger.info("Initializing default users for in-memory mode...")

        # Create admin user
        success, user_id = auth_manager.create_user(
            "admin",
            "admin@legeai.com",
            "AdminPass123!",
            UserRole.ADMIN
        )
        if success:
            logger.info(f"✓ Admin user created: {user_id}")
        else:
            logger.warning(f"Admin user creation failed: {user_id}")

        # Create manager user
        success, user_id = auth_manager.create_user(
            "manager",
            "manager@legeai.com",
            "ManagerPass123!",
            UserRole.MANAGER
        )
        if success:
            logger.info(f"✓ Manager user created: {user_id}")
        else:
            logger.warning(f"Manager user creation failed: {user_id}")

        # Create agent user
        success, user_id = auth_manager.create_user(
            "agent1",
            "agent1@legeai.com",
            "AgentPass123!",
            UserRole.AGENT
        )
        if success:
            logger.info(f"✓ Agent user created: {user_id}")
        else:
            logger.warning(f"Agent user creation failed: {user_id}")

        logger.info("Default users initialization complete.")

app.include_router(leads_router, prefix="/v1")
app.include_router(documents_router, prefix="/v1")

app.include_router(routing_router, prefix="/v1")
app.include_router(webhooks_router, prefix="/v1")
app.include_router(docuseal_webhooks_router, prefix="/v1")


@app.get("/v1/health")
async def health():
    return {"status": "ok"}



