from logging.config import fileConfig
import os
import sys
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Ensure project root is on sys.path so "backend" package can be imported
# even when running Alembic from the backend/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from project root before reading DATABASE_URL
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# add your model's MetaData object here
# for 'autogenerate' support
from backend.models.base import Base

# Import all models so Alembic can detect them
from backend.models.user import User
from backend.models.session import UserSession
from backend.models.lead import Lead
from backend.models.score import Score
from backend.models.document import Document
from backend.models.customer import Customer
from backend.models.policy import Policy
from backend.models.claim import Claim
from backend.models.communication import Communication
from backend.models.report import Report

# Import marketing automation models
from backend.models.marketing_automation import (
    Campaign,
    AudienceSegment,
    MarketingTemplate,
    AutomationTrigger,
    CampaignAnalytics,
    CampaignSend
)

# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def _get_url() -> str:
    url = os.getenv("DATABASE_URL") or config.get_main_option("sqlalchemy.url")
    if url and url.startswith("postgresql+asyncpg"):
        # Use sync driver for Alembic
        url = url.replace("postgresql+asyncpg", "postgresql+psycopg2")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

