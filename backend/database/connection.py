import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/leadgen",
)

engine = create_async_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# FastAPI dependency-style session provider
async def session_dep() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db() -> None:
    # Alembic will handle migrations. This hook can be used for sanity checks.
    try:
        async with engine.begin() as conn:
            await conn.run_sync(lambda conn: None)
    except Exception as e:
        # Do not crash if DB is unavailable (dev mode). APIs can still run in in-memory mode.
        import logging
        logging.getLogger(__name__).warning(f"DB unavailable or not initialized: {e}")

