"""
Storage — db.py

SQLAlchemy async database session factory.

Provides:
  - async engine backed by aiosqlite (SQLite for now)
  - AsyncSession factory
  - FastAPI dependency: get_db()
  - init_db() — creates all tables on first startup
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from engine.config import get_settings
from engine.storage.models import Base

logger = logging.getLogger(__name__)

# ── Engine (lazy-initialized) ──────────────────────────────────────────────────
_engine = None
_session_factory: async_sessionmaker | None = None


def _get_db_url() -> str:
    """
    Return the database URL, ensuring the parent directory exists for SQLite.
    """
    settings = get_settings()
    url = settings.db_url

    # For SQLite, make sure the data/ directory exists
    if "sqlite" in url:
        # Extract the file path from the URL
        # e.g. "sqlite+aiosqlite:///./data/circuit_breaker.db" → "./data/circuit_breaker.db"
        parts = url.split("///")
        if len(parts) == 2:
            db_path = Path(parts[1])
            db_path.parent.mkdir(parents=True, exist_ok=True)

    return url


def get_engine():
    """Return (or create) the async SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        url = _get_db_url()
        _engine = create_async_engine(
            url,
            echo=False,          # set True to log every SQL statement
            future=True,
            pool_pre_ping=True,
        )
        logger.info(f"Database engine created: {url}")
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Return (or create) the async session factory singleton."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
    return _session_factory


async def init_db() -> None:
    """
    Create all tables if they don't exist.
    Called once on application startup.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized.")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields an async DB session per request.

    Usage:
        @router.post("/runs")
        async def create_run(db: AsyncSession = Depends(get_db)):
            ...
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
