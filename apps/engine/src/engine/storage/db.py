"""
Storage — db.py (Day 2 implementation).

SQLAlchemy async database session factory.
Provides a FastAPI dependency for injecting DB sessions into route handlers.

Stub for Day 1 — full implementation in Day 2.
"""

# TODO (Day 2): Implement:
#   - create_async_engine() using settings.db_url
#   - AsyncSessionLocal = async_sessionmaker(...)
#   - async def get_db() -> AsyncGenerator[AsyncSession, None]
#     (FastAPI dependency)
#   - async def init_db() — creates all tables on startup
