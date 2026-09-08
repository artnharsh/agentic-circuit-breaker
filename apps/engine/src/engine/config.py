"""
Engine configuration — loaded from .env via pydantic-settings.
All thresholds and keys live here. Never hardcode values in business logic.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "mock"] = "mock"

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-haiku-4-5", alias="ANTHROPIC_MODEL")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    recursion_limit: int = Field(default=6, alias="RECURSION_LIMIT")

    # ── Circuit Breaker Thresholds ────────────────────────────────────────────
    similarity_half_open: float = Field(default=0.85, alias="SIMILARITY_HALF_OPEN")
    similarity_open: float = Field(default=0.95, alias="SIMILARITY_OPEN")
    similarity_window: int = Field(default=2, alias="SIMILARITY_WINDOW")

    # ── Storage ───────────────────────────────────────────────────────────────
    db_url: str = Field(
        default="sqlite+aiosqlite:///./data/circuit_breaker.db",
        alias="DB_URL",
    )

    # ── Corpus ─────────────────────────────────────────────────────
    corpus_dir: str = Field(default="", alias="CORPUS_DIR")
    # Empty string = auto-detect relative to this file's repo root

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
