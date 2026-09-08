"""
FastAPI application entrypoint for the Engine service.

Includes:
  - /health  (liveness probe)
  - /runs    (pipeline trigger)
  - /docs    (auto-generated Swagger UI)
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engine.api.routes import health, runs
from engine.config import get_settings

# ── Logging ────────────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agentic Circuit Breaker — Engine",
    description=(
        "Real-time middleware that detects semantic thrashing in stateful "
        "multi-agent RAG pipelines and forces graceful summarization."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (permissive for local dev; tighten for production) ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(runs.router)


# ── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("=" * 60)
    logger.info("Agentic Circuit Breaker — Engine v0.1.0")
    logger.info(f"  LLM Provider : {settings.llm_provider}")
    logger.info(f"  Recursion Limit: {settings.recursion_limit}")
    logger.info(f"  DB URL       : {settings.db_url}")
    logger.info(f"  Corpus Dir   : {settings.corpus_dir}")
    logger.info("  API Docs     : http://localhost:8000/docs")
    logger.info("=" * 60)
