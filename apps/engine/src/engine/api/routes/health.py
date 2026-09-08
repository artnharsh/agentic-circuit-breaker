"""GET /health — liveness probe."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    """Returns service status. Used by Docker healthcheck and load balancers."""
    return {"status": "ok", "version": "0.1.0", "service": "agentic-circuit-breaker-engine"}
