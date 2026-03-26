"""GET /health endpoint.

Returns the health status of the Pipeline Service including
connectivity status of the LLM provider and MCP servers.
Conforms to the health endpoint defined in internal-api.yaml.
"""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request

from src.config import SERVICE_START_TIME, get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    """Return Pipeline Service health status.

    Probes the LLM provider to report real connectivity status.

    Args:
        request: The incoming FastAPI request (used to access app state).

    Returns:
        dict: Health response with status, version, uptime, and components.
    """
    logger.debug("health check requested", action="health_check")

    settings = get_settings()
    now = datetime.now(UTC)
    uptime_seconds = int((now - SERVICE_START_TIME).total_seconds())

    llm_provider = getattr(request.app.state, "llm_provider", None)
    if llm_provider is not None:
        check = await llm_provider.health_check()
        llm_status = {
            "provider": settings.LLM_PROVIDER,
            **check.to_dict(),
        }
    else:
        llm_status = {
            "status": "unknown",
            "provider": settings.LLM_PROVIDER,
            "checked_at": None,
        }

    overall_status = "healthy" if llm_status["status"] == "healthy" else "degraded"

    response = {
        "status": overall_status,
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "components": {
            "llm_provider": llm_status,
            "mcp_servers": [],
        },
    }

    logger.info(
        "health check completed",
        action="health_check",
        status=overall_status,
        llm_status=llm_status["status"],
        uptime_seconds=uptime_seconds,
    )
    return response
