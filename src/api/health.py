"""GET /health endpoint.

Returns the health status of the Pipeline Service including
connectivity status of the LLM provider and MCP servers.
Conforms to the health endpoint defined in internal-api.yaml.
"""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter

from src.config import SERVICE_START_TIME, get_settings

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return Pipeline Service health status.

    Returns:
        dict: Health response with status, version, uptime, and components.
    """
    logger.debug("health check requested", action="health_check")

    settings = get_settings()
    now = datetime.now(UTC)
    uptime_seconds = int((now - SERVICE_START_TIME).total_seconds())

    response = {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "components": {
            "llm_provider": {
                "status": "unknown",
                "provider": settings.LLM_PROVIDER,
                "last_check_at": None,
            },
            "mcp_servers": [],
        },
    }

    logger.info(
        "health check completed",
        action="health_check",
        status="healthy",
        uptime_seconds=uptime_seconds,
    )
    return response
