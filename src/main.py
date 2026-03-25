"""FastAPI application entry point.

Creates the FastAPI app, registers all route handlers, configures
structlog, and provides a uvicorn runner for direct execution.
Startable with: uv run uvicorn src.main:app --reload --port 8100
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api.execute import router as execute_router
from src.api.health import router as health_router
from src.api.interpret import router as interpret_router
from src.config import configure_logging, get_settings

configure_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown events.

    Args:
        application: The FastAPI application instance.
    """
    settings = get_settings()

    from src.llm.factory import create_llm_provider

    application.state.llm_provider = create_llm_provider(settings)

    logger.info(
        "pipeline service started",
        action="startup",
        port=settings.PORT,
        llm_provider=settings.LLM_PROVIDER,
        log_level=settings.LOG_LEVEL,
    )
    yield
    logger.info("pipeline service shutting down", action="shutdown")


app = FastAPI(
    title="Tekmar Pipeline Service",
    version="1.0.0",
    description="AI-powered infrastructure query and analysis engine",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(interpret_router)
app.include_router(execute_router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True,
    )
