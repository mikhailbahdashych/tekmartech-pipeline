"""Application configuration and structured logging setup.

Loads all environment variables via Pydantic Settings and configures
structlog for JSON output (production) or human-readable output (development).
"""

from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

import structlog
from pydantic_settings import BaseSettings

SERVICE_START_TIME: datetime = datetime.now(UTC)


class Settings(BaseSettings):
    """Pipeline Service configuration loaded from environment variables.

    All fields have sensible defaults for local development. API keys
    default to None and are only required when the corresponding
    LLM_PROVIDER is selected.
    """

    PORT: int = 8100
    LLM_PROVIDER: Literal["anthropic", "openai", "ollama"] = "ollama"
    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    OPENAI_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1
    MCP_SERVERS_CONFIG: str = "./mcp_servers.yaml"
    EXECUTION_TIMEOUT_SECONDS: int = 300
    STEP_TIMEOUT_SECONDS: int = 30
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    Returns:
        Settings: The application configuration.
    """
    return Settings()


def configure_logging() -> None:
    """Configure structlog for the application.

    Uses JSON rendering when LOG_LEVEL is not DEBUG (production),
    and human-readable console output when LOG_LEVEL is DEBUG
    (development).
    """
    settings = get_settings()
    is_development = settings.LOG_LEVEL.upper() == "DEBUG"

    if is_development:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
