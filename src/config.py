"""Application configuration and structured logging setup.

Loads configuration exclusively from .env.development or .env.production.
The env file is selected by the PIPELINE_ENV environment variable
("development" or "production"). Every required variable must be present
in the env file — there are no fallback defaults.

Provider-specific validation: only the variables relevant to the selected
LLM_PROVIDER are required (e.g., ANTHROPIC_API_KEY is not required when
LLM_PROVIDER=ollama).
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog

SERVICE_START_TIME: datetime = datetime.now(UTC)

# Variables that every env file must define, regardless of provider.
_COMMON_REQUIRED = [
    "LLM_PROVIDER",
    "LLM_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "PORT",
    "LOG_LEVEL",
    "EXECUTION_TIMEOUT_SECONDS",
    "STEP_TIMEOUT_SECONDS",
    "MCP_SERVERS_CONFIG",
]

# Variables required per LLM provider.
_PROVIDER_REQUIRED: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"],
    "openai": ["OPENAI_API_KEY"],
    "ollama": ["OLLAMA_BASE_URL", "OLLAMA_MODEL"],
}

_VALID_PROVIDERS = set(_PROVIDER_REQUIRED.keys())


class Settings:
    """Pipeline Service configuration.

    All values are loaded from the env file selected by PIPELINE_ENV.
    No defaults — if a required variable is missing, the application
    refuses to start.
    """

    def __init__(self, env_vars: dict[str, str]) -> None:
        self.PIPELINE_ENV: str = env_vars["PIPELINE_ENV"]
        self.LLM_PROVIDER: str = env_vars["LLM_PROVIDER"]
        self.PORT: int = int(env_vars["PORT"])
        self.LOG_LEVEL: str = env_vars["LOG_LEVEL"]
        self.LLM_MAX_TOKENS: int = int(env_vars["LLM_MAX_TOKENS"])
        self.LLM_TEMPERATURE: float = float(env_vars["LLM_TEMPERATURE"])
        self.MCP_SERVERS_CONFIG: str = env_vars["MCP_SERVERS_CONFIG"]
        self.EXECUTION_TIMEOUT_SECONDS: int = int(env_vars["EXECUTION_TIMEOUT_SECONDS"])
        self.STEP_TIMEOUT_SECONDS: int = int(env_vars["STEP_TIMEOUT_SECONDS"])

        # Provider-specific (only the selected provider's vars are populated)
        self.ANTHROPIC_API_KEY: str | None = env_vars.get("ANTHROPIC_API_KEY")
        self.ANTHROPIC_MODEL: str | None = env_vars.get("ANTHROPIC_MODEL")
        self.OPENAI_API_KEY: str | None = env_vars.get("OPENAI_API_KEY")
        self.OLLAMA_BASE_URL: str | None = env_vars.get("OLLAMA_BASE_URL")
        self.OLLAMA_MODEL: str | None = env_vars.get("OLLAMA_MODEL")


def _load_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict.

    Handles comments, blank lines, and optional quoting of values.

    Args:
        path: Path to the .env file.

    Returns:
        Dict of variable name to value.
    """
    env_vars: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip surrounding quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        env_vars[key] = value
    return env_vars


def _validate_and_load() -> Settings:
    """Load the env file, validate all required variables, and return Settings.

    Exits the process with a clear error message if:
    - PIPELINE_ENV is not "development" or "production"
    - The env file does not exist
    - Any required variable is missing
    - LLM_PROVIDER is not a recognized value

    Returns:
        A fully populated Settings instance.
    """
    pipeline_env = os.environ.get("PIPELINE_ENV", "development").lower()
    if pipeline_env not in ("development", "production"):
        print(
            f"FATAL: PIPELINE_ENV must be 'development' or 'production', got '{pipeline_env}'",
            file=sys.stderr,
        )
        sys.exit(1)

    env_file = Path(f".env.{pipeline_env}")
    if not env_file.exists():
        print(
            f"FATAL: Environment file '{env_file}' not found. "
            f"Create it before starting the application.",
            file=sys.stderr,
        )
        sys.exit(1)

    env_vars = _load_env_file(env_file)
    env_vars["PIPELINE_ENV"] = pipeline_env

    # OS environment variables override the file (standard behavior)
    for key in env_vars:
        if key in os.environ:
            env_vars[key] = os.environ[key]

    # Validate common required variables
    missing = [var for var in _COMMON_REQUIRED if not env_vars.get(var)]
    if missing:
        print(
            f"FATAL: Missing required variables in '{env_file}': {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate LLM_PROVIDER value
    provider = env_vars["LLM_PROVIDER"]
    if provider not in _VALID_PROVIDERS:
        print(
            f"FATAL: LLM_PROVIDER must be one of {sorted(_VALID_PROVIDERS)}, got '{provider}'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate provider-specific variables
    provider_missing = [var for var in _PROVIDER_REQUIRED[provider] if not env_vars.get(var)]
    if provider_missing:
        print(
            f"FATAL: LLM_PROVIDER='{provider}' requires: {', '.join(provider_missing)} "
            f"(missing in '{env_file}')",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Pipeline Service [{pipeline_env}] — "
        f"loaded from .env.{pipeline_env} "
        f"(LLM_PROVIDER={provider}, LOG_LEVEL={env_vars['LOG_LEVEL']}, PORT={env_vars['PORT']})",
        file=sys.stderr,
    )

    return Settings(env_vars)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the application settings singleton.

    On first call, loads and validates the env file. Subsequent calls
    return the cached instance.

    Returns:
        The validated Settings instance.
    """
    global _settings  # noqa: PLW0603
    if _settings is None:
        _settings = _validate_and_load()
    return _settings


def configure_logging() -> None:
    """Configure structlog for the application.

    Uses human-readable console output when LOG_LEVEL is DEBUG
    (development), and JSON rendering otherwise (production).
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
