"""LLM provider factory.

Creates the appropriate LLM provider instance based on the
LLM_PROVIDER configuration. Called once during application
startup; the resulting instance is shared across all requests.
"""

import structlog

from src.config import Settings
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create an LLM provider instance based on configuration.

    Args:
        settings: Application settings containing provider config.

    Returns:
        LLMProvider: Configured provider instance ready for use.

    Raises:
        ValueError: If the provider is unknown or required config is missing.
    """
    provider_name = settings.LLM_PROVIDER

    if provider_name == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
                "Set it in your .env file."
            )
        logger.info(
            "creating anthropic LLM provider",
            action="create_provider",
            model=settings.ANTHROPIC_MODEL,
        )
        return AnthropicProvider(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    if provider_name == "ollama":
        logger.info(
            "creating ollama LLM provider",
            action="create_provider",
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        return OllamaProvider(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE,
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. Supported values: anthropic, ollama"
    )
