"""Anthropic Claude LLM provider implementation.

Implements stream_completion using the Anthropic Python SDK's
async streaming API (Messages API with stream=True).
"""

from collections.abc import AsyncIterator

import anthropic
import structlog

from src.llm.exceptions import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from src.llm.provider import HealthCheckResult, LLMProvider

logger = structlog.get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic Claude API.

    Uses the async Messages API with streaming enabled. The model
    is configurable via the ANTHROPIC_MODEL environment variable.

    Attributes:
        _client: The async Anthropic API client.
        _model: The model identifier to use.
        _temperature: Temperature for response generation.
    """

    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model identifier (e.g., "claude-sonnet-4-20250514").
            temperature: Temperature for plan generation.
        """
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def health_check(self) -> HealthCheckResult:
        """Check Anthropic API connectivity by listing models.

        Uses the models list endpoint which verifies both
        connectivity and API key validity without token usage.

        Returns:
            HealthCheckResult with status and details.
        """
        try:
            await self._client.models.list(limit=1)
            return HealthCheckResult(
                status="healthy",
                details=f"Anthropic API reachable, model '{self._model}'",
            )
        except anthropic.AuthenticationError:
            return HealthCheckResult(
                status="unhealthy",
                details="Invalid or missing API key",
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError):
            return HealthCheckResult(
                status="unhealthy",
                details="Cannot connect to Anthropic API",
            )
        except Exception as exc:
            return HealthCheckResult(status="unhealthy", details=str(exc))

    async def stream_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield text chunks from the Anthropic Claude API.

        Args:
            system_prompt: The system-level instructions.
            messages: Conversation messages with role/content dicts.
            max_tokens: Maximum tokens in the response.

        Yields:
            Text chunks as Claude generates them.

        Raises:
            LLMUnavailableError: API connection failed or 5xx error.
            LLMTimeoutError: API request timed out.
            LLMAuthenticationError: Invalid API key.
            LLMRateLimitError: Rate limit exceeded.
            LLMProviderError: Any other API error.
        """
        logger.debug(
            "starting anthropic stream",
            action="anthropic_stream",
            model=self._model,
        )

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.APITimeoutError as exc:
            logger.error(
                "anthropic API request timed out",
                action="anthropic_stream",
                error=str(exc),
            )
            raise LLMTimeoutError(f"Anthropic API request timed out: {exc}") from exc

        except anthropic.APIConnectionError as exc:
            logger.error(
                "anthropic API connection failed",
                action="anthropic_stream",
                error=str(exc),
            )
            raise LLMUnavailableError(f"Cannot connect to Anthropic API: {exc}") from exc

        except anthropic.AuthenticationError as exc:
            logger.error(
                "anthropic API authentication failed",
                action="anthropic_stream",
                error=str(exc),
            )
            raise LLMAuthenticationError(f"Anthropic authentication failed: {exc}") from exc

        except anthropic.RateLimitError as exc:
            logger.warn(
                "anthropic API rate limit exceeded",
                action="anthropic_stream",
                error=str(exc),
            )
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {exc}") from exc

        except anthropic.APIStatusError as exc:
            logger.error(
                "anthropic API status error",
                action="anthropic_stream",
                status_code=exc.status_code,
                error=str(exc),
            )
            if exc.status_code >= 500:
                raise LLMUnavailableError(
                    f"Anthropic API server error ({exc.status_code}): {exc}"
                ) from exc
            raise LLMProviderError(f"Anthropic API error ({exc.status_code}): {exc}") from exc

        except anthropic.APIError as exc:
            logger.error(
                "anthropic API error",
                action="anthropic_stream",
                error=str(exc),
            )
            raise LLMProviderError(f"Anthropic API error: {exc}") from exc

        logger.info(
            "anthropic stream completed",
            action="anthropic_stream",
            model=self._model,
        )
