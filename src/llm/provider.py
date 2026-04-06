"""Abstract base class for LLM provider implementations.

Defines the single interface that all LLM providers must implement.
The Interpreter calls stream_completion without knowing which provider
is running. Conforms to the LLM Provider Abstraction described in CLAUDE.md.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, datetime


class HealthCheckResult:
    """Result of an LLM provider health check.

    Attributes:
        status: "healthy" or "unhealthy".
        checked_at: ISO 8601 timestamp of when the check was performed.
        details: Optional message with additional context.
    """

    def __init__(self, status: str, details: str | None = None) -> None:
        self.status = status
        self.checked_at = datetime.now(UTC).isoformat()
        self.details = details

    def to_dict(self) -> dict:
        """Serialize to a dict for the health endpoint response."""
        result: dict = {
            "status": self.status,
            "checked_at": self.checked_at,
        }
        if self.details:
            result["details"] = self.details
        return result


class LLMProvider(ABC):
    """Abstract base class for LLM provider implementations.

    Every provider (Anthropic, Ollama, and future ones) implements
    the stream_completion method. The Interpreter calls it without
    knowing which provider is active.
    """

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Check whether the provider is reachable and operational.

        Each provider implements a lightweight, non-billable probe
        (e.g., listing models) to verify connectivity and auth.

        Returns:
            HealthCheckResult with status and timestamp.
        """
        ...

    @abstractmethod
    async def stream_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield text chunks as the LLM generates them.

        Args:
            system_prompt: The system-level instructions for the LLM.
            messages: List of conversation messages, each with "role"
                and "content" keys. Minimum one message with role "user".
                For single-turn: [{"role": "user", "content": query_text}].
                For multi-turn: alternating user/assistant messages.
            max_tokens: Maximum tokens in the LLM response.

        Yields:
            Text chunks from the LLM response as they are generated.

        Raises:
            LLMUnavailableError: The LLM service is unreachable.
            LLMTimeoutError: The LLM did not respond in time.
            LLMAuthenticationError: Invalid or missing API key.
            LLMRateLimitError: Rate limit exceeded.
            LLMProviderError: Any other provider-level error.
        """
        yield ""  # pragma: no cover
