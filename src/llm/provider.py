"""Abstract base class for LLM provider implementations.

Defines the single interface that all LLM providers must implement.
The Interpreter calls stream_completion without knowing which provider
is running. Conforms to the LLM Provider Abstraction described in CLAUDE.md.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    """Abstract base class for LLM provider implementations.

    Every provider (Anthropic, Ollama, and future ones) implements
    the stream_completion method. The Interpreter calls it without
    knowing which provider is active.
    """

    @abstractmethod
    async def stream_completion(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield text chunks as the LLM generates them.

        Args:
            system_prompt: The system-level instructions for the LLM.
            user_message: The user's message (query_text).
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
