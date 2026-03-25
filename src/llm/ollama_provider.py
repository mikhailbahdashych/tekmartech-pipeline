"""Ollama local LLM provider implementation.

Implements stream_completion using httpx to call the Ollama REST API
(POST /api/generate with stream: true). Parses the streaming NDJSON
response and yields text chunks.
"""

import json
from collections.abc import AsyncIterator

import httpx
import structlog

from src.llm.exceptions import (
    LLMProviderError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from src.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using the Ollama local inference server.

    Communicates with Ollama via its REST API. The base URL and model
    are configurable via environment variables.

    Attributes:
        _base_url: Ollama server base URL.
        _model: Model name to use (e.g., "llama3").
        _temperature: Temperature for response generation.
    """

    def __init__(self, base_url: str, model: str, temperature: float) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL (e.g., "http://localhost:11434").
            model: Ollama model name.
            temperature: Temperature for plan generation.
        """
        self._base_url = base_url
        self._model = model
        self._temperature = temperature

    async def stream_completion(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield text chunks from the Ollama API.

        Args:
            system_prompt: The system-level instructions.
            user_message: The user's message (query_text).
            max_tokens: Maximum tokens in the response.

        Yields:
            Text chunks as Ollama generates them.

        Raises:
            LLMUnavailableError: Cannot connect to Ollama server.
            LLMTimeoutError: Ollama request timed out.
            LLMProviderError: Any other error.
        """
        logger.debug(
            "starting ollama stream",
            action="ollama_stream",
            model=self._model,
            base_url=self._base_url,
        )

        payload = {
            "model": self._model,
            "system": system_prompt,
            "prompt": user_message,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with (
                httpx.AsyncClient(base_url=self._base_url, timeout=120.0) as client,
                client.stream("POST", "/api/generate", json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk_data = json.loads(line)
                    text = chunk_data.get("response", "")
                    if text:
                        yield text
                    if chunk_data.get("done", False):
                        break

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.error(
                "cannot connect to ollama server",
                action="ollama_stream",
                base_url=self._base_url,
                error=str(exc),
            )
            raise LLMUnavailableError(
                f"Cannot connect to Ollama at {self._base_url}: {exc}"
            ) from exc

        except httpx.ReadTimeout as exc:
            logger.error(
                "ollama request timed out",
                action="ollama_stream",
                error=str(exc),
            )
            raise LLMTimeoutError(f"Ollama request timed out: {exc}") from exc

        except httpx.HTTPStatusError as exc:
            logger.error(
                "ollama HTTP error",
                action="ollama_stream",
                status_code=exc.response.status_code,
                error=str(exc),
            )
            raise LLMProviderError(
                f"Ollama HTTP error ({exc.response.status_code}): {exc}"
            ) from exc

        except httpx.HTTPError as exc:
            logger.error(
                "ollama HTTP error",
                action="ollama_stream",
                error=str(exc),
            )
            raise LLMProviderError(f"Ollama error: {exc}") from exc

        logger.info(
            "ollama stream completed",
            action="ollama_stream",
            model=self._model,
        )
