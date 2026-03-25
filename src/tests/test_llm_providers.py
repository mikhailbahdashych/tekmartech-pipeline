"""Tests for LLM provider implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.exceptions import (
    LLMAuthenticationError,
    LLMTimeoutError,
    LLMUnavailableError,
)

# =============================================================================
# Anthropic provider tests
# =============================================================================


@pytest.mark.asyncio
async def test_anthropic_streams_text_deltas():
    """AnthropicProvider yields text chunks from the streaming API."""

    from src.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key", model="test-model", temperature=0.1)

    # Mock the streaming context manager
    mock_text_stream = AsyncMock()
    mock_text_stream.__aiter__ = lambda self: self
    chunks = iter(["Hello ", "world", "!"])
    mock_text_stream.__anext__ = AsyncMock(side_effect=lambda: next(chunks, StopAsyncIteration()))

    # Actually, let's use a simpler approach with async generator
    async def mock_text_gen():
        for text in ["Hello ", "world", "!"]:
            yield text

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock()
    mock_stream_ctx.__aenter__.return_value = mock_stream_ctx
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_stream_ctx.text_stream = mock_text_gen()

    with patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx):
        result = []
        async for chunk in provider.stream_completion("system", "user", 100):
            result.append(chunk)

    assert result == ["Hello ", "world", "!"]


@pytest.mark.asyncio
async def test_anthropic_connection_error_raises_unavailable():
    """AnthropicProvider raises LLMUnavailableError on connection failure."""
    import anthropic

    from src.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key", model="test-model", temperature=0.1)

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(
        side_effect=anthropic.APIConnectionError(request=MagicMock())
    )
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx),
        pytest.raises(LLMUnavailableError),
    ):
        async for _ in provider.stream_completion("system", "user", 100):
            pass


@pytest.mark.asyncio
async def test_anthropic_timeout_raises_timeout():
    """AnthropicProvider raises LLMTimeoutError on timeout."""
    import anthropic

    from src.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key", model="test-model", temperature=0.1)

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(
        side_effect=anthropic.APITimeoutError(request=MagicMock())
    )
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx),
        pytest.raises(LLMTimeoutError),
    ):
        async for _ in provider.stream_completion("system", "user", 100):
            pass


@pytest.mark.asyncio
async def test_anthropic_auth_error_raises_authentication():
    """AnthropicProvider raises LLMAuthenticationError on auth failure."""
    import anthropic

    from src.llm.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="bad-key", model="test-model", temperature=0.1)

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(
        side_effect=anthropic.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )
    )
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx),
        pytest.raises(LLMAuthenticationError),
    ):
        async for _ in provider.stream_completion("system", "user", 100):
            pass


# =============================================================================
# Ollama provider tests
# =============================================================================


@pytest.mark.asyncio
async def test_ollama_streams_text_from_ndjson():
    """OllamaProvider yields text from Ollama NDJSON streaming response."""
    from src.llm.ollama_provider import OllamaProvider

    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3", temperature=0.1)

    ndjson_lines = [
        '{"response": "Hello ", "done": false}',
        '{"response": "world", "done": false}',
        '{"response": "!", "done": true}',
    ]

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()

    async def mock_aiter_lines():
        for line in ndjson_lines:
            yield line

    mock_response.aiter_lines = mock_aiter_lines

    mock_client = AsyncMock()
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        result = []
        async for chunk in provider.stream_completion("system", "user", 100):
            result.append(chunk)

    assert result == ["Hello ", "world", "!"]


@pytest.mark.asyncio
async def test_ollama_connection_error_raises_unavailable():
    """OllamaProvider raises LLMUnavailableError when Ollama is not running."""
    from src.llm.ollama_provider import OllamaProvider

    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3", temperature=0.1)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client),
        pytest.raises(LLMUnavailableError, match="Cannot connect"),
    ):
        async for _ in provider.stream_completion("system", "user", 100):
            pass
