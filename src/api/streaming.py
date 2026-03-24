"""Shared NDJSON streaming infrastructure for /interpret and /execute.

Provides a helper that wraps an async generator of Pydantic event models
into a FastAPI StreamingResponse with application/x-ndjson content type.
"""

from collections.abc import AsyncGenerator

import structlog
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


async def _serialize_events(
    event_generator: AsyncGenerator[BaseModel, None],
) -> AsyncGenerator[str, None]:
    """Serialize Pydantic event models to NDJSON lines.

    Args:
        event_generator: Async generator yielding Pydantic BaseModel instances.

    Yields:
        JSON-serialized event string followed by a newline character.
    """
    async for event in event_generator:
        yield event.model_dump_json() + "\n"


def ndjson_stream_response(
    event_generator: AsyncGenerator[BaseModel, None],
) -> StreamingResponse:
    """Create a FastAPI StreamingResponse for an NDJSON event stream.

    Wraps an async generator of Pydantic event models into a
    StreamingResponse with content-type application/x-ndjson. Each
    event is serialized to JSON via Pydantic's model_dump_json()
    followed by a newline character.

    Args:
        event_generator: Async generator yielding Pydantic event models.

    Returns:
        StreamingResponse: FastAPI response object for NDJSON streaming.
    """
    return StreamingResponse(
        content=_serialize_events(event_generator),
        media_type="application/x-ndjson",
    )
