"""POST /interpret endpoint with real LLM-powered interpretation.

Validates the request, then streams NDJSON events as the LLM analyzes
the query and produces a structured execution plan.
Conforms to the /interpret endpoint defined in internal-api.yaml.
"""

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.streaming import ndjson_stream_response
from src.config import get_settings
from src.models.query_plan import QueryPlan
from src.models.tool_catalog import ToolCatalog
from src.orchestrator.interpret_orchestrator import interpret_stream

logger = structlog.get_logger(__name__)
router = APIRouter()


class InterpretRequest(BaseModel):
    """Request body for POST /interpret.

    Attributes:
        query_id: UUID of the query record.
        query_text: The user's natural language question.
        tool_catalog: Filtered set of available MCP tools.
        query_plan_templates: Optional verified plan templates.
    """

    query_id: str
    query_text: str
    tool_catalog: ToolCatalog
    query_plan_templates: list[QueryPlan] | None = None


def _catalog_has_tools(request: InterpretRequest) -> bool:
    """Check if the request's tool catalog contains at least one tool.

    Args:
        request: The validated interpretation request.

    Returns:
        True if at least one integration has at least one tool.
    """
    return any(len(integration.tools) > 0 for integration in request.tool_catalog.integrations)


@router.post("/interpret")
async def interpret(request: Request) -> JSONResponse:
    """Interpret a natural language query and stream NDJSON events.

    Validates the request body before streaming. Returns a 400 JSON
    error for invalid requests, or a 200 NDJSON stream for valid ones.

    Pre-stream validation:
    - Request body must be valid JSON with required fields
    - Tool catalog must contain at least one tool

    Args:
        request: The incoming FastAPI request.

    Returns:
        JSONResponse on validation error, or StreamingResponse with NDJSON.
    """
    logger.debug("interpret request received", action="interpret_validate")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "validation.invalid_request",
                    "message": "Request body must be valid JSON.",
                    "stage": "validation",
                }
            },
        )

    for field in ("query_id", "query_text", "tool_catalog"):
        if field not in body:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "validation.missing_field",
                        "message": f"Required field '{field}' is missing.",
                        "details": {"field": field},
                        "stage": "validation",
                    }
                },
            )

    try:
        interpret_request = InterpretRequest(**body)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "validation.invalid_request",
                    "message": f"Invalid request body: {exc}",
                    "stage": "validation",
                }
            },
        )

    # Pre-stream validation: catalog must contain tools
    if not interpret_request.tool_catalog.integrations or not _catalog_has_tools(interpret_request):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "interpretation.invalid_catalog",
                    "message": "Tool catalog is empty or contains no valid "
                    "tool definitions. Cannot generate a plan.",
                    "stage": "interpretation",
                }
            },
        )

    logger.info(
        "interpret request validated, starting stream",
        action="interpret_validate",
        query_id=interpret_request.query_id,
    )

    provider = request.app.state.llm_provider
    settings = get_settings()

    return ndjson_stream_response(
        interpret_stream(
            query_id=interpret_request.query_id,
            query_text=interpret_request.query_text,
            tool_catalog=interpret_request.tool_catalog,
            provider=provider,
            settings=settings,
        )
    )
