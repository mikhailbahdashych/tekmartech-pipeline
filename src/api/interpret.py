"""POST /interpret endpoint with mock streaming implementation.

Validates the request, then streams NDJSON events simulating the
interpretation flow: started → text deltas → plan generated.
Conforms to the /interpret endpoint defined in internal-api.yaml.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.streaming import ndjson_stream_response
from src.models.query_plan import PlanStep, QueryPlan
from src.models.stream_events import (
    InterpretationError,
    InterpretationPlanGenerated,
    InterpretationStarted,
    InterpretationTextDelta,
)
from src.models.tool_catalog import ToolCatalog

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


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


async def _mock_interpret_stream(
    request: InterpretRequest,
) -> AsyncGenerator[BaseModel, None]:
    """Generate mock interpretation NDJSON events.

    Streams: interpretation_started → 3x text_delta → plan_generated.
    Includes error handling to ensure a terminal event is always emitted.

    Args:
        request: The validated interpretation request.

    Yields:
        Pydantic event models for NDJSON serialization.
    """
    query_id = request.query_id

    try:
        logger.debug(
            "starting mock interpretation",
            action="interpret_stream",
            query_id=query_id,
        )

        yield InterpretationStarted(
            query_id=query_id,
            timestamp=_now_iso(),
        )

        delta_texts = [
            "Analyzing your query...",
            "Identifying relevant tools...",
            "Generating execution plan...",
        ]
        full_text = ""

        for text in delta_texts:
            await asyncio.sleep(0.1)
            full_text += text + " "
            yield InterpretationTextDelta(
                query_id=query_id,
                text_delta=text,
                timestamp=_now_iso(),
            )

        mock_plan = QueryPlan(
            plan_id="mock-plan-001",
            plan_version="1.0",
            steps=[
                PlanStep(
                    step_id="step_1",
                    tool_name="aws.iam_list_users",
                    integration_id="integration-001",
                    parameters={"max_results": 1000},
                    description="List all IAM users from the AWS account",
                    output_alias="all_iam_users",
                ),
                PlanStep(
                    step_id="step_2",
                    tool_name="aws.iam_get_account_summary",
                    integration_id="integration-001",
                    parameters={},
                    description="Get IAM account summary for MFA statistics",
                    output_alias="account_summary",
                ),
            ],
            estimated_tool_calls=2,
            summary="Retrieve all IAM users and account summary to identify users without MFA.",
        )

        yield InterpretationPlanGenerated(
            query_id=query_id,
            query_plan=mock_plan,
            plan_summary=mock_plan.summary,
            estimated_duration_seconds=5,
            full_interpretation_text=full_text.strip(),
            timestamp=_now_iso(),
        )

        logger.info(
            "mock interpretation completed",
            action="interpret_stream",
            query_id=query_id,
        )

    except Exception as exc:
        logger.error(
            "unexpected error during interpretation stream",
            action="interpret_stream",
            query_id=query_id,
            error=str(exc),
        )
        yield InterpretationError(
            query_id=query_id,
            error_code="interpretation.internal_error",
            error_message=f"Internal error during interpretation: {exc}",
            timestamp=_now_iso(),
        )


@router.post("/interpret")
async def interpret(request: Request) -> JSONResponse:
    """Interpret a natural language query and stream NDJSON events.

    Validates the request body before streaming. Returns a 400 JSON
    error for invalid requests, or a 200 NDJSON stream for valid ones.

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

    logger.info(
        "interpret request validated, starting stream",
        action="interpret_validate",
        query_id=interpret_request.query_id,
    )

    return ndjson_stream_response(_mock_interpret_stream(interpret_request))
