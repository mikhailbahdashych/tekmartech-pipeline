"""POST /execute endpoint with mock streaming implementation.

Validates the request, then streams NDJSON events simulating the
execution flow: started → step events → completed.
Conforms to the /execute endpoint defined in internal-api.yaml.
"""

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.streaming import ndjson_stream_response
from src.models.query_plan import QueryPlan
from src.models.result_data import (
    IntegrationQueried,
    ResultColumn,
    ResultData,
    ResultMetadata,
    ResultTable,
)
from src.models.stream_events import (
    ExecutionCompleted,
    ExecutionError,
    ExecutionStarted,
    StepCompleted,
    StepStarted,
)
from src.models.transparency_log import TransparencyLog, TransparencyLogEntry

logger = structlog.get_logger(__name__)
router = APIRouter()


class ExecuteRequest(BaseModel):
    """Request body for POST /execute.

    Attributes:
        query_id: UUID of the query record.
        query_plan: The approved query plan to execute.
        credentials: Map of integration_id to credential_envelope.
    """

    query_id: str
    query_plan: QueryPlan
    credentials: dict


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _data_hash(data: dict) -> str:
    """Compute SHA-256 hash of JSON-serialized data.

    Args:
        data: Dictionary to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


async def _mock_execute_stream(
    request: ExecuteRequest,
) -> AsyncGenerator[BaseModel, None]:
    """Generate mock execution NDJSON events.

    Streams: execution_started → (step_started + step_completed) per step
    → execution_completed. Includes error handling to ensure a terminal
    event is always emitted.

    Args:
        request: The validated execution request.

    Yields:
        Pydantic event models for NDJSON serialization.
    """
    query_id = request.query_id
    plan = request.query_plan
    steps = plan.steps
    execution_start = _now_iso()
    log_entries: list[TransparencyLogEntry] = []

    try:
        logger.debug(
            "starting mock execution",
            action="execute_stream",
            query_id=query_id,
            total_steps=len(steps),
        )

        yield ExecutionStarted(
            query_id=query_id,
            plan_id=plan.plan_id,
            total_steps=len(steps),
            timestamp=execution_start,
        )

        total_records = 0
        for idx, step in enumerate(steps, start=1):
            step_start = _now_iso()
            yield StepStarted(
                query_id=query_id,
                step_id=step.step_id,
                step_index=idx,
                tool_name=step.tool_name,
                tool_display_name=step.tool_name.replace(".", " ").replace("_", " ").title(),
                integration_display_name=f"Integration {step.integration_id}",
                description=step.description,
                timestamp=step_start,
            )

            await asyncio.sleep(0.1)

            mock_data = {"records": [{"id": i, "name": f"record_{i}"} for i in range(5)]}
            mock_hash = _data_hash(mock_data)
            record_count = len(mock_data["records"])
            total_records += record_count

            log_entries.append(
                TransparencyLogEntry(
                    entry_id=f"entry-{idx}",
                    step_id=step.step_id,
                    invocation_id=f"invocation-{idx}",
                    tool_name=step.tool_name,
                    integration_id=step.integration_id,
                    parameters=step.parameters,
                    credential_mode="direct",
                    status="success",
                    started_at=step_start,
                    completed_at=_now_iso(),
                    duration_ms=100,
                    external_api_calls=1,
                    data_hash=mock_hash,
                    is_retry=False,
                )
            )

            yield StepCompleted(
                query_id=query_id,
                step_id=step.step_id,
                duration_ms=100,
                record_count=record_count,
                summary=f"{record_count} records retrieved from {step.tool_name}",
                data_hash=mock_hash,
                timestamp=_now_iso(),
            )

        execution_end = _now_iso()
        transparency_log = TransparencyLog(
            query_id=query_id,
            plan_id=plan.plan_id,
            execution_started_at=execution_start,
            execution_completed_at=execution_end,
            total_duration_ms=len(steps) * 100,
            total_tool_invocations=len(steps),
            total_external_api_calls=len(steps),
            execution_status="completed",
            entries=log_entries,
        )

        result_data = ResultData(
            format_version="1.0",
            tables=[
                ResultTable(
                    table_id="table_1",
                    title="Query Results",
                    columns=[
                        ResultColumn(key="id", label="ID", data_type="number"),
                        ResultColumn(key="name", label="Name", data_type="string"),
                    ],
                    rows=[[i, f"record_{i}"] for i in range(total_records)],
                    row_count=total_records,
                    source_step_ids=[s.step_id for s in steps],
                )
            ],
            metadata=ResultMetadata(
                total_records=total_records,
                integrations_queried=[
                    IntegrationQueried(
                        integration_id=steps[0].integration_id,
                        display_name=f"Integration {steps[0].integration_id}",
                        server_type="aws",
                    )
                ],
                execution_duration_ms=len(steps) * 100,
            ),
        )

        yield ExecutionCompleted(
            query_id=query_id,
            execution_status="completed",
            result_data=result_data,
            result_summary=f"Retrieved {total_records} records across {len(steps)} steps.",
            transparency_log=transparency_log,
            timestamp=execution_end,
        )

        logger.info(
            "mock execution completed",
            action="execute_stream",
            query_id=query_id,
            total_steps=len(steps),
        )

    except Exception as exc:
        logger.error(
            "unexpected error during execution stream",
            action="execute_stream",
            query_id=query_id,
            error=str(exc),
        )
        yield ExecutionError(
            query_id=query_id,
            error_code="execution.internal_error",
            error_message=f"Internal error during execution: {exc}",
            timestamp=_now_iso(),
        )


@router.post("/execute")
async def execute(request: Request) -> JSONResponse:
    """Execute an approved query plan and stream NDJSON events.

    Validates the request body before streaming. Returns a 400 JSON
    error for invalid requests, or a 200 NDJSON stream for valid ones.

    Args:
        request: The incoming FastAPI request.

    Returns:
        JSONResponse on validation error, or StreamingResponse with NDJSON.
    """
    logger.debug("execute request received", action="execute_validate")

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

    for field in ("query_id", "query_plan", "credentials"):
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
        execute_request = ExecuteRequest(**body)
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
        "execute request validated, starting stream",
        action="execute_validate",
        query_id=execute_request.query_id,
    )

    return ndjson_stream_response(_mock_execute_stream(execute_request))
