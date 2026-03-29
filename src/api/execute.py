"""POST /execute endpoint with real execution engine.

Validates the request, then streams NDJSON events as the execution
engine processes each plan step against MCP servers.
Conforms to the /execute endpoint defined in internal-api.yaml.
"""

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.streaming import ndjson_stream_response
from src.config import get_settings
from src.models.credential_envelope import CredentialEnvelope
from src.models.query_plan import QueryPlan
from src.orchestrator.execute_orchestrator import execute_stream

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
    credentials: dict[str, dict]


@router.post("/execute")
async def execute(request: Request) -> JSONResponse:
    """Execute an approved query plan and stream NDJSON events.

    Validates the request body before streaming. Returns a 400 JSON
    error for invalid requests, or a 200 NDJSON stream for valid ones.

    Pre-stream validation:
    - Request body must be valid JSON with required fields
    - Every integration_id in the plan must have credentials

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

    # Parse credential envelopes
    parsed_credentials: dict[str, CredentialEnvelope] = {}
    try:
        for int_id, cred_data in execute_request.credentials.items():
            parsed_credentials[int_id] = CredentialEnvelope(**cred_data)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "execution.credential_error",
                    "message": f"Invalid credential format: {exc}",
                    "stage": "execution",
                }
            },
        )

    # Validate every integration_id in the plan has credentials
    plan_integration_ids = {step.integration_id for step in execute_request.query_plan.steps}
    missing_creds = plan_integration_ids - set(parsed_credentials.keys())
    if missing_creds:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "execution.credential_error",
                    "message": "Missing credentials for integration(s): "
                    f"{', '.join(sorted(missing_creds))}",
                    "details": {"missing_integration_ids": sorted(missing_creds)},
                    "stage": "execution",
                }
            },
        )

    logger.info(
        "execute request validated, starting stream",
        action="execute_validate",
        query_id=execute_request.query_id,
    )

    server_registry = request.app.state.server_registry
    settings = get_settings()

    return ndjson_stream_response(
        execute_stream(
            query_id=execute_request.query_id,
            query_plan=execute_request.query_plan,
            credentials=parsed_credentials,
            server_registry=server_registry,
            settings=settings,
        )
    )
