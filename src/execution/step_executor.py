"""Single step executor.

Executes a single plan step: resolves templates, looks up credentials,
invokes the MCP tool, applies transforms, and builds a transparency log entry.
"""

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import structlog

from src.config import Settings
from src.execution.template_resolver import TemplateResolutionError, resolve_parameters
from src.execution.transform_engine import apply_transform
from src.mcp.client import MCPClient
from src.models.credential_envelope import CredentialEnvelope
from src.models.query_plan import PlanStep
from src.models.tool_invocation import ToolInvocationRequest
from src.models.transparency_log import TransparencyLogEntry

logger = structlog.get_logger(__name__)

_SENSITIVE_PATTERNS = frozenset(
    {
        "password",
        "token",
        "secret",
        "key",
        "credential",
        "api_key",
        "access_key",
        "pat",
        "secret_access_key",
        "personal_access_token",
        "session_token",
    }
)


@dataclass
class StepResult:
    """Result of executing a single plan step.

    Attributes:
        step_id: The plan step identifier.
        status: Whether the step succeeded or failed.
        data: The tool output data (on success).
        record_count: Number of records returned.
        summary: Brief description of the result.
        data_hash: SHA-256 hash of the data.
        duration_ms: Time taken in milliseconds.
        error_code: Error code (on failure).
        error_message: Error description (on failure).
        retryable: Whether the failure is retryable.
        log_entries: Transparency log entries for this step.
    """

    step_id: str
    status: Literal["success", "error"]
    data: dict | None = None
    record_count: int = 0
    summary: str = ""
    data_hash: str | None = None
    duration_ms: int = 0
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    log_entries: list[TransparencyLogEntry] = field(default_factory=list)


def redact_parameters(params: dict) -> dict:
    """Redact sensitive values from parameters for transparency logging.

    Replaces values of keys that match sensitive patterns with "[REDACTED]".

    Args:
        params: The parameters dict to redact.

    Returns:
        A new dict with sensitive values replaced.
    """
    redacted = {}
    for key, value in params.items():
        if key.lower() in _SENSITIVE_PATTERNS:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = redact_parameters(value)
        else:
            redacted[key] = value
    return redacted


def _compute_data_hash(data: dict) -> str:
    """Compute SHA-256 hash of JSON-serialized data.

    Args:
        data: Dictionary to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _count_records(data: dict) -> int:
    """Count records in tool output by finding the largest array.

    Args:
        data: The tool output data.

    Returns:
        The count of items in the largest array found, or 0.
    """
    max_count = 0
    for value in data.values():
        if isinstance(value, list):
            max_count = max(max_count, len(value))
    return max_count


async def execute_step(
    step: PlanStep,
    credentials: dict[str, CredentialEnvelope],
    step_results: dict[str, dict],
    mcp_client_factory: Callable[[str], MCPClient],
    settings: Settings,
) -> StepResult:
    """Execute a single plan step.

    Resolves template references, looks up credentials, invokes the MCP tool,
    applies transforms, and builds a transparency log entry.

    Args:
        step: The plan step to execute.
        credentials: Map of integration_id to credential_envelope.
        step_results: Map of completed step_id to output data.
        mcp_client_factory: Factory returning an MCPClient context manager.
        settings: Application settings.

    Returns:
        StepResult with data, log entry, and status.
    """
    started_at = datetime.now(UTC)
    invocation_id = str(uuid.uuid4())

    logger.debug(
        "executing step",
        action="execute_step",
        step_id=step.step_id,
        tool_name=step.tool_name,
        integration_id=step.integration_id,
    )

    # Look up credentials
    envelope = credentials.get(step.integration_id)
    if envelope is None:
        return _build_failed_result(
            step,
            invocation_id,
            started_at,
            "execution.credential_error",
            f"No credentials provided for integration '{step.integration_id}'",
            retryable=False,
        )

    # Resolve template references
    try:
        resolved_params = resolve_parameters(step.parameters, step_results)
    except TemplateResolutionError as exc:
        return _build_failed_result(
            step,
            invocation_id,
            started_at,
            "execution.plan_invalid",
            f"Template resolution failed: {exc}",
            retryable=False,
        )

    # Build invocation request
    request = ToolInvocationRequest(
        tool_name=step.tool_name,
        parameters=resolved_params,
        credentials=envelope,
        invocation_id=invocation_id,
        timeout_seconds=settings.STEP_TIMEOUT_SECONDS,
    )

    # Invoke MCP tool
    try:
        mcp_client = mcp_client_factory(envelope.server_type)
        async with mcp_client as client:
            response = await client.invoke_tool(request)
    except Exception as exc:
        logger.error(
            "MCP invocation failed",
            action="execute_step",
            step_id=step.step_id,
            error=str(exc),
        )
        return _build_failed_result(
            step,
            invocation_id,
            started_at,
            "execution.tool_invocation_failed",
            f"Tool invocation failed: {exc}",
            retryable=True,
            credential_mode=envelope.credential_mode,
        )

    completed_at = datetime.now(UTC)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    # Handle error response from MCP server
    if response.status == "error":
        error_code = response.error.code if response.error else "internal.server_error"
        error_message = response.error.message if response.error else "Unknown error"
        retryable = (
            response.error.details.retryable if response.error and response.error.details else False
        )

        log_entry = _build_log_entry(
            step,
            invocation_id,
            resolved_params,
            envelope.credential_mode,
            "error",
            started_at,
            completed_at,
            duration_ms,
            response.metadata.external_api_calls,
            error_code=error_code,
            error_message=error_message,
        )

        return StepResult(
            step_id=step.step_id,
            status="error",
            duration_ms=duration_ms,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable or False,
            log_entries=[log_entry],
        )

    # Success — apply transform if present
    data = response.data or {}
    if step.transform:
        data = apply_transform(data, step.transform)

    data_hash = _compute_data_hash(data)
    record_count = _count_records(data)

    log_entry = _build_log_entry(
        step,
        invocation_id,
        resolved_params,
        envelope.credential_mode,
        "success",
        started_at,
        completed_at,
        duration_ms,
        response.metadata.external_api_calls,
        data_hash=data_hash,
    )

    logger.info(
        "step executed successfully",
        action="execute_step",
        step_id=step.step_id,
        record_count=record_count,
        duration_ms=duration_ms,
    )

    return StepResult(
        step_id=step.step_id,
        status="success",
        data=data,
        record_count=record_count,
        summary=f"{record_count} records retrieved from {step.tool_name}",
        data_hash=data_hash,
        duration_ms=duration_ms,
        log_entries=[log_entry],
    )


def _build_failed_result(
    step: PlanStep,
    invocation_id: str,
    started_at: datetime,
    error_code: str,
    error_message: str,
    retryable: bool,
    credential_mode: str = "direct",
) -> StepResult:
    """Build a StepResult for a failed step.

    Args:
        step: The plan step.
        invocation_id: The invocation UUID.
        started_at: When execution began.
        error_code: Machine-readable error code.
        error_message: Human-readable error description.
        retryable: Whether the failure is retryable.
        credential_mode: The credential transit mode.

    Returns:
        StepResult with error status and log entry.
    """
    completed_at = datetime.now(UTC)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    log_entry = _build_log_entry(
        step,
        invocation_id,
        step.parameters,
        credential_mode,
        "error",
        started_at,
        completed_at,
        duration_ms,
        0,
        error_code=error_code,
        error_message=error_message,
    )

    return StepResult(
        step_id=step.step_id,
        status="error",
        duration_ms=duration_ms,
        error_code=error_code,
        error_message=error_message,
        retryable=retryable,
        log_entries=[log_entry],
    )


def _build_log_entry(
    step: PlanStep,
    invocation_id: str,
    parameters: dict,
    credential_mode: str,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: int,
    external_api_calls: int,
    error_code: str | None = None,
    error_message: str | None = None,
    data_hash: str | None = None,
) -> TransparencyLogEntry:
    """Build a transparency log entry for a step invocation.

    Args:
        step: The plan step.
        invocation_id: UUID for this invocation.
        parameters: Resolved parameters (will be redacted).
        credential_mode: "broker" or "direct".
        status: "success", "error", or "partial".
        started_at: When invocation began.
        completed_at: When invocation ended.
        duration_ms: Wall-clock time.
        external_api_calls: Number of external API calls.
        error_code: Error code if failed.
        error_message: Error message if failed.
        data_hash: SHA-256 hash of data if succeeded.

    Returns:
        TransparencyLogEntry with all fields populated.
    """
    return TransparencyLogEntry(
        entry_id=str(uuid.uuid4()),
        step_id=step.step_id,
        invocation_id=invocation_id,
        tool_name=step.tool_name,
        integration_id=step.integration_id,
        parameters=redact_parameters(parameters),
        credential_mode=credential_mode,
        status=status,
        error_code=error_code,
        error_message=error_message,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_ms=duration_ms,
        external_api_calls=external_api_calls,
        data_hash=data_hash,
        is_retry=False,
    )
