"""Execution orchestrator — coordinates the full /execute flow.

This async generator wires the execution engine, output engine, and
MCP client together. It produces the complete NDJSON event stream
with a guaranteed terminal event.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from src.config import Settings
from src.execution.engine import ExecutionContext, execute_plan
from src.mcp.client import MCPClient
from src.mcp.server_registry import ServerRegistry
from src.models.credential_envelope import CredentialEnvelope
from src.models.query_plan import QueryPlan
from src.models.stream_events import (
    ExecutionCompleted,
    ExecutionError,
    ExecutionFailed,
    ExecutionStarted,
)
from src.output.csv_generator import generate_csv
from src.output.formatter import format_results
from src.output.summary_generator import generate_summary
from src.output.transparency_log_builder import build_transparency_log

logger = structlog.get_logger(__name__)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


async def execute_stream(
    query_id: str,
    query_plan: QueryPlan,
    credentials: dict[str, CredentialEnvelope],
    server_registry: ServerRegistry,
    settings: Settings,
) -> AsyncGenerator[BaseModel, None]:
    """Generate the execution NDJSON event stream.

    Coordinates: engine execution → output formatting → terminal event.
    Guarantees exactly one terminal event regardless of outcome.

    Args:
        query_id: UUID of the query being executed.
        query_plan: The approved query plan to execute.
        credentials: Map of integration_id to credential_envelope.
        server_registry: Registry of available MCP servers.
        settings: Application settings.

    Yields:
        Pydantic event models for NDJSON serialization.
    """
    execution_started_at = _now_iso()

    logger.debug(
        "starting execution stream",
        action="execute_stream",
        query_id=query_id,
        total_steps=len(query_plan.steps),
    )

    try:
        yield ExecutionStarted(
            query_id=query_id,
            plan_id=query_plan.plan_id,
            total_steps=len(query_plan.steps),
            timestamp=execution_started_at,
        )

        context = ExecutionContext()

        def mcp_client_factory(server_type: str) -> MCPClient:
            """Create an MCPClient for the given server type.

            Args:
                server_type: The MCP server type to connect to.

            Returns:
                An MCPClient instance (use as async context manager).

            Raises:
                RuntimeError: If no server is registered for the type.
            """
            config = server_registry.get_server_config(server_type)
            if config is None:
                raise RuntimeError(f"No MCP server registered for type '{server_type}'")
            return MCPClient(config)

        try:
            async with asyncio.timeout(settings.EXECUTION_TIMEOUT_SECONDS):
                async for event in execute_plan(
                    query_id=query_id,
                    plan=query_plan,
                    credentials=credentials,
                    mcp_client_factory=mcp_client_factory,
                    settings=settings,
                    context=context,
                ):
                    yield event

        except TimeoutError:
            logger.error(
                "execution timed out",
                action="execute_stream",
                query_id=query_id,
                timeout=settings.EXECUTION_TIMEOUT_SECONDS,
            )
            execution_completed_at = _now_iso()
            transparency_log = build_transparency_log(
                query_id=query_id,
                plan_id=query_plan.plan_id,
                execution_started_at=execution_started_at,
                execution_completed_at=execution_completed_at,
                entries=context.log_entries,
                execution_status="failed",
            )
            yield ExecutionError(
                query_id=query_id,
                error_code="execution.timeout",
                error_message=f"Execution timed out after {settings.EXECUTION_TIMEOUT_SECONDS}s",
                transparency_log=transparency_log,
                timestamp=execution_completed_at,
            )
            return

        except Exception as exc:
            logger.error(
                "unexpected error during execution",
                action="execute_stream",
                query_id=query_id,
                error=str(exc),
            )
            execution_completed_at = _now_iso()
            transparency_log = build_transparency_log(
                query_id=query_id,
                plan_id=query_plan.plan_id,
                execution_started_at=execution_started_at,
                execution_completed_at=execution_completed_at,
                entries=context.log_entries,
                execution_status="failed",
            )
            yield ExecutionError(
                query_id=query_id,
                error_code="execution.internal_error",
                error_message=f"Unexpected error during execution: {exc}",
                transparency_log=transparency_log,
                timestamp=execution_completed_at,
            )
            return

        # Build terminal event
        execution_completed_at = _now_iso()

        if context.steps_succeeded == 0:
            # All steps failed
            transparency_log = build_transparency_log(
                query_id=query_id,
                plan_id=query_plan.plan_id,
                execution_started_at=execution_started_at,
                execution_completed_at=execution_completed_at,
                entries=context.log_entries,
                execution_status="failed",
            )
            yield ExecutionFailed(
                query_id=query_id,
                error_code="execution.all_steps_failed",
                error_message="All execution steps failed. No usable data was collected.",
                transparency_log=transparency_log,
                timestamp=execution_completed_at,
            )
            return

        # Some or all steps succeeded — produce results
        execution_status = "completed" if context.steps_failed == 0 else "partial"
        execution_duration_ms = _compute_duration_ms(execution_started_at, execution_completed_at)

        try:
            result_data = format_results(
                step_results=context.step_results,
                plan=query_plan,
                execution_duration_ms=execution_duration_ms,
            )
            result_summary = generate_summary(
                result_data=result_data,
                steps_succeeded=context.steps_succeeded,
                steps_failed=context.steps_failed,
            )
            export_csv = generate_csv(result_data)
        except Exception as exc:
            logger.error(
                "output formatting failed",
                action="execute_stream",
                query_id=query_id,
                error=str(exc),
            )
            transparency_log = build_transparency_log(
                query_id=query_id,
                plan_id=query_plan.plan_id,
                execution_started_at=execution_started_at,
                execution_completed_at=execution_completed_at,
                entries=context.log_entries,
                execution_status="failed",
            )
            yield ExecutionError(
                query_id=query_id,
                error_code="execution.internal_error",
                error_message=f"Failed to format results: {exc}",
                transparency_log=transparency_log,
                timestamp=execution_completed_at,
            )
            return

        transparency_log = build_transparency_log(
            query_id=query_id,
            plan_id=query_plan.plan_id,
            execution_started_at=execution_started_at,
            execution_completed_at=execution_completed_at,
            entries=context.log_entries,
            execution_status=execution_status,
        )

        yield ExecutionCompleted(
            query_id=query_id,
            execution_status=execution_status,
            result_data=result_data,
            result_summary=result_summary,
            transparency_log=transparency_log,
            export_csv=export_csv,
            timestamp=execution_completed_at,
        )

        logger.info(
            "execution stream completed",
            action="execute_stream",
            query_id=query_id,
            execution_status=execution_status,
            steps_succeeded=context.steps_succeeded,
            steps_failed=context.steps_failed,
        )

    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "execution stream cancelled by client disconnect",
            action="execute_stream",
            query_id=query_id,
        )
        return


def _compute_duration_ms(started_at: str, completed_at: str) -> int:
    """Compute duration in milliseconds between two ISO timestamps.

    Args:
        started_at: ISO 8601 start time.
        completed_at: ISO 8601 end time.

    Returns:
        Duration in milliseconds.
    """
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
        return int((end - start).total_seconds() * 1000)
    except (ValueError, TypeError):
        return 0
