"""Execution engine — processes a query plan step by step.

Executes steps sequentially, emitting step_started/step_completed/step_failed
events. Populates an ExecutionContext with results and log entries for the
orchestrator to use when building the terminal event. Error containment
ensures a single step failure does not crash the entire execution.

Architectural Invariant #7: This engine contains no AI. It reads the plan
literally and executes it deterministically.
"""

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from src.config import Settings
from src.execution.step_executor import StepResult, execute_step
from src.mcp.client import MCPClient
from src.models.credential_envelope import CredentialEnvelope
from src.models.query_plan import QueryPlan
from src.models.stream_events import StepCompleted, StepFailed, StepStarted
from src.models.transparency_log import TransparencyLogEntry

logger = structlog.get_logger(__name__)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass
class ExecutionContext:
    """Mutable context shared between the engine and orchestrator.

    The engine populates this during execution; the orchestrator reads
    it after the engine generator is exhausted.

    Attributes:
        step_results: Map of step_id to output data (successful steps only).
        log_entries: All transparency log entries in chronological order.
        steps_succeeded: Count of successful steps.
        steps_failed: Count of failed steps.
        total_steps: Total steps in the plan.
    """

    step_results: dict[str, dict] = field(default_factory=dict)
    log_entries: list[TransparencyLogEntry] = field(default_factory=list)
    steps_succeeded: int = 0
    steps_failed: int = 0
    total_steps: int = 0


async def execute_plan(
    query_id: str,
    plan: QueryPlan,
    credentials: dict[str, CredentialEnvelope],
    mcp_client_factory: Callable[[str], MCPClient],
    settings: Settings,
    context: ExecutionContext,
) -> AsyncGenerator[BaseModel, None]:
    """Execute a query plan step by step.

    Yields StepStarted, StepCompleted, and StepFailed events. Does NOT
    yield ExecutionStarted or terminal events (those are the orchestrator's
    responsibility). Populates the context with results and log entries.

    Args:
        query_id: UUID of the query being executed.
        plan: The query plan to execute.
        credentials: Map of integration_id to credential_envelope.
        mcp_client_factory: Factory returning MCPClient context managers.
        settings: Application settings.
        context: Mutable context to populate with results.

    Yields:
        StepStarted, StepCompleted, or StepFailed Pydantic event models.
    """
    context.total_steps = len(plan.steps)

    logger.debug(
        "starting plan execution",
        action="execute_plan",
        query_id=query_id,
        total_steps=context.total_steps,
    )

    for idx, step in enumerate(plan.steps, start=1):
        # Check depends_on: skip step if any dependency failed
        if step.depends_on:
            missing_deps = [dep for dep in step.depends_on if dep not in context.step_results]
            if missing_deps:
                logger.warn(
                    "skipping step due to failed dependencies",
                    action="execute_plan",
                    step_id=step.step_id,
                    missing_deps=missing_deps,
                )
                context.steps_failed += 1
                yield StepStarted(
                    query_id=query_id,
                    step_id=step.step_id,
                    step_index=idx,
                    tool_name=step.tool_name,
                    tool_display_name=step.tool_name.replace(".", " ").replace("_", " ").title(),
                    integration_display_name=f"Integration {step.integration_id}",
                    description=step.description,
                    timestamp=_now_iso(),
                )
                yield StepFailed(
                    query_id=query_id,
                    step_id=step.step_id,
                    duration_ms=0,
                    error_code="execution.plan_invalid",
                    error_message=f"Dependencies not met: {', '.join(missing_deps)}",
                    retryable=False,
                    timestamp=_now_iso(),
                )
                continue

        # Emit step_started
        yield StepStarted(
            query_id=query_id,
            step_id=step.step_id,
            step_index=idx,
            tool_name=step.tool_name,
            tool_display_name=step.tool_name.replace(".", " ").replace("_", " ").title(),
            integration_display_name=f"Integration {step.integration_id}",
            description=step.description,
            timestamp=_now_iso(),
        )

        # Execute the step with error containment
        try:
            result: StepResult = await execute_step(
                step=step,
                credentials=credentials,
                step_results=context.step_results,
                mcp_client_factory=mcp_client_factory,
                settings=settings,
            )
        except Exception as exc:
            logger.error(
                "unexpected error executing step",
                action="execute_plan",
                step_id=step.step_id,
                error=str(exc),
            )
            context.steps_failed += 1
            yield StepFailed(
                query_id=query_id,
                step_id=step.step_id,
                duration_ms=0,
                error_code="execution.internal_error",
                error_message=f"Unexpected error: {exc}",
                retryable=False,
                timestamp=_now_iso(),
            )
            continue

        # Collect log entries (Invariant #6: every invocation produces an entry)
        context.log_entries.extend(result.log_entries)

        if result.status == "success":
            context.steps_succeeded += 1
            if result.data is not None:
                context.step_results[step.step_id] = result.data

            yield StepCompleted(
                query_id=query_id,
                step_id=step.step_id,
                duration_ms=result.duration_ms,
                record_count=result.record_count,
                summary=result.summary,
                data_hash=result.data_hash or "",
                timestamp=_now_iso(),
            )
        else:
            context.steps_failed += 1
            yield StepFailed(
                query_id=query_id,
                step_id=step.step_id,
                duration_ms=result.duration_ms,
                error_code=result.error_code or "execution.internal_error",
                error_message=result.error_message or "Unknown error",
                retryable=result.retryable,
                timestamp=_now_iso(),
            )

    logger.info(
        "plan execution completed",
        action="execute_plan",
        query_id=query_id,
        succeeded=context.steps_succeeded,
        failed=context.steps_failed,
    )
