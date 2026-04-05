"""Interpretation orchestrator — coordinates the full /interpret flow.

This async generator replaces the mock implementation. It builds a prompt,
streams the LLM response as text deltas, parses the plan, validates it
against the catalog, and emits the appropriate terminal event. Every
stream is guaranteed to end with exactly one terminal event.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from src.config import Settings
from src.interpretation.plan_parser import (
    parse_plan_from_llm_output,
    validate_plan_against_catalog,
)
from src.interpretation.prompt_builder import build_interpretation_prompt
from src.llm.exceptions import (
    LLMProviderError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from src.llm.provider import LLMProvider
from src.models.stream_events import (
    InterpretationError,
    InterpretationFailed,
    InterpretationPlanGenerated,
    InterpretationStarted,
    InterpretationTextDelta,
)
from src.models.tool_catalog import ToolCatalog

logger = structlog.get_logger(__name__)


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def _estimate_duration(estimated_tool_calls: int) -> int:
    """Estimate execution duration in seconds.

    Args:
        estimated_tool_calls: Number of expected MCP tool invocations.

    Returns:
        Estimated duration in seconds (~3 seconds per tool call).
    """
    return max(estimated_tool_calls * 3, 1)


async def interpret_stream(
    query_id: str,
    query_text: str,
    tool_catalog: ToolCatalog,
    provider: LLMProvider,
    settings: Settings,
) -> AsyncGenerator[BaseModel, None]:
    """Generate the interpretation NDJSON event stream.

    Coordinates: prompt building → LLM streaming → plan parsing →
    catalog validation → terminal event emission. Guarantees exactly
    one terminal event is emitted regardless of success or failure.

    Args:
        query_id: UUID of the query being interpreted.
        query_text: The user's natural language question.
        tool_catalog: Filtered tool catalog for this query.
        provider: The LLM provider to use for completion.
        settings: Application settings.

    Yields:
        Pydantic event models for NDJSON serialization.
    """
    logger.debug(
        "starting interpretation stream",
        action="interpret_stream",
        query_id=query_id,
    )

    try:
        # Always emit the started event first
        yield InterpretationStarted(
            query_id=query_id,
            timestamp=_now_iso(),
        )

        # Build prompt
        try:
            system_prompt, user_message = build_interpretation_prompt(query_text, tool_catalog)
        except ValueError as exc:
            logger.error(
                "prompt building failed",
                action="interpret_stream",
                query_id=query_id,
                error=str(exc),
            )
            yield InterpretationFailed(
                query_id=query_id,
                error_code="interpretation.unsatisfiable_request",
                error_message=str(exc),
                full_interpretation_text=None,
                timestamp=_now_iso(),
            )
            return

        # Stream LLM response
        # Stop sending text deltas once "### Plan" appears — everything
        # after that heading is machine-readable and should not reach the
        # frontend. The plan data arrives in the terminal event instead.
        # We track how much of full_text has been sent to the frontend and
        # hold back the last few characters in case the marker is split
        # across chunk boundaries.
        full_text = ""
        sent_length = 0
        plan_heading_seen = False
        _PLAN_MARKER = "### Plan"
        try:
            async for chunk in provider.stream_completion(
                system_prompt, user_message, settings.LLM_MAX_TOKENS
            ):
                full_text += chunk
                if not plan_heading_seen:
                    marker_pos = full_text.find(_PLAN_MARKER)
                    if marker_pos != -1:
                        plan_heading_seen = True
                        unsent = full_text[sent_length:marker_pos]
                        if unsent:
                            yield InterpretationTextDelta(
                                query_id=query_id,
                                text_delta=unsent,
                                timestamp=_now_iso(),
                            )
                    else:
                        # Hold back enough to cover a partial marker at the tail
                        safe_end = max(sent_length, len(full_text) - len(_PLAN_MARKER) + 1)
                        unsent = full_text[sent_length:safe_end]
                        if unsent:
                            yield InterpretationTextDelta(
                                query_id=query_id,
                                text_delta=unsent,
                                timestamp=_now_iso(),
                            )
                        sent_length = safe_end

            # Flush any buffered tail if the marker never appeared
            if not plan_heading_seen and sent_length < len(full_text):
                yield InterpretationTextDelta(
                    query_id=query_id,
                    text_delta=full_text[sent_length:],
                    timestamp=_now_iso(),
                )

        except LLMTimeoutError as exc:
            logger.error(
                "LLM timed out during interpretation",
                action="interpret_stream",
                query_id=query_id,
                error=str(exc),
            )
            yield InterpretationError(
                query_id=query_id,
                error_code="interpretation.model_timeout",
                error_message=str(exc),
                timestamp=_now_iso(),
            )
            return

        except LLMUnavailableError as exc:
            logger.error(
                "LLM unavailable during interpretation",
                action="interpret_stream",
                query_id=query_id,
                error=str(exc),
            )
            yield InterpretationError(
                query_id=query_id,
                error_code="interpretation.model_unavailable",
                error_message=str(exc),
                timestamp=_now_iso(),
            )
            return

        except LLMProviderError as exc:
            logger.error(
                "LLM provider error during interpretation",
                action="interpret_stream",
                query_id=query_id,
                error=str(exc),
            )
            yield InterpretationError(
                query_id=query_id,
                error_code="interpretation.internal_error",
                error_message=str(exc),
                timestamp=_now_iso(),
            )
            return

        except Exception as exc:
            logger.error(
                "unexpected error during LLM streaming",
                action="interpret_stream",
                query_id=query_id,
                error=str(exc),
            )
            yield InterpretationError(
                query_id=query_id,
                error_code="interpretation.internal_error",
                error_message=f"Unexpected error during interpretation: {exc}",
                timestamp=_now_iso(),
            )
            return

        # Parse plan from accumulated text
        result = parse_plan_from_llm_output(full_text)

        if result.query_plan is None:
            logger.info(
                "no plan produced by LLM",
                action="interpret_stream",
                query_id=query_id,
                error_message=result.error_message,
            )
            yield InterpretationFailed(
                query_id=query_id,
                error_code="interpretation.unsatisfiable_request",
                error_message=result.error_message
                or "The AI could not produce an execution plan for this query.",
                full_interpretation_text=result.analysis_text,
                timestamp=_now_iso(),
            )
            return

        # Validate plan against catalog (Invariant #5)
        validation_errors = validate_plan_against_catalog(result.query_plan, tool_catalog)
        if validation_errors:
            error_details = "; ".join(validation_errors)
            logger.warn(
                "plan references tools not in catalog",
                action="interpret_stream",
                query_id=query_id,
                validation_errors=validation_errors,
            )
            yield InterpretationFailed(
                query_id=query_id,
                error_code="interpretation.unsatisfiable_request",
                error_message=f"Plan references tools or integrations "
                f"not in the catalog: {error_details}",
                full_interpretation_text=result.analysis_text,
                timestamp=_now_iso(),
            )
            return

        # Success
        logger.info(
            "interpretation completed successfully",
            action="interpret_stream",
            query_id=query_id,
            step_count=len(result.query_plan.steps),
        )
        yield InterpretationPlanGenerated(
            query_id=query_id,
            query_plan=result.query_plan,
            plan_summary=result.query_plan.summary,
            estimated_duration_seconds=_estimate_duration(result.query_plan.estimated_tool_calls),
            full_interpretation_text=result.analysis_text,
            timestamp=_now_iso(),
        )

    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "interpretation stream cancelled by client disconnect",
            action="interpret_stream",
            query_id=query_id,
        )
        return
