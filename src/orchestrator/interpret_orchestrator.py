"""Interpretation orchestrator — coordinates the full /interpret flow.

This async generator builds a prompt, streams the LLM response as text
deltas, parses the output (plan or clarification), validates it against
the catalog, and emits the appropriate terminal event. Supports multi-turn
conversation via optional conversation_history. Every stream is guaranteed
to end with exactly one terminal event.
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
from src.models.conversation import AssistantTurn, UserTurn
from src.models.stream_events import (
    InterpretationClarificationNeeded,
    InterpretationError,
    InterpretationFailed,
    InterpretationPlanGenerated,
    InterpretationStarted,
    InterpretationTextDelta,
)
from src.models.tool_catalog import ToolCatalog

logger = structlog.get_logger(__name__)

# Markers that indicate the start of machine-readable output.
# Text deltas stop streaming to the frontend once any of these appear.
_STOP_MARKERS = ["### Structured Output", "---PLAN_START---", "---CLARIFICATION_START---"]
_MAX_MARKER_LEN = max(len(m) for m in _STOP_MARKERS)


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


def _find_first_marker(text: str) -> int | None:
    """Find the position of the first stop marker in the text.

    Args:
        text: The accumulated LLM output text.

    Returns:
        The index of the first marker found, or None if no marker is present.
    """
    first_pos = None
    for marker in _STOP_MARKERS:
        pos = text.find(marker)
        if pos != -1 and (first_pos is None or pos < first_pos):
            first_pos = pos
    return first_pos


async def interpret_stream(
    query_id: str,
    query_text: str,
    tool_catalog: ToolCatalog,
    provider: LLMProvider,
    settings: Settings,
    conversation_history: list[AssistantTurn | UserTurn] | None = None,
) -> AsyncGenerator[BaseModel, None]:
    """Generate the interpretation NDJSON event stream.

    Coordinates: prompt building → LLM streaming → output parsing
    (plan or clarification) → catalog validation → terminal event.
    Guarantees exactly one terminal event regardless of outcome.

    Args:
        query_id: UUID of the query being interpreted.
        query_text: The user's natural language question.
        tool_catalog: Filtered tool catalog for this query.
        provider: The LLM provider to use for completion.
        settings: Application settings.
        conversation_history: Previous turns for multi-turn, or None.

    Yields:
        Pydantic event models for NDJSON serialization.
    """
    logger.debug(
        "starting interpretation stream",
        action="interpret_stream",
        query_id=query_id,
        has_history=conversation_history is not None and len(conversation_history) > 0,
    )

    try:
        # Always emit the started event first
        yield InterpretationStarted(
            query_id=query_id,
            timestamp=_now_iso(),
        )

        # Build prompt
        try:
            system_prompt, messages = build_interpretation_prompt(
                query_text, tool_catalog, conversation_history
            )
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
        # Stop sending text deltas once any stop marker appears — everything
        # after is machine-readable (plan or clarification JSON) and should
        # not reach the frontend. We track how much of full_text has been
        # sent and hold back characters to handle markers split across chunks.
        full_text = ""
        sent_length = 0
        cutoff_found = False
        try:
            async for chunk in provider.stream_completion(
                system_prompt, messages, settings.LLM_MAX_TOKENS
            ):
                full_text += chunk
                if not cutoff_found:
                    marker_pos = _find_first_marker(full_text)
                    if marker_pos is not None:
                        cutoff_found = True
                        unsent = full_text[sent_length:marker_pos]
                        if unsent:
                            yield InterpretationTextDelta(
                                query_id=query_id,
                                text_delta=unsent,
                                timestamp=_now_iso(),
                            )
                    else:
                        # Hold back enough to cover a partial marker at the tail
                        safe_end = max(sent_length, len(full_text) - _MAX_MARKER_LEN + 1)
                        unsent = full_text[sent_length:safe_end]
                        if unsent:
                            yield InterpretationTextDelta(
                                query_id=query_id,
                                text_delta=unsent,
                                timestamp=_now_iso(),
                            )
                        sent_length = safe_end

            # Flush any buffered tail if no marker ever appeared
            if not cutoff_found and sent_length < len(full_text):
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

        # Parse output (plan or clarification) from accumulated text
        result = parse_plan_from_llm_output(full_text)

        # Clarification path (new: multi-turn)
        if result.clarification is not None:
            logger.info(
                "clarification requested by LLM",
                action="interpret_stream",
                query_id=query_id,
                clarification_id=result.clarification.clarification_id,
            )
            yield InterpretationClarificationNeeded(
                query_id=query_id,
                clarification_id=result.clarification.clarification_id,
                question=result.clarification.question,
                options=result.clarification.options,
                allows_free_text=result.clarification.allows_free_text,
                full_interpretation_text=result.analysis_text,
                timestamp=_now_iso(),
            )
            return

        # No plan and no clarification — failure
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

        # Success — plan generated
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
