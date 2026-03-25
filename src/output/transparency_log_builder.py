"""Transparency log builder.

Assembles a complete TransparencyLog from individual entries collected
during execution. Conforms to the transparency_log schema in
mcp-tool-interface.yaml section 4.
"""

from datetime import datetime

import structlog

from src.models.transparency_log import TransparencyLog, TransparencyLogEntry

logger = structlog.get_logger(__name__)


def build_transparency_log(
    query_id: str,
    plan_id: str,
    execution_started_at: str,
    execution_completed_at: str,
    entries: list[TransparencyLogEntry],
    execution_status: str,
) -> TransparencyLog:
    """Assemble a complete transparency log from execution entries.

    Computes aggregate metrics from the individual entries.

    Args:
        query_id: UUID of the query.
        plan_id: UUID of the executed plan.
        execution_started_at: ISO 8601 timestamp when execution began.
        execution_completed_at: ISO 8601 timestamp when execution ended.
        entries: Individual log entries in chronological order.
        execution_status: Overall outcome ("completed", "partial", "failed").

    Returns:
        Complete TransparencyLog with all aggregates computed.
    """
    logger.debug(
        "building transparency log",
        action="build_transparency_log",
        entry_count=len(entries),
    )

    # Compute duration from timestamps
    try:
        start = datetime.fromisoformat(execution_started_at)
        end = datetime.fromisoformat(execution_completed_at)
        total_duration_ms = int((end - start).total_seconds() * 1000)
    except (ValueError, TypeError):
        total_duration_ms = sum(e.duration_ms for e in entries)

    total_external_api_calls = sum(e.external_api_calls for e in entries)

    log = TransparencyLog(
        query_id=query_id,
        plan_id=plan_id,
        execution_started_at=execution_started_at,
        execution_completed_at=execution_completed_at,
        total_duration_ms=total_duration_ms,
        total_tool_invocations=len(entries),
        total_external_api_calls=total_external_api_calls,
        execution_status=execution_status,
        entries=entries,
    )

    logger.info(
        "transparency log built",
        action="build_transparency_log",
        total_invocations=len(entries),
        total_api_calls=total_external_api_calls,
    )
    return log
