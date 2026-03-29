"""Summary generator.

Produces a human-readable summary of query results. Uses programmatic
templates, not AI (Invariant #7).
"""

import structlog

from src.models.result_data import ResultData

logger = structlog.get_logger(__name__)


def generate_summary(
    result_data: ResultData,
    steps_succeeded: int,
    steps_failed: int,
) -> str:
    """Generate a human-readable result summary.

    Args:
        result_data: The formatted result data.
        steps_succeeded: Number of successful steps.
        steps_failed: Number of failed steps.

    Returns:
        A summary string suitable for display to the user.
    """
    total_steps = steps_succeeded + steps_failed
    total_records = result_data.metadata.total_records
    table_count = len(result_data.tables)
    integration_count = len(result_data.metadata.integrations_queried)
    duration_s = result_data.metadata.execution_duration_ms / 1000

    parts = [f"Retrieved {total_records} records across {table_count} tables."]
    parts.append(f"{steps_succeeded} of {total_steps} steps completed successfully.")

    if integration_count > 0:
        names = [iq.display_name for iq in result_data.metadata.integrations_queried]
        parts.append(f"Queried {integration_count} integration(s): {', '.join(names)}.")

    parts.append(f"Execution took {duration_s:.1f}s.")

    summary = " ".join(parts)

    logger.debug(
        "summary generated",
        action="generate_summary",
        total_records=total_records,
    )
    return summary
