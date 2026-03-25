"""Result formatter.

Transforms raw step results into a ResultData structure with tables,
columns, and metadata. Conforms to the result_data schema in
internal-api.yaml. Contains no AI (Invariant #7).
"""

import structlog

from src.models.query_plan import QueryPlan
from src.models.result_data import (
    IntegrationQueried,
    ResultColumn,
    ResultData,
    ResultMetadata,
    ResultTable,
)

logger = structlog.get_logger(__name__)


def _infer_data_type(value: object) -> str:
    """Infer the result_column data_type from a Python value.

    Args:
        value: A sample value from the data.

    Returns:
        One of: "string", "number", "boolean", "datetime", "array".
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str) and len(value) >= 19 and "T" in value:
        return "datetime"
    return "string"


def _label_from_key(key: str) -> str:
    """Generate a human-readable label from a snake_case key.

    Args:
        key: The column key (e.g., "last_login_at").

    Returns:
        A formatted label (e.g., "Last login at").
    """
    return key.replace("_", " ").capitalize()


def _find_primary_array(data: dict) -> tuple[str, list[dict]]:
    """Find the primary data array in tool output.

    Looks for the largest array of dicts in the data.

    Args:
        data: The tool output data.

    Returns:
        Tuple of (array_key, array_of_dicts). Returns ("items", []) if none found.
    """
    best_key = "items"
    best_array: list[dict] = []

    for key, value in data.items():
        if not (isinstance(value, list) and len(value) > len(best_array)):
            continue
        if value and isinstance(value[0], dict):
            best_key = key
            best_array = value

    return best_key, best_array


def format_results(
    step_results: dict[str, dict],
    plan: QueryPlan,
    execution_duration_ms: int,
) -> ResultData:
    """Transform raw step results into a structured ResultData.

    Creates one table per step that produced data. Infers column
    definitions from the first row of each array.

    Args:
        step_results: Map of step_id to tool output data.
        plan: The executed query plan.
        execution_duration_ms: Total execution time.

    Returns:
        ResultData with tables, columns, and metadata.
    """
    logger.debug(
        "formatting results",
        action="format_results",
        step_count=len(step_results),
    )

    tables: list[ResultTable] = []
    total_records = 0
    integration_ids_seen: set[str] = set()
    table_index = 0

    # Build a step lookup for descriptions
    step_lookup = {s.step_id: s for s in plan.steps}

    for step_id, data in step_results.items():
        _, array_data = _find_primary_array(data)
        if not array_data:
            continue

        table_index += 1
        step = step_lookup.get(step_id)
        title = step.description if step else f"Results from {step_id}"
        integration_ids_seen.add(step.integration_id if step else "unknown")

        # Infer columns from first row
        first_row = array_data[0]
        columns = [
            ResultColumn(
                key=key,
                label=_label_from_key(key),
                data_type=_infer_data_type(value),
            )
            for key, value in first_row.items()
        ]
        column_keys = [c.key for c in columns]

        # Build rows as positional arrays
        rows = [[item.get(key) for key in column_keys] for item in array_data]

        tables.append(
            ResultTable(
                table_id=f"table_{table_index}",
                title=title,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                source_step_ids=[step_id],
            )
        )
        total_records += len(rows)

    # Build integrations_queried from plan steps
    integrations_queried = []
    seen = set()
    for step in plan.steps:
        if step.integration_id not in seen:
            seen.add(step.integration_id)
            integrations_queried.append(
                IntegrationQueried(
                    integration_id=step.integration_id,
                    display_name=f"Integration {step.integration_id}",
                    server_type=step.tool_name.split(".")[0]
                    if "." in step.tool_name
                    else "unknown",
                )
            )

    result_data = ResultData(
        tables=tables,
        metadata=ResultMetadata(
            total_records=total_records,
            integrations_queried=integrations_queried,
            execution_duration_ms=execution_duration_ms,
        ),
    )

    logger.info(
        "results formatted",
        action="format_results",
        table_count=len(tables),
        total_records=total_records,
    )
    return result_data
