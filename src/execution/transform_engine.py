"""Transform engine for applying filter and select_fields to tool output.

Applies the optional transform directive from a plan step to the tool's
raw output. Supports filter conditions (10 operators) and field selection.
Conforms to the transform schema in mcp-tool-interface.yaml.
"""

import copy

import structlog

from src.models.query_plan import Transform

logger = structlog.get_logger(__name__)


def _navigate_to_array(data: dict, array_path: str) -> tuple[dict, str, list]:
    """Navigate to an array within data using a dot-path.

    Args:
        data: The root data dict.
        array_path: Dot-separated path to the array (e.g., "users").

    Returns:
        Tuple of (parent_container, final_key, array).

    Raises:
        ValueError: If the path does not lead to a list.
    """
    parts = array_path.split(".")
    current = data
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Path '{array_path}': key '{part}' not found")

    final_key = parts[-1]
    if isinstance(current, dict) and final_key in current:
        arr = current[final_key]
        if not isinstance(arr, list):
            raise ValueError(f"Path '{array_path}' does not point to an array")
        return current, final_key, arr

    raise ValueError(f"Path '{array_path}': key '{final_key}' not found")


def _evaluate_condition(item: dict, field: str, operator: str, value: object) -> bool:
    """Evaluate a filter condition against a single item.

    Args:
        item: The data item to test.
        field: The field within the item to test.
        operator: The comparison operator.
        value: The value to compare against.

    Returns:
        True if the item matches the condition.
    """
    item_value = item.get(field)

    if operator == "equals":
        return item_value == value
    if operator == "not_equals":
        return item_value != value
    if operator == "greater_than":
        return item_value is not None and item_value > value
    if operator == "less_than":
        return item_value is not None and item_value < value
    if operator == "is_null":
        return item_value is None
    if operator == "is_not_null":
        return item_value is not None
    if operator == "contains":
        return item_value is not None and value in str(item_value)
    if operator == "not_contains":
        return item_value is None or value not in str(item_value)
    if operator == "before":
        return item_value is not None and str(item_value) < str(value)
    if operator == "after":
        return item_value is not None and str(item_value) > str(value)

    logger.warn("unknown filter operator", action="evaluate_condition", operator=operator)
    return True


def _apply_filter(data: dict, array_path: str, field: str, operator: str, value: object) -> dict:
    """Apply a filter condition to an array within data.

    Args:
        data: The data dict containing the array.
        array_path: Dot-path to the array to filter.
        field: Field to test within each item.
        operator: Comparison operator.
        value: Value to compare against.

    Returns:
        Modified data with the filtered array.
    """
    try:
        parent, key, arr = _navigate_to_array(data, array_path)
    except ValueError as exc:
        logger.warn("filter array path not found", action="apply_filter", error=str(exc))
        return data

    filtered = [item for item in arr if _evaluate_condition(item, field, operator, value)]
    parent[key] = filtered

    logger.debug(
        "filter applied",
        action="apply_filter",
        original_count=len(arr),
        filtered_count=len(filtered),
    )
    return data


def _apply_select_fields(data: dict, select_fields: list[str], array_path: str | None) -> dict:
    """Apply field selection to items in the result.

    Args:
        data: The data dict.
        select_fields: List of field names to retain.
        array_path: Dot-path to the array whose items to project.

    Returns:
        Modified data with only selected fields in each item.
    """
    if not array_path:
        # If no array_path from filter, try to find the first list in data
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                array_path = key
                break

    if not array_path:
        return data

    try:
        parent, key, arr = _navigate_to_array(data, array_path)
    except ValueError:
        return data

    projected = [{f: item.get(f) for f in select_fields} for item in arr if isinstance(item, dict)]
    parent[key] = projected
    return data


def apply_transform(data: dict, transform: Transform) -> dict:
    """Apply filter and select_fields transforms to tool output.

    Filter is applied first, then select_fields. Works on a deep copy
    to avoid mutating the original data.

    Args:
        data: The raw tool output data.
        transform: The Transform directive from the plan step.

    Returns:
        Transformed data dict.
    """
    logger.debug("applying transform", action="apply_transform")
    result = copy.deepcopy(data)
    array_path = None

    if transform.filter:
        condition = transform.filter.condition
        array_path = transform.filter.array_path
        result = _apply_filter(
            result,
            array_path,
            condition.field,
            condition.operator,
            condition.value,
        )

    if transform.select_fields:
        result = _apply_select_fields(result, transform.select_fields, array_path)

    logger.debug("transform applied", action="apply_transform")
    return result
