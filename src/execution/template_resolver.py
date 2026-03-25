"""Template resolver for step parameter references.

Resolves {{step_id.path}} and {{item.field}} template references in
plan step parameters. Used by the step executor before invoking MCP tools.
Conforms to the Step Parameter Templates section in mcp-tool-interface.yaml.
"""

import copy
import re

import structlog

logger = structlog.get_logger(__name__)

TEMPLATE_PATTERN = re.compile(r"\{\{(.+?)\}\}")


class TemplateResolutionError(Exception):
    """Raised when a template reference cannot be resolved.

    This becomes a step_failed event, not a crash.
    """


def _navigate_path(data: dict, path_parts: list[str], full_ref: str) -> object:
    """Navigate a dot-path through a nested data structure.

    Args:
        data: The root data dict to navigate.
        path_parts: The dot-separated path segments.
        full_ref: The full template reference string for error messages.

    Returns:
        The value at the end of the path.

    Raises:
        TemplateResolutionError: If the path cannot be resolved.
    """
    current = data
    for part in path_parts:
        if isinstance(current, dict):
            if part not in current:
                raise TemplateResolutionError(
                    f"Path '{'.'.join(path_parts)}' not found in reference '{full_ref}': "
                    f"key '{part}' does not exist"
                )
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
                current = current[index]
            except (ValueError, IndexError) as exc:
                raise TemplateResolutionError(
                    f"Cannot index into list at '{part}' in reference '{full_ref}': {exc}"
                ) from exc
        else:
            raise TemplateResolutionError(
                f"Cannot navigate further at '{part}' in reference '{full_ref}': "
                f"value is {type(current).__name__}, not dict or list"
            )
    return current


def _resolve_reference(
    ref: str,
    step_results: dict[str, dict],
    iteration_item: dict | None,
    item_alias: str | None,
) -> object:
    """Resolve a single template reference.

    Args:
        ref: The reference string inside {{...}} (e.g., "step_1.users.0.name").
        step_results: Map of step_id to step output data.
        iteration_item: The current iteration item, if in iterate_over context.
        item_alias: The alias for the iteration item.

    Returns:
        The resolved value.

    Raises:
        TemplateResolutionError: If the reference cannot be resolved.
    """
    parts = ref.strip().split(".")

    # Handle {{item.field}} references in iterate_over context
    if item_alias and parts[0] == item_alias:
        if iteration_item is None:
            raise TemplateResolutionError(
                f"Reference '{ref}' uses item alias '{item_alias}' "
                f"but no iteration item is available"
            )
        if len(parts) == 1:
            return iteration_item
        return _navigate_path(iteration_item, parts[1:], ref)

    # Handle {{step_id.path}} references
    step_id = parts[0]
    if step_id not in step_results:
        raise TemplateResolutionError(
            f"Reference '{ref}' refers to step '{step_id}' "
            f"which has not completed or does not exist"
        )

    if len(parts) == 1:
        return step_results[step_id]
    return _navigate_path(step_results[step_id], parts[1:], ref)


def _resolve_value(
    value: object,
    step_results: dict[str, dict],
    iteration_item: dict | None,
    item_alias: str | None,
) -> object:
    """Recursively resolve template references in a value.

    Args:
        value: The value to process (string, dict, list, or primitive).
        step_results: Map of step_id to step output data.
        iteration_item: The current iteration item.
        item_alias: The alias for the iteration item.

    Returns:
        The value with all template references resolved.
    """
    if isinstance(value, str):
        matches = list(TEMPLATE_PATTERN.finditer(value))
        if not matches:
            return value
        # If the entire string is a single template, return the resolved value directly
        # (preserving non-string types like lists and dicts)
        if len(matches) == 1 and matches[0].group(0) == value:
            return _resolve_reference(matches[0].group(1), step_results, iteration_item, item_alias)
        # Otherwise, substitute within the string
        result = value
        for match in reversed(matches):
            resolved = _resolve_reference(match.group(1), step_results, iteration_item, item_alias)
            result = result[: match.start()] + str(resolved) + result[match.end() :]
        return result

    if isinstance(value, dict):
        return {
            k: _resolve_value(v, step_results, iteration_item, item_alias) for k, v in value.items()
        }

    if isinstance(value, list):
        return [_resolve_value(item, step_results, iteration_item, item_alias) for item in value]

    return value


def resolve_parameters(
    parameters: dict,
    step_results: dict[str, dict],
    iteration_item: dict | None = None,
    item_alias: str | None = None,
) -> dict:
    """Resolve all template references in step parameters.

    Replaces {{step_id.path}} and {{item.field}} references with
    actual values from previous step results or the current iteration item.

    Args:
        parameters: The step's parameter dict (may contain templates).
        step_results: Map of completed step_id to step output data.
        iteration_item: The current iteration item (for iterate_over steps).
        item_alias: The alias for the iteration item (e.g., "repo").

    Returns:
        A new dict with all template references resolved.

    Raises:
        TemplateResolutionError: If any reference cannot be resolved.
    """
    logger.debug(
        "resolving parameter templates",
        action="resolve_templates",
        param_count=len(parameters),
    )
    resolved = _resolve_value(copy.deepcopy(parameters), step_results, iteration_item, item_alias)
    logger.debug(
        "parameter templates resolved",
        action="resolve_templates",
    )
    return resolved
