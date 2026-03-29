"""Plan parser for extracting query plans from LLM output.

Extracts the natural language analysis and structured query plan from
the complete LLM response text. The LLM produces only intelligence fields
(tool_name, integration_id, parameters, description, output_alias, summary).
The parser injects all mechanical fields (plan_id, plan_version, step_id,
estimated_tool_calls) and validates against the Pydantic model and the
tool catalog (Architectural Invariant #5).
"""

import json
import uuid
from dataclasses import dataclass

import structlog

from src.models.query_plan import QueryPlan
from src.models.tool_catalog import ToolCatalog

logger = structlog.get_logger(__name__)

PLAN_START_DELIMITER = "---PLAN_START---"
PLAN_END_DELIMITER = "---PLAN_END---"


@dataclass
class ParseResult:
    """Result of parsing LLM output for a query plan.

    Attributes:
        analysis_text: The natural language analysis (before the plan block).
        query_plan: The parsed and validated QueryPlan, or None if not found.
        error_message: Explanation if parsing or validation failed.
    """

    analysis_text: str
    query_plan: QueryPlan | None
    error_message: str | None


def _extract_plan_json(full_text: str) -> tuple[str, str | None]:
    """Extract plan JSON using ---PLAN_START--- / ---PLAN_END--- delimiters.

    Args:
        full_text: The complete LLM output.

    Returns:
        Tuple of (analysis_text, plan_json_string or None).
    """
    start_idx = full_text.find(PLAN_START_DELIMITER)
    if start_idx == -1:
        return full_text.strip(), None

    analysis_text = full_text[:start_idx].strip()
    after_start = full_text[start_idx + len(PLAN_START_DELIMITER) :]

    end_idx = after_start.find(PLAN_END_DELIMITER)
    plan_json = after_start.strip() if end_idx == -1 else after_start[:end_idx].strip()

    return analysis_text, plan_json


def _validate_llm_fields(plan_dict: dict) -> str | None:
    """Validate that all LLM-produced fields are present and correct.

    Checks the raw dict before constructing any Pydantic model.

    Args:
        plan_dict: The parsed JSON dict from the LLM.

    Returns:
        Error message string if validation fails, None if all fields are valid.
    """
    # Top-level fields
    steps = plan_dict.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return "Plan has no steps."

    summary = plan_dict.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return "Plan has no summary."

    # Per-step fields
    for i, step in enumerate(steps):
        step_label = f"Step {i + 1}"

        tool_name = step.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return f"{step_label} has no tool_name."

        integration_id = step.get("integration_id")
        if not isinstance(integration_id, str) or not integration_id.strip():
            return f"{step_label} has no integration_id."

        # parameters: default to {} if missing
        if "parameters" not in step or step["parameters"] is None:
            step["parameters"] = {}
        if not isinstance(step["parameters"], dict):
            return f"{step_label} has invalid parameters (must be an object)."

        description = step.get("description")
        if not isinstance(description, str) or not description.strip():
            return f"{step_label} has no description."

        output_alias = step.get("output_alias")
        if not isinstance(output_alias, str) or not output_alias.strip():
            return f"{step_label} has no output_alias."

    return None


def _inject_mechanical_fields(plan_dict: dict) -> str | None:
    """Inject system-controlled fields into the validated plan dict.

    Sets plan_id, plan_version, step_ids, and estimated_tool_calls.
    Validates cross-references (depends_on, iterate_over.source_step).

    Args:
        plan_dict: The validated plan dict (mutated in place).

    Returns:
        Error message string if cross-reference validation fails, None on success.
    """
    plan_dict["plan_id"] = str(uuid.uuid4())
    plan_dict["plan_version"] = "1.0"

    steps = plan_dict["steps"]

    # Assign step_ids
    for i, step in enumerate(steps):
        step["step_id"] = f"step_{i + 1}"

    # Calculate estimated_tool_calls
    estimated = 0
    for step in steps:
        if step.get("iterate_over"):
            estimated += 10
        else:
            estimated += 1
        if step.get("paginate", True) is not False:
            estimated += 1
    plan_dict["estimated_tool_calls"] = estimated

    # Validate cross-references
    valid_step_ids = {step["step_id"] for step in steps}
    step_id_indices = {step["step_id"]: i for i, step in enumerate(steps)}

    for step in steps:
        step_id = step["step_id"]
        step_idx = step_id_indices[step_id]

        # Validate depends_on
        if step.get("depends_on"):
            for dep_id in step["depends_on"]:
                if dep_id not in valid_step_ids:
                    return f"Step {step_id} references non-existent step {dep_id}."
                if step_id_indices[dep_id] >= step_idx:
                    return f"Step {step_id} references non-existent step {dep_id}."

        # Validate iterate_over.source_step
        if step.get("iterate_over"):
            source = step["iterate_over"].get("source_step", "")
            if source not in valid_step_ids:
                return f"Step {step_id} references non-existent step {source}."
            if step_id_indices[source] >= step_idx:
                return f"Step {step_id} references non-existent step {source}."

    return None


def parse_plan_from_llm_output(full_text: str) -> ParseResult:
    """Parse the LLM output to extract analysis text and query plan.

    Strict 5-step process:
    1. Extract JSON between ---PLAN_START--- and ---PLAN_END---.
    2. Validate LLM-produced fields (tool_name, integration_id, etc.).
    3. Inject mechanical fields (plan_id, plan_version, step_id, etc.).
    4. Construct the Pydantic QueryPlan model.
    5. (Catalog validation is done separately via validate_plan_against_catalog.)

    Args:
        full_text: The complete accumulated LLM output text.

    Returns:
        ParseResult with analysis_text, query_plan (or None), and error_message.
    """
    logger.debug(
        "parsing plan from LLM output",
        action="parse_plan",
        text_length=len(full_text),
    )

    # Step 1: Extract JSON
    analysis_text, plan_json = _extract_plan_json(full_text)

    if plan_json is None:
        logger.info(
            "no plan block found in LLM output",
            action="parse_plan",
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message="No plan block found in LLM output.",
        )

    try:
        plan_dict = json.loads(plan_json)
    except json.JSONDecodeError as exc:
        logger.warn(
            "invalid JSON in plan block",
            action="parse_plan",
            error=str(exc),
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message=f"Invalid JSON in plan block: {exc}",
        )

    # Step 2: Validate LLM-produced fields
    validation_error = _validate_llm_fields(plan_dict)
    if validation_error:
        logger.warn(
            "LLM field validation failed",
            action="parse_plan",
            error=validation_error,
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message=validation_error,
        )

    # Step 3: Inject mechanical fields
    injection_error = _inject_mechanical_fields(plan_dict)
    if injection_error:
        logger.warn(
            "cross-reference validation failed",
            action="parse_plan",
            error=injection_error,
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message=injection_error,
        )

    logger.info(
        "mechanical fields injected",
        action="parse_plan",
        plan_id=plan_dict["plan_id"],
        plan_version=plan_dict["plan_version"],
        step_ids=[s["step_id"] for s in plan_dict["steps"]],
        estimated_tool_calls=plan_dict["estimated_tool_calls"],
    )

    # Step 4: Construct the Pydantic model
    try:
        query_plan = QueryPlan.model_validate(plan_dict)
    except Exception as exc:
        logger.warn(
            "plan JSON does not match QueryPlan schema",
            action="parse_plan",
            error=str(exc),
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message=f"Plan does not conform to required schema: {exc}",
        )

    logger.info(
        "plan parsed successfully",
        action="parse_plan",
        step_count=len(query_plan.steps),
    )

    return ParseResult(
        analysis_text=analysis_text,
        query_plan=query_plan,
        error_message=None,
    )


def validate_plan_against_catalog(
    plan: QueryPlan,
    catalog: ToolCatalog,
) -> list[str]:
    """Validate that a query plan only references tools and integrations in the catalog.

    Enforces Architectural Invariant #5: the Interpreter MUST NOT reference
    tools outside the provided catalog.

    Args:
        plan: The parsed query plan to validate.
        catalog: The tool catalog to validate against.

    Returns:
        List of validation error strings. Empty list means the plan is valid.
    """
    logger.debug(
        "validating plan against catalog",
        action="validate_plan",
        step_count=len(plan.steps),
    )

    errors: list[str] = []

    # Build lookup: integration_id -> set of tool_names
    catalog_index: dict[str, set[str]] = {}
    for integration in catalog.integrations:
        catalog_index[integration.integration_id] = {tool.tool_name for tool in integration.tools}

    # Check for duplicate step_ids
    step_ids = [step.step_id for step in plan.steps]
    seen_ids: set[str] = set()
    for step_id in step_ids:
        if step_id in seen_ids:
            errors.append(f"Duplicate step_id: '{step_id}'")
        seen_ids.add(step_id)

    # Validate each step
    for step in plan.steps:
        # Check integration_id exists
        if step.integration_id not in catalog_index:
            errors.append(
                f"Step '{step.step_id}' references unknown integration_id: '{step.integration_id}'"
            )
            continue

        # Check tool_name exists in the correct integration
        integration_tools = catalog_index[step.integration_id]
        if step.tool_name not in integration_tools:
            # Check if tool exists in any integration
            all_tools = {tool_name for tools in catalog_index.values() for tool_name in tools}
            if step.tool_name in all_tools:
                errors.append(
                    f"Step '{step.step_id}': tool '{step.tool_name}' exists "
                    f"but does not belong to integration '{step.integration_id}'"
                )
            else:
                errors.append(
                    f"Step '{step.step_id}' references unknown tool_name: '{step.tool_name}'"
                )

        # Check depends_on references
        if step.depends_on:
            for dep_id in step.depends_on:
                if dep_id not in seen_ids:
                    errors.append(f"Step '{step.step_id}' depends_on unknown step_id: '{dep_id}'")

    if errors:
        logger.warn(
            "plan validation failed",
            action="validate_plan",
            error_count=len(errors),
        )
    else:
        logger.info(
            "plan validation passed",
            action="validate_plan",
        )

    return errors
