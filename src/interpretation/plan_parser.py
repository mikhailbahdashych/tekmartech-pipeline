"""Plan parser for extracting query plans from LLM output.

Extracts the natural language analysis and structured query plan from
the complete LLM response text. Validates the plan against the Pydantic
model and the tool catalog (Architectural Invariant #5).
"""

import json
import re
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


def _extract_with_delimiters(full_text: str) -> tuple[str, str | None]:
    """Extract plan JSON using ---PLAN_START--- / ---PLAN_END--- delimiters.

    Args:
        full_text: The complete LLM output.

    Returns:
        Tuple of (analysis_text, plan_json_string or None).
    """
    start_idx = full_text.find(PLAN_START_DELIMITER)
    if start_idx == -1:
        return full_text, None

    analysis_text = full_text[:start_idx].strip()
    after_start = full_text[start_idx + len(PLAN_START_DELIMITER) :]

    end_idx = after_start.find(PLAN_END_DELIMITER)
    plan_json = after_start.strip() if end_idx == -1 else after_start[:end_idx].strip()

    return analysis_text, plan_json


def _extract_with_code_fence(full_text: str) -> tuple[str, str | None]:
    """Extract plan JSON from the last ```json code fence.

    Args:
        full_text: The complete LLM output.

    Returns:
        Tuple of (analysis_text, plan_json_string or None).
    """
    pattern = r"```(?:json)?\s*\n(.*?)```"
    matches = list(re.finditer(pattern, full_text, re.DOTALL))
    if not matches:
        return full_text, None

    last_match = matches[-1]
    analysis_text = full_text[: last_match.start()].strip()
    plan_json = last_match.group(1).strip()

    return analysis_text, plan_json


def _extract_last_json_object(full_text: str) -> tuple[str, str | None]:
    """Extract the last JSON object from the text by finding matching braces.

    Args:
        full_text: The complete LLM output.

    Returns:
        Tuple of (analysis_text, plan_json_string or None).
    """
    last_close = full_text.rfind("}")
    if last_close == -1:
        return full_text, None

    depth = 0
    for i in range(last_close, -1, -1):
        if full_text[i] == "}":
            depth += 1
        elif full_text[i] == "{":
            depth -= 1
        if depth == 0:
            analysis_text = full_text[:i].strip()
            plan_json = full_text[i : last_close + 1]
            return analysis_text, plan_json

    return full_text, None


def parse_plan_from_llm_output(full_text: str) -> ParseResult:
    """Parse the LLM output to extract analysis text and query plan.

    Tries multiple extraction strategies in order of priority:
    1. Custom delimiters (---PLAN_START--- / ---PLAN_END---)
    2. Markdown JSON code fences
    3. Last bare JSON object

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

    # Strategy 1: Custom delimiters
    analysis_text, plan_json = _extract_with_delimiters(full_text)
    extraction_method = "delimiters"

    # Strategy 2: Code fence fallback
    if plan_json is None:
        analysis_text, plan_json = _extract_with_code_fence(full_text)
        extraction_method = "code_fence"

    # Strategy 3: Last JSON object fallback
    if plan_json is None:
        analysis_text, plan_json = _extract_last_json_object(full_text)
        extraction_method = "bare_json"

    # No plan found
    if plan_json is None:
        logger.info(
            "no plan block found in LLM output",
            action="parse_plan",
        )
        return ParseResult(
            analysis_text=full_text.strip(),
            query_plan=None,
            error_message="No plan block found in LLM output. "
            "The AI may have determined the query cannot be answered.",
        )

    # Parse JSON
    try:
        plan_data = json.loads(plan_json)
    except json.JSONDecodeError as exc:
        logger.warn(
            "invalid JSON in plan block",
            action="parse_plan",
            extraction_method=extraction_method,
            error=str(exc),
        )
        return ParseResult(
            analysis_text=analysis_text,
            query_plan=None,
            error_message=f"Plan block contains invalid JSON: {exc}",
        )

    # Validate against Pydantic model
    try:
        query_plan = QueryPlan(**plan_data)
    except Exception as exc:
        logger.warn(
            "plan JSON does not match QueryPlan schema",
            action="parse_plan",
            extraction_method=extraction_method,
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
        extraction_method=extraction_method,
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
