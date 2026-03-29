"""System prompt construction for the Interpreter.

Translates a tool_catalog into an LLM prompt that instructs the AI
to analyze an infrastructure query and produce a structured query plan.
The prompt tells the LLM to produce ONLY the intelligence fields (tool
selection, parameters, reasoning). Mechanical fields (plan_id, plan_version,
step_id, estimated_tool_calls) are injected by the parser after the fact.
"""

import json

import structlog

from src.models.tool_catalog import ToolCatalog

logger = structlog.get_logger(__name__)

ROLE_SECTION = """\
You are an infrastructure query planner. You analyze questions about IT \
infrastructure and produce execution plans using the available tools listed \
below. You MUST only reference tools and integration IDs from the catalog \
provided.\
"""

OUTPUT_FORMAT_SECTION = """\
## Response Format

Respond in two parts.

PART 1: Write a brief analysis (2-4 sentences) explaining what the user \
is asking and which tools you will use.

PART 2: Output a JSON execution plan between these exact markers:

---PLAN_START---
<your JSON here>
---PLAN_END---

The JSON must have this exact structure:
{
  "steps": [ ... ],
  "summary": "..."
}

Each step in the "steps" array must have exactly these fields:
- "tool_name": exact tool name from the catalog (e.g., "github.list_repositories")
- "integration_id": exact integration ID from the catalog (the UUID shown in the catalog)
- "parameters": object matching the tool's input parameters (use {} if none required)
- "description": one sentence explaining what this step does
- "output_alias": short snake_case label for this step's output (e.g., "org_repos")

Optional step fields (include only when needed):
- "depends_on": array of step numbers from earlier steps, e.g., ["step_1"]
- "transform": { "filter": { "array_path": "...", "condition": { "field": "...", \
"operator": "equals", "value": ... } }, "select_fields": ["field1", "field2"] }
- "iterate_over": { "source_step": "step_1", "array_path": "items", \
"item_alias": "item" }
- "paginate": false (only if you want just the first page; default is true)

The "summary" field is a one-sentence explanation of the entire plan for a \
non-technical person.

Do NOT include plan_id, plan_version, step_id, or estimated_tool_calls. \
These are added automatically by the system.\
"""

CONSTRAINTS_SECTION = """\
## Rules
1. Only use tool_name values from the catalog above.
2. Only use integration_id values from the catalog above. They are UUIDs.
3. Each tool must be paired with the integration_id that owns it.
4. If the question cannot be answered with the available tools, explain why \
and do NOT output ---PLAN_START---.
5. Use the fewest steps possible.
6. Do NOT include plan_id, plan_version, step_id, or estimated_tool_calls \
in your JSON.\
"""


def _build_dynamic_example(tool_catalog: ToolCatalog) -> str:
    """Build an example plan section using real catalog values.

    Uses the first integration with at least one tool to construct
    an example, so the LLM sees consistent IDs between the catalog
    and the example. This prevents smaller models from copying
    hardcoded fake IDs.

    Args:
        tool_catalog: The filtered tool catalog for this query.

    Returns:
        Formatted example section string.
    """
    integration = None
    for integ in tool_catalog.integrations:
        if integ.tools:
            integration = integ
            break

    if integration is None:
        return ""

    tool = integration.tools[0]
    tool_action = tool.tool_name.split(".")[-1].replace("_", " ")

    return f"""## Example using your catalog

---PLAN_START---
{{
  "steps": [
    {{
      "tool_name": "{tool.tool_name}",
      "integration_id": "{integration.integration_id}",
      "parameters": {{}},
      "description": "{tool_action.capitalize()} from {integration.display_name}",
      "output_alias": "{tool.tool_name.split('.')[-1]}_data"
    }}
  ],
  "summary": "Use {tool.display_name} to {tool_action} from {integration.display_name}."
}}
---PLAN_END---"""


def _format_tool_catalog(tool_catalog: ToolCatalog) -> str:
    """Format the tool catalog as a readable section for the system prompt.

    Args:
        tool_catalog: The filtered tool catalog for this query.

    Returns:
        Formatted string listing all integrations and their tools.
    """
    sections = ["## Available Tool Catalog\n"]

    for integration in tool_catalog.integrations:
        sections.append(
            f"### Integration: {integration.display_name}\n"
            f"- Integration ID: `{integration.integration_id}`\n"
            f"- Server type: {integration.server_type}\n"
        )

        if not integration.tools:
            sections.append("No tools available for this integration.\n")
            continue

        sections.append("#### Tools:\n")
        for tool in integration.tools:
            sections.append(
                f"**{tool.tool_name}** — {tool.display_name}\n"
                f"- Category: {tool.category}\n"
                f"- Description: {tool.description}\n"
                f"- Input parameters:\n```json\n"
                f"{json.dumps(tool.input_schema, indent=2)}\n```\n"
            )
            if tool.output_schema:
                sections.append(
                    f"- Output schema:\n```json\n{json.dumps(tool.output_schema, indent=2)}\n```\n"
                )

    return "\n".join(sections)


def _has_tools(tool_catalog: ToolCatalog) -> bool:
    """Check if the tool catalog contains at least one tool.

    Args:
        tool_catalog: The tool catalog to check.

    Returns:
        True if at least one integration has at least one tool.
    """
    return any(len(integration.tools) > 0 for integration in tool_catalog.integrations)


def build_interpretation_prompt(
    query_text: str,
    tool_catalog: ToolCatalog,
) -> tuple[str, str]:
    """Build the system prompt and user message for the Interpreter.

    Args:
        query_text: The user's natural language question.
        tool_catalog: The filtered tool catalog for this query.

    Returns:
        Tuple of (system_prompt, user_message).

    Raises:
        ValueError: If the tool catalog has no integrations or no tools.
    """
    logger.debug(
        "building interpretation prompt",
        action="build_prompt",
        integration_count=len(tool_catalog.integrations),
    )

    if not tool_catalog.integrations or not _has_tools(tool_catalog):
        raise ValueError(
            "Tool catalog is empty or contains no tools. "
            "Cannot generate a plan without available tools."
        )

    catalog_section = _format_tool_catalog(tool_catalog)
    example_section = _build_dynamic_example(tool_catalog)

    system_prompt = "\n\n".join(
        [
            ROLE_SECTION,
            catalog_section,
            OUTPUT_FORMAT_SECTION,
            example_section,
            CONSTRAINTS_SECTION,
        ]
    )

    logger.info(
        "interpretation prompt built",
        action="build_prompt",
        prompt_length=len(system_prompt),
    )

    return system_prompt, query_text
