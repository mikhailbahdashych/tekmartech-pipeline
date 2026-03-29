"""System prompt construction for the Interpreter.

Translates a tool_catalog into an LLM prompt that instructs the AI
to analyze an infrastructure query and produce a structured query plan.
The prompt quality directly determines plan quality.
"""

import json

import structlog

from src.models.tool_catalog import ToolCatalog

logger = structlog.get_logger(__name__)

ROLE_SECTION = """\
You are an infrastructure query planner for Tekmar, an AI-powered security \
and compliance analysis tool. Your task is to analyze a user's natural \
language question about their infrastructure and produce a structured \
execution plan that will answer it.

You have access to a set of MCP tools organized by integration. Each \
integration represents a connected external system (e.g., an AWS account, \
a GitHub organization). You MUST only use tools from the catalog provided \
below.\
"""

OUTPUT_FORMAT_SECTION = """\
## Your Response Format

Structure your response in two parts:

**Part 1 — Analysis (natural language)**
Write a clear analysis explaining:
- What the user is asking for
- Which tools and integrations you will use and why
- How the steps connect (what data flows from one step to the next)
- Any assumptions you are making

**Part 2 — Execution Plan (structured JSON)**
After your analysis, output the structured plan between these exact delimiters:

---PLAN_START---
{
  "plan_id": "auto",
  "plan_version": "auto",
  "steps": [
    {
      "step_id": "step_1",
      "tool_name": "<tool_name from catalog>",
      "integration_id": "<integration_id that owns this tool>",
      "parameters": {},
      "description": "<what this step does and why>",
      "output_alias": "<snake_case name for this step's output>"
    }
  ],
  "estimated_tool_calls": 0,
  "summary": "<human-readable summary of the entire plan>"
}
---PLAN_END---

### System-assigned fields (do NOT change these values):
- **plan_id**: Always set to "auto"
- **plan_version**: Always set to "auto"
- **step_id**: Always use "step_1", "step_2", "step_3", etc. in order
- **estimated_tool_calls**: Always set to 0

These fields are overwritten by the system after parsing. Focus on the \
fields that matter: tool_name, integration_id, parameters, and the plan logic.

### Plan field requirements:
- **tool_name**: Must exactly match a tool_name from the catalog below
- **integration_id**: Must exactly match the integration_id that owns the tool
- **parameters**: Must conform to the tool's input_schema
- **description**: Clear explanation of what this step does
- **output_alias**: Descriptive snake_case name (e.g., "all_iam_users")
- **summary**: Clear explanation a non-technical person can understand

### Optional step fields:
- **depends_on**: Array of step_ids that must complete first
- **transform**: Filter or select fields from results
  - filter: { "array_path": "users", "condition": { "field": "mfa_enabled", \
"operator": "equals", "value": false } }
  - select_fields: ["username", "mfa_enabled"]
- **iterate_over**: Execute per item from a previous step
  - { "source_step": "step_1", "array_path": "repositories", "item_alias": "repo" }
- **paginate**: Boolean, default true. Set false to only get the first page.

### Referencing previous step output:
Use template syntax: "{{step_1.all_iam_users.users}}" to reference the \
"users" field from step_1's output.\
"""

CONSTRAINTS_SECTION = """\
## Constraints

1. You MUST only use tool_name values that appear in the catalog above.
2. You MUST only use integration_id values that appear in the catalog above.
3. Each tool MUST be paired with the integration_id of the integration that owns it.
4. If the user's question cannot be answered with the available tools, explain \
why in your analysis and do NOT output a ---PLAN_START--- block.
5. Keep plans as simple as possible — use the minimum number of steps needed.
6. Do NOT copy integration_id values from examples. The only valid \
integration_id values are those listed in the Available Tool Catalog section. \
Each integration_id is a UUID like '4f405c15-a2a8-4c62-8363-b462b5bb7abe', \
not a short string like 'int-aws-001'.
7. Do NOT modify system-assigned fields (plan_id, plan_version, \
estimated_tool_calls). Leave them exactly as shown in the template.\
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
    # Find first integration with tools
    integration = None
    for integ in tool_catalog.integrations:
        if integ.tools:
            integration = integ
            break

    if integration is None:
        return ""

    tool = integration.tools[0]

    return f"""\
## Example

The example below uses real integration and tool values from YOUR catalog \
above. Always use the exact integration_id and tool_name values from the catalog.

For this catalog, a simple single-step plan would look like:

---PLAN_START---
{{
  "plan_id": "auto",
  "plan_version": "auto",
  "steps": [
    {{
      "step_id": "step_1",
      "tool_name": "{tool.tool_name}",
      "integration_id": "{integration.integration_id}",
      "parameters": {{}},
      "description": "Retrieve data using {tool.display_name} from {integration.display_name}",
      "output_alias": "{tool.tool_name.split(".")[-1]}_results"
    }}
  ],
  "estimated_tool_calls": 0,
  "summary": "Query {integration.display_name} using {tool.display_name}."
}}
---PLAN_END---\
"""


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
