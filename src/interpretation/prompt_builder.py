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
  "plan_id": "<generate a unique UUID>",
  "plan_version": "1.0",
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
  "estimated_tool_calls": <integer>,
  "summary": "<human-readable summary of the entire plan>"
}
---PLAN_END---

### Plan field requirements:
- **plan_id**: A UUID you generate (e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
- **plan_version**: Always "1.0"
- **steps**: Ordered array, at least one step. Steps execute sequentially.
- **step_id**: Convention "step_1", "step_2", etc.
- **tool_name**: Must exactly match a tool_name from the catalog below
- **integration_id**: Must exactly match the integration_id that owns the tool
- **parameters**: Must conform to the tool's input_schema
- **description**: Clear explanation of what this step does
- **output_alias**: Descriptive snake_case name (e.g., "all_iam_users")
- **estimated_tool_calls**: Total expected MCP invocations including pagination
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
5. estimated_tool_calls should account for pagination and iteration expansions.
6. Keep plans as simple as possible — use the minimum number of steps needed.\
"""

EXAMPLE_SECTION = """\
## Example

For a query "Show me all IAM users without MFA" with an AWS integration \
(id: "int-aws-001") that has tools aws.iam_list_users and \
aws.iam_get_account_summary, a good plan would be:

---PLAN_START---
{
  "plan_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "plan_version": "1.0",
  "steps": [
    {
      "step_id": "step_1",
      "tool_name": "aws.iam_list_users",
      "integration_id": "int-aws-001",
      "parameters": {"max_results": 1000},
      "description": "Retrieve all IAM users with their MFA status and metadata",
      "output_alias": "all_iam_users",
      "transform": {
        "filter": {
          "array_path": "users",
          "condition": {
            "field": "mfa_enabled",
            "operator": "equals",
            "value": false
          }
        },
        "select_fields": ["username", "user_id", "mfa_enabled", "last_login_at", "created_at"]
      }
    }
  ],
  "estimated_tool_calls": 1,
  "summary": "List all IAM users and filter for those without MFA enabled."
}
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

    system_prompt = "\n\n".join(
        [
            ROLE_SECTION,
            catalog_section,
            OUTPUT_FORMAT_SECTION,
            EXAMPLE_SECTION,
            CONSTRAINTS_SECTION,
        ]
    )

    logger.info(
        "interpretation prompt built",
        action="build_prompt",
        prompt_length=len(system_prompt),
    )

    return system_prompt, query_text
