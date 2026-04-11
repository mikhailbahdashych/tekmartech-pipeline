"""System prompt construction for the Interpreter.

Translates a tool_catalog into an LLM prompt that instructs the AI
to analyze an infrastructure query and produce either a structured
query plan or a clarification question. Supports multi-turn conversation
by converting conversation_history into an LLM messages list.

The prompt tells the LLM to produce ONLY the intelligence fields (tool
selection, parameters, reasoning). Mechanical fields (plan_id, plan_version,
step_id, estimated_tool_calls, clarification_id) are injected by the parser.
"""

import json

import structlog

from src.models.conversation import AssistantTurn, UserTurn
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

Your response MUST have exactly two sections in this order:

### Analysis
Write 2-4 sentences explaining:
- What the user is asking for
- Which tool(s) you will use and from which integration
- Any assumptions you are making

### Structured Output
Your response must end with EXACTLY ONE of these blocks:

**Option A — If you have enough information to create a plan:**

---PLAN_START---
<your plan JSON here>
---PLAN_END---

The plan JSON must have this exact structure:
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
These are added automatically by the system.

**Option B — If the query is ambiguous and you need more information:**

---CLARIFICATION_START---
{
  "question": "Your clarification question in natural language",
  "options": [
    { "option_id": "opt_1", "label": "Short label", "description": "Optional explanation" },
    { "option_id": "opt_2", "label": "Another choice" }
  ],
  "allows_free_text": true
}
---CLARIFICATION_END---

IMPORTANT: Output EITHER a plan OR a clarification, NEVER both.\
"""

CLARIFICATION_GUIDANCE_SECTION = """\
## When to Ask for Clarification

Ask a clarification question ONLY when:
- The query is genuinely ambiguous (could mean different things)
- Multiple integrations of the same type exist and the user did not specify \
which one (e.g., "check our AWS" when Production AWS and Staging AWS are both connected)
- The query could be answered at very different levels of detail and you need \
to know what the user wants

Do NOT ask for clarification when:
- The query is clear and can be answered directly
- There is only one integration of the relevant type
- You can make a reasonable default assumption

When in doubt, generate a plan. Users prefer getting results they can refine \
over being asked unnecessary questions.

Keep clarification questions concise (1-2 sentences). Provide 2-4 options \
when possible. Always set allows_free_text to true unless the choices are \
strictly enumerated.\
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
in your JSON.
7. Ask for clarification only when genuinely ambiguous. Prefer producing a plan.\
"""


def _build_dynamic_example(tool_catalog: ToolCatalog) -> str:
    """Build an example plan section using real catalog values.

    Uses the first integration with at least one tool to construct
    an example, so the LLM sees consistent IDs between the catalog
    and the example. If the catalog has 2+ integrations, also shows
    a brief clarification example using real integration names.

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

    example = f"""## Example using your catalog

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

    # Add clarification example if catalog has 2+ integrations with tools
    integrations_with_tools = [i for i in tool_catalog.integrations if i.tools]
    if len(integrations_with_tools) >= 2:
        first = integrations_with_tools[0]
        second = integrations_with_tools[1]
        example += f"""

If you need clarification, for example to ask which integration to query:

---CLARIFICATION_START---
{{
  "question": "Which integration would you like me to query?",
  "options": [
    {{ "option_id": "opt_1", "label": "{first.display_name}" }},
    {{ "option_id": "opt_2", "label": "{second.display_name}" }}
  ],
  "allows_free_text": true
}}
---CLARIFICATION_END---"""

    return example


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


def _format_user_turn(turn: UserTurn) -> str:
    """Convert a UserTurn model to a natural language message string.

    Args:
        turn: The user turn from conversation history.

    Returns:
        Formatted string representing the user's response.
    """
    if turn.response_type == "option_selection" and turn.selected_option_id:
        content = f"I choose: {turn.selected_option_id}"
        if turn.text:
            content += f". {turn.text}"
        return content
    return turn.text or ""


def _build_messages(
    query_text: str,
    conversation_history: list[AssistantTurn | UserTurn] | None,
) -> list[dict[str, str]]:
    """Build the LLM messages list from query text and conversation history.

    For first turn (no history): returns a single user message with the query.
    For continuation turns: reconstructs the full conversation as alternating
    user/assistant messages, ending with the user's latest response.

    Args:
        query_text: The original user query.
        conversation_history: Previous conversation turns, or None.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": query_text}]

    if not conversation_history:
        return messages

    for turn in conversation_history:
        if isinstance(turn, AssistantTurn):
            messages.append({"role": "assistant", "content": turn.content})
        elif isinstance(turn, UserTurn):
            content = _format_user_turn(turn)
            messages.append({"role": "user", "content": content})

    # Final user message to prompt the LLM to continue
    messages.append({
        "role": "user",
        "content": "Based on my response, please continue your analysis and generate the execution plan.",
    })

    return messages


def build_interpretation_prompt(
    query_text: str,
    tool_catalog: ToolCatalog,
    conversation_history: list[AssistantTurn | UserTurn] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Build the system prompt and messages list for the Interpreter.

    For single-turn (no history): system prompt + one user message.
    For multi-turn (with history): system prompt + full conversation messages.

    Args:
        query_text: The user's natural language question.
        tool_catalog: The filtered tool catalog for this query.
        conversation_history: Previous turns for multi-turn, or None.

    Returns:
        Tuple of (system_prompt, messages_list).

    Raises:
        ValueError: If the tool catalog has no integrations or no tools.
    """
    logger.debug(
        "building interpretation prompt",
        action="build_prompt",
        integration_count=len(tool_catalog.integrations),
        has_history=conversation_history is not None and len(conversation_history) > 0,
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
            CLARIFICATION_GUIDANCE_SECTION,
            CONSTRAINTS_SECTION,
        ]
    )

    messages = _build_messages(query_text, conversation_history)

    logger.info(
        "interpretation prompt built",
        action="build_prompt",
        prompt_length=len(system_prompt),
        message_count=len(messages),
    )

    return system_prompt, messages
