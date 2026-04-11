"""Tests for src.interpretation.prompt_builder."""

import pytest

from src.interpretation.prompt_builder import (
    CLARIFICATION_GUIDANCE_SECTION,
    CONSTRAINTS_SECTION,
    OUTPUT_FORMAT_SECTION,
    _build_dynamic_example,
    _build_messages,
    _format_user_turn,
    build_interpretation_prompt,
)
from src.models.conversation import (
    AssistantTurn,
    ClarificationOption,
    ClarificationQuestion,
    UserTurn,
)
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition

INTEGRATION_ID = "4f405c15-a2a8-4c62-8363-b462b5bb7abe"
INTEGRATION_ID_2 = "b1c2d3e4-f5a6-7890-abcd-ef1234567890"


def _make_catalog() -> ToolCatalog:
    return ToolCatalog(
        integrations=[
            CatalogIntegration(
                integration_id=INTEGRATION_ID,
                server_type="github",
                display_name="Production GitHub",
                tools=[
                    ToolDefinition(
                        tool_name="github.list_repositories",
                        display_name="List Repositories",
                        description="List all repositories in the org.",
                        category="configuration",
                        input_schema={"type": "object", "properties": {}},
                        output_schema={"type": "object"},
                    ),
                ],
            ),
        ]
    )


def _make_multi_integration_catalog() -> ToolCatalog:
    return ToolCatalog(
        integrations=[
            CatalogIntegration(
                integration_id=INTEGRATION_ID,
                server_type="aws",
                display_name="Production AWS",
                tools=[
                    ToolDefinition(
                        tool_name="aws.iam_list_users",
                        display_name="List IAM Users",
                        description="List all IAM users.",
                        category="identity",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ],
            ),
            CatalogIntegration(
                integration_id=INTEGRATION_ID_2,
                server_type="aws",
                display_name="Staging AWS",
                tools=[
                    ToolDefinition(
                        tool_name="aws.iam_list_users",
                        display_name="List IAM Users",
                        description="List all IAM users.",
                        category="identity",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ],
            ),
        ]
    )


def _make_empty_catalog() -> ToolCatalog:
    return ToolCatalog(integrations=[])


class TestOutputFormatExcludesMechanicalFields:
    """The prompt must NOT instruct the LLM to produce mechanical fields."""

    def test_no_plan_id_in_output_format(self):
        assert '"plan_id"' not in OUTPUT_FORMAT_SECTION

    def test_no_estimated_tool_calls_in_output_format(self):
        assert '"estimated_tool_calls"' not in OUTPUT_FORMAT_SECTION

    def test_no_plan_version_in_output_format(self):
        assert '"plan_version"' not in OUTPUT_FORMAT_SECTION

    def test_no_step_id_in_output_format(self):
        assert '"step_id"' not in OUTPUT_FORMAT_SECTION


class TestOutputFormatIncludesLLMFields:
    """The prompt MUST instruct the LLM to produce intelligence fields."""

    def test_has_tool_name(self):
        assert '"tool_name"' in OUTPUT_FORMAT_SECTION

    def test_has_integration_id(self):
        assert '"integration_id"' in OUTPUT_FORMAT_SECTION

    def test_has_parameters(self):
        assert '"parameters"' in OUTPUT_FORMAT_SECTION

    def test_has_description(self):
        assert '"description"' in OUTPUT_FORMAT_SECTION

    def test_has_output_alias(self):
        assert '"output_alias"' in OUTPUT_FORMAT_SECTION


class TestOutputFormatIncludesClarification:
    """The prompt must teach the LLM about both plan and clarification output."""

    def test_has_plan_start_marker(self):
        assert "---PLAN_START---" in OUTPUT_FORMAT_SECTION

    def test_has_clarification_start_marker(self):
        assert "---CLARIFICATION_START---" in OUTPUT_FORMAT_SECTION

    def test_has_clarification_end_marker(self):
        assert "---CLARIFICATION_END---" in OUTPUT_FORMAT_SECTION

    def test_mentions_either_plan_or_clarification(self):
        assert "EITHER" in OUTPUT_FORMAT_SECTION
        assert "NEVER both" in OUTPUT_FORMAT_SECTION


class TestClarificationGuidance:
    """Clarification guidance section exists and has the right content."""

    def test_mentions_ambiguous(self):
        assert "ambiguous" in CLARIFICATION_GUIDANCE_SECTION

    def test_mentions_prefer_plan(self):
        assert "generate a plan" in CLARIFICATION_GUIDANCE_SECTION


class TestConstraintsExcludesMechanicalFields:
    """Constraints section reinforces: do not include mechanical fields."""

    def test_constraints_mention_exclusion(self):
        assert "plan_id" in CONSTRAINTS_SECTION
        assert "step_id" in CONSTRAINTS_SECTION
        assert "estimated_tool_calls" in CONSTRAINTS_SECTION

    def test_constraints_mention_clarification_preference(self):
        assert "clarification" in CONSTRAINTS_SECTION


class TestDynamicExample:
    """Dynamic example uses real catalog values and excludes mechanical fields."""

    def test_uses_real_integration_id(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert INTEGRATION_ID in example

    def test_uses_real_tool_name(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "github.list_repositories" in example

    def test_no_plan_id_in_example(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "plan_id" not in example

    def test_no_step_id_in_example(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "step_id" not in example

    def test_no_estimated_tool_calls_in_example(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "estimated_tool_calls" not in example

    def test_no_plan_version_in_example(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "plan_version" not in example

    def test_empty_catalog_returns_empty(self):
        catalog = _make_empty_catalog()
        example = _build_dynamic_example(catalog)
        assert example == ""

    def test_multi_integration_includes_clarification_example(self):
        catalog = _make_multi_integration_catalog()
        example = _build_dynamic_example(catalog)
        assert "---CLARIFICATION_START---" in example
        assert "Production AWS" in example
        assert "Staging AWS" in example

    def test_single_integration_no_clarification_example(self):
        catalog = _make_catalog()
        example = _build_dynamic_example(catalog)
        assert "---CLARIFICATION_START---" not in example


class TestBuildInterpretationPrompt:
    """Integration: full prompt assembly."""

    def test_returns_tuple_with_messages_list(self):
        catalog = _make_catalog()
        system, messages = build_interpretation_prompt("List all repos", catalog)
        assert isinstance(system, str)
        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "List all repos"

    def test_system_prompt_contains_catalog(self):
        catalog = _make_catalog()
        system, _ = build_interpretation_prompt("query", catalog)
        assert INTEGRATION_ID in system
        assert "github.list_repositories" in system

    def test_system_prompt_contains_both_markers(self):
        catalog = _make_catalog()
        system, _ = build_interpretation_prompt("query", catalog)
        assert "---PLAN_START---" in system
        assert "---CLARIFICATION_START---" in system

    def test_raises_on_empty_catalog(self):
        catalog = _make_empty_catalog()
        with pytest.raises(ValueError, match="empty"):
            build_interpretation_prompt("query", catalog)

    def test_raises_on_no_tools(self):
        catalog = ToolCatalog(
            integrations=[
                CatalogIntegration(
                    integration_id=INTEGRATION_ID,
                    server_type="github",
                    display_name="GitHub",
                    tools=[],
                )
            ]
        )
        with pytest.raises(ValueError, match="no tools"):
            build_interpretation_prompt("query", catalog)


class TestBuildMessages:
    """Messages list construction for single and multi-turn."""

    def test_first_turn_single_message(self):
        messages = _build_messages("What repos exist?", None)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "What repos exist?"}

    def test_first_turn_empty_history(self):
        messages = _build_messages("query", [])
        assert len(messages) == 1

    def test_continuation_includes_history(self):
        history = [
            AssistantTurn(
                role="assistant",
                content="I need to know which AWS account.",
                clarification=ClarificationQuestion(
                    clarification_id="abc",
                    question="Which AWS?",
                    options=[
                        ClarificationOption(option_id="opt_1", label="Prod"),
                        ClarificationOption(option_id="opt_2", label="Staging"),
                    ],
                ),
            ),
            UserTurn(
                role="user",
                response_type="option_selection",
                selected_option_id="opt_1",
            ),
        ]
        messages = _build_messages("Check security", history)
        # Original query + assistant + user + continuation prompt
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Check security"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "user"
        assert "continue" in messages[3]["content"].lower()

    def test_continuation_with_history_passed_to_builder(self):
        catalog = _make_catalog()
        history = [
            AssistantTurn(role="assistant", content="Analysis text."),
            UserTurn(role="user", response_type="free_text", text="Just list them all."),
        ]
        _, messages = build_interpretation_prompt("List repos", catalog, history)
        assert len(messages) == 4


class TestFormatUserTurn:
    """User turn formatting."""

    def test_option_selection(self):
        turn = UserTurn(
            role="user",
            response_type="option_selection",
            selected_option_id="opt_1",
        )
        assert "opt_1" in _format_user_turn(turn)

    def test_option_selection_with_text(self):
        turn = UserTurn(
            role="user",
            response_type="option_selection",
            selected_option_id="opt_2",
            text="And also check staging.",
        )
        result = _format_user_turn(turn)
        assert "opt_2" in result
        assert "staging" in result.lower()

    def test_free_text(self):
        turn = UserTurn(
            role="user",
            response_type="free_text",
            text="Check the production account.",
        )
        assert _format_user_turn(turn) == "Check the production account."

    def test_free_text_empty(self):
        turn = UserTurn(role="user", response_type="free_text")
        assert _format_user_turn(turn) == ""
