"""Tests for src.interpretation.plan_parser."""

import json
import uuid

import pytest

from src.interpretation.plan_parser import (
    parse_plan_from_llm_output,
    validate_plan_against_catalog,
)
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition

INTEGRATION_ID = "4f405c15-a2a8-4c62-8363-b462b5bb7abe"
TOOL_NAME = "github.list_repositories"


def _make_catalog() -> ToolCatalog:
    return ToolCatalog(
        integrations=[
            CatalogIntegration(
                integration_id=INTEGRATION_ID,
                server_type="github",
                display_name="Production GitHub",
                tools=[
                    ToolDefinition(
                        tool_name=TOOL_NAME,
                        display_name="List Repositories",
                        description="List all repositories.",
                        category="configuration",
                        input_schema={"type": "object", "properties": {}},
                    ),
                    ToolDefinition(
                        tool_name="github.get_branch_protection",
                        display_name="Get Branch Protection",
                        description="Get branch protection rules.",
                        category="configuration",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ],
            ),
        ]
    )


def _wrap_plan(plan_dict: dict, analysis: str = "Analysis text here.") -> str:
    """Wrap a plan dict into full LLM output with delimiters."""
    return f"{analysis}\n\n---PLAN_START---\n{json.dumps(plan_dict)}\n---PLAN_END---"


def _wrap_clarification(clar_dict: dict, analysis: str = "Analysis text here.") -> str:
    """Wrap a clarification dict into full LLM output with delimiters."""
    return f"{analysis}\n\n---CLARIFICATION_START---\n{json.dumps(clar_dict)}\n---CLARIFICATION_END---"


def _minimal_plan(**overrides) -> dict:
    """Build a minimal valid LLM plan dict (no mechanical fields)."""
    plan = {
        "steps": [
            {
                "tool_name": TOOL_NAME,
                "integration_id": INTEGRATION_ID,
                "parameters": {},
                "description": "List all repositories",
                "output_alias": "org_repos",
            }
        ],
        "summary": "List repositories from GitHub.",
    }
    plan.update(overrides)
    return plan


def _minimal_clarification(**overrides) -> dict:
    """Build a minimal valid clarification dict."""
    clar = {
        "question": "Which AWS account should I query?",
        "options": [
            {"option_id": "opt_1", "label": "Production AWS"},
            {"option_id": "opt_2", "label": "Staging AWS"},
        ],
        "allows_free_text": True,
    }
    clar.update(overrides)
    return clar


class TestValidParse:
    """Parser correctly handles well-formed LLM output with a plan."""

    def test_parses_valid_plan(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is not None
        assert result.clarification is None
        assert result.error_message is None

    def test_extracts_analysis_text(self):
        text = _wrap_plan(_minimal_plan(), analysis="My analysis.")
        result = parse_plan_from_llm_output(text)
        assert result.analysis_text == "My analysis."

    def test_injects_plan_id_as_uuid(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan
        uuid.UUID(plan.plan_id)

    def test_injects_plan_version(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        assert result.query_plan.plan_version == "1.0"

    def test_injects_sequential_step_ids(self):
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step one",
                    "output_alias": "step_one_data",
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step two",
                    "output_alias": "step_two_data",
                },
            ],
            "summary": "Two step plan.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan
        assert plan.steps[0].step_id == "step_1"
        assert plan.steps[1].step_id == "step_2"

    def test_estimated_tool_calls_positive(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        assert result.query_plan.estimated_tool_calls > 0

    def test_ignores_llm_provided_mechanical_fields(self):
        plan_dict = _minimal_plan()
        plan_dict["plan_id"] = "bad-id"
        plan_dict["plan_version"] = "99.0"
        plan_dict["estimated_tool_calls"] = 999
        plan_dict["steps"][0]["step_id"] = "wrong_step"
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan
        assert plan.plan_id != "bad-id"
        assert plan.plan_version == "1.0"
        assert plan.estimated_tool_calls != 999
        assert plan.steps[0].step_id == "step_1"


class TestMissingFields:
    """Parser rejects plans with missing LLM-produced fields."""

    def test_missing_tool_name(self):
        plan_dict = _minimal_plan()
        del plan_dict["steps"][0]["tool_name"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert result.clarification is None
        assert "tool_name" in result.error_message

    def test_missing_integration_id(self):
        plan_dict = _minimal_plan()
        del plan_dict["steps"][0]["integration_id"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "integration_id" in result.error_message

    def test_missing_description(self):
        plan_dict = _minimal_plan()
        del plan_dict["steps"][0]["description"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "description" in result.error_message

    def test_missing_output_alias(self):
        plan_dict = _minimal_plan()
        del plan_dict["steps"][0]["output_alias"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "output_alias" in result.error_message

    def test_missing_summary(self):
        plan_dict = _minimal_plan()
        del plan_dict["summary"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "summary" in result.error_message

    def test_empty_steps_array(self):
        plan_dict = _minimal_plan(steps=[])
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "no steps" in result.error_message

    def test_missing_parameters_defaults_to_empty_dict(self):
        plan_dict = _minimal_plan()
        del plan_dict["steps"][0]["parameters"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is not None
        assert result.query_plan.steps[0].parameters == {}


class TestExtractionErrors:
    """Parser handles malformed LLM output."""

    def test_no_plan_start_marker(self):
        text = "Just some analysis text without any plan."
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert result.clarification is None
        assert "No plan block" in result.error_message

    def test_invalid_json_between_markers(self):
        text = "Analysis.\n\n---PLAN_START---\n{not valid json}\n---PLAN_END---"
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "Invalid JSON" in result.error_message

    def test_analysis_text_preserved_on_error(self):
        text = "My analysis.\n\n---PLAN_START---\n{bad}\n---PLAN_END---"
        result = parse_plan_from_llm_output(text)
        assert result.analysis_text == "My analysis."


class TestCrossReferences:
    """Parser validates depends_on and iterate_over references."""

    def test_depends_on_later_step_errors(self):
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step one",
                    "output_alias": "step_one_data",
                    "depends_on": ["step_2"],
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step two",
                    "output_alias": "step_two_data",
                },
            ],
            "summary": "Plan with forward reference.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "non-existent step" in result.error_message

    def test_depends_on_nonexistent_step_errors(self):
        plan_dict = _minimal_plan()
        plan_dict["steps"][0]["depends_on"] = ["step_99"]
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "non-existent step" in result.error_message

    def test_iterate_over_forward_reference_errors(self):
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step one with iterate",
                    "output_alias": "step_one_data",
                    "iterate_over": {
                        "source_step": "step_2",
                        "array_path": "items",
                        "item_alias": "item",
                    },
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step two",
                    "output_alias": "step_two_data",
                },
            ],
            "summary": "Plan with forward iterate.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert "non-existent step" in result.error_message

    def test_valid_depends_on_earlier_step(self):
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step one",
                    "output_alias": "step_one_data",
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Step two",
                    "output_alias": "step_two_data",
                    "depends_on": ["step_1"],
                },
            ],
            "summary": "Plan with valid depends_on.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is not None

    def test_depends_on_using_output_alias_resolved(self):
        """LLM uses output_alias instead of step_id in depends_on — parser resolves it."""
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "List repos",
                    "output_alias": "org_repos",
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Check protection",
                    "output_alias": "protections",
                    "depends_on": ["org_repos"],
                },
            ],
            "summary": "Plan where LLM used output_alias in depends_on.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is not None
        assert result.query_plan.steps[1].depends_on == ["step_1"]

    def test_iterate_over_using_output_alias_resolved(self):
        """LLM uses output_alias in iterate_over.source_step — parser resolves it."""
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "List repos",
                    "output_alias": "org_repos",
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Check per repo",
                    "output_alias": "protections",
                    "iterate_over": {
                        "source_step": "org_repos",
                        "array_path": "repositories",
                        "item_alias": "repo",
                    },
                },
            ],
            "summary": "Plan where LLM used output_alias in iterate_over.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is not None
        assert result.query_plan.steps[1].iterate_over.source_step == "step_1"


class TestEstimatedToolCalls:
    """estimated_tool_calls calculation."""

    def test_iterate_over_multiplies(self):
        plan_dict = {
            "steps": [
                {
                    "tool_name": TOOL_NAME,
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "List repos",
                    "output_alias": "repos",
                },
                {
                    "tool_name": "github.get_branch_protection",
                    "integration_id": INTEGRATION_ID,
                    "parameters": {},
                    "description": "Check protection per repo",
                    "output_alias": "protections",
                    "iterate_over": {
                        "source_step": "step_1",
                        "array_path": "repositories",
                        "item_alias": "repo",
                    },
                },
            ],
            "summary": "Check branch protection for all repos.",
        }
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan
        assert plan.estimated_tool_calls == 13

    def test_paginate_false_no_extra(self):
        plan_dict = _minimal_plan()
        plan_dict["steps"][0]["paginate"] = False
        text = _wrap_plan(plan_dict)
        result = parse_plan_from_llm_output(text)
        assert result.query_plan.estimated_tool_calls == 1


class TestClarificationParse:
    """Parser correctly handles LLM output with clarification blocks."""

    def test_parses_valid_clarification(self):
        text = _wrap_clarification(_minimal_clarification())
        result = parse_plan_from_llm_output(text)
        assert result.clarification is not None
        assert result.query_plan is None
        assert result.error_message is None
        assert result.clarification.question == "Which AWS account should I query?"
        assert len(result.clarification.options) == 2

    def test_clarification_id_is_uuid(self):
        text = _wrap_clarification(_minimal_clarification())
        result = parse_plan_from_llm_output(text)
        uuid.UUID(result.clarification.clarification_id)

    def test_clarification_extracts_analysis_text(self):
        text = _wrap_clarification(_minimal_clarification(), analysis="My reasoning.")
        result = parse_plan_from_llm_output(text)
        assert result.analysis_text == "My reasoning."

    def test_clarification_no_options(self):
        clar = {"question": "What do you want to check?", "allows_free_text": True}
        text = _wrap_clarification(clar)
        result = parse_plan_from_llm_output(text)
        assert result.clarification is not None
        assert result.clarification.options is None
        assert result.clarification.allows_free_text is True

    def test_clarification_allows_free_text_defaults_true(self):
        clar = {"question": "Which one?"}
        text = _wrap_clarification(clar)
        result = parse_plan_from_llm_output(text)
        assert result.clarification.allows_free_text is True

    def test_clarification_normalizes_id_to_option_id(self):
        """LLM uses 'id' instead of 'option_id' — parser normalizes."""
        clar = {
            "question": "Which?",
            "options": [
                {"id": "a", "label": "Option A"},
                {"id": "b", "label": "Option B"},
            ],
        }
        text = _wrap_clarification(clar)
        result = parse_plan_from_llm_output(text)
        assert result.clarification.options[0].option_id == "a"
        assert result.clarification.options[1].option_id == "b"

    def test_clarification_generates_option_ids_if_missing(self):
        clar = {
            "question": "Which?",
            "options": [{"label": "First"}, {"label": "Second"}],
        }
        text = _wrap_clarification(clar)
        result = parse_plan_from_llm_output(text)
        assert result.clarification.options[0].option_id == "opt_1"
        assert result.clarification.options[1].option_id == "opt_2"

    def test_both_plan_and_clarification_is_error(self):
        plan_block = f"---PLAN_START---\n{json.dumps(_minimal_plan())}\n---PLAN_END---"
        clar_block = f"---CLARIFICATION_START---\n{json.dumps(_minimal_clarification())}\n---CLARIFICATION_END---"
        text = f"Analysis.\n\n{plan_block}\n\n{clar_block}"
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert result.clarification is None
        assert "both" in result.error_message.lower()

    def test_neither_plan_nor_clarification_is_error(self):
        text = "Just analysis text, no structured output."
        result = parse_plan_from_llm_output(text)
        assert result.query_plan is None
        assert result.clarification is None
        assert result.error_message is not None

    def test_malformed_clarification_json(self):
        text = "Analysis.\n\n---CLARIFICATION_START---\n{bad json}\n---CLARIFICATION_END---"
        result = parse_plan_from_llm_output(text)
        assert result.clarification is None
        assert "Invalid JSON" in result.error_message

    def test_clarification_missing_question(self):
        clar = {"options": [{"option_id": "a", "label": "A"}]}
        text = _wrap_clarification(clar)
        result = parse_plan_from_llm_output(text)
        assert result.clarification is None
        assert "no question" in result.error_message.lower()


class TestCatalogValidation:
    """validate_plan_against_catalog catches bad tool/integration references."""

    def test_unknown_tool_name(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan

        plan.steps[0].tool_name = "aws.iam_list_users"
        catalog = _make_catalog()
        errors = validate_plan_against_catalog(plan, catalog)
        assert len(errors) > 0
        assert "tool_name" in errors[0]

    def test_unknown_integration_id(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan

        plan.steps[0].integration_id = "00000000-0000-0000-0000-000000000000"
        catalog = _make_catalog()
        errors = validate_plan_against_catalog(plan, catalog)
        assert len(errors) > 0
        assert "integration_id" in errors[0]

    def test_tool_wrong_integration(self):
        catalog = ToolCatalog(
            integrations=[
                CatalogIntegration(
                    integration_id="aaaa",
                    server_type="github",
                    display_name="GitHub",
                    tools=[
                        ToolDefinition(
                            tool_name="github.list_repositories",
                            display_name="List Repos",
                            description="List repos.",
                            category="configuration",
                        )
                    ],
                ),
                CatalogIntegration(
                    integration_id="bbbb",
                    server_type="aws",
                    display_name="AWS",
                    tools=[
                        ToolDefinition(
                            tool_name="aws.iam_list_users",
                            display_name="List Users",
                            description="List users.",
                            category="identity",
                        )
                    ],
                ),
            ]
        )
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        plan = result.query_plan
        plan.steps[0].integration_id = "bbbb"
        errors = validate_plan_against_catalog(plan, catalog)
        assert len(errors) > 0
        assert "does not belong" in errors[0]

    def test_valid_plan_passes(self):
        text = _wrap_plan(_minimal_plan())
        result = parse_plan_from_llm_output(text)
        catalog = _make_catalog()
        errors = validate_plan_against_catalog(result.query_plan, catalog)
        assert errors == []
