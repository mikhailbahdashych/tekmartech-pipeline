"""Tests for src.interpretation.prompt_builder."""

import pytest

from src.interpretation.prompt_builder import (
    CONSTRAINTS_SECTION,
    OUTPUT_FORMAT_SECTION,
    _build_dynamic_example,
    build_interpretation_prompt,
)
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition

INTEGRATION_ID = "4f405c15-a2a8-4c62-8363-b462b5bb7abe"


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


class TestConstraintsExcludesMechanicalFields:
    """Constraints section reinforces: do not include mechanical fields."""

    def test_constraints_mention_exclusion(self):
        assert "plan_id" in CONSTRAINTS_SECTION
        assert "step_id" in CONSTRAINTS_SECTION
        assert "estimated_tool_calls" in CONSTRAINTS_SECTION


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


class TestBuildInterpretationPrompt:
    """Integration: full prompt assembly."""

    def test_returns_tuple(self):
        catalog = _make_catalog()
        system, user = build_interpretation_prompt("List all repos", catalog)
        assert isinstance(system, str)
        assert user == "List all repos"

    def test_system_prompt_contains_catalog(self):
        catalog = _make_catalog()
        system, _ = build_interpretation_prompt("query", catalog)
        assert INTEGRATION_ID in system
        assert "github.list_repositories" in system

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
