"""Tests for the interpretation prompt builder."""

import pytest

from src.interpretation.prompt_builder import build_interpretation_prompt
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition


def _make_catalog(*integrations: CatalogIntegration) -> ToolCatalog:
    return ToolCatalog(integrations=list(integrations))


def _make_integration(
    integration_id: str = "int-001",
    server_type: str = "aws",
    display_name: str = "Production AWS",
    tools: list[ToolDefinition] | None = None,
) -> CatalogIntegration:
    if tools is None:
        tools = [
            ToolDefinition(
                tool_name=f"{server_type}.list_users",
                display_name="List Users",
                description="Lists all users",
                category="identity",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object"},
            )
        ]
    return CatalogIntegration(
        integration_id=integration_id,
        server_type=server_type,
        display_name=display_name,
        tools=tools,
    )


def test_build_prompt_includes_all_tools():
    """System prompt includes all tool names and integration IDs."""
    catalog = _make_catalog(
        _make_integration(
            integration_id="int-aws",
            server_type="aws",
            tools=[
                ToolDefinition(
                    tool_name="aws.iam_list_users",
                    display_name="List IAM Users",
                    description="Lists IAM users",
                    category="identity",
                ),
                ToolDefinition(
                    tool_name="aws.s3_list_buckets",
                    display_name="List S3 Buckets",
                    description="Lists S3 buckets",
                    category="storage",
                ),
            ],
        ),
        _make_integration(
            integration_id="int-github",
            server_type="github",
            tools=[
                ToolDefinition(
                    tool_name="github.list_organization_members",
                    display_name="List Org Members",
                    description="Lists org members",
                    category="identity",
                ),
            ],
        ),
    )

    system_prompt, user_message = build_interpretation_prompt("test query", catalog)

    assert "aws.iam_list_users" in system_prompt
    assert "aws.s3_list_buckets" in system_prompt
    assert "github.list_organization_members" in system_prompt
    assert "int-aws" in system_prompt
    assert "int-github" in system_prompt


def test_build_prompt_user_message_is_query_text():
    """User message is exactly the query text, unmodified."""
    catalog = _make_catalog(_make_integration())
    _, user_message = build_interpretation_prompt("Show me all IAM users", catalog)
    assert user_message == "Show me all IAM users"


def test_build_prompt_includes_output_format():
    """System prompt includes output format instructions with delimiters."""
    catalog = _make_catalog(_make_integration())
    system_prompt, _ = build_interpretation_prompt("test", catalog)
    assert "---PLAN_START---" in system_prompt
    assert "---PLAN_END---" in system_prompt
    assert "plan_id" in system_prompt
    assert "step_id" in system_prompt


def test_build_prompt_includes_input_schema():
    """System prompt includes tool input_schema JSON."""
    catalog = _make_catalog(
        _make_integration(
            tools=[
                ToolDefinition(
                    tool_name="aws.test_tool",
                    display_name="Test",
                    description="Test tool",
                    category="identity",
                    input_schema={
                        "type": "object",
                        "properties": {"max_results": {"type": "integer"}},
                    },
                ),
            ],
        )
    )
    system_prompt, _ = build_interpretation_prompt("test", catalog)
    assert "max_results" in system_prompt


def test_build_prompt_empty_catalog_raises():
    """Empty catalog raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        build_interpretation_prompt("test", ToolCatalog(integrations=[]))


def test_build_prompt_catalog_with_empty_tools_raises():
    """Catalog with integrations but no tools raises ValueError."""
    catalog = _make_catalog(_make_integration(integration_id="int-1", tools=[]))
    with pytest.raises(ValueError, match="empty"):
        build_interpretation_prompt("test", catalog)
