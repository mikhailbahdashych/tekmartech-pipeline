"""Tests for the plan parser and catalog validation."""

import json

from src.interpretation.plan_parser import (
    parse_plan_from_llm_output,
    validate_plan_against_catalog,
)
from src.models.query_plan import QueryPlan
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition

VALID_PLAN_JSON = json.dumps(
    {
        "plan_id": "test-plan-001",
        "plan_version": "1.0",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "aws.iam_list_users",
                "integration_id": "int-aws-001",
                "parameters": {"max_results": 1000},
                "description": "List all IAM users",
                "output_alias": "all_users",
            }
        ],
        "estimated_tool_calls": 1,
        "summary": "List all IAM users from the AWS account.",
    }
)


def _make_catalog() -> ToolCatalog:
    return ToolCatalog(
        integrations=[
            CatalogIntegration(
                integration_id="int-aws-001",
                server_type="aws",
                display_name="Production AWS",
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
            CatalogIntegration(
                integration_id="int-github-001",
                server_type="github",
                display_name="GitHub Org",
                tools=[
                    ToolDefinition(
                        tool_name="github.list_organization_members",
                        display_name="List Org Members",
                        description="Lists members",
                        category="identity",
                    ),
                ],
            ),
        ]
    )


# =============================================================================
# Plan extraction tests
# =============================================================================


def test_parse_with_delimiters():
    """Parses plan from ---PLAN_START--- / ---PLAN_END--- delimiters."""
    llm_output = (
        f"Here is my analysis of the query.\n\n---PLAN_START---\n{VALID_PLAN_JSON}\n---PLAN_END---"
    )
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is not None
    assert result.query_plan.plan_id == "test-plan-001"
    assert result.error_message is None
    assert "analysis" in result.analysis_text.lower()


def test_parse_with_code_fence_fallback():
    """Parses plan from markdown ```json code fence."""
    llm_output = f"Here is my analysis.\n\n```json\n{VALID_PLAN_JSON}\n```"
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is not None
    assert result.query_plan.plan_id == "test-plan-001"
    assert result.error_message is None


def test_parse_with_bare_json_fallback():
    """Parses plan from bare JSON object at end of text."""
    llm_output = f"Here is my analysis.\n\n{VALID_PLAN_JSON}"
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is not None
    assert result.query_plan.plan_id == "test-plan-001"


def test_parse_no_plan_block():
    """Returns None plan when LLM output has no plan."""
    llm_output = "I cannot answer this question because no tools are available."
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is None
    assert result.error_message is not None
    assert result.analysis_text == llm_output


def test_parse_invalid_json():
    """Returns error when plan block contains invalid JSON."""
    llm_output = "Analysis text.\n\n---PLAN_START---\n{invalid json}\n---PLAN_END---"
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is None
    assert "invalid JSON" in result.error_message


def test_parse_invalid_plan_schema():
    """Returns error when JSON is valid but missing required plan fields."""
    invalid_plan = json.dumps({"plan_id": "test", "steps": []})
    llm_output = f"Analysis.\n\n---PLAN_START---\n{invalid_plan}\n---PLAN_END---"
    result = parse_plan_from_llm_output(llm_output)
    assert result.query_plan is None
    assert "schema" in result.error_message.lower()


def test_parse_extracts_analysis_text():
    """Analysis text is everything before the plan block."""
    analysis = "Step 1: We will list users.\nStep 2: We will filter by MFA."
    llm_output = f"{analysis}\n\n---PLAN_START---\n{VALID_PLAN_JSON}\n---PLAN_END---"
    result = parse_plan_from_llm_output(llm_output)
    assert result.analysis_text == analysis


# =============================================================================
# Catalog validation tests
# =============================================================================


def test_validate_plan_valid():
    """Valid plan passes validation with empty error list."""
    plan = QueryPlan(**json.loads(VALID_PLAN_JSON))
    errors = validate_plan_against_catalog(plan, _make_catalog())
    assert errors == []


def test_validate_plan_unknown_tool():
    """Plan step with unknown tool_name is rejected."""
    plan_data = json.loads(VALID_PLAN_JSON)
    plan_data["steps"][0]["tool_name"] = "aws.nonexistent_tool"
    plan = QueryPlan(**plan_data)
    errors = validate_plan_against_catalog(plan, _make_catalog())
    assert len(errors) == 1
    assert "unknown tool_name" in errors[0].lower() or "nonexistent_tool" in errors[0]


def test_validate_plan_unknown_integration():
    """Plan step with unknown integration_id is rejected."""
    plan_data = json.loads(VALID_PLAN_JSON)
    plan_data["steps"][0]["integration_id"] = "int-nonexistent"
    plan = QueryPlan(**plan_data)
    errors = validate_plan_against_catalog(plan, _make_catalog())
    assert len(errors) == 1
    assert "int-nonexistent" in errors[0]


def test_validate_plan_tool_wrong_integration():
    """Plan step with tool belonging to a different integration is rejected."""
    plan_data = json.loads(VALID_PLAN_JSON)
    # aws.iam_list_users belongs to int-aws-001, not int-github-001
    plan_data["steps"][0]["integration_id"] = "int-github-001"
    plan = QueryPlan(**plan_data)
    errors = validate_plan_against_catalog(plan, _make_catalog())
    assert len(errors) == 1
    assert "does not belong" in errors[0]


def test_validate_plan_duplicate_step_ids():
    """Plan with duplicate step_ids is rejected."""
    plan_data = json.loads(VALID_PLAN_JSON)
    plan_data["steps"].append(
        {
            "step_id": "step_1",
            "tool_name": "aws.s3_list_buckets",
            "integration_id": "int-aws-001",
            "parameters": {},
            "description": "Duplicate step",
            "output_alias": "buckets",
        }
    )
    plan = QueryPlan(**plan_data)
    errors = validate_plan_against_catalog(plan, _make_catalog())
    assert any("duplicate" in e.lower() for e in errors)
