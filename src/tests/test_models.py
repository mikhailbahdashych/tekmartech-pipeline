"""Tests for Pydantic model validation.

Verifies that models accept valid data, reject invalid data, and
enforce contract constraints (field types, required fields, literals).
"""

import pytest
from pydantic import ValidationError

from src.models.query_plan import PlanStep, QueryPlan
from src.models.stream_events import (
    ExecutionStarted,
    InterpretationStarted,
    StepCompleted,
)
from src.models.tool_catalog import ToolDefinition


def test_tool_definition_valid_data():
    """ToolDefinition accepts valid data with all required fields."""
    tool = ToolDefinition(
        tool_name="aws.iam_list_users",
        display_name="List IAM Users",
        description="Lists all IAM users",
        category="identity",
        input_schema={"type": "object", "properties": {}},
        output_schema={"type": "object", "properties": {}},
    )
    assert tool.tool_name == "aws.iam_list_users"
    assert tool.category == "identity"
    assert tool.rate_limit is None
    assert tool.pagination is None


def test_tool_definition_rejects_invalid_category():
    """ToolDefinition rejects an invalid category value."""
    with pytest.raises(ValidationError):
        ToolDefinition(
            tool_name="test.tool",
            display_name="Test",
            description="Test tool",
            category="invalid_category",
            input_schema={},
            output_schema={},
        )


def test_query_plan_rejects_empty_steps():
    """QueryPlan requires at least one step (min_length=1)."""
    with pytest.raises(ValidationError):
        QueryPlan(
            plan_id="plan-001",
            steps=[],
            estimated_tool_calls=0,
            summary="Empty plan",
        )


def test_query_plan_valid_data():
    """QueryPlan accepts valid data with steps."""
    plan = QueryPlan(
        plan_id="plan-001",
        steps=[
            PlanStep(
                step_id="step_1",
                tool_name="aws.iam_list_users",
                integration_id="int-001",
                parameters={},
                description="List users",
                output_alias="users",
            )
        ],
        estimated_tool_calls=1,
        summary="List all users",
    )
    assert plan.plan_id == "plan-001"
    assert plan.plan_version == "1.0"
    assert len(plan.steps) == 1


def test_stream_event_interpretation_started_literal():
    """InterpretationStarted event field is always 'interpretation_started'."""
    event = InterpretationStarted(
        query_id="test-id",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert event.event == "interpretation_started"


def test_stream_event_execution_started_literal():
    """ExecutionStarted event field is always 'execution_started'."""
    event = ExecutionStarted(
        query_id="test-id",
        plan_id="plan-001",
        total_steps=3,
        timestamp="2026-01-01T00:00:00Z",
    )
    assert event.event == "execution_started"
    assert event.total_steps == 3


def test_stream_event_step_completed_literal():
    """StepCompleted status field is always 'success'."""
    event = StepCompleted(
        query_id="test-id",
        step_id="step_1",
        duration_ms=150,
        record_count=10,
        summary="10 records",
        data_hash="abc123",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert event.status == "success"
    assert event.event == "step_completed"
