"""Tests for the execution engine."""

import pytest

from src.config import get_settings
from src.execution.engine import ExecutionContext, execute_plan
from src.models.credential_envelope import CredentialEnvelope
from src.models.query_plan import PlanStep, QueryPlan
from src.models.tool_invocation import (
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolResponseMetadata,
)


def _make_plan(*steps: PlanStep) -> QueryPlan:
    return QueryPlan(
        plan_id="test-plan",
        steps=list(steps),
        estimated_tool_calls=len(steps),
        summary="Test plan",
    )


def _make_step(step_id: str = "step_1", depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        tool_name="test.list_items",
        integration_id="int-001",
        parameters={},
        description=f"Test step {step_id}",
        output_alias=f"output_{step_id}",
        depends_on=depends_on,
    )


def _make_credentials() -> dict[str, CredentialEnvelope]:
    return {
        "int-001": CredentialEnvelope(
            server_type="test",
            credential_mode="direct",
            credential_data={"token": "test-token"},
        )
    }


class MockMCPClient:
    """Mock MCP client that returns a predefined response."""

    def __init__(self, data: dict | None = None, status: str = "success"):
        self._data = data or {"items": [{"id": 1}, {"id": 2}]}
        self._status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def invoke_tool(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        return ToolInvocationResponse(
            invocation_id=request.invocation_id,
            status=self._status,
            data=self._data if self._status != "error" else None,
            metadata=ToolResponseMetadata(
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                duration_ms=100,
                external_api_calls=1,
            ),
        )


class FailingMCPClient(MockMCPClient):
    """Mock MCP client that raises an exception."""

    async def invoke_tool(self, request):
        raise RuntimeError("Connection refused")


async def _collect_events(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_engine_two_step_happy_path():
    """Two-step plan executes both steps successfully."""
    plan = _make_plan(_make_step("step_1"), _make_step("step_2"))
    context = ExecutionContext()

    events = await _collect_events(
        execute_plan(
            query_id="q1",
            plan=plan,
            credentials=_make_credentials(),
            mcp_client_factory=lambda st: MockMCPClient(),
            settings=get_settings(),
            context=context,
        )
    )

    # 2 step_started + 2 step_completed = 4 events
    assert len(events) == 4
    assert events[0].event == "step_started"
    assert events[1].event == "step_completed"
    assert events[2].event == "step_started"
    assert events[3].event == "step_completed"
    assert context.steps_succeeded == 2
    assert context.steps_failed == 0


@pytest.mark.asyncio
async def test_engine_step_failure_continues():
    """First step fails, second still executes."""
    plan = _make_plan(_make_step("step_1"), _make_step("step_2"))
    context = ExecutionContext()

    call_count = 0

    def alternating_factory(st):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return FailingMCPClient()
        return MockMCPClient()

    events = await _collect_events(
        execute_plan(
            query_id="q1",
            plan=plan,
            credentials=_make_credentials(),
            mcp_client_factory=alternating_factory,
            settings=get_settings(),
            context=context,
        )
    )

    event_types = [e.event for e in events]
    assert "step_failed" in event_types
    assert "step_completed" in event_types
    assert context.steps_succeeded == 1
    assert context.steps_failed == 1


@pytest.mark.asyncio
async def test_engine_all_steps_fail():
    """All steps fail — context reflects total failure."""
    plan = _make_plan(_make_step("step_1"), _make_step("step_2"))
    context = ExecutionContext()

    await _collect_events(
        execute_plan(
            query_id="q1",
            plan=plan,
            credentials=_make_credentials(),
            mcp_client_factory=lambda st: FailingMCPClient(),
            settings=get_settings(),
            context=context,
        )
    )

    assert context.steps_succeeded == 0
    assert context.steps_failed == 2


@pytest.mark.asyncio
async def test_engine_depends_on_skips_on_failure():
    """Step with depends_on is skipped when dependency failed."""
    plan = _make_plan(
        _make_step("step_1"),
        _make_step("step_2", depends_on=["step_1"]),
    )
    context = ExecutionContext()

    events = await _collect_events(
        execute_plan(
            query_id="q1",
            plan=plan,
            credentials=_make_credentials(),
            mcp_client_factory=lambda st: FailingMCPClient(),
            settings=get_settings(),
            context=context,
        )
    )

    # step_1 fails, step_2 skipped due to dependency
    assert context.steps_failed == 2
    failed_events = [e for e in events if e.event == "step_failed"]
    assert len(failed_events) == 2
    assert "Dependencies not met" in failed_events[1].error_message


@pytest.mark.asyncio
async def test_engine_log_entries_collected():
    """Transparency log entries are collected for all steps."""
    plan = _make_plan(_make_step("step_1"))
    context = ExecutionContext()

    await _collect_events(
        execute_plan(
            query_id="q1",
            plan=plan,
            credentials=_make_credentials(),
            mcp_client_factory=lambda st: MockMCPClient(),
            settings=get_settings(),
            context=context,
        )
    )

    assert len(context.log_entries) >= 1
    assert context.log_entries[0].tool_name == "test.list_items"
