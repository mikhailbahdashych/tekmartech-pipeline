"""Tests for the interpretation orchestrator."""

import json

import pytest

from src.config import get_settings
from src.llm.exceptions import LLMTimeoutError, LLMUnavailableError
from src.llm.provider import LLMProvider
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition
from src.orchestrator.interpret_orchestrator import interpret_stream

MOCK_PLAN_JSON = json.dumps(
    {
        "plan_id": "mock-plan-001",
        "plan_version": "1.0",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "aws.iam_list_users",
                "integration_id": "int-aws-001",
                "parameters": {},
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
                ],
            )
        ]
    )


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that yields predefined text chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def stream_completion(self, system_prompt, user_message, max_tokens):
        for chunk in self._chunks:
            yield chunk


class ErrorLLMProvider(LLMProvider):
    """Mock LLM provider that raises an exception."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def stream_completion(self, system_prompt, user_message, max_tokens):
        raise self._error
        yield  # Make it a generator


async def _collect_events(stream):
    """Collect all events from an async generator."""
    events = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_orchestrator_happy_path():
    """Successful interpretation: started → deltas → plan_generated."""
    chunks = [
        "Here is my analysis of your query. ",
        "I will list all IAM users.\n\n",
        f"---PLAN_START---\n{MOCK_PLAN_JSON}\n---PLAN_END---",
    ]
    provider = MockLLMProvider(chunks)
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-001",
            query_text="List all IAM users",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    assert events[0].event == "interpretation_started"

    deltas = [e for e in events if e.event == "interpretation_text_delta"]
    assert len(deltas) == 3

    terminal = events[-1]
    assert terminal.event == "interpretation_plan_generated"
    assert terminal.query_plan.plan_id == "mock-plan-001"
    assert terminal.plan_summary == "List all IAM users from the AWS account."
    assert "analysis" in terminal.full_interpretation_text.lower()


@pytest.mark.asyncio
async def test_orchestrator_no_plan_emits_failed():
    """LLM produces no plan block → interpretation_failed."""
    provider = MockLLMProvider(["I cannot answer this question with the available tools."])
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-002",
            query_text="Query about unsupported system",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    terminal = events[-1]
    assert terminal.event == "interpretation_failed"
    assert terminal.error_code == "interpretation.unsatisfiable_request"


@pytest.mark.asyncio
async def test_orchestrator_invalid_catalog_reference_emits_failed():
    """Plan references tool not in catalog → interpretation_failed."""
    bad_plan = json.dumps(
        {
            "plan_id": "bad-plan",
            "plan_version": "1.0",
            "steps": [
                {
                    "step_id": "step_1",
                    "tool_name": "aws.nonexistent_tool",
                    "integration_id": "int-aws-001",
                    "parameters": {},
                    "description": "Invalid step",
                    "output_alias": "bad_output",
                }
            ],
            "estimated_tool_calls": 1,
            "summary": "Bad plan",
        }
    )
    chunks = [f"Analysis.\n\n---PLAN_START---\n{bad_plan}\n---PLAN_END---"]
    provider = MockLLMProvider(chunks)
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-003",
            query_text="Test",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    terminal = events[-1]
    assert terminal.event == "interpretation_failed"
    msg = terminal.error_message.lower()
    assert "not in the catalog" in msg or "nonexistent" in msg


@pytest.mark.asyncio
async def test_orchestrator_llm_timeout_emits_error():
    """LLM timeout → interpretation_error with model_timeout code."""
    provider = ErrorLLMProvider(LLMTimeoutError("Request timed out"))
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-004",
            query_text="Test",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    assert events[0].event == "interpretation_started"
    terminal = events[-1]
    assert terminal.event == "interpretation_error"
    assert terminal.error_code == "interpretation.model_timeout"


@pytest.mark.asyncio
async def test_orchestrator_llm_unavailable_emits_error():
    """LLM unavailable → interpretation_error with model_unavailable code."""
    provider = ErrorLLMProvider(LLMUnavailableError("Cannot connect"))
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-005",
            query_text="Test",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    terminal = events[-1]
    assert terminal.event == "interpretation_error"
    assert terminal.error_code == "interpretation.model_unavailable"


@pytest.mark.asyncio
async def test_orchestrator_unexpected_error_emits_error():
    """Unexpected exception → interpretation_error with internal_error code."""
    provider = ErrorLLMProvider(RuntimeError("Something broke"))
    settings = get_settings()

    events = await _collect_events(
        interpret_stream(
            query_id="test-query-006",
            query_text="Test",
            tool_catalog=_make_catalog(),
            provider=provider,
            settings=settings,
        )
    )

    terminal = events[-1]
    assert terminal.event == "interpretation_error"
    assert terminal.error_code == "interpretation.internal_error"
