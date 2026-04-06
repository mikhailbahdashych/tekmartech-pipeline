"""Tests for src.orchestrator.interpret_orchestrator."""

import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from src.llm.provider import HealthCheckResult, LLMProvider
from src.models.conversation import (
    AssistantTurn,
    ClarificationOption,
    ClarificationQuestion,
    UserTurn,
)
from src.models.stream_events import (
    InterpretationClarificationNeeded,
    InterpretationFailed,
    InterpretationPlanGenerated,
    InterpretationStarted,
    InterpretationTextDelta,
)
from src.models.tool_catalog import CatalogIntegration, ToolCatalog, ToolDefinition
from src.orchestrator.interpret_orchestrator import interpret_stream

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
                ],
            ),
        ]
    )


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.LLM_MAX_TOKENS = 4096
    return settings


def _plan_json() -> str:
    return json.dumps({
        "steps": [
            {
                "tool_name": TOOL_NAME,
                "integration_id": INTEGRATION_ID,
                "parameters": {},
                "description": "List all repositories",
                "output_alias": "org_repos",
            }
        ],
        "summary": "List GitHub repositories.",
    })


def _clarification_json() -> str:
    return json.dumps({
        "question": "Which AWS account?",
        "options": [
            {"option_id": "opt_1", "label": "Production"},
            {"option_id": "opt_2", "label": "Staging"},
        ],
        "allows_free_text": True,
    })


class MockProvider(LLMProvider):
    """LLM provider that yields pre-defined chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(status="healthy")

    async def stream_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class MockProviderCapture(LLMProvider):
    """LLM provider that captures the messages it receives and yields chunks."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.captured_messages: list[dict[str, str]] | None = None

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(status="healthy")

    async def stream_completion(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        self.captured_messages = messages
        for chunk in self._chunks:
            yield chunk


async def _collect_events(stream):
    """Collect all events from an async generator."""
    events = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
class TestPlanGeneration:
    """Orchestrator produces plan_generated for valid plan output."""

    async def test_plan_generated_event_sequence(self):
        llm_output = f"### Analysis\nAnalysis text.\n\n### Structured Output\n---PLAN_START---\n{_plan_json()}\n---PLAN_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "List repos", _make_catalog(), provider, _make_settings())
        )

        assert isinstance(events[0], InterpretationStarted)
        assert isinstance(events[-1], InterpretationPlanGenerated)
        assert events[-1].query_plan is not None
        assert events[-1].query_plan.steps[0].tool_name == TOOL_NAME

    async def test_text_deltas_contain_analysis(self):
        llm_output = f"Analysis text here.\n\n### Structured Output\n---PLAN_START---\n{_plan_json()}\n---PLAN_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        deltas = [e for e in events if isinstance(e, InterpretationTextDelta)]
        full_text = "".join(d.text_delta for d in deltas)
        assert "Analysis text here." in full_text

    async def test_plan_json_not_in_deltas(self):
        llm_output = f"Analysis.\n\n### Structured Output\n---PLAN_START---\n{_plan_json()}\n---PLAN_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        deltas = [e for e in events if isinstance(e, InterpretationTextDelta)]
        full_text = "".join(d.text_delta for d in deltas)
        assert "PLAN_START" not in full_text
        assert "tool_name" not in full_text


@pytest.mark.asyncio
class TestClarificationGeneration:
    """Orchestrator produces clarification_needed for clarification output."""

    async def test_clarification_event_emitted(self):
        llm_output = f"I need more info.\n\n---CLARIFICATION_START---\n{_clarification_json()}\n---CLARIFICATION_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "Check security", _make_catalog(), provider, _make_settings())
        )

        assert isinstance(events[0], InterpretationStarted)
        assert isinstance(events[-1], InterpretationClarificationNeeded)
        assert events[-1].question == "Which AWS account?"
        assert len(events[-1].options) == 2

    async def test_clarification_has_uuid_id(self):
        import uuid

        llm_output = f"Need info.\n\n---CLARIFICATION_START---\n{_clarification_json()}\n---CLARIFICATION_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        clar_event = events[-1]
        uuid.UUID(clar_event.clarification_id)

    async def test_clarification_json_not_in_deltas(self):
        llm_output = f"Analysis text.\n\n---CLARIFICATION_START---\n{_clarification_json()}\n---CLARIFICATION_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        deltas = [e for e in events if isinstance(e, InterpretationTextDelta)]
        full_text = "".join(d.text_delta for d in deltas)
        assert "CLARIFICATION_START" not in full_text

    async def test_clarification_full_text_in_event(self):
        llm_output = f"My analysis here.\n\n---CLARIFICATION_START---\n{_clarification_json()}\n---CLARIFICATION_END---"
        provider = MockProvider([llm_output])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        clar_event = events[-1]
        assert "My analysis here." in clar_event.full_interpretation_text


@pytest.mark.asyncio
class TestMultiTurnConversation:
    """Orchestrator passes conversation history to prompt builder and LLM."""

    async def test_continuation_messages_include_history(self):
        """Verify the LLM receives messages that include conversation history."""
        llm_output = f"Continuing.\n\n### Structured Output\n---PLAN_START---\n{_plan_json()}\n---PLAN_END---"
        provider = MockProviderCapture([llm_output])

        history = [
            AssistantTurn(
                role="assistant",
                content="I need to know which account.",
                clarification=ClarificationQuestion(
                    clarification_id="clar-1",
                    question="Which AWS?",
                    options=[ClarificationOption(option_id="opt_1", label="Prod")],
                ),
            ),
            UserTurn(
                role="user",
                response_type="option_selection",
                selected_option_id="opt_1",
            ),
        ]

        events = await _collect_events(
            interpret_stream(
                "q1", "Check security", _make_catalog(), provider, _make_settings(),
                conversation_history=history,
            )
        )

        # Should have plan_generated as terminal event
        assert isinstance(events[-1], InterpretationPlanGenerated)

        # Verify messages sent to provider include history
        assert provider.captured_messages is not None
        assert len(provider.captured_messages) >= 3  # original + assistant + user + continuation
        roles = [m["role"] for m in provider.captured_messages]
        assert roles[0] == "user"  # original query
        assert "assistant" in roles

    async def test_no_history_single_message(self):
        llm_output = f"Analysis.\n\n### Structured Output\n---PLAN_START---\n{_plan_json()}\n---PLAN_END---"
        provider = MockProviderCapture([llm_output])

        await _collect_events(
            interpret_stream("q1", "List repos", _make_catalog(), provider, _make_settings())
        )

        assert provider.captured_messages is not None
        assert len(provider.captured_messages) == 1
        assert provider.captured_messages[0]["role"] == "user"


@pytest.mark.asyncio
class TestFailurePaths:
    """Orchestrator handles failure cases correctly."""

    async def test_no_structured_output_emits_failed(self):
        provider = MockProvider(["Just some analysis with no plan or clarification."])

        events = await _collect_events(
            interpret_stream("q1", "query", _make_catalog(), provider, _make_settings())
        )

        assert isinstance(events[-1], InterpretationFailed)
