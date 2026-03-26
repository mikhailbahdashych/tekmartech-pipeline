"""Tests for API endpoints and NDJSON streaming infrastructure.

Tests health, interpret, and execute endpoints using httpx.AsyncClient
with FastAPI's test client. Verifies NDJSON event sequences match
the contracts defined in internal-api.yaml.
"""

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import src.orchestrator.execute_orchestrator as _eo
from src.llm.provider import HealthCheckResult, LLMProvider
from src.main import app
from src.mcp.server_registry import MCPServerConfig
from src.models.tool_invocation import (
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolResponseMetadata,
)

# =============================================================================
# Mock LLM provider for API tests
# =============================================================================

MOCK_PLAN_JSON = json.dumps(
    {
        "plan_id": "test-plan-001",
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


class MockLLMProvider(LLMProvider):
    """Mock provider that returns analysis text and a valid plan."""

    async def health_check(self):
        return HealthCheckResult(status="healthy", details="mock provider")

    async def stream_completion(self, system_prompt, user_message, max_tokens):
        yield "Analyzing your query. "
        yield "I will list all IAM users.\n\n"
        yield f"---PLAN_START---\n{MOCK_PLAN_JSON}\n---PLAN_END---"


SAMPLE_CATALOG = {
    "integrations": [
        {
            "integration_id": "int-aws-001",
            "server_type": "aws",
            "display_name": "Production AWS",
            "tools": [
                {
                    "tool_name": "aws.iam_list_users",
                    "display_name": "List IAM Users",
                    "description": "Lists all IAM users",
                    "category": "identity",
                    "input_schema": {},
                    "output_schema": {},
                }
            ],
        }
    ]
}


class MockMCPClient:
    """Mock MCP client that returns predefined tool responses."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def invoke_tool(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        return ToolInvocationResponse(
            invocation_id=request.invocation_id,
            status="success",
            data={"items": [{"id": 1, "name": "item-1"}, {"id": 2, "name": "item-2"}]},
            metadata=ToolResponseMetadata(
                started_at="2026-01-01T00:00:00Z",
                completed_at="2026-01-01T00:00:01Z",
                duration_ms=100,
                external_api_calls=1,
            ),
        )


class MockServerRegistry:
    """Mock server registry for testing."""

    def get_server_config(self, server_type: str) -> MCPServerConfig:
        return MCPServerConfig(server_type=server_type, command="mock", args=[])

    def get_all_server_types(self) -> list[str]:
        return ["mock"]


_original_mcp_client = _eo.MCPClient


class _PatchedMCPClient(MockMCPClient):
    def __init__(self, *args, **kwargs):
        pass


@pytest_asyncio.fixture
async def client():
    """Create an async HTTP client with mock LLM provider and server registry."""
    app.state.llm_provider = MockLLMProvider()
    app.state.server_registry = MockServerRegistry()
    _eo.MCPClient = _PatchedMCPClient
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    _eo.MCPClient = _original_mcp_client


def _parse_ndjson(content: str) -> list[dict]:
    """Parse NDJSON content into a list of dicts."""
    lines = content.strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


# =============================================================================
# Health endpoint tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_returns_correct_structure(client: AsyncClient):
    """GET /health returns status, version, uptime_seconds, and components."""
    response = await client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert isinstance(data["uptime_seconds"], int)
    assert data["uptime_seconds"] >= 0
    assert "llm_provider" in data["components"]
    assert data["components"]["llm_provider"]["status"] == "healthy"
    assert data["components"]["llm_provider"]["checked_at"] is not None
    assert isinstance(data["components"]["mcp_servers"], list)


# =============================================================================
# Interpret endpoint tests
# =============================================================================


@pytest.mark.asyncio
async def test_interpret_rejects_missing_query_id(client: AsyncClient):
    """POST /interpret returns 400 when query_id is missing."""
    response = await client.post(
        "/interpret",
        json={"query_text": "test", "tool_catalog": SAMPLE_CATALOG},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "validation.missing_field"
    assert "query_id" in error["message"]


@pytest.mark.asyncio
async def test_interpret_rejects_empty_body(client: AsyncClient):
    """POST /interpret returns 400 for empty body."""
    response = await client.post("/interpret", json={})
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_interpret_rejects_empty_catalog(client: AsyncClient):
    """POST /interpret returns 400 for empty tool catalog."""
    response = await client.post(
        "/interpret",
        json={
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
            "query_text": "Show me all IAM users without MFA",
            "tool_catalog": {"integrations": []},
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "interpretation.invalid_catalog"


@pytest.mark.asyncio
async def test_interpret_streams_correct_ndjson_sequence(client: AsyncClient):
    """POST /interpret streams: started → text_deltas → plan_generated."""
    response = await client.post(
        "/interpret",
        json={
            "query_id": "550e8400-e29b-41d4-a716-446655440000",
            "query_text": "Show me all IAM users without MFA",
            "tool_catalog": SAMPLE_CATALOG,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"

    events = _parse_ndjson(response.text)

    # First event must be interpretation_started
    assert events[0]["event"] == "interpretation_started"
    assert events[0]["query_id"] == "550e8400-e29b-41d4-a716-446655440000"

    # Middle events are text_delta
    delta_events = [e for e in events if e["event"] == "interpretation_text_delta"]
    assert len(delta_events) >= 1

    # Last event must be the terminal event
    terminal = events[-1]
    assert terminal["event"] == "interpretation_plan_generated"
    assert "query_plan" in terminal
    assert "plan_summary" in terminal
    assert "full_interpretation_text" in terminal
    assert terminal["query_plan"]["plan_id"] == "test-plan-001"


# =============================================================================
# Execute endpoint tests
# =============================================================================

VALID_EXECUTE_BODY = {
    "query_id": "550e8400-e29b-41d4-a716-446655440000",
    "query_plan": {
        "plan_id": "plan-001",
        "plan_version": "1.0",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "aws.iam_list_users",
                "integration_id": "int-001",
                "parameters": {},
                "description": "List IAM users",
                "output_alias": "users",
            },
            {
                "step_id": "step_2",
                "tool_name": "aws.iam_get_account_summary",
                "integration_id": "int-001",
                "parameters": {},
                "description": "Get account summary",
                "output_alias": "summary",
            },
        ],
        "estimated_tool_calls": 2,
        "summary": "List users and get summary",
    },
    "credentials": {
        "int-001": {
            "server_type": "aws",
            "credential_mode": "direct",
            "credential_data": {
                "access_key_id": "test",
                "secret_access_key": "test",
                "region": "us-east-1",
            },
        }
    },
}


@pytest.mark.asyncio
async def test_execute_rejects_missing_fields(client: AsyncClient):
    """POST /execute returns 400 when required fields are missing."""
    response = await client.post("/execute", json={"query_id": "test"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "validation.missing_field"


@pytest.mark.asyncio
async def test_execute_streams_correct_ndjson_sequence(client: AsyncClient):
    """POST /execute streams: started → step pairs → completed."""
    response = await client.post("/execute", json=VALID_EXECUTE_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"

    events = _parse_ndjson(response.text)

    # First event: execution_started
    assert events[0]["event"] == "execution_started"
    assert events[0]["total_steps"] == 2
    assert events[0]["plan_id"] == "plan-001"

    # Step events come in pairs: step_started + step_completed
    step_events = [e for e in events if e["event"].startswith("step_")]
    assert len(step_events) == 4  # 2 started + 2 completed

    assert step_events[0]["event"] == "step_started"
    assert step_events[0]["step_index"] == 1
    assert step_events[1]["event"] == "step_completed"
    assert step_events[1]["status"] == "success"

    assert step_events[2]["event"] == "step_started"
    assert step_events[2]["step_index"] == 2
    assert step_events[3]["event"] == "step_completed"

    # Last event: execution_completed
    terminal = events[-1]
    assert terminal["event"] == "execution_completed"
    assert terminal["execution_status"] == "completed"
    assert "result_data" in terminal
    assert "transparency_log" in terminal
    assert "result_summary" in terminal
    assert terminal["transparency_log"]["total_tool_invocations"] == 2


# =============================================================================
# NDJSON streaming helper test
# =============================================================================


@pytest.mark.asyncio
async def test_ndjson_streaming_produces_valid_json_lines():
    """The NDJSON streaming helper produces valid newline-delimited JSON."""
    from src.api.streaming import _serialize_events
    from src.models.stream_events import InterpretationStarted

    async def mock_generator():
        yield InterpretationStarted(
            query_id="test-id",
            timestamp="2026-01-01T00:00:00Z",
        )
        yield InterpretationStarted(
            query_id="test-id-2",
            timestamp="2026-01-01T00:00:01Z",
        )

    lines = []
    async for line in _serialize_events(mock_generator()):
        lines.append(line)

    assert len(lines) == 2
    for line in lines:
        assert line.endswith("\n")
        parsed = json.loads(line)
        assert parsed["event"] == "interpretation_started"
