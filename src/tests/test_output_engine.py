"""Tests for output engine modules."""

from src.models.query_plan import PlanStep, QueryPlan
from src.models.result_data import (
    IntegrationQueried,
    ResultColumn,
    ResultData,
    ResultMetadata,
    ResultTable,
)
from src.models.transparency_log import TransparencyLogEntry
from src.output.csv_generator import generate_csv
from src.output.formatter import format_results
from src.output.summary_generator import generate_summary
from src.output.transparency_log_builder import build_transparency_log


def _make_plan() -> QueryPlan:
    return QueryPlan(
        plan_id="plan-001",
        steps=[
            PlanStep(
                step_id="step_1",
                tool_name="github.list_repositories",
                integration_id="int-001",
                parameters={},
                description="List repositories",
                output_alias="repos",
            )
        ],
        estimated_tool_calls=1,
        summary="List repos",
    )


# =============================================================================
# Formatter tests
# =============================================================================


def test_format_results_single_table():
    """Single step result produces one table with correct columns."""
    step_results = {
        "step_1": {
            "repositories": [
                {"name": "repo-a", "visibility": "public", "stars": 10},
                {"name": "repo-b", "visibility": "private", "stars": 5},
            ]
        }
    }
    result = format_results(step_results, _make_plan(), execution_duration_ms=500)
    assert len(result.tables) == 1
    assert result.tables[0].row_count == 2
    assert result.metadata.total_records == 2
    column_keys = [c.key for c in result.tables[0].columns]
    assert "name" in column_keys
    assert "visibility" in column_keys


def test_format_results_empty():
    """No step results produces empty tables."""
    result = format_results({}, _make_plan(), execution_duration_ms=0)
    assert result.tables == []
    assert result.metadata.total_records == 0


def test_format_results_column_type_inference():
    """Column types are inferred from Python value types."""
    step_results = {
        "step_1": {
            "items": [
                {"name": "test", "count": 42, "active": True, "tags": ["a", "b"]},
            ]
        }
    }
    result = format_results(step_results, _make_plan(), execution_duration_ms=100)
    col_types = {c.key: c.data_type for c in result.tables[0].columns}
    assert col_types["name"] == "string"
    assert col_types["count"] == "number"
    assert col_types["active"] == "boolean"
    assert col_types["tags"] == "array"


# =============================================================================
# Summary generator tests
# =============================================================================


def test_generate_summary_completed():
    """Summary includes record count, steps, and integrations."""
    result_data = ResultData(
        tables=[
            ResultTable(
                table_id="t1",
                title="Test",
                row_count=10,
                columns=[ResultColumn(key="id", label="ID", data_type="number")],
                rows=[[i] for i in range(10)],
                source_step_ids=["step_1"],
            )
        ],
        metadata=ResultMetadata(
            total_records=10,
            integrations_queried=[
                IntegrationQueried(
                    integration_id="int-1", display_name="GitHub", server_type="github"
                )
            ],
            execution_duration_ms=1500,
        ),
    )
    summary = generate_summary(result_data, steps_succeeded=2, steps_failed=0)
    assert "10 records" in summary
    assert "2 of 2" in summary
    assert "GitHub" in summary


# =============================================================================
# CSV generator tests
# =============================================================================


def test_generate_csv_single_table():
    """CSV includes table title, headers, and data."""
    result_data = ResultData(
        tables=[
            ResultTable(
                table_id="t1",
                title="Users",
                columns=[
                    ResultColumn(key="name", label="Name", data_type="string"),
                    ResultColumn(key="age", label="Age", data_type="number"),
                ],
                rows=[["alice", 30], ["bob", 25]],
                row_count=2,
                source_step_ids=["step_1"],
            )
        ],
        metadata=ResultMetadata(
            total_records=2, integrations_queried=[], execution_duration_ms=100
        ),
    )
    csv_str = generate_csv(result_data)
    assert csv_str is not None
    assert "Users" in csv_str
    assert "Name" in csv_str
    assert "alice" in csv_str


def test_generate_csv_empty():
    """No tables returns None."""
    result_data = ResultData(
        tables=[],
        metadata=ResultMetadata(total_records=0, integrations_queried=[], execution_duration_ms=0),
    )
    assert generate_csv(result_data) is None


# =============================================================================
# Transparency log builder tests
# =============================================================================


def test_build_transparency_log_aggregates():
    """Log builder computes correct totals from entries."""
    entries = [
        TransparencyLogEntry(
            entry_id="e1",
            step_id="step_1",
            invocation_id="inv-1",
            tool_name="test.tool",
            integration_id="int-1",
            parameters={},
            credential_mode="direct",
            status="success",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
            duration_ms=1000,
            external_api_calls=3,
        ),
        TransparencyLogEntry(
            entry_id="e2",
            step_id="step_2",
            invocation_id="inv-2",
            tool_name="test.tool",
            integration_id="int-1",
            parameters={},
            credential_mode="direct",
            status="error",
            error_code="external.api_error",
            error_message="Not found",
            started_at="2026-01-01T00:00:01Z",
            completed_at="2026-01-01T00:00:02Z",
            duration_ms=500,
            external_api_calls=1,
        ),
    ]
    log = build_transparency_log(
        query_id="q1",
        plan_id="p1",
        execution_started_at="2026-01-01T00:00:00Z",
        execution_completed_at="2026-01-01T00:00:02Z",
        entries=entries,
        execution_status="partial",
    )
    assert log.total_tool_invocations == 2
    assert log.total_external_api_calls == 4
    assert log.execution_status == "partial"
    assert log.total_duration_ms == 2000
