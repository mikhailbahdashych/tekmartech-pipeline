"""Pydantic models for transparency log schemas.

Conforms to the transparency_log and transparency_log_entry schemas
defined in mcp-tool-interface.yaml section 4.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TransparencyLogEntry(BaseModel):
    """A single record of an MCP tool invocation within a query execution.

    Conforms to the transparency_log_entry schema in mcp-tool-interface.yaml.

    Attributes:
        entry_id: UUID for this log entry.
        step_id: The plan step that triggered this invocation.
        invocation_id: UUID correlating with the actual tool call.
        tool_name: The MCP tool that was invoked.
        integration_id: The integration the tool was executed against.
        parameters: Resolved parameters (sensitive values redacted).
        credential_mode: Whether broker or direct transit was used.
        status: Outcome of this invocation.
        error_code: Error code if status is error or partial.
        error_message: Error message if status is error or partial.
        started_at: ISO 8601 timestamp when invocation began.
        completed_at: ISO 8601 timestamp when invocation ended.
        duration_ms: Wall-clock time in milliseconds.
        external_api_calls: Number of external API calls made.
        data_hash: SHA-256 hash of returned data.
        is_retry: Whether this was a retry of a previous attempt.
        retry_of: Entry ID of the original failed invocation if retrying.
    """

    entry_id: str
    step_id: str
    invocation_id: str
    tool_name: str
    integration_id: str
    parameters: dict = Field(default_factory=dict)
    credential_mode: Literal["broker", "direct"]
    status: Literal["success", "error", "partial"]
    error_code: str | None = None
    error_message: str | None = None
    started_at: str
    completed_at: str
    duration_ms: int
    external_api_calls: int
    data_hash: str | None = None
    is_retry: bool = False
    retry_of: str | None = None


class TransparencyLog(BaseModel):
    """Complete execution record for a query plan.

    Conforms to the transparency_log schema in mcp-tool-interface.yaml.

    Attributes:
        query_id: UUID of the query this log belongs to.
        plan_id: UUID of the plan that was executed.
        execution_started_at: ISO 8601 timestamp when execution began.
        execution_completed_at: ISO 8601 timestamp when execution ended.
        total_duration_ms: Total wall-clock time in milliseconds.
        total_tool_invocations: Total MCP tool invocations made.
        total_external_api_calls: Total external API calls across all invocations.
        execution_status: Overall outcome of the execution.
        entries: One entry per MCP tool invocation, in chronological order.
    """

    query_id: str
    plan_id: str
    execution_started_at: str
    execution_completed_at: str
    total_duration_ms: int
    total_tool_invocations: int
    total_external_api_calls: int
    execution_status: Literal["completed", "partial", "failed"]
    entries: list[TransparencyLogEntry] = Field(default_factory=list)
