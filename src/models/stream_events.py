"""Pydantic models for all NDJSON stream event types.

Defines event models for both /interpret and /execute endpoints.
Each event has a literal `event` field matching the contract value
in internal-api.yaml. Events are serialized to JSON and streamed
as newline-delimited JSON (NDJSON).
"""

from typing import Literal

from pydantic import BaseModel

from src.models.conversation import ClarificationOption
from src.models.query_plan import QueryPlan
from src.models.result_data import ResultData
from src.models.transparency_log import TransparencyLog

# =============================================================================
# /interpret stream events
# =============================================================================


class InterpretationStarted(BaseModel):
    """First event in the interpretation stream.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query being interpreted.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_started"] = "interpretation_started"
    query_id: str
    timestamp: str


class InterpretationTextDelta(BaseModel):
    """Streams a chunk of the AI's analysis text.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query being interpreted.
        text_delta: A chunk of the AI's response text.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_text_delta"] = "interpretation_text_delta"
    query_id: str
    text_delta: str
    timestamp: str


class InterpretationPlanGenerated(BaseModel):
    """Terminal success event with the complete query plan.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        query_plan: The generated execution plan.
        plan_summary: Human-readable explanation of the plan.
        estimated_duration_seconds: Estimated execution time.
        full_interpretation_text: Complete AI analysis text.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_plan_generated"] = "interpretation_plan_generated"
    query_id: str
    query_plan: QueryPlan
    plan_summary: str
    estimated_duration_seconds: int
    full_interpretation_text: str
    timestamp: str


class InterpretationFailed(BaseModel):
    """Terminal event when interpretation fails with an explainable reason.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        error_code: Machine-readable error code.
        error_message: Human-readable failure explanation.
        full_interpretation_text: Text generated before failure.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_failed"] = "interpretation_failed"
    query_id: str
    error_code: str
    error_message: str
    full_interpretation_text: str | None = None
    timestamp: str


class InterpretationClarificationNeeded(BaseModel):
    """Terminal event when the Interpreter needs clarification from the user.

    Ends the current streaming response. The Application API stores this
    as a conversation turn and waits for the user's response before
    calling POST /interpret again with conversation_history.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        clarification_id: UUID for this clarification request.
        question: The AI's clarification question in natural language.
        options: Structured choices for the user, if applicable.
        allows_free_text: Whether the user can type a free-text response.
        full_interpretation_text: Complete AI analysis text from this turn.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_clarification_needed"] = "interpretation_clarification_needed"
    query_id: str
    clarification_id: str
    question: str
    options: list[ClarificationOption] | None = None
    allows_free_text: bool = True
    full_interpretation_text: str
    timestamp: str


class InterpretationError(BaseModel):
    """Terminal event for unexpected infrastructure errors during interpretation.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        error_code: Machine-readable error code.
        error_message: Human-readable error description.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["interpretation_error"] = "interpretation_error"
    query_id: str
    error_code: str
    error_message: str
    timestamp: str


# =============================================================================
# /execute stream events
# =============================================================================


class ExecutionStarted(BaseModel):
    """First event in the execution stream.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        plan_id: UUID of the plan being executed.
        total_steps: Number of steps in the plan.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["execution_started"] = "execution_started"
    query_id: str
    plan_id: str
    total_steps: int
    timestamp: str


class StepStarted(BaseModel):
    """Emitted when execution of a plan step begins.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        step_id: Identifier of the step within the plan.
        step_index: 1-based position in the plan.
        tool_name: The MCP tool being invoked.
        tool_display_name: Human-readable tool name.
        integration_display_name: Which integration is being queried.
        description: The step description from the plan.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["step_started"] = "step_started"
    query_id: str
    step_id: str
    step_index: int
    tool_name: str
    tool_display_name: str
    integration_display_name: str
    description: str
    timestamp: str


class StepCompleted(BaseModel):
    """Emitted when a plan step finishes successfully.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        step_id: Identifier of the step within the plan.
        status: Always "success" for completed steps.
        duration_ms: Time taken in milliseconds.
        record_count: Number of records returned.
        summary: Brief summary of results.
        data_hash: SHA-256 hash of returned data.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["step_completed"] = "step_completed"
    query_id: str
    step_id: str
    status: Literal["success"] = "success"
    duration_ms: int
    record_count: int
    summary: str
    data_hash: str
    timestamp: str


class StepFailed(BaseModel):
    """Emitted when a plan step fails.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        step_id: Identifier of the step within the plan.
        status: Always "error" for failed steps.
        duration_ms: Time taken in milliseconds.
        error_code: Tool error code from the MCP response.
        error_message: Human-readable error description.
        retryable: Whether the step could be retried.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["step_failed"] = "step_failed"
    query_id: str
    step_id: str
    status: Literal["error"] = "error"
    duration_ms: int
    error_code: str
    error_message: str
    retryable: bool
    timestamp: str


class ExecutionCompleted(BaseModel):
    """Terminal success event with complete results.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        execution_status: "completed" or "partial".
        result_data: Structured query results.
        result_summary: Human-readable summary.
        transparency_log: Complete execution log.
        export_csv: CSV-formatted results (optional).
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["execution_completed"] = "execution_completed"
    query_id: str
    execution_status: Literal["completed", "partial"]
    result_data: ResultData
    result_summary: str
    transparency_log: TransparencyLog
    export_csv: str | None = None
    timestamp: str


class ExecutionFailed(BaseModel):
    """Terminal event when execution fails entirely.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        execution_status: Always "failed".
        error_code: Machine-readable error code.
        error_message: Human-readable error description.
        transparency_log: Partial log showing what was attempted.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["execution_failed"] = "execution_failed"
    query_id: str
    execution_status: Literal["failed"] = "failed"
    error_code: str
    error_message: str
    transparency_log: TransparencyLog | None = None
    timestamp: str


class ExecutionError(BaseModel):
    """Terminal event for unexpected errors during execution.

    Attributes:
        event: Literal event type identifier.
        query_id: UUID of the query.
        error_code: Machine-readable error code.
        error_message: Human-readable error description.
        transparency_log: Whatever log was collected.
        timestamp: ISO 8601 timestamp.
    """

    event: Literal["execution_error"] = "execution_error"
    query_id: str
    error_code: str
    error_message: str
    transparency_log: TransparencyLog | None = None
    timestamp: str
