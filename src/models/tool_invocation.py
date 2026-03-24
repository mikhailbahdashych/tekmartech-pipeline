"""Pydantic models for MCP tool invocation request and response schemas.

Conforms to the tool_invocation_request and tool_invocation_response
schemas defined in mcp-tool-interface.yaml section 3.
"""

from typing import Literal

from pydantic import BaseModel

from src.models.credential_envelope import CredentialEnvelope


class ToolInvocationRequest(BaseModel):
    """Payload sent to an MCP server to invoke a specific tool.

    Conforms to the tool_invocation_request schema in mcp-tool-interface.yaml.

    Attributes:
        tool_name: The tool to invoke.
        parameters: Resolved parameters for this invocation.
        credentials: Credential envelope for authentication.
        invocation_id: UUID for this specific invocation.
        timeout_seconds: Maximum time in seconds for the MCP server.
    """

    tool_name: str
    parameters: dict
    credentials: CredentialEnvelope
    invocation_id: str
    timeout_seconds: int = 30


class ToolInvocationErrorDetails(BaseModel):
    """Additional structured error context from a tool invocation.

    Attributes:
        external_status_code: HTTP status code from the external API.
        external_error: Sanitized error message from the external API.
        retryable: Whether the Execution Engine should retry.
    """

    external_status_code: int | None = None
    external_error: str | None = None
    retryable: bool | None = None


class ToolInvocationError(BaseModel):
    """Error information from a failed tool invocation.

    Attributes:
        code: Machine-readable error code (e.g., "auth.invalid_credentials").
        message: Human-readable error description.
        details: Additional structured error context.
    """

    code: str
    message: str
    details: ToolInvocationErrorDetails | None = None


class ToolResponseMetadata(BaseModel):
    """Execution metadata captured for the transparency log.

    Attributes:
        started_at: When the MCP server began processing.
        completed_at: When the MCP server finished processing.
        duration_ms: Wall-clock time in milliseconds.
        external_api_calls: Number of API calls made to the external system.
        data_hash: SHA-256 hash of the returned data.
    """

    started_at: str
    completed_at: str
    duration_ms: int
    external_api_calls: int
    data_hash: str | None = None


class ToolInvocationResponse(BaseModel):
    """Payload returned by an MCP server after executing a tool invocation.

    Conforms to the tool_invocation_response schema in mcp-tool-interface.yaml.

    Attributes:
        invocation_id: Echoed from the request for correlation.
        status: Outcome of the invocation.
        data: Tool output data (present on success/partial).
        error: Error information (present on error/partial).
        metadata: Execution metadata for the transparency log.
    """

    invocation_id: str
    status: Literal["success", "error", "partial"]
    data: dict | None = None
    error: ToolInvocationError | None = None
    metadata: ToolResponseMetadata
