"""MCP client for tool invocation via stdio transport.

Wraps the MCP Python SDK to connect to MCP servers as child processes,
send tool invocation requests, and receive responses. Each MCPClient
instance manages a single server process connection.
"""

import json
from datetime import UTC, datetime

import structlog
from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters
from src.mcp.server_registry import MCPServerConfig
from src.models.tool_invocation import (
    ToolInvocationError,
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolResponseMetadata,
)

logger = structlog.get_logger(__name__)


class MCPClient:
    """Connects to an MCP server via stdio and invokes tools.

    Used as an async context manager. Spawns the MCP server process
    on entry and cleans it up on exit.

    Args:
        config: The server launch configuration.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """Initialize the MCP client.

        Args:
            config: Server configuration with command, args, and cwd.
        """
        self._config = config
        self._session: ClientSession | None = None
        self._stdio_cm = None
        self._session_cm = None

    async def __aenter__(self) -> "MCPClient":
        """Spawn the MCP server process and initialize the session.

        Returns:
            The initialized MCPClient ready for tool invocations.
        """
        logger.debug(
            "connecting to MCP server",
            action="mcp_connect",
            server_type=self._config.server_type,
            command=self._config.command,
        )

        params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args,
            cwd=self._config.cwd,
        )

        self._stdio_cm = stdio_client(params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read_stream, write_stream)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        logger.info(
            "MCP server connected",
            action="mcp_connect",
            server_type=self._config.server_type,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the session and terminate the server process."""
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)

        logger.debug(
            "MCP server disconnected",
            action="mcp_disconnect",
            server_type=self._config.server_type,
        )

    async def invoke_tool(self, request: ToolInvocationRequest) -> ToolInvocationResponse:
        """Invoke a tool on the connected MCP server.

        Sends the full ToolInvocationRequest as the arguments dict
        to the MCP call_tool method. The server returns a TextContent
        containing a JSON-serialized ToolInvocationResponse.

        Args:
            request: The tool invocation request with parameters and credentials.

        Returns:
            The parsed ToolInvocationResponse from the MCP server.

        Raises:
            RuntimeError: If the client session is not initialized.
        """
        if self._session is None:
            raise RuntimeError("MCPClient session not initialized. Use as async context manager.")

        logger.debug(
            "invoking MCP tool",
            action="mcp_invoke",
            tool_name=request.tool_name,
            invocation_id=request.invocation_id,
            server_type=self._config.server_type,
        )

        started_at = datetime.now(UTC)

        try:
            result = await self._session.call_tool(request.tool_name, request.model_dump())

            # Extract the TextContent response
            if result.content and len(result.content) > 0:
                text_content = result.content[0]
                response_data = json.loads(text_content.text)
                response = ToolInvocationResponse(**response_data)
            else:
                response = _build_error_response(
                    request.invocation_id,
                    "internal.server_error",
                    "MCP server returned empty response",
                    started_at,
                )

        except json.JSONDecodeError as exc:
            logger.error(
                "failed to parse MCP server response",
                action="mcp_invoke",
                tool_name=request.tool_name,
                error=str(exc),
            )
            response = _build_error_response(
                request.invocation_id,
                "internal.server_error",
                f"Invalid response from MCP server: {exc}",
                started_at,
            )

        except Exception as exc:
            logger.error(
                "MCP tool invocation failed",
                action="mcp_invoke",
                tool_name=request.tool_name,
                error=str(exc),
            )
            response = _build_error_response(
                request.invocation_id,
                "internal.server_error",
                f"MCP invocation error: {exc}",
                started_at,
            )

        logger.info(
            "MCP tool invocation completed",
            action="mcp_invoke",
            tool_name=request.tool_name,
            invocation_id=request.invocation_id,
            status=response.status,
        )
        return response


def _build_error_response(
    invocation_id: str,
    error_code: str,
    error_message: str,
    started_at: datetime,
) -> ToolInvocationResponse:
    """Build an error ToolInvocationResponse.

    Args:
        invocation_id: The invocation ID to echo.
        error_code: Machine-readable error code.
        error_message: Human-readable error description.
        started_at: When the invocation began.

    Returns:
        A ToolInvocationResponse with error status.
    """
    completed_at = datetime.now(UTC)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    return ToolInvocationResponse(
        invocation_id=invocation_id,
        status="error",
        error=ToolInvocationError(code=error_code, message=error_message),
        metadata=ToolResponseMetadata(
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=duration_ms,
            external_api_calls=0,
        ),
    )
