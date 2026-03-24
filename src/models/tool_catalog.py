"""Pydantic models for MCP tool catalog schemas.

Conforms to the tool_catalog, catalog_integration, and tool_definition
schemas defined in mcp-tool-interface.yaml section 1.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RateLimit(BaseModel):
    """Advisory rate limit information for a tool.

    Attributes:
        max_calls_per_second: Maximum invocations per second. May be fractional.
        max_calls_per_minute: Maximum invocations per minute.
    """

    max_calls_per_second: float | None = None
    max_calls_per_minute: int | None = None


class PaginationConfig(BaseModel):
    """Pagination support declaration for a tool.

    Attributes:
        supported: Whether this tool returns paginated results.
        cursor_parameter: Input parameter name that accepts a pagination cursor.
        cursor_response_field: Dot-path to the next-page cursor in output.
    """

    supported: bool
    cursor_parameter: str | None = None
    cursor_response_field: str | None = None


class ToolDefinition(BaseModel):
    """A single read-only operation offered by an MCP server.

    Conforms to the tool_definition schema in mcp-tool-interface.yaml.

    Attributes:
        tool_name: Unique identifier, convention: {server_type}.{operation}.
        display_name: Human-readable name for plans and logs.
        description: Detailed description of what the tool does.
        category: Functional category for tool selection.
        input_schema: JSON Schema describing accepted parameters.
        output_schema: JSON Schema describing returned data structure.
        rate_limit: Advisory rate limit information.
        pagination: Pagination support declaration.
    """

    tool_name: str
    display_name: str
    description: str
    category: Literal[
        "identity",
        "access_management",
        "audit_log",
        "configuration",
        "network",
        "storage",
        "compute",
    ]
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    rate_limit: RateLimit | None = None
    pagination: PaginationConfig | None = None


class CatalogIntegration(BaseModel):
    """A single integration's contribution to the tool catalog.

    Conforms to the catalog_integration schema in mcp-tool-interface.yaml.

    Attributes:
        integration_id: UUID of the integration record from the database.
        server_type: MCP server type (e.g., "aws", "github").
        display_name: Human-readable name the admin gave this integration.
        tools: The tools available through this integration.
    """

    integration_id: str
    server_type: str
    display_name: str
    tools: list[ToolDefinition] = Field(default_factory=list)


class ToolCatalog(BaseModel):
    """Filtered set of MCP tools available for a specific query.

    Conforms to the tool_catalog schema in mcp-tool-interface.yaml.

    Attributes:
        integrations: One entry per in-scope integration with its tools.
    """

    integrations: list[CatalogIntegration] = Field(default_factory=list)
