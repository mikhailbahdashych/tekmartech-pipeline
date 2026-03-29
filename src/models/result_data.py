"""Pydantic models for result data schemas.

Conforms to the result_data, result_table, and result_column schemas
defined in internal-api.yaml.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResultColumn(BaseModel):
    """A column definition within a result table.

    Conforms to the result_column schema in internal-api.yaml.

    Attributes:
        key: Machine-readable column identifier (snake_case).
        label: Human-readable column header for display.
        data_type: Data type of values in this column.
        sortable: Whether the Interface Layer should allow sorting.
    """

    key: str
    label: str
    data_type: Literal["string", "number", "boolean", "datetime", "array"]
    sortable: bool = True


class ResultTable(BaseModel):
    """A single table of results within a query's result_data.

    Conforms to the result_table schema in internal-api.yaml.

    Attributes:
        table_id: Unique identifier within the result set.
        title: Descriptive title displayed as section header.
        description: Additional context about what this table shows.
        columns: Column definitions in display order.
        rows: Data rows, each an array of values matching columns positionally.
        row_count: Number of rows in this table.
        source_step_ids: Plan step_ids that produced data in this table.
    """

    table_id: str
    title: str
    description: str | None = None
    columns: list[ResultColumn] = Field(min_length=1)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int
    source_step_ids: list[str] = Field(default_factory=list)


class IntegrationQueried(BaseModel):
    """Summary of an integration that contributed to the results.

    Attributes:
        integration_id: UUID of the integration.
        display_name: Human-readable name.
        server_type: MCP server type.
    """

    integration_id: str
    display_name: str
    server_type: str


class ResultMetadata(BaseModel):
    """Contextual information about the results as a whole.

    Attributes:
        total_records: Total number of records across all tables.
        integrations_queried: Summary of which integrations contributed.
        execution_duration_ms: How long execution took.
    """

    total_records: int
    integrations_queried: list[IntegrationQueried] = Field(default_factory=list)
    execution_duration_ms: int


class ResultData(BaseModel):
    """Structured output of a query execution.

    Conforms to the result_data schema in internal-api.yaml. Designed
    to be directly renderable by the Interface Layer.

    Attributes:
        format_version: Schema version of the result format.
        tables: Query results organized as one or more tables.
        metadata: Contextual information about the results.
    """

    format_version: str = "1.0"
    tables: list[ResultTable] = Field(default_factory=list)
    metadata: ResultMetadata
