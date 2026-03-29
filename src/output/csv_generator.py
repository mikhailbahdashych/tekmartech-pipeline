"""CSV generator.

Converts ResultData into a CSV-formatted string for export.
Each table becomes a section with headers and data rows.
"""

import csv
import io

import structlog

from src.models.result_data import ResultData

logger = structlog.get_logger(__name__)


def generate_csv(result_data: ResultData) -> str | None:
    """Generate a CSV string from result data.

    Each table produces a section: table title, column headers,
    then data rows. Multiple tables are separated by a blank line.

    Args:
        result_data: The formatted result data.

    Returns:
        CSV-formatted string, or None if no tables exist.
    """
    if not result_data.tables:
        return None

    logger.debug(
        "generating CSV export",
        action="generate_csv",
        table_count=len(result_data.tables),
    )

    output = io.StringIO()
    writer = csv.writer(output)

    for i, table in enumerate(result_data.tables):
        if i > 0:
            writer.writerow([])  # Blank line between tables

        writer.writerow([table.title])
        writer.writerow([col.label for col in table.columns])

        for row in table.rows:
            writer.writerow(row)

    result = output.getvalue()

    logger.info(
        "CSV export generated",
        action="generate_csv",
        size_bytes=len(result),
    )
    return result
