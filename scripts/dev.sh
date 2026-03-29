#!/usr/bin/env bash
# Start the pipeline service in development mode.
# Usage: ./scripts/dev.sh

set -euo pipefail

export PIPELINE_ENV=development
exec uv run uvicorn src.main:app --reload --port "${PORT:-8100}"
