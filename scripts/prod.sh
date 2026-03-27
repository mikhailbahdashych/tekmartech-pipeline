#!/usr/bin/env bash
# Start the pipeline service in production mode.
# Usage: ./scripts/prod.sh

set -euo pipefail

export PIPELINE_ENV=production
exec uv run uvicorn src.main:app --port "${PORT:-8100}"
