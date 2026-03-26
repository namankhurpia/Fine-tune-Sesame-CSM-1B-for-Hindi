#!/usr/bin/env bash
# =============================================================================
# Download FLEURS Hindi and preprocess into conversation format.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "Downloading & preprocessing Hindi speech data..."
uv run python -m src.download_data "$@"
