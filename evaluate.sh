#!/usr/bin/env bash
# =============================================================================
# Evaluation script. Whisper ASR round-trip WER on generated audio.
#
# Usage:
#   bash evaluate.sh                               # Default config
#   bash evaluate.sh --config configs/mac.yaml     # Mac profile
#   bash evaluate.sh --config configs/gpu.yaml     # GPU profile
#   bash evaluate.sh --whisper_model medium         # Override whisper size
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "Evaluating generated audio..."
uv run python -m src.evaluate "$@"
