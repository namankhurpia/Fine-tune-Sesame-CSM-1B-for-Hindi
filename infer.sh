#!/usr/bin/env bash
# =============================================================================
# Inference script. Generates Hindi speech from fine-tuned model.
#
# Usage:
#   bash infer.sh                                  # Default config
#   bash infer.sh --config configs/mac.yaml        # Mac profile
#   bash infer.sh --config configs/gpu.yaml        # GPU profile
#   bash infer.sh --baseline                       # Also generate baseline
#   bash infer.sh --prompt "नमस्ते"                 # Custom prompt
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running: export HF_TOKEN=hf_xxx}"

echo "Generating Hindi speech..."
uv run python -m src.infer "$@"
