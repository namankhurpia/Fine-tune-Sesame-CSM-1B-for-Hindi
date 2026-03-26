#!/usr/bin/env bash
# =============================================================================
# Training script.
#
# Usage:
#   bash train.sh                                  # Default config (auto-detect)
#   bash train.sh --config configs/mac.yaml        # Mac profile
#   bash train.sh --config configs/gpu.yaml        # GPU profile
#   bash train.sh --quick                          # Sanity check (10 samples)
#   bash train.sh --config configs/mac.yaml --quick
#   CUDA_VISIBLE_DEVICES=0 bash train.sh --config configs/gpu.yaml
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running: export HF_TOKEN=hf_xxx}"

echo "============================================================"
echo "  Starting training..."
echo "============================================================"
uv run python -m src.train "$@"
