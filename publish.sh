#!/usr/bin/env bash
# =============================================================================
# Publish trained LoRA adapter to HuggingFace Hub.
#
# Usage:
#   bash publish.sh                                          # Interactive
#   bash publish.sh --repo namankhurpia/csm-1b-hindi-lora    # Specify repo
#   bash publish.sh --config configs/gpu.yaml                # GPU adapter
#   bash publish.sh --tag v2-500samples                      # Version tag
#   bash publish.sh --private                                # Private repo
#   bash publish.sh --merged                                 # Push full merged model
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running: export HF_TOKEN=hf_xxx}"

uv run python -m src.publish "$@"
