#!/usr/bin/env bash
# =============================================================================
# Environment setup: installs uv, creates venv, patches CSM for PEFT.
#
# PyTorch wheels: default "auto" uses CUDA 12.8 builds if nvidia-smi exists,
# else CPU-only (avoids missing libcudart.so.* on CPU VMs).
# CUDA 12.8 supports RTX 50-series (Blackwell) + all older GPUs.
#   SETUP_TORCH=cpu   bash setup.sh   # force CPU (recommended for CPU-only VMs)
#   SETUP_TORCH=cuda  bash setup.sh   # force CUDA 12.8 wheels
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash setup.sh  # override index
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "============================================================"
echo "  Hindi Speech-to-Speech Fine-Tuning — Setup"
echo "============================================================"

# --- uv ---
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[1/4] uv $(uv self version 2>/dev/null || uv --version)"

# --- venv (pin to project .venv so conda/base Python does not leak into uv run) ---
echo "[2/4] Creating venv & installing dependencies..."
uv venv --python 3.11 --quiet 2>/dev/null || uv venv --quiet
export VIRTUAL_ENV="$ROOT/.venv"
export UV_PYTHON="$ROOT/.venv/bin/python"

# --- PyTorch / torchaudio: explicit index (PyPI default can pull CUDA 13+ and break on CPU VMs) ---
# RTX 50-series (Blackwell, sm_120) requires CUDA 12.8+ wheels.
# Older GPUs (Ampere/Hopper) also work fine with cu128.
SETUP_TORCH="${SETUP_TORCH:-auto}"
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    TORCH_INDEX="$TORCH_INDEX_URL"
elif [[ "$SETUP_TORCH" == "cpu" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
elif [[ "$SETUP_TORCH" == "cuda" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [[ "$SETUP_TORCH" == "auto" ]] && command -v nvidia-smi &>/dev/null; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi
echo "  PyTorch index: $TORCH_INDEX (SETUP_TORCH=$SETUP_TORCH)"

uv pip install "torch>=2.4.0" "torchaudio>=2.4.0" --index-url "$TORCH_INDEX"
uv pip install -r requirements.txt --quiet

# --- Patch CSM ---
echo "[3/4] Patching CSM depth decoder for PEFT..."
uv run python -c "from src.model import _patch_csm_inplace_op; _patch_csm_inplace_op()"

# --- Verify ---
echo "[4/4] Verifying installation..."
uv run python -c "
import torch, torchaudio, transformers, peft, datasets
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT:         {peft.__version__}')
print(f'  Datasets:     {datasets.__version__}')
device = 'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'
print(f'  Device:       {device}')
"

echo ""
echo "Setup complete. Next steps:"
echo "  bash download_data.sh   # Download Hindi dataset"
echo "  bash train.sh           # Start training"
echo "============================================================"
