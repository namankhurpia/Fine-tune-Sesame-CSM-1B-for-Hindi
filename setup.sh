#!/usr/bin/env bash
# =============================================================================
# Environment setup: installs uv, creates venv, patches CSM for PEFT.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

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

# --- venv + deps ---
echo "[2/4] Creating venv & installing dependencies..."
uv venv --python 3.11 --quiet 2>/dev/null || uv venv --quiet
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
