# Hindi Speech-to-Speech Fine-Tuning

Fine-tune [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) for Hindi conversational speech generation using LoRA on consumer hardware (Apple Silicon / single GPU).

## Architecture

```
User speech -> [Whisper ASR] -> Hindi text -> [CSM-1B + LoRA] -> Hindi audio
                                                    |
                                              Mimi codec (frozen)
                                              Backbone LM (LoRA)
                                              Depth decoder (frozen)
```

CSM is a **text-to-speech** model with a Mimi neural codec and two LLaMA-style decoders. We freeze the codec and depth decoder, applying **LoRA** to the backbone's attention layers only. This reduces trainable parameters from 1.7B to **~3-7M (<0.5%)**, making training feasible on 16GB devices.

## Project Structure

```
.
├── config.yaml              # Default config (auto-detects hardware)
├── configs/
│   ├── mac.yaml             # Mac Apple Silicon profile (16GB)
│   └── gpu.yaml             # CUDA GPU profile (24GB+)
├── setup.sh                 # Environment setup
├── download_data.sh         # Data pipeline
├── train.sh                 # Training
├── infer.sh                 # Inference
├── evaluate.sh              # WER evaluation
├── publish.sh               # Publish model to HuggingFace Hub
├── src/
│   ├── config.py            # Config loader
│   ├── model.py             # Model loading, LoRA, CSM patch
│   ├── data.py              # Dataset, collator, preprocessing
│   ├── train.py             # Training loop (HF Trainer)
│   ├── infer.py             # Audio generation
│   ├── evaluate.py          # Whisper ASR evaluation
│   ├── download_data.py     # Data download entry point
│   └── publish.py           # HuggingFace Hub publisher
├── dataset/                 # Custom dataset creation toolkit (see dataset/README.md)
├── data/                    # Downloaded datasets (gitignored)
├── outputs/                 # Checkpoints + generated audio (gitignored)
└── requirements.txt
```

## Quick Start

### 1. Setup

```bash
git clone <repo-url> && cd voice_model

# Install everything (uv + Python 3.11 + dependencies + CSM patch)
bash setup.sh
```

### 2. Authenticate with HuggingFace

The CSM-1B model is gated. You need to:
1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model license at [sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
export HF_TOKEN=hf_your_token_here
```

### 3. Download Data

Downloads [Google FLEURS Hindi](https://huggingface.co/datasets/google/fleurs), resamples to 24kHz, and formats into CSM conversation pairs.

```bash
# Mac — 200 train + 100 val samples (small, fits in 16GB)
bash download_data.sh --config configs/mac.yaml

# GPU — 2120 train + 239 val samples (full FLEURS Hindi)
bash download_data.sh --config configs/gpu.yaml

# Auto-detect hardware (uses default config.yaml)
bash download_data.sh
```

### 4. Train

```bash
# --- Mac Apple Silicon ---
bash train.sh --config configs/mac.yaml --quick    # Sanity check: 10 samples, 2 epochs
bash train.sh --config configs/mac.yaml            # Full: 200 samples, 5 epochs

# --- GPU (24GB+) ---
bash train.sh --config configs/gpu.yaml --quick    # Sanity check: 10 samples, 2 epochs
bash train.sh --config configs/gpu.yaml            # Full: 2120 samples, 10 epochs

# Auto-detect hardware
bash train.sh --quick
bash train.sh
```

### 5. Generate Audio

```bash
# --- Mac ---
bash infer.sh --config configs/mac.yaml                              # Fine-tuned model
bash infer.sh --config configs/mac.yaml --baseline                   # Compare baseline vs fine-tuned
bash infer.sh --config configs/mac.yaml --prompt "नमस्ते, कैसे हैं आप?"

# --- GPU ---
bash infer.sh --config configs/gpu.yaml
bash infer.sh --config configs/gpu.yaml --baseline
bash infer.sh --config configs/gpu.yaml --prompt "नमस्ते, कैसे हैं आप?"
```

### 6. Evaluate

```bash
bash evaluate.sh --config configs/mac.yaml     # Mac (uses Whisper base)
bash evaluate.sh --config configs/gpu.yaml     # GPU (uses Whisper medium)
```

Runs Whisper ASR on generated audio and computes Word Error Rate (WER) against the original Hindi text.

### 7. Publish to HuggingFace Hub

```bash
# Push LoRA adapter with version tag
bash publish.sh --config configs/gpu.yaml --tag v1-fleurs

# Push with specific repo name
bash publish.sh --repo your-username/csm-1b-hindi-lora --tag v1

# Push full merged model (larger, no PEFT dependency to load)
bash publish.sh --config configs/gpu.yaml --merged --tag v1-merged

# Private repo
bash publish.sh --config configs/gpu.yaml --tag v1 --private
```

Generates a model card with training stats, usage code, and architecture details. Creates a git tag for versioning. See [Publishing Models](#publishing-models) for full details.

## Config Profiles

The project ships with two hardware profiles under `configs/`. Pass `--config` to **any** script to use one:

```bash
# Every script accepts --config
bash download_data.sh --config configs/mac.yaml
bash train.sh --config configs/mac.yaml
bash infer.sh --config configs/mac.yaml
bash evaluate.sh --config configs/mac.yaml
bash publish.sh --config configs/mac.yaml --tag v1

# Same for GPU
bash download_data.sh --config configs/gpu.yaml
bash train.sh --config configs/gpu.yaml
bash infer.sh --config configs/gpu.yaml
bash evaluate.sh --config configs/gpu.yaml
bash publish.sh --config configs/gpu.yaml --tag v1
```

### Mac vs GPU — side by side

| Setting | `configs/mac.yaml` | `configs/gpu.yaml` | Why |
|---------|-------------------|-------------------|-----|
| **Device** | `mps` | `cuda` | Hardware target |
| **dtype** | `float32` | `bfloat16` | MPS can't do bf16; CUDA is 2x faster in half precision |
| **LoRA rank** | `r=16`, `alpha=32` | `r=32`, `alpha=64` | GPU has memory for a larger adapter |
| **Train samples** | `200` | `2120` (full FLEURS) | GPU can handle the full dataset |
| **Max audio** | `5s` | `10s` | Longer clips OOM on 16GB |
| **Batch size** | `1` | `4` | GPU parallelism |
| **Grad accumulation** | `4` (eff. BS=4) | `2` (eff. BS=8) | Larger effective batch on GPU |
| **Epochs** | `5` | `10` | More passes needed for larger dataset |
| **Learning rate** | `2e-5` | `3e-5` | Higher LR with larger effective batch |
| **Grad checkpointing** | `true` | `false` | Mac needs it to fit in 16GB; GPU trades memory for speed |
| **pin_memory** | `false` | `true` | Broken on MPS; faster host-to-device on CUDA |
| **Dataloader workers** | `0` | `4` | MPS doesn't benefit; CUDA does |
| **Inference tokens** | `50` (~4s audio) | `250` (~20s audio) | Generation is 5-10x faster on CUDA |
| **Whisper eval model** | `base` | `medium` | Better accuracy when speed isn't a bottleneck |
| **Output dir** | `outputs/mac/` | `outputs/gpu/` | Separate outputs so you can compare |

### Default config

`config.yaml` at the project root uses `"auto"` for device and dtype, so it works anywhere. The dedicated profiles are pre-tuned for their target hardware — use them when you know where you're running.

### Customizing

Copy a profile and edit it:

```bash
cp configs/gpu.yaml configs/a100.yaml
# Edit configs/a100.yaml — increase batch_size, disable gradient_checkpointing, etc.
bash train.sh --config configs/a100.yaml
```

## Training Visualization

Training metrics are logged to **Weights & Biases** and **TensorBoard** simultaneously by default (`logging.tool: "both"` in config). You'll see real-time:
- Training loss curve
- Evaluation loss
- Learning rate schedule
- System metrics (GPU utilization, memory)

### Weights & Biases

```bash
uv run wandb login           # One-time: paste API key from https://wandb.ai/authorize
bash train.sh                # Metrics stream to wandb automatically
```

View your dashboard at [wandb.ai](https://wandb.ai).

### TensorBoard

```bash
# Logs are written alongside checkpoints
uv run tensorboard --logdir outputs/mac/tensorboard    # Mac profile
uv run tensorboard --logdir outputs/gpu/tensorboard    # GPU profile
```

### Options

Set `logging.tool` in your config file:
- `"both"` — wandb + TensorBoard (default)
- `"wandb"` — wandb only
- `"tensorboard"` — TensorBoard only
- `"none"` — disable logging

## Mac Quick Start (Apple Silicon)

```bash
bash setup.sh
export HF_TOKEN=hf_your_token_here

bash download_data.sh --config configs/mac.yaml       # 200 train samples
bash train.sh --config configs/mac.yaml --quick        # Sanity check (~50 min)
bash train.sh --config configs/mac.yaml                # Full training (~5 hrs)
bash infer.sh --config configs/mac.yaml --baseline
bash evaluate.sh --config configs/mac.yaml
```

## GPU Environment

### Single GPU

```bash
export HF_TOKEN=hf_your_token_here
CUDA_VISIBLE_DEVICES=0 bash train.sh --config configs/gpu.yaml
```

### Multi-GPU

```bash
uv run accelerate launch --num_processes 2 -m src.train --config configs/gpu.yaml
```

### Cloud quick start (Vast.ai / RunPod / Lambda / Colab)

```bash
# 1. Setup
git clone <repo-url> && cd voice_model
bash setup.sh

# 2. Authenticate
export HF_TOKEN=hf_your_token_here
uv run huggingface-cli login --token $HF_TOKEN

# 3. Download full FLEURS Hindi dataset (2120 train samples)
bash download_data.sh --config configs/gpu.yaml

# 4. Train (full: ~1hr on 24GB GPU)
bash train.sh --config configs/gpu.yaml

# 5. Inference + Evaluation
bash infer.sh --config configs/gpu.yaml --baseline
bash evaluate.sh --config configs/gpu.yaml

# 6. Publish to HuggingFace Hub before destroying the instance
bash publish.sh --config configs/gpu.yaml --tag v1
```

Use `tmux` to protect long-running commands from SSH disconnects.

## How It Works

### Data Pipeline

1. **Download**: FLEURS Hindi dataset from HuggingFace (read speech with Devanagari transcriptions)
2. **Resample**: 16kHz to 24kHz (Mimi codec requirement)
3. **Pair**: Consecutive utterances become 2-turn conversations:
   - Speaker 0: text + audio (context)
   - Speaker 1: text only (generation target)
4. **Save**: JSONL format with audio arrays serialized inline

### Training

1. Load CSM-1B base model
2. **Freeze** the Mimi codec (encoder/decoder for audio tokens)
3. **Apply LoRA** to backbone attention (`q_proj`, `v_proj`, `k_proj`, `o_proj`)
4. **Freeze** depth decoder (has an in-place op patched for PEFT compatibility)
5. Train with HF Trainer: gradient checkpointing, gradient accumulation, mixed precision on GPU

### CSM Patch

The CSM depth decoder has an in-place operation that breaks autograd when LoRA is applied:
```python
# Original — fails with PEFT
inputs_embeds[:, 0] = backbone_last_hidden_state

# Patched — out-of-place, autograd-safe
inputs_embeds = torch.cat([backbone_last_hidden_state.unsqueeze(1), inputs_embeds[:, 1:]], dim=1)
```

This is applied automatically by `setup.sh`. If you reinstall transformers, re-run `bash setup.sh`.

### Inference

Loads the base model + LoRA adapter, merges weights, and generates audio autoregressively. Each token = one audio frame decoded by the Mimi codec.

### Evaluation

Round-trip test: generated audio -> Whisper Hindi ASR -> compare transcription to original text -> WER.

## Publishing Models

Push your trained adapter to HuggingFace Hub for sharing, versioning, and easy loading.

```bash
bash publish.sh --config configs/gpu.yaml --tag v1-fleurs
```

### Options

| Flag | Example | Description |
|------|---------|-------------|
| `--tag` | `--tag v2-500samples` | Version tag (creates a git tag on HuggingFace) |
| `--repo` | `--repo user/my-model` | Custom repo name (default: `<user>/csm-1b-hindi-lora`) |
| `--private` | `--private` | Create a private repo |
| `--merged` | `--merged` | Push full merged model instead of LoRA adapter |
| `--config` | `--config configs/gpu.yaml` | Config file (to find adapter path) |
| `--adapter_path` | `--adapter_path outputs/gpu/final` | Override adapter path directly |

### What gets published

- **LoRA adapter files** (~15-30 MB) — or full merged model (~4 GB with `--merged`)
- **Model card** (auto-generated README.md) with:
  - Training stats (final loss, epochs, steps)
  - LoRA config details
  - Ready-to-copy Python usage code
  - Architecture diagram
- **Git tag** for versioning

### Versioning workflow

```bash
# First training run — FLEURS 200 samples
bash publish.sh --tag v1-fleurs-200

# Retrain with more data
bash publish.sh --tag v2-fleurs-full

# Train on custom dataset
bash publish.sh --tag v3-custom-500
```

### Loading a published model

```python
from transformers import CsmForConditionalGeneration
from peft import PeftModel

base = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b")
model = PeftModel.from_pretrained(base, "your-username/csm-1b-hindi-lora")
model = model.merge_and_unload()
```

## Custom Dataset Creation

The `dataset/` subfolder contains a complete toolkit for building your own Hindi speech dataset. See [`dataset/README.md`](dataset/README.md) for the full guide.

### Approaches

| Approach | Method | Speed | Quality |
|----------|--------|-------|---------|
| **A** | Synthetic TTS (F5-Hindi / MMS-TTS) | Fastest | Good, synthetic voice |
| **B** | Record your own voice | Slow | Best, natural prosody |
| **C** | Existing audio + Whisper transcription | Medium | Natural, depends on source |
| **D** | Mix all sources | — | Best overall results |

### Quick example (synthetic data)

```bash
cd dataset/
uv pip install -r requirements.txt

uv run python scripts/01_collect_text.py --source iitb --count 500
uv run python scripts/02_synthesize_audio.py --tts mms
uv run python scripts/05_build_dataset.py --source synthesized
uv run python scripts/06_validate.py

# Copy to training pipeline
cp output/*_conversations.jsonl ../data/processed/
cd .. && bash train.sh --quick
```

## Hardware Requirements

| Setup | Training | Inference |
|-------|----------|-----------|
| **Mac M2 16GB** | ~50 min quick test, ~5 hrs full (200 samples) | ~30s per sample |
| **Single GPU 24GB** (A10/3090) | ~10 min quick, ~1 hr full (2120 samples) | ~5s per sample |
| **A100 80GB** | ~5 min quick, ~30 min full | ~2s per sample |

## Known Limitations

- **CSM is TTS, not full speech-to-speech**: For the complete loop, chain with Whisper ASR on the input side
- **Small dataset**: 200 FLEURS samples produce limited Hindi quality — scale to thousands for real results
- **Read speech**: FLEURS is not conversational; add dialogue datasets for better conversational output
- **English bias**: CSM-1B was trained primarily on English; Hindi output may retain English prosody

## Scaling Up

Once the pipeline is proven:

1. **More data**: Use `configs/gpu.yaml` which already sets `num_train_samples: 2120` (full FLEURS Hindi)
2. **Additional datasets**: Swap `dataset_id` in your config or add custom Hindi conversation audio
3. **Larger LoRA**: GPU profile already uses `r=32`; go to 64 with more VRAM
4. **Full speech-to-speech**: Consider [Kyutai Moshi](https://github.com/kyutai-labs/moshi) on cloud GPUs for true duplex interaction

## License

This project fine-tunes the [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) model. See their license for model usage terms. FLEURS data is CC-BY-4.0.
