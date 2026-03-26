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
├── src/
│   ├── config.py            # Config loader
│   ├── model.py             # Model loading, LoRA, CSM patch
│   ├── data.py              # Dataset, collator, preprocessing
│   ├── train.py             # Training loop (HF Trainer)
│   ├── infer.py             # Audio generation
│   ├── evaluate.py          # Whisper ASR evaluation
│   └── download_data.py     # Data download entry point
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

```bash
bash download_data.sh
```

Downloads [Google FLEURS Hindi](https://huggingface.co/datasets/google/fleurs) (200 train + 100 val samples by default), resamples to 24kHz, and formats into CSM conversation pairs.

### 4. Train

```bash
# Sanity check — 10 samples, 2 epochs (~50 min on M2 Mac)
bash train.sh --quick

# Full training with default config
bash train.sh
```

### 5. Generate Audio

```bash
bash infer.sh                              # Fine-tuned model
bash infer.sh --baseline                   # Compare baseline vs fine-tuned
bash infer.sh --prompt "नमस्ते, कैसे हैं आप?"  # Custom prompt
```

### 6. Evaluate

```bash
bash evaluate.sh
```

Runs Whisper ASR on generated audio and computes Word Error Rate (WER) against the original Hindi text.

## Config Profiles

The project ships with two hardware profiles under `configs/`. Pass `--config` to any script to use one:

```bash
bash train.sh --config configs/mac.yaml        # Apple Silicon
bash train.sh --config configs/gpu.yaml        # CUDA GPU
bash train.sh --config configs/mac.yaml --quick # Quick test on Mac
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

### Cloud quick start (Colab / Lambda / RunPod)

```bash
git clone <repo-url> && cd voice_model
bash setup.sh
export HF_TOKEN=hf_xxx
bash download_data.sh --config configs/gpu.yaml
bash train.sh --config configs/gpu.yaml
bash infer.sh --config configs/gpu.yaml --baseline
bash evaluate.sh --config configs/gpu.yaml
```

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
