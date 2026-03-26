"""
Centralized configuration loader.

Loads config.yaml and resolves hardware-dependent defaults (dtype, device).
"""

import sys
from pathlib import Path

import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Resolve device
    hw = cfg["hardware"]
    if hw["device"] == "auto":
        if torch.cuda.is_available():
            hw["device"] = "cuda"
        elif torch.backends.mps.is_available():
            hw["device"] = "mps"
        else:
            hw["device"] = "cpu"

    # Resolve dtype
    if hw["dtype"] == "auto":
        hw["dtype"] = "bfloat16" if hw["device"] == "cuda" else "float32"

    hw["torch_dtype"] = getattr(torch, hw["dtype"])

    # Resolve logging tool
    log_tool = cfg["logging"]["tool"]
    if log_tool not in ("wandb", "tensorboard", "both", "none"):
        print(f"[config] Unknown logging tool '{log_tool}', falling back to 'none'")
        cfg["logging"]["tool"] = "none"

    return cfg


def print_config(cfg: dict, config_path: str = None) -> None:
    device = cfg["hardware"]["device"]
    dtype = cfg["hardware"]["dtype"]
    lora = cfg["model"]["lora"]
    t = cfg["training"]
    d = cfg["data"]

    print("=" * 60)
    print("  Hindi Speech-to-Speech Fine-Tuning")
    print("=" * 60)
    if config_path:
        print(f"  Profile:     {config_path}")
    print(f"  Model:       {cfg['model']['base_id']}")
    print(f"  Device:      {device}  |  dtype: {dtype}")
    print(f"  LoRA:        r={lora['r']}, alpha={lora['alpha']}")
    print(f"  Data:        {d['num_train_samples']} train, max {d['max_audio_sec']}s audio")
    print(f"  Epochs:      {t['epochs']}  |  BS: {t['batch_size']}x{t['gradient_accumulation_steps']}")
    print(f"  LR:          {t['learning_rate']}  |  Warmup: {t['warmup_steps']}")
    print(f"  Grad ckpt:   {t.get('gradient_checkpointing', True)}")
    print(f"  Logging:     {cfg['logging']['tool']}")
    print(f"  Output:      {t['output_dir']}")
    print("=" * 60)
