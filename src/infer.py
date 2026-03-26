"""
Inference entry point. Generates Hindi speech from text prompts.

Usage:
    python -m src.infer                     # Fine-tuned model
    python -m src.infer --baseline          # Baseline comparison
    python -m src.infer --prompt "कुछ हिंदी टेक्स्ट"
"""

import argparse
import os
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch

from .config import load_config
from .model import load_for_inference


def generate_one(model, processor, text: str, device: str, cfg: dict):
    """Generate audio for a single Hindi text prompt."""
    conversation = [
        {"role": "0", "content": [{"type": "text", "text": text}]},
    ]
    inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        audio = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=cfg["inference"]["max_new_tokens"],
            do_sample=cfg["inference"]["do_sample"],
        )
    return audio


def run_generation(model, processor, prompts, device, cfg, prefix, out_dir):
    """Generate audio for all prompts and save."""
    for i, text in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {text[:50]}...", flush=True)
        t0 = time.time()
        audio = generate_one(model, processor, text, device, cfg)
        elapsed = time.time() - t0
        out_path = out_dir / f"{prefix}_{i:02d}.wav"
        processor.save_audio(audio, str(out_path))
        print(f"           -> {out_path} ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Hindi speech")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--baseline", action="store_true", help="Also run baseline model")
    parser.add_argument("--prompt", type=str, help="Single custom prompt (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = cfg["hardware"]["device"]
    out_dir = Path(cfg["inference"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [args.prompt] if args.prompt else cfg["inference"]["prompts"]

    # Fine-tuned model
    adapter_path = Path(cfg["inference"]["adapter_path"])
    if adapter_path.exists():
        print(f"\n--- Fine-tuned model (LoRA: {adapter_path}) ---")
        model, processor = load_for_inference(cfg, use_adapter=True)
        run_generation(model, processor, prompts, device, cfg, "finetuned", out_dir)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"No adapter at {adapter_path} — skipping fine-tuned inference.")

    # Baseline model
    if args.baseline:
        print(f"\n--- Baseline model ({cfg['model']['base_id']}) ---")
        model, processor = load_for_inference(cfg, use_adapter=False)
        run_generation(model, processor, prompts, device, cfg, "baseline", out_dir)

    print(f"\nAll audio saved to {out_dir}/")


if __name__ == "__main__":
    main()
