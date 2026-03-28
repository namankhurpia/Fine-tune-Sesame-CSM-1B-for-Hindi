"""
Publish trained LoRA adapter (or merged model) to HuggingFace Hub.

Usage:
    python -m src.publish                                          # Interactive
    python -m src.publish --repo user/csm-1b-hindi-lora            # Specify repo
    python -m src.publish --config configs/gpu.yaml                # GPU adapter path
    python -m src.publish --tag v2-500samples                      # Version tag
    python -m src.publish --private                                # Private repo
    python -m src.publish --merged                                 # Push full merged model
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def get_adapter_info(adapter_path: str) -> dict:
    """Read adapter config for model card metadata."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def get_training_info(adapter_path: str) -> dict:
    """Try to read training state for metadata."""
    state_path = Path(adapter_path).parent / "trainer_state.json"
    if not state_path.exists():
        # Check one level up
        state_path = Path(adapter_path).parent / "trainer_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        train_losses = [e["loss"] for e in log_history if "loss" in e]
        eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
        return {
            "total_steps": state.get("global_step", "?"),
            "final_train_loss": f"{train_losses[-1]:.4f}" if train_losses else "?",
            "final_eval_loss": f"{eval_losses[-1]:.4f}" if eval_losses else "?",
            "epochs": state.get("epoch", "?"),
        }
    return {}


def build_model_card(
    repo_id: str,
    adapter_info: dict,
    training_info: dict,
    tag: str | None,
    is_merged: bool,
) -> str:
    """Generate a HuggingFace model card (README.md)."""
    r = adapter_info.get("r", "?")
    alpha = adapter_info.get("lora_alpha", "?")
    base_model = adapter_info.get("base_model_name_or_path", "sesame/csm-1b")
    target_modules = adapter_info.get("target_modules", [])

    version_line = f"\n**Version:** `{tag}`\n" if tag else ""
    model_type = "Full merged model" if is_merged else "LoRA adapter"

    train_section = ""
    if training_info:
        train_section = f"""
## Training Results

| Metric | Value |
|--------|-------|
| Total steps | {training_info.get('total_steps', '?')} |
| Epochs | {training_info.get('epochs', '?')} |
| Final train loss | {training_info.get('final_train_loss', '?')} |
| Final eval loss | {training_info.get('final_eval_loss', '?')} |
"""

    card = f"""---
base_model: {base_model}
language:
  - hi
library_name: {'transformers' if is_merged else 'peft'}
license: other
pipeline_tag: text-to-speech
tags:
  - hindi
  - tts
  - lora
  - csm
  - sesame
  - speech
{f'  - {tag}' if tag else ''}
---

# CSM-1B Hindi Fine-Tuned — {model_type}

Fine-tuned [{base_model}](https://huggingface.co/{base_model}) for **Hindi text-to-speech** using LoRA on the backbone attention layers.
{version_line}
## Model Details

| Property | Value |
|----------|-------|
| Base model | [{base_model}](https://huggingface.co/{base_model}) |
| Type | {model_type} |
| Language | Hindi (hi) |
| LoRA rank | {r} |
| LoRA alpha | {alpha} |
| Target modules | {', '.join(target_modules) if isinstance(target_modules, list) else target_modules} |
| Dataset | [Google FLEURS Hindi](https://huggingface.co/datasets/google/fleurs) |
{train_section}
## Usage

### Load as LoRA adapter (recommended)

```python
from transformers import AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel

base_model = CsmForConditionalGeneration.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base_model, "{repo_id}")
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained("{base_model}")

model.eval()
model.to("cuda")  # or "mps" for Mac

# Generate Hindi speech
conversation = [
    {{"role": "0", "content": [
        {{"type": "text", "text": "नमस्ते, आज मौसम बहुत अच्छा है।"}},
    ]}},
    {{"role": "1", "content": [
        {{"type": "text", "text": "हाँ, बहुत सुहावना दिन है।"}},
    ]}},
]

inputs = processor.apply_chat_template(
    conversation, tokenize=True, return_dict=True,
)
inputs = {{k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}}

output = model.generate(**inputs, max_new_tokens=250, do_sample=False)
audio = processor.decode(output)
```

## Architecture

```
Hindi text -> [CSM-1B + LoRA] -> Hindi audio
                    |
              Mimi codec (frozen)
              Backbone LM (LoRA on q/k/v/o_proj)
              Depth decoder (frozen)
```

## Training Pipeline

Trained with the [Hindi Speech-to-Speech Fine-Tuning](https://github.com/namankhurpia/Fine-tune-Sesame-CSM-1B-for-Hindi) pipeline.

## License

This model is a fine-tuned adapter for [{base_model}](https://huggingface.co/{base_model}). See the base model's license for usage terms. Training data (FLEURS) is CC-BY-4.0.
"""
    return card


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--config", default="config.yaml", help="Config file for adapter path")
    parser.add_argument("--repo", default=None, help="HuggingFace repo ID (e.g., user/model-name)")
    parser.add_argument("--adapter_path", default=None, help="Override adapter path")
    parser.add_argument("--tag", default=None, help="Version tag (e.g., v1, v2-500samples)")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--merged", action="store_true", help="Push full merged model instead of adapter")
    args = parser.parse_args()

    # Load config for defaults
    from .config import load_config
    cfg = load_config(args.config)

    adapter_path = args.adapter_path or cfg["inference"]["adapter_path"]

    if not Path(adapter_path).exists():
        print(f"Error: Adapter not found at {adapter_path}")
        print(f"  Train first: bash train.sh --config {args.config}")
        return

    # Resolve repo name
    api = HfApi()
    user = api.whoami()["name"]

    if args.repo:
        repo_id = args.repo
    else:
        repo_id = f"{user}/csm-1b-hindi-lora"
        if args.tag:
            repo_id = f"{user}/csm-1b-hindi-lora-{args.tag}"

    print("=" * 60)
    print("  Publish to HuggingFace Hub")
    print("=" * 60)
    print(f"  Repo:     {repo_id}")
    print(f"  Adapter:  {adapter_path}")
    print(f"  Tag:      {args.tag or '(none)'}")
    print(f"  Private:  {args.private}")
    print(f"  Type:     {'Merged model' if args.merged else 'LoRA adapter'}")
    print("=" * 60)

    # Create repo
    create_repo(repo_id, exist_ok=True, private=args.private)
    print(f"\n  Repo created/exists: https://huggingface.co/{repo_id}")

    if args.merged:
        # Merge LoRA into base and push the full model
        print("\n  Merging LoRA into base model...")
        from .model import load_for_inference
        model, processor = load_for_inference(cfg, use_adapter=True)
        print("  Pushing merged model (this may take a few minutes)...")
        model.push_to_hub(repo_id, private=args.private)
        processor.push_to_hub(repo_id, private=args.private)
        print(f"  Merged model pushed!")
    else:
        # Push just the adapter files
        print("\n  Uploading LoRA adapter...")
        upload_folder(
            repo_id=repo_id,
            folder_path=adapter_path,
            commit_message=f"Upload LoRA adapter{f' ({args.tag})' if args.tag else ''}",
        )
        print(f"  Adapter uploaded!")

    # Generate and upload model card
    adapter_info = get_adapter_info(adapter_path)
    training_info = get_training_info(adapter_path)
    card = build_model_card(repo_id, adapter_info, training_info, args.tag, args.merged)

    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card",
    )
    print(f"  Model card uploaded!")

    # Create a git tag if version specified
    if args.tag:
        try:
            api.create_tag(repo_id, tag=args.tag, tag_message=f"Version {args.tag}")
            print(f"  Tag created: {args.tag}")
        except Exception as e:
            print(f"  Tag creation skipped: {e}")

    print(f"\n  Published: https://huggingface.co/{repo_id}")
    print(f"\n  Share this link with anyone to showcase your model!")
    if not args.merged:
        print(f"\n  To load:")
        print(f"    from peft import PeftModel")
        print(f"    model = PeftModel.from_pretrained(base_model, \"{repo_id}\")")


if __name__ == "__main__":
    main()
