"""
Training entry point.

Usage:
    python -m src.train                        # Full training
    python -m src.train --quick                # Sanity check (10 samples, 2 epochs)
    python -m src.train --config my_config.yaml
"""

import argparse
import os
import platform
from pathlib import Path

# MPS-specific env vars (no-op on CUDA/CPU)
if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from transformers import Trainer, TrainingArguments

from .config import load_config, print_config
from .data import CSMCollator, ConversationDataset
from .model import load_base_model, setup_lora


def _config_for_wandb(cfg: dict) -> dict:
    """Deep copy config with W&B-serializable values (e.g. torch.dtype)."""
    import copy

    c = copy.deepcopy(cfg)
    hw = c.get("hardware", {})
    if "torch_dtype" in hw:
        hw["torch_dtype"] = str(hw["torch_dtype"])
    return c


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CSM-1B on Hindi")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="10 samples, 2 epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print_config(cfg, config_path=args.config)

    # --- Model ---
    model, processor = load_base_model(cfg)
    model = setup_lora(model, cfg)

    # --- Data ---
    data_cfg = cfg["data"]
    train_ds = ConversationDataset(
        f"{data_cfg['processed_dir']}/train_conversations.jsonl",
        max_audio_sec=data_cfg["max_audio_sec"],
        sr=data_cfg["sample_rate"],
    )
    val_ds = ConversationDataset(
        f"{data_cfg['processed_dir']}/val_conversations.jsonl",
        max_audio_sec=data_cfg["max_audio_sec"],
        sr=data_cfg["sample_rate"],
    )

    if args.quick:
        train_ds.conversations = train_ds.conversations[:10]
        val_ds.conversations = val_ds.conversations[:5]
        print(f"QUICK MODE: {len(train_ds)} train, {len(val_ds)} val")

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    collator = CSMCollator(processor, cfg["hardware"]["device"], dtype=cfg["hardware"]["torch_dtype"])

    # --- Logging (W&B + TensorBoard via HuggingFace Trainer report_to) ---
    log_cfg = cfg["logging"]
    log_tool = log_cfg["tool"]
    uses_wandb = log_tool in ("wandb", "both")
    uses_tensorboard = log_tool in ("tensorboard", "both")

    if log_tool == "none":
        report_to = "none"
    else:
        report_to = []
        if uses_wandb:
            report_to.append("wandb")
        if uses_tensorboard:
            report_to.append("tensorboard")

    # --- Training args ---
    t = cfg["training"]
    hw = cfg["hardware"]
    epochs = 2 if args.quick else t["epochs"]

    run_name = log_cfg.get("run_name")
    if run_name is not None and isinstance(run_name, str) and not run_name.strip():
        run_name = None

    if uses_wandb:
        os.environ["WANDB_PROJECT"] = log_cfg["project"]
        import wandb

        wandb.init(
            project=log_cfg["project"],
            name=run_name,
            config=_config_for_wandb(cfg),
        )

    if uses_tensorboard:
        tb_dir = log_cfg.get("tensorboard_log_dir") or f"{t['output_dir']}/tensorboard"
        tb_path = str(Path(tb_dir).resolve())
        os.environ["TENSORBOARD_LOGGING_DIR"] = tb_path
        print(f"TensorBoard log dir: {tb_path}")
        print(f"  View with: tensorboard --logdir {tb_path!r}")

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=epochs,
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        fp16=False,
        bf16=(hw["device"] == "cuda" and hw["dtype"] == "bfloat16"),
        remove_unused_columns=False,
        dataloader_pin_memory=hw["pin_memory"],
        dataloader_num_workers=hw["dataloader_workers"],
        report_to=report_to,
        run_name=run_name,
        logging_dir=f"{t['output_dir']}/logs",
        logging_first_step=True,
        save_total_limit=t["save_total_limit"],
    )

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"\nTraining for {epochs} epochs...")
    trainer.train()

    # --- Save ---
    final_path = Path(t["output_dir"]) / "final"
    print(f"\nSaving to {final_path}/")
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))

    if uses_wandb:
        import wandb

        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
