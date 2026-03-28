"""
Model loading, LoRA setup, and CSM source patching.
"""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoProcessor, CsmForConditionalGeneration


def _patch_csm_inplace_op():
    """
    Apply all necessary patches to CSM source for PEFT + GPU compatibility.

    Patch 1 — In-place op in depth decoder (breaks PEFT autograd):
        inputs_embeds[:, 0] = backbone_last_hidden_state
      → torch.cat([...]) out-of-place

    Patch 2 — Clone embed output (related to patch 1):
        inputs_embeds = self.embed_tokens(...)
      → self.embed_tokens(...).clone()

    Patch 3 — dtype mismatch in audio embedding merge (bfloat16 vs float32):
        inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask]
      → audio_embeds[...].to(inputs_embeds.dtype)

    Called from setup.sh and auto-applied on model load.
    """
    import inspect

    import transformers.models.csm.modeling_csm as csm_mod

    src_file = inspect.getsourcefile(csm_mod)
    with open(src_file) as f:
        code = f.read()

    modified = False

    # Patch 1: in-place op in depth decoder
    old_inplace = "inputs_embeds[:, 0] = backbone_last_hidden_state"
    new_concat = (
        "inputs_embeds = torch.cat("
        "[backbone_last_hidden_state.unsqueeze(1), inputs_embeds[:, 1:]], dim=1)"
    )
    if old_inplace in code:
        code = code.replace(old_inplace, new_concat)
        modified = True
        print("  [patch] Fixed in-place op in depth decoder")

    # Patch 2: clone embed output
    old_embed = "inputs_embeds = self.embed_tokens(input_ids + offset)"
    new_embed = "inputs_embeds = self.embed_tokens(input_ids + offset).clone()"
    if old_embed in code:
        code = code.replace(old_embed, new_embed)
        modified = True
        print("  [patch] Added .clone() to embed_tokens output")

    # Patch 3: dtype mismatch — Mimi codec can output float32 even when model is bfloat16
    old_audio_assign = "inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask]"
    new_audio_assign = "inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask].to(inputs_embeds.dtype)"
    if old_audio_assign in code and ".to(inputs_embeds.dtype)" not in code:
        code = code.replace(old_audio_assign, new_audio_assign)
        modified = True
        print("  [patch] Fixed audio embedding dtype mismatch")

    if modified:
        with open(src_file, "w") as f:
            f.write(code)
        print(f"  [patch] Wrote patches to {src_file}")

        # Reload the module so patches take effect in this process
        import importlib
        importlib.reload(csm_mod)
    else:
        print("  [patch] All patches already applied — skipping")


def load_base_model(cfg: dict):
    """Load base CSM model and processor."""
    model_id = cfg["model"]["base_id"]
    dtype = cfg["hardware"]["torch_dtype"]
    device = cfg["hardware"]["device"]

    # Auto-apply patches (idempotent — skips if already patched)
    _patch_csm_inplace_op()

    print(f"Loading model: {model_id} (dtype={cfg['hardware']['dtype']}, device={device})")
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    return model, processor


def setup_lora(model, cfg: dict):
    """Freeze codec + depth decoder, apply LoRA to backbone attention."""
    lora_cfg = cfg["model"]["lora"]

    # Freeze codec entirely
    model.codec_model.eval()
    for p in model.codec_model.parameters():
        p.requires_grad = False

    # Apply LoRA to backbone attention layers
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Freeze depth decoder (has in-place ops that conflict with autograd)
    for name, p in model.named_parameters():
        if "depth_decoder" in name:
            p.requires_grad = False

    model.print_trainable_parameters()
    return model


def load_for_inference(cfg: dict, use_adapter: bool = True):
    """Load model for inference, optionally with LoRA adapter merged in."""
    model_id = cfg["model"]["base_id"]
    dtype = cfg["hardware"]["torch_dtype"]
    device = cfg["hardware"]["device"]
    adapter_path = cfg["inference"]["adapter_path"]

    # Auto-apply patches
    _patch_csm_inplace_op()

    processor = AutoProcessor.from_pretrained(model_id)

    if use_adapter:
        print(f"Loading base model + LoRA adapter from: {adapter_path}")
        base = CsmForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
    else:
        print(f"Loading baseline model: {model_id}")
        model = CsmForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)

    model.to(device)
    model.eval()
    return model, processor
