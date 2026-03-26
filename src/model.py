"""
Model loading, LoRA setup, and CSM depth-decoder patching.
"""

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoProcessor, CsmForConditionalGeneration


def _patch_csm_inplace_op():
    """
    Patch CSM depth decoder to replace an in-place op that breaks PEFT autograd.

    The original code does:
        inputs_embeds[:, 0] = backbone_last_hidden_state
    which fails when LoRA makes the backbone output require grad.

    We patch it to use torch.cat (out-of-place) and .clone() on the embed output.
    """
    import inspect

    import transformers.models.csm.modeling_csm as csm_mod

    src_file = inspect.getsourcefile(csm_mod)
    with open(src_file) as f:
        code = f.read()

    old_inplace = "inputs_embeds[:, 0] = backbone_last_hidden_state"
    new_concat = (
        "inputs_embeds = torch.cat("
        "[backbone_last_hidden_state.unsqueeze(1), inputs_embeds[:, 1:]], dim=1)"
    )
    old_embed = "inputs_embeds = self.embed_tokens(input_ids + offset)"
    new_embed = "inputs_embeds = self.embed_tokens(input_ids + offset).clone()"

    if old_inplace not in code:
        print("  [patch] CSM already patched or source changed — skipping")
        return

    code = code.replace(old_inplace, new_concat)
    code = code.replace(old_embed, new_embed)

    with open(src_file, "w") as f:
        f.write(code)
    print("  [patch] CSM depth decoder patched for PEFT compatibility")

    # Reload the module so the patch takes effect in this process
    import importlib
    importlib.reload(csm_mod)


def load_base_model(cfg: dict):
    """Load base CSM model and processor."""
    model_id = cfg["model"]["base_id"]
    dtype = cfg["hardware"]["torch_dtype"]
    device = cfg["hardware"]["device"]

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
