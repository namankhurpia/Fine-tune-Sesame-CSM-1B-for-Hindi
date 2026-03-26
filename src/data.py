"""
Data downloading, preprocessing, and training dataset/collator.
"""

import json
from pathlib import Path

import numpy as np
from datasets import Audio, load_dataset, load_from_disk
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Download & preprocess
# ---------------------------------------------------------------------------

def download_fleurs_hindi(cfg: dict) -> None:
    """Download FLEURS Hindi, resample to 24kHz, save to disk."""
    data_cfg = cfg["data"]
    n_train = data_cfg["num_train_samples"]
    n_val = data_cfg["num_val_samples"]
    sr = data_cfg["sample_rate"]
    out_dir = Path(data_cfg["processed_dir"])

    print(f"Downloading FLEURS Hindi: {n_train} train + {n_val} val samples...")
    train_ds = load_dataset(
        data_cfg["dataset_id"], data_cfg["dataset_config"],
        split=f"train[:{n_train}]", trust_remote_code=True,
    )
    val_ds = load_dataset(
        data_cfg["dataset_id"], data_cfg["dataset_config"],
        split=f"validation[:{n_val}]", trust_remote_code=True,
    )

    print(f"Resampling to {sr} Hz...")
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=sr))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=sr))

    # Inspect
    s = train_ds[0]
    a = s["audio"]
    print(f"  Sample: '{s['transcription'][:60]}...'")
    print(f"  Audio:  {a['sampling_rate']} Hz, {len(a['array'])/a['sampling_rate']:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(out_dir / "fleurs_hindi_train"))
    val_ds.save_to_disk(str(out_dir / "fleurs_hindi_val"))
    print(f"Saved to {out_dir}/")


def build_conversations(cfg: dict) -> None:
    """Convert FLEURS samples into CSM 2-turn conversation JSONL."""
    data_cfg = cfg["data"]
    out_dir = Path(data_cfg["processed_dir"])

    for split, name in [("fleurs_hindi_train", "train"), ("fleurs_hindi_val", "val")]:
        ds = load_from_disk(str(out_dir / split))
        samples = list(ds)
        convos = []

        for i in range(0, len(samples) - 1, 2):
            ctx, tgt = samples[i], samples[i + 1]
            convos.append({
                "conversation": [
                    {
                        "role": "0",
                        "content": [
                            {"type": "text", "text": ctx["transcription"]},
                            {"type": "audio", "path": ctx["audio"]["array"].tolist()},
                        ],
                    },
                    {
                        "role": "1",
                        "content": [
                            {"type": "text", "text": tgt["transcription"]},
                        ],
                    },
                ],
                "target_text": tgt["transcription"],
            })

        out_path = out_dir / f"{name}_conversations.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for c in convos:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        print(f"  {name}: {len(convos)} conversations -> {out_path}")


# ---------------------------------------------------------------------------
# Training dataset & collator
# ---------------------------------------------------------------------------

class ConversationDataset(Dataset):
    """Loads conversations from JSONL for training."""

    def __init__(self, jsonl_path: str, max_audio_sec: float = 5.0, sr: int = 24000):
        self.max_samples = int(max_audio_sec * sr)
        self.conversations = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.conversations.append(json.loads(line))

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        item = self.conversations[idx]
        convo = item["conversation"]

        # Deserialize audio arrays and truncate
        for turn in convo:
            for content in turn["content"]:
                if content["type"] == "audio":
                    audio = np.array(content["path"], dtype=np.float32)
                    if len(audio) > self.max_samples:
                        audio = audio[: self.max_samples]
                    content["path"] = audio

        return {"conversation": convo}


class CSMCollator:
    """Collates conversations into model inputs using CSM processor."""

    def __init__(self, processor, device: str):
        self.processor = processor
        self.device = device

    def __call__(self, features):
        convo = features[0]["conversation"]  # batch_size=1
        inputs = self.processor.apply_chat_template(
            convo, tokenize=True, return_dict=True, output_labels=True,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
