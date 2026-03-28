"""
Step 5: Build CSM-format JSONL dataset from audio + text pairs.

Reads a manifest file (filename<TAB>text) and audio files, then builds
the 2-turn conversation JSONL format required by the training pipeline.

Usage:
    # From synthesized audio
    python scripts/05_build_dataset.py --source synthesized

    # From recorded audio
    python scripts/05_build_dataset.py --source recorded

    # From custom manifest + audio directory
    python scripts/05_build_dataset.py --manifest audio/raw/manifest.txt --audio_dir audio/raw/

    # Custom train/val split
    python scripts/05_build_dataset.py --source synthesized --train_ratio 0.9
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


def load_manifest(manifest_path: str, audio_dir: str = None) -> list[dict]:
    """Load manifest file and pair with audio files.

    Returns list of {"text": str, "audio_path": Path} dicts.
    """
    manifest = Path(manifest_path)
    if audio_dir is None:
        audio_dir = manifest.parent

    pairs = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            filename, text = parts
            audio_path = Path(audio_dir) / filename
            if audio_path.exists():
                pairs.append({"text": text.strip(), "audio_path": audio_path})
            else:
                print(f"  Warning: {audio_path} not found, skipping")

    return pairs


def load_audio(path: Path, max_sec: float = 10.0) -> np.ndarray:
    """Load and validate audio file."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != TARGET_SR:
        target_len = int(len(audio) * TARGET_SR / sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # Truncate
    max_samples = int(max_sec * TARGET_SR)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


def build_conversations(pairs: list[dict], max_audio_sec: float = 10.0) -> list[dict]:
    """Convert audio-text pairs into CSM 2-turn conversations.

    Each conversation:
      - Speaker 0: context (text + audio) from pair[i]
      - Speaker 1: target (text only) from pair[i+1]
    """
    conversations = []

    for i in range(0, len(pairs) - 1, 2):
        ctx = pairs[i]
        tgt = pairs[i + 1]

        try:
            audio = load_audio(ctx["audio_path"], max_sec=max_audio_sec)
        except Exception as e:
            print(f"  Error loading {ctx['audio_path']}: {e}")
            continue

        if len(audio) < TARGET_SR * 0.3:  # Skip clips under 0.3s
            continue

        convo = {
            "conversation": [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": ctx["text"]},
                        {"type": "audio", "path": audio.tolist()},
                    ],
                },
                {
                    "role": "1",
                    "content": [
                        {"type": "text", "text": tgt["text"]},
                    ],
                },
            ],
            "target_text": tgt["text"],
        }
        conversations.append(convo)

    return conversations


def save_jsonl(conversations: list[dict], path: Path) -> None:
    """Save conversations as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build CSM JSONL dataset")
    parser.add_argument(
        "--source",
        choices=["synthesized", "recorded", "custom"],
        default="custom",
        help="Audio source type",
    )
    parser.add_argument("--manifest", default=None, help="Path to manifest.txt")
    parser.add_argument("--audio_dir", default=None, help="Audio file directory")
    parser.add_argument("--output_dir", default="output/", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.85, help="Train split ratio")
    parser.add_argument("--max_audio_sec", type=float, default=10.0, help="Max audio seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 5: Build CSM Dataset")
    print("=" * 50)

    # Resolve manifest and audio_dir based on source
    if args.source == "synthesized":
        manifest = args.manifest or "audio/synthesized/manifest.txt"
        audio_dir = args.audio_dir or "audio/synthesized/"
    elif args.source == "recorded":
        manifest = args.manifest or "audio/recorded/manifest.txt"
        audio_dir = args.audio_dir or "audio/recorded/"
    else:
        if not args.manifest:
            print("  Error: --manifest required for custom source")
            return
        manifest = args.manifest
        audio_dir = args.audio_dir

    print(f"  Manifest: {manifest}")
    print(f"  Audio dir: {audio_dir}")

    # Load pairs
    pairs = load_manifest(manifest, audio_dir)
    print(f"  Loaded {len(pairs)} text-audio pairs")

    if len(pairs) < 2:
        print("  Error: Need at least 2 pairs to build conversations")
        return

    # Shuffle before pairing
    random.seed(args.seed)
    random.shuffle(pairs)

    # Build conversations
    conversations = build_conversations(pairs, max_audio_sec=args.max_audio_sec)
    print(f"  Built {len(conversations)} conversations")

    if not conversations:
        print("  Error: No valid conversations built")
        return

    # Split
    random.shuffle(conversations)
    split_idx = int(len(conversations) * args.train_ratio)
    train_convos = conversations[:split_idx]
    val_convos = conversations[split_idx:]

    # Ensure at least 1 in each split
    if not val_convos and len(train_convos) > 1:
        val_convos = [train_convos.pop()]

    out_dir = Path(args.output_dir)

    train_path = out_dir / "train_conversations.jsonl"
    val_path = out_dir / "val_conversations.jsonl"

    save_jsonl(train_convos, train_path)
    save_jsonl(val_convos, val_path)

    print(f"\n  Train: {len(train_convos)} conversations -> {train_path}")
    print(f"  Val:   {len(val_convos)} conversations -> {val_path}")

    # Print stats
    total_audio_sec = 0
    for c in conversations:
        for turn in c["conversation"]:
            for content in turn["content"]:
                if content["type"] == "audio":
                    total_audio_sec += len(content["path"]) / TARGET_SR

    print(f"\n  Total audio in dataset: {total_audio_sec / 60:.1f} minutes")
    print(f"\n  To use with training pipeline:")
    print(f"    1. Copy output files to ../data/processed/")
    print(f"       cp {train_path} ../data/processed/")
    print(f"       cp {val_path} ../data/processed/")
    print(f"    2. Run training:")
    print(f"       cd .. && bash train.sh")
    print(f"\n  Or validate first:")
    print(f"    python scripts/06_validate.py --input {out_dir}")


if __name__ == "__main__":
    main()
