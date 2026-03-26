"""
Evaluation entry point. Whisper ASR round-trip WER on generated audio.

Usage:
    python -m src.evaluate
    python -m src.evaluate --whisper_model medium
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
from jiwer import wer

from .config import load_config

# Ensure SSL certs work for Whisper model download
os.environ.setdefault("SSL_CERT_FILE", "")
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass


def load_audio_sf(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV with soundfile and resample (no ffmpeg required)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        target_len = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def evaluate_files(audio_dir: Path, prefix: str, ground_truth: list, whisper_model):
    """Evaluate audio files matching prefix against ground truth."""
    files = sorted(audio_dir.glob(f"{prefix}_*.wav"))
    if not files:
        print(f"  No {prefix}_*.wav files found.")
        return []

    results = []
    for i, path in enumerate(files):
        if i >= len(ground_truth):
            break

        audio = load_audio_sf(str(path), target_sr=16000)
        audio = whisper.pad_or_trim(torch.from_numpy(audio)).numpy()
        result = whisper_model.transcribe(audio, language="hi")
        transcription = result["text"].strip()
        reference = ground_truth[i]
        error = wer(reference, transcription) if transcription else 1.0

        results.append({"file": path.name, "ref": reference, "hyp": transcription, "wer": error})
        print(f"  {path.name}:")
        print(f"    Ref: {reference}")
        print(f"    Hyp: {transcription}")
        print(f"    WER: {error:.0%}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hindi audio via Whisper")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--whisper_model", type=str, help="Override whisper model size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_dir = Path(cfg["evaluation"]["audio_dir"])
    whisper_size = args.whisper_model or cfg["evaluation"]["whisper_model"]
    ground_truth = cfg["inference"]["prompts"]

    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}. Run inference first.")
        return

    print(f"Loading Whisper {whisper_size}...")
    w = whisper.load_model(whisper_size)

    print("\n=== Baseline ===")
    bl = evaluate_files(audio_dir, "baseline", ground_truth, w)

    print("\n=== Fine-tuned ===")
    ft = evaluate_files(audio_dir, "finetuned", ground_truth, w)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, res in [("Baseline", bl), ("Fine-tuned", ft)]:
        if res:
            avg = sum(r["wer"] for r in res) / len(res)
            print(f"  {label:12s}  avg WER = {avg:.0%}  ({len(res)} samples)")
    if bl and ft:
        delta = (sum(r["wer"] for r in ft) / len(ft)) - (sum(r["wer"] for r in bl) / len(bl))
        print(f"  {'Delta':12s}  {delta:+.0%} ({'worse' if delta > 0 else 'better'})")
    print("\nManual listening recommended — WER doesn't capture prosody/naturalness.")


if __name__ == "__main__":
    main()
