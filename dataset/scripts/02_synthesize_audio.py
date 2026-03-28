"""
Step 2: Synthesize Hindi audio from text using TTS models.

Supported TTS backends:
  - F5-Hindi (SPRINGLab/F5-Hindi-24KHz) — best quality, 24kHz native
  - MMS-TTS (facebook/mms-tts-hin) — lighter, lower quality fallback

Usage:
    python scripts/02_synthesize_audio.py                                # Default: F5-Hindi
    python scripts/02_synthesize_audio.py --tts mms                      # Use MMS-TTS
    python scripts/02_synthesize_audio.py --input text/my_sentences.txt  # Custom input
    python scripts/02_synthesize_audio.py --ref_audio speaker.wav        # Voice cloning (F5 only)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

TARGET_SR = 24000


def load_f5_hindi(model_id: str = "SPRINGLab/F5-Hindi-24KHz"):
    """Load F5-Hindi TTS model."""
    print(f"  Loading F5-Hindi from {model_id}...")

    try:
        from f5_tts.api import F5TTS

        tts = F5TTS(model_type="F5-TTS", ckpt_file=model_id)
        return ("f5", tts)
    except ImportError:
        print("  f5-tts not installed. Install with: uv pip install f5-tts")
        print("  Falling back to MMS-TTS...")
        return load_mms_hindi()
    except Exception as e:
        print(f"  F5-Hindi load failed: {e}")
        print("  Falling back to MMS-TTS...")
        return load_mms_hindi()


def load_mms_hindi(model_id: str = "facebook/mms-tts-hin"):
    """Load Facebook MMS-TTS Hindi model."""
    print(f"  Loading MMS-TTS from {model_id}...")

    from transformers import VitsModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    return ("mms", (model, tokenizer))


def synthesize_f5(tts, text: str, ref_audio: str = None, ref_text: str = None) -> np.ndarray:
    """Generate audio with F5-Hindi."""
    wav, sr, _ = tts.infer(
        ref_file=ref_audio or "",
        ref_text=ref_text or "",
        gen_text=text,
    )
    # Resample if needed
    if sr != TARGET_SR:
        wav = resample(wav, sr, TARGET_SR)
    return wav


def synthesize_mms(model_and_tok, text: str) -> np.ndarray:
    """Generate audio with MMS-TTS."""
    import torch

    model, tokenizer = model_and_tok
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    wav = output.waveform.squeeze().cpu().numpy()
    model_sr = model.config.sampling_rate

    # Resample to 24kHz
    if model_sr != TARGET_SR:
        wav = resample(wav, model_sr, TARGET_SR)
    return wav


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling via linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Synthesize Hindi audio from text")
    parser.add_argument("--input", default="text/collected_sentences.txt", help="Input text file")
    parser.add_argument("--output_dir", default="audio/synthesized/", help="Output audio directory")
    parser.add_argument("--tts", choices=["f5", "mms"], default="f5", help="TTS backend")
    parser.add_argument("--ref_audio", default=None, help="Reference audio for voice cloning (F5 only)")
    parser.add_argument("--ref_text", default=None, help="Transcription of reference audio (F5 only)")
    parser.add_argument("--start", type=int, default=0, help="Start from sentence index (for resuming)")
    parser.add_argument("--limit", type=int, default=None, help="Max sentences to process")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 2: Synthesize Hindi Audio")
    print("=" * 50)

    # Load sentences
    sentences = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)

    if args.start > 0:
        sentences = sentences[args.start:]
    if args.limit:
        sentences = sentences[: args.limit]

    print(f"  Sentences to process: {len(sentences)}")

    # Load TTS
    if args.tts == "f5":
        backend, tts = load_f5_hindi()
    else:
        backend, tts = load_mms_hindi()

    print(f"  Using backend: {backend}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also save a manifest mapping filename -> text
    manifest_path = out_dir / "manifest.txt"
    manifest_file = open(manifest_path, "a", encoding="utf-8")

    total_duration = 0.0
    failed = 0

    for i, text in enumerate(sentences):
        idx = args.start + i
        wav_path = out_dir / f"{idx:05d}.wav"

        if wav_path.exists():
            print(f"  [{idx}] Already exists, skipping")
            continue

        try:
            t0 = time.time()
            if backend == "f5":
                audio = synthesize_f5(tts, text, args.ref_audio, args.ref_text)
            else:
                audio = synthesize_mms(tts, text)

            # Save
            sf.write(str(wav_path), audio, TARGET_SR)
            duration = len(audio) / TARGET_SR
            total_duration += duration
            elapsed = time.time() - t0

            print(f"  [{idx}] {duration:.1f}s audio, {elapsed:.1f}s gen | {text[:50]}...")
            manifest_file.write(f"{wav_path.name}\t{text}\n")

        except Exception as e:
            print(f"  [{idx}] FAILED: {e} | {text[:50]}...")
            failed += 1
            continue

    manifest_file.close()

    print(f"\n  Done! {len(sentences) - failed}/{len(sentences)} succeeded")
    print(f"  Total audio: {total_duration / 60:.1f} minutes")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Next: python scripts/05_build_dataset.py --source synthesized")


if __name__ == "__main__":
    main()
