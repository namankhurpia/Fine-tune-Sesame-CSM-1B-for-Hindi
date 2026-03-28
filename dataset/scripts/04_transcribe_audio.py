"""
Step 4 (Alternative): Transcribe existing Hindi audio files with Whisper.

Use this when you have raw Hindi audio (from YouTube, podcasts, etc.)
but no transcription. Whisper generates Devanagari text.

Usage:
    python scripts/04_transcribe_audio.py --input_dir audio/raw/
    python scripts/04_transcribe_audio.py --input_dir audio/raw/ --whisper_model large
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 16kHz for Whisper."""
    if orig_sr == 16000:
        return audio
    target_len = int(len(audio) * 16000 / orig_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def resample_to_24k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample to 24kHz for CSM."""
    if orig_sr == TARGET_SR:
        return audio
    target_len = int(len(audio) * TARGET_SR / orig_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Transcribe Hindi audio with Whisper")
    parser.add_argument("--input_dir", required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: input_dir)")
    parser.add_argument("--whisper_model", default="medium", help="Whisper model size")
    parser.add_argument("--language", default="hi", help="Language code")
    parser.add_argument("--resample_24k", action="store_true", help="Also save 24kHz copies")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 4: Transcribe Hindi Audio")
    print("=" * 50)

    # Fix SSL certs
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

    import whisper

    print(f"  Loading Whisper {args.whisper_model}...")
    model = whisper.load_model(args.whisper_model)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    )

    print(f"  Found {len(audio_files)} audio files")

    manifest_path = output_dir / "manifest.txt"
    manifest_file = open(manifest_path, "w", encoding="utf-8")

    resampled_dir = None
    if args.resample_24k:
        resampled_dir = output_dir / "resampled_24k"
        resampled_dir.mkdir(parents=True, exist_ok=True)

    for i, audio_path in enumerate(audio_files):
        print(f"\n  [{i + 1}/{len(audio_files)}] {audio_path.name}")

        # Load audio
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Mono

        duration = len(audio) / sr
        print(f"    Duration: {duration:.1f}s, SR: {sr}Hz")

        # Transcribe (Whisper needs 16kHz float32)
        audio_16k = resample_to_16k(audio.astype(np.float32), sr)
        result = model.transcribe(
            audio_16k,
            language=args.language,
            fp16=False,
        )

        text = result["text"].strip()
        print(f"    Text: {text}")

        # Save to manifest
        manifest_file.write(f"{audio_path.name}\t{text}\n")

        # Optionally save 24kHz copy
        if resampled_dir:
            audio_24k = resample_to_24k(audio.astype(np.float32), sr)
            out_path = resampled_dir / f"{audio_path.stem}.wav"
            sf.write(str(out_path), audio_24k, TARGET_SR)

    manifest_file.close()
    print(f"\n  Transcriptions saved to {manifest_path}")
    print(f"\n  Next: python scripts/05_build_dataset.py --manifest {manifest_path} --audio_dir {input_dir}")


if __name__ == "__main__":
    main()
