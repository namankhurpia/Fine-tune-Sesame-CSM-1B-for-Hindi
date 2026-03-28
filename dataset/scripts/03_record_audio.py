"""
Step 3 (Alternative): Record your own Hindi audio via microphone.

This script shows sentences one at a time and records your voice.
Press Enter to start recording, Enter again to stop.

Usage:
    python scripts/03_record_audio.py                               # Default sentences
    python scripts/03_record_audio.py --input text/my_sentences.txt # Custom text
    python scripts/03_record_audio.py --start 20                    # Resume from sentence 20
"""

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


def record_audio(sr: int = 24000, max_duration: float = 15.0) -> np.ndarray:
    """Record audio from microphone until Enter is pressed or max_duration."""
    try:
        import sounddevice as sd
    except ImportError:
        print("  Install sounddevice: uv pip install sounddevice")
        sys.exit(1)

    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [Audio status: {status}]")
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )

    input("  Press ENTER to start recording...")
    print("  Recording... press ENTER to stop.")
    stream.start()

    # Wait for Enter or timeout
    def wait_for_enter():
        input()
        stop_event.set()

    thread = threading.Thread(target=wait_for_enter, daemon=True)
    thread.start()
    thread.join(timeout=max_duration)
    stop_event.set()

    stream.stop()
    stream.close()

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()

    # Trim silence from start/end (simple energy-based)
    threshold = 0.01
    above = np.abs(audio) > threshold
    if above.any():
        first = np.argmax(above)
        last = len(above) - np.argmax(above[::-1])
        # Add small padding
        pad = int(0.1 * sr)
        first = max(0, first - pad)
        last = min(len(audio), last + pad)
        audio = audio[first:last]

    return audio


def main():
    parser = argparse.ArgumentParser(description="Record Hindi audio via microphone")
    parser.add_argument("--input", default="text/sample_sentences.txt", help="Text file with sentences")
    parser.add_argument("--output_dir", default="audio/recorded/", help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start from sentence index")
    parser.add_argument("--max_duration", type=float, default=15.0, help="Max recording seconds")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 3: Record Hindi Audio")
    print("=" * 50)
    print("  For each sentence:")
    print("    1. Read the Hindi text")
    print("    2. Press Enter to start recording")
    print("    3. Speak the sentence clearly")
    print("    4. Press Enter to stop")
    print("    Type 'skip' to skip, 'quit' to exit")
    print("=" * 50)

    # Load sentences
    sentences = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.txt"
    manifest_file = open(manifest_path, "a", encoding="utf-8")

    recorded = 0
    for i in range(args.start, len(sentences)):
        text = sentences[i]
        wav_path = out_dir / f"{i:05d}.wav"

        print(f"\n  [{i + 1}/{len(sentences)}]")
        print(f"  Text: {text}")

        if wav_path.exists():
            resp = input("  Already recorded. Re-record? (y/n/skip/quit): ").strip().lower()
            if resp == "quit":
                break
            if resp not in ("y", "yes"):
                continue

        cmd = input("  Ready? (Enter=record, skip, quit): ").strip().lower()
        if cmd == "quit":
            break
        if cmd == "skip":
            continue

        audio = record_audio(sr=TARGET_SR, max_duration=args.max_duration)

        if len(audio) < TARGET_SR * 0.3:  # Less than 0.3s
            print("  Too short! Skipping.")
            continue

        sf.write(str(wav_path), audio, TARGET_SR)
        duration = len(audio) / TARGET_SR
        print(f"  Saved: {wav_path} ({duration:.1f}s)")
        manifest_file.write(f"{wav_path.name}\t{text}\n")
        recorded += 1

    manifest_file.close()
    print(f"\n  Recorded {recorded} audio files in {out_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"\n  Next: python scripts/05_build_dataset.py --source recorded")


if __name__ == "__main__":
    main()
