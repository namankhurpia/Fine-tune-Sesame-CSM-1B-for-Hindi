"""
Step 6: Validate the dataset before training.

Checks:
  - JSONL format is correct
  - Audio arrays are valid (correct dtype, reasonable length, not silent)
  - Text is non-empty Devanagari
  - Conversation structure matches CSM expectations
  - Reports statistics

Usage:
    python scripts/06_validate.py                          # Default output/
    python scripts/06_validate.py --input output/
    python scripts/06_validate.py --input ../data/processed/
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

TARGET_SR = 24000


def check_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    return bool(re.search(r"[\u0900-\u097F]", text))


def validate_conversation(convo: dict, idx: int) -> list[str]:
    """Validate a single conversation entry. Returns list of warnings."""
    warnings = []

    if "conversation" not in convo:
        warnings.append(f"[{idx}] Missing 'conversation' key")
        return warnings

    turns = convo["conversation"]
    if len(turns) != 2:
        warnings.append(f"[{idx}] Expected 2 turns, got {len(turns)}")
        return warnings

    # Check speaker 0 (context: text + audio)
    turn0 = turns[0]
    if turn0.get("role") != "0":
        warnings.append(f"[{idx}] Turn 0 role should be '0', got '{turn0.get('role')}'")

    has_text = False
    has_audio = False
    audio_duration = 0

    for content in turn0.get("content", []):
        if content["type"] == "text":
            has_text = True
            text = content["text"]
            if not text.strip():
                warnings.append(f"[{idx}] Turn 0 text is empty")
            elif not check_devanagari(text):
                warnings.append(f"[{idx}] Turn 0 text has no Devanagari: '{text[:30]}...'")

        elif content["type"] == "audio":
            has_audio = True
            audio = content["path"]
            if not isinstance(audio, list):
                warnings.append(f"[{idx}] Audio should be a list, got {type(audio)}")
                continue

            arr = np.array(audio, dtype=np.float32)
            audio_duration = len(arr) / TARGET_SR

            if len(arr) == 0:
                warnings.append(f"[{idx}] Audio is empty")
            elif audio_duration < 0.3:
                warnings.append(f"[{idx}] Audio too short: {audio_duration:.2f}s")
            elif audio_duration > 30:
                warnings.append(f"[{idx}] Audio very long: {audio_duration:.1f}s")

            # Check for silence
            rms = np.sqrt(np.mean(arr ** 2))
            if rms < 1e-5:
                warnings.append(f"[{idx}] Audio appears silent (RMS={rms:.2e})")

            # Check for clipping
            if np.max(np.abs(arr)) > 1.0:
                warnings.append(f"[{idx}] Audio has values > 1.0 (may be clipped)")

    if not has_text:
        warnings.append(f"[{idx}] Turn 0 missing text")
    if not has_audio:
        warnings.append(f"[{idx}] Turn 0 missing audio")

    # Check speaker 1 (target: text only)
    turn1 = turns[1]
    if turn1.get("role") != "1":
        warnings.append(f"[{idx}] Turn 1 role should be '1', got '{turn1.get('role')}'")

    has_target_text = False
    for content in turn1.get("content", []):
        if content["type"] == "text":
            has_target_text = True
            text = content["text"]
            if not text.strip():
                warnings.append(f"[{idx}] Turn 1 (target) text is empty")
            elif not check_devanagari(text):
                warnings.append(f"[{idx}] Turn 1 text has no Devanagari: '{text[:30]}...'")

    if not has_target_text:
        warnings.append(f"[{idx}] Turn 1 missing target text")

    # Check target_text field
    if "target_text" not in convo:
        warnings.append(f"[{idx}] Missing 'target_text' field")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Validate CSM dataset")
    parser.add_argument("--input", default="output/", help="Directory with JSONL files")
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 6: Validate Dataset")
    print("=" * 50)

    input_dir = Path(args.input)
    jsonl_files = sorted(input_dir.glob("*_conversations.jsonl"))

    if not jsonl_files:
        print(f"  No *_conversations.jsonl files found in {input_dir}/")
        sys.exit(1)

    total_convos = 0
    total_warnings = 0
    total_errors = 0
    total_audio_sec = 0
    all_text_lengths = []

    for jsonl_path in jsonl_files:
        print(f"\n  File: {jsonl_path.name}")
        print(f"  {'─' * 40}")

        conversations = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    convo = json.loads(line)
                    conversations.append(convo)
                except json.JSONDecodeError as e:
                    print(f"  ERROR line {line_num}: Invalid JSON: {e}")
                    total_errors += 1

        warnings_count = 0
        audio_durations = []
        text_lens = []

        for i, convo in enumerate(conversations):
            warnings = validate_conversation(convo, i)
            for w in warnings:
                print(f"    WARN: {w}")
                warnings_count += 1

            # Collect stats
            for turn in convo.get("conversation", []):
                for content in turn.get("content", []):
                    if content["type"] == "audio" and isinstance(content.get("path"), list):
                        dur = len(content["path"]) / TARGET_SR
                        audio_durations.append(dur)
                        total_audio_sec += dur
                    if content["type"] == "text":
                        text_lens.append(len(content.get("text", "")))

        total_convos += len(conversations)
        total_warnings += warnings_count
        all_text_lengths.extend(text_lens)

        # Per-file stats
        print(f"\n  Conversations: {len(conversations)}")
        print(f"  Warnings: {warnings_count}")
        if audio_durations:
            print(f"  Audio: {np.mean(audio_durations):.1f}s avg, "
                  f"{np.min(audio_durations):.1f}s min, "
                  f"{np.max(audio_durations):.1f}s max")
        if text_lens:
            print(f"  Text: {np.mean(text_lens):.0f} chars avg, "
                  f"{min(text_lens)} min, {max(text_lens)} max")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"  SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total conversations: {total_convos}")
    print(f"  Total audio: {total_audio_sec / 60:.1f} minutes")
    print(f"  Warnings: {total_warnings}")
    print(f"  Errors: {total_errors}")

    if total_warnings == 0 and total_errors == 0:
        print(f"\n  PASS - Dataset looks good!")
        print(f"\n  Copy to training pipeline:")
        print(f"    cp {input_dir}/*_conversations.jsonl ../data/processed/")
        print(f"    cd .. && bash train.sh --quick")
    else:
        print(f"\n  ISSUES FOUND - Review warnings above before training")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
