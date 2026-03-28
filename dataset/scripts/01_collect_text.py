"""
Step 1: Collect Hindi text from various sources.

Sources supported:
  1. Local text files (text/sample_sentences.txt or text/*.txt)
  2. IIT Bombay Hindi-English parallel corpus (Hindi side only)
  3. AI4Bharat IndicNLP sentences

Usage:
    python scripts/01_collect_text.py                          # Use sample_sentences.txt
    python scripts/01_collect_text.py --source iitb --count 500
    python scripts/01_collect_text.py --source file --input my_text.txt
"""

import argparse
import re
from pathlib import Path


def load_from_file(path: str) -> list[str]:
    """Load sentences from a text file (one per line, # comments ignored)."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
    print(f"  Loaded {len(sentences)} sentences from {path}")
    return sentences


def load_from_directory(directory: str) -> list[str]:
    """Load sentences from all .txt files in a directory."""
    sentences = []
    for txt_file in sorted(Path(directory).glob("*.txt")):
        sentences.extend(load_from_file(str(txt_file)))
    return sentences


def download_iitb(count: int = 500) -> list[str]:
    """Download Hindi sentences from IIT Bombay parallel corpus."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Install datasets: uv pip install datasets")
        return []

    print(f"  Downloading IIT Bombay Hindi corpus ({count} sentences)...")
    ds = load_dataset(
        "cfilt/iitb-english-hindi",
        split=f"train[:{count}]",
        trust_remote_code=True,
    )
    sentences = [row["translation"]["hi"] for row in ds]
    print(f"  Got {len(sentences)} Hindi sentences")
    return sentences


def download_indicnlp(count: int = 500) -> list[str]:
    """Download Hindi sentences from AI4Bharat IndicNLP corpus."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Install datasets: uv pip install datasets")
        return []

    print(f"  Downloading IndicNLP Hindi sentences ({count})...")
    try:
        ds = load_dataset(
            "ai4bharat/IndicSentenceSummarization",
            "hi",
            split=f"train[:{count}]",
            trust_remote_code=True,
        )
        sentences = [row["text"] for row in ds]
    except Exception as e:
        print(f"  IndicNLP download failed: {e}")
        print("  Falling back to IIT Bombay corpus...")
        return download_iitb(count)

    print(f"  Got {len(sentences)} Hindi sentences")
    return sentences


def clean_sentences(sentences: list[str]) -> list[str]:
    """Basic cleaning: remove empty lines, excessive whitespace, very short/long."""
    cleaned = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        # Skip too short (< 5 chars) or too long (> 300 chars)
        if len(s) < 5 or len(s) > 300:
            continue
        cleaned.append(s)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Collect Hindi text for dataset")
    parser.add_argument(
        "--source",
        choices=["file", "dir", "iitb", "indicnlp"],
        default="file",
        help="Text source (default: file)",
    )
    parser.add_argument(
        "--input",
        default="text/sample_sentences.txt",
        help="Input file or directory path",
    )
    parser.add_argument("--count", type=int, default=500, help="Number of sentences to download")
    parser.add_argument(
        "--output",
        default="text/collected_sentences.txt",
        help="Output file path",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Step 1: Collect Hindi Text")
    print("=" * 50)

    if args.source == "file":
        sentences = load_from_file(args.input)
    elif args.source == "dir":
        sentences = load_from_directory(args.input)
    elif args.source == "iitb":
        sentences = download_iitb(args.count)
    elif args.source == "indicnlp":
        sentences = download_indicnlp(args.count)

    sentences = clean_sentences(sentences)
    print(f"\n  After cleaning: {len(sentences)} sentences")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

    print(f"  Saved to {out_path}")
    print(f"\n  Next: python scripts/02_synthesize_audio.py --input {out_path}")


if __name__ == "__main__":
    main()
