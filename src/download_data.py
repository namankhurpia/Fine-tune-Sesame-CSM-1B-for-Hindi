"""
Data download and preprocessing entry point.

Usage:
    python -m src.download_data
    python -m src.download_data --config config.yaml
"""

import argparse

from .config import load_config
from .data import build_conversations, download_fleurs_hindi


def main():
    parser = argparse.ArgumentParser(description="Download & preprocess Hindi data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Step 1/2: Downloading FLEURS Hindi...")
    download_fleurs_hindi(cfg)

    print("\nStep 2/2: Building conversations...")
    build_conversations(cfg)

    print("\nData pipeline complete.")


if __name__ == "__main__":
    main()
