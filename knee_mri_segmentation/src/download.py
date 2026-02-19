#!/usr/bin/env python3
"""
Download knee MRI cartilage segmentation dataset from Kaggle.
Requires `kaggle` CLI configured (or manual download).
"""

import argparse
import os
import shutil
import subprocess
import sys

DEFAULT_RAW = "data/raw/knee_mri_cartilage"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_kaggle_dataset(dataset: str, out_dir: str) -> None:
    """
    Download Kaggle dataset via CLI (if available).
    Falls back to manual instructions on error.
    """
    ensure_dir(out_dir)
    try:
        cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", out_dir, "--unzip"]
        subprocess.run(cmd, check=True)
        print(f"Dataset {dataset} downloaded and extracted to {out_dir}")
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        print(f"ERROR: Kaggle CLI unavailable or download failed: {err}")
        print("\n=== Manual Download Instructions ===")
        print(f"1. Visit https://www.kaggle.com/datasets/{dataset}")
        print("2. Download the dataset ZIP manually.")
        print(f"3. Extract contents to: {os.path.abspath(out_dir)}")
        print("4. Re-run the preprocessor after extraction.\n")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download knee MRI cartilage dataset.")
    parser.add_argument("--output", default=DEFAULT_RAW, help="Output directory.")
    parser.add_argument(
        "--dataset",
        default="ujjwalsinha01/3d-knee-mri-cartilage-segmentation",
        help="Kaggle dataset identifier.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_kaggle_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
