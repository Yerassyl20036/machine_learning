#!/usr/bin/env python3
"""
Preprocess knee MRI cartilage dataset and build train/test split metadata.
Assumes data is in 'data/raw/knee_mri_cartilage/Osteoarthritis dataset'.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DEFAULT_RAW = "data/raw/knee_mri_cartilage/Osteoarthritis dataset"
DEFAULT_OUT = "data/processed/knee_mri_cartilage"
DEFAULT_META = "data/processed/knee_mri_cartilage/metadata.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def collect_samples(raw_root: str, limit_per_split: int = 200) -> pd.DataFrame:
    """
    Collect PNG samples from nested class directories train/val/test/{0,1,2,3,4}.
    Returns DataFrame with columns [split, class_label, image_path, mask_path].
    """
    rows = []
    splits = ["train", "val", "test"]
    class_ids = [0, 1, 2, 3, 4]  # background, femoral, tibial, bone, defect

    for split in splits:
        split_dir = Path(raw_root) / split
        if not split_dir.exists():
            continue
        count = 0
        for class_id in class_ids:
            class_dir = split_dir / str(class_id)
            if not class_dir.exists():
                continue
            images = sorted(class_dir.glob("*.png"))
            for img_path in images:
                # In this dataset, image == mask (single-channel .png with class labels)
                # We'll extract the data to process as needed.
                # For simplicity, assume every PNG is a mask+image combined or just mask.
                # Actually this dataset may store slices separately; adapt if structure differs.
                # Here assume image == mask (grayscale label map), we'll load and process.
                rows.append(
                    {
                        "split": split,
                        "class_label": class_id,
                        "image_path": str(img_path.relative_to(raw_root)),
                        "mask_path": str(img_path.relative_to(raw_root)),
                    }
                )
                count += 1
                if count >= limit_per_split:
                    break
            if count >= limit_per_split:
                break

    return pd.DataFrame(rows)


def export_slices(df: pd.DataFrame, raw_root: str, out_dir: str) -> pd.DataFrame:
    """
    Copy or convert slices to output directory for easy access.
    We'll save normalized images + masks as .npy for CV pipeline.
    """
    ensure_dir(out_dir)
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    ensure_dir(img_dir)
    ensure_dir(mask_dir)

    updated_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting slices"):
        src = Path(raw_root) / row["image_path"]
        if not src.exists():
            continue

        # Load the PNG (assuming it's a grayscale or label image; adapt to MRI intensity if needed)
        img = np.array(Image.open(src))

        # In the OAI dataset, the slice might be an intensity image or a mask.
        # For this pipeline, we'll treat the labeled PNG as a mask.
        # If you need intensity (the actual MRI scan), you'll need to locate paired .nii or other format.
        # We'll work with what's available: assume PNG == mask (label).

        # Normalize and save as image (for visualization):
        # For simplicity, copy mask to both image and mask (you can replace with real intensity if available)
        img_norm = img.astype(np.float32)

        out_idx = int(idx)
        img_out = os.path.join(img_dir, f"{out_idx:05d}.npy")
        mask_out = os.path.join(mask_dir, f"{out_idx:05d}.npy")

        np.save(img_out, img_norm)
        np.save(mask_out, img.astype(np.uint8))

        updated_rows.append(
            {
                "id": out_idx,
                "split": row["split"],
                "class_label": row["class_label"],
                "image_path": img_out,
                "mask_path": mask_out,
            }
        )

    return pd.DataFrame(updated_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess knee MRI dataset.")
    parser.add_argument("--raw", default=DEFAULT_RAW, help="Raw dataset root.")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV path.")
    parser.add_argument("--limit", type=int, default=200, help="Limit samples per split.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.raw):
        print(f"ERROR: Raw directory not found: {args.raw}")
        print("Run 'python main.py --mode knee-download' or download manually.")
        return

    df = collect_samples(args.raw, args.limit)
    print(f"Collected {len(df)} samples from {args.raw}")

    df_exported = export_slices(df, args.raw, args.output)

    ensure_dir(os.path.dirname(args.metadata))
    df_exported.to_csv(args.metadata, index=False)
    print(f"Metadata saved: {args.metadata}")


if __name__ == "__main__":
    main()
