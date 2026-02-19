#!/usr/bin/env python3
"""
Generate analysis figures and summary tables for the knee MRI segmentation report.
Produces class distribution charts, per-class metrics bar plots, etc.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_META = "data/processed/knee_mri_cartilage/metadata.csv"
DEFAULT_METRICS = "results/knee_segmentation/metrics.csv"
DEFAULT_OUT = "results/knee_segmentation/figures"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_class_distribution(meta: pd.DataFrame, out_path: str) -> None:
    """
    Plot distribution of classes across train/val/test splits.
    """
    class_names = {0: "Background", 1: "Femoral", 2: "Tibial", 3: "Bone", 4: "Defect"}
    splits = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for ax, split in zip(axes, splits):
        subset = meta[meta["split"] == split]
        counts = subset["class_label"].value_counts().sort_index()

        labels = [class_names.get(int(c), str(c)) for c in counts.index]
        ax.bar(labels, counts.values, color="#5A9BD5")
        ax.set_title(f"{split.capitalize()} Split")
        ax.set_xlabel("Class")
        ax.set_ylabel("Sample Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metrics_bar(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Bar chart of segmentation metrics (mIoU, Dice, pixel_acc, etc.).
    """
    if metrics.empty:
        return

    row = metrics.iloc[0]
    keys = ["miou", "dice", "pixel_acc", "mean_acc", "fw_iou"]
    values = [row.get(k, 0) for k in keys]

    plt.figure(figsize=(6, 4))
    plt.bar(keys, values, color="#76B947")
    plt.ylim(0, 1)
    plt.title("Baseline Segmentation Metrics (Test Set)")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate analysis figures for knee MRI report.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Metrics CSV.")
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output figures directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output)

    if os.path.exists(args.metadata):
        meta = pd.read_csv(args.metadata)
        class_dist_path = os.path.join(args.output, "class_distribution.png")
        plot_class_distribution(meta, class_dist_path)
        print(f"Class distribution saved: {class_dist_path}")

    if os.path.exists(args.metrics):
        metrics = pd.read_csv(args.metrics)
        metrics_bar_path = os.path.join(args.output, "metrics_bar.png")
        plot_metrics_bar(metrics, metrics_bar_path)
        print(f"Metrics bar chart saved: {metrics_bar_path}")


if __name__ == "__main__":
    main()
