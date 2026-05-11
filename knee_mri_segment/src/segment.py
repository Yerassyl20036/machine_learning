#!/usr/bin/env python3
"""
Baseline CV segmentation for knee MRI: Otsu thresholding + morphology + connected components.
Evaluates against provided masks and saves metrics + visualization samples.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

try:
    from .metrics import segmentation_metrics
except ImportError:
    from metrics import segmentation_metrics

DEFAULT_META = "data/processed/knee_mri_cartilage/metadata.csv"
DEFAULT_RESULTS = "results/knee_segmentation"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def segment_otsu_morphology(img: np.ndarray) -> np.ndarray:
    """
    Simple baseline: normalize → Otsu threshold → morphology → connected components.
    Returns multi-class segmentation map (0=bg, 1=bone, 2=cartilage, etc.).
    This is a toy example; adapt to real MRI domain knowledge.
    """
    # Normalize to 0-255
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu binary threshold
    _, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels = cv2.connectedComponents(closed)

    # Map components to simplified classes:
    # background (0), largest component → bone (1), smaller → cartilage (2), etc.
    seg = np.zeros_like(labels, dtype=np.uint8)
    component_sizes = [np.sum(labels == i) for i in range(num_labels)]

    if num_labels > 1:
        largest_idx = np.argmax(component_sizes[1:]) + 1
        seg[labels == largest_idx] = 1  # bone
        for i in range(1, num_labels):
            if i != largest_idx:
                seg[labels == i] = 2  # cartilage/other

    return seg


def segment_improved(img: np.ndarray, return_steps: bool = False):
    """
    Improved double-Otsu segmentation:
      1. Normalize → Otsu threshold → tissue vs background
      2. Second Otsu WITHIN tissue → bone (medium) vs cartilage (bright)
      3. Morphological refinement

    Returns:
      - ndarray (0=bg, 1=bone, 2=cartilage) when return_steps=False
      - (ndarray, steps_dict) when return_steps=True
        steps_dict keys: "normalized", "tissue_binary", "bone_cart_raw", "t1", "t2"
    """
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 1: Otsu — tissue vs background
    t1, binary = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Second Otsu within tissue pixels only
    fg_pixels = img_norm[binary > 0]
    if len(fg_pixels) > 100:
        t2, _ = cv2.threshold(fg_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t2 = int(t1) + 20

    # Assign classes: bone = medium tissue, cartilage = bright tissue
    bone_raw = ((binary > 0) & (img_norm < int(t2))).astype(np.uint8)
    cart_raw = ((binary > 0) & (img_norm >= int(t2))).astype(np.uint8)

    # Step 3: Morphological refinement
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    bone = cv2.morphologyEx(bone_raw, cv2.MORPH_CLOSE, k5)
    bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, k3)

    cart = cv2.morphologyEx(cart_raw, cv2.MORPH_CLOSE, k3)
    cart[bone == 1] = 0  # cartilage cannot overlap bone

    result = np.zeros(img.shape, dtype=np.uint8)
    result[bone == 1] = 1
    result[cart == 1] = 2

    if not return_steps:
        return result

    raw_vis = np.zeros_like(img_norm)
    raw_vis[bone_raw == 1] = 128   # bone raw → gray
    raw_vis[cart_raw == 1] = 255   # cartilage raw → white

    steps = {
        "normalized":     img_norm,
        "tissue_binary":  binary,
        "bone_cart_raw":  raw_vis,
        "t1": int(t1),
        "t2": int(t2),
    }
    return result, steps


def evaluate_dataset(meta: pd.DataFrame, split: str = "test", num_classes: int = 5) -> dict:
    """
    Run baseline segmentation and compute metrics against ground truth.
    """
    subset = meta[meta["split"] == split]
    all_preds = []
    all_targets = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Evaluating {split}"):
        img = np.load(row["image_path"])
        mask = np.load(row["mask_path"])

        pred = segment_otsu_morphology(img)

        # Resize pred to match mask if needed
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        all_preds.append(pred.flatten())
        all_targets.append(mask.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute segmentation metrics
    metrics = segmentation_metrics(all_preds, all_targets, num_classes=num_classes)
    return metrics


def save_sample_overlays(meta: pd.DataFrame, out_dir: str, max_samples: int = 6) -> None:
    """
    Save example RGB + GT + Pred overlays.
    """
    ensure_dir(out_dir)
    subset = meta[meta["split"] == "test"].head(max_samples)

    for idx, (_, row) in enumerate(subset.iterrows()):
        img = np.load(row["image_path"])
        mask = np.load(row["mask_path"])
        pred = segment_otsu_morphology(img)

        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Visualize
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="gray")
        plt.title("MRI Slice")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="tab10", vmin=0, vmax=4)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="tab10", vmin=0, vmax=4)
        plt.title("Prediction (Otsu+Morphology)")
        plt.axis("off")

        out_path = os.path.join(out_dir, f"knee_seg_{idx:03d}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_confusion_matrix(meta: pd.DataFrame, out_path: str, num_classes: int = 5) -> None:
    """
    Plot confusion matrix for test set.
    """
    subset = meta[meta["split"] == "test"]
    all_preds = []
    all_targets = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Computing confusion"):
        img = np.load(row["image_path"])
        mask = np.load(row["mask_path"])
        pred = segment_otsu_morphology(img)

        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        all_preds.append(pred.flatten())
        all_targets.append(mask.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(num_classes), ["BG", "Fem", "Tib", "Bone", "Def"])
    plt.yticks(range(num_classes), ["BG", "Fem", "Tib", "Bone", "Def"])

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline segmentation for knee MRI.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV.")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Results directory.")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.metadata):
        print(f"ERROR: Metadata not found: {args.metadata}")
        print("Run 'python main.py --mode knee-preprocess' first.")
        return

    meta = pd.read_csv(args.metadata)
    ensure_dir(args.results)

    print("Evaluating baseline segmentation...")
    metrics = evaluate_dataset(meta, split="test", num_classes=args.num_classes)

    # Save metrics
    df_metrics = pd.DataFrame([metrics])
    metrics_path = os.path.join(args.results, "metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Metrics saved: {metrics_path}")
    print("Metrics:", metrics)

    # Save sample overlays
    samples_dir = os.path.join(args.results, "samples")
    save_sample_overlays(meta, samples_dir, max_samples=6)
    print(f"Sample overlays saved: {samples_dir}")

    # Plot confusion matrix
    cm_path = os.path.join(args.results, "confusion_matrix.png")
    plot_confusion_matrix(meta, cm_path, num_classes=args.num_classes)
    print(f"Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    main()
