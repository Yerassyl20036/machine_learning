from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_synthetic_dataset(base_dir: Path, train_n: int = 80, test_n: int = 24, w: int = 96, h: int = 72) -> Path:
    rng = np.random.default_rng(42)

    rgb_dir = base_dir / "rgb"
    label_dir = base_dir / "labels"
    depth_dir = base_dir / "depth"
    ensure_dir(rgb_dir)
    ensure_dir(label_dir)
    ensure_dir(depth_dir)

    records = []
    total = train_n + test_n

    yy, xx = np.mgrid[0:h, 0:w]

    for i in range(total):
        # Three-class synthetic mask: background + rectangle + circle.
        label = np.zeros((h, w), dtype=np.uint8)

        x0 = int(rng.integers(w // 8, w // 3))
        y0 = int(rng.integers(h // 8, h // 3))
        x1 = int(rng.integers(w // 2, w - w // 10))
        y1 = int(rng.integers(h // 2, h - h // 10))
        label[y0:y1, x0:x1] = 1

        cx = int(rng.integers(w // 3, 2 * w // 3))
        cy = int(rng.integers(h // 3, 2 * h // 3))
        r = int(rng.integers(min(w, h) // 10, min(w, h) // 5))
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        label[circle] = 2

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = (label == 1) * 190 + (label == 2) * 40 + 20
        rgb[..., 1] = (label == 1) * 30 + (label == 2) * 180 + 20
        rgb[..., 2] = (label == 1) * 60 + (label == 2) * 70 + 25
        rgb = np.clip(rgb + rng.normal(0, 10, size=rgb.shape), 0, 255).astype(np.uint8)

        # Smooth depth with small class-dependent offsets.
        base_depth = 0.8 + 0.003 * xx + 0.004 * yy
        depth = base_depth + (label == 1) * 0.05 - (label == 2) * 0.03
        depth = depth + rng.normal(0, 0.005, size=depth.shape)
        depth = np.clip(depth, 0.1, 10.0).astype(np.float32)

        rgb_path = rgb_dir / f"rgb_{i:05d}.png"
        label_path = label_dir / f"label_{i:05d}.png"
        depth_path = depth_dir / f"depth_{i:05d}.npy"

        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(label).save(label_path)
        np.save(depth_path, depth)

        split = "train" if i < train_n else "test"
        records.append(
            {
                "id": i,
                "split": split,
                "rgb_path": str(rgb_path),
                "label_path": str(label_path),
                "depth_path": str(depth_path),
            }
        )

    meta = pd.DataFrame(records)
    meta_path = base_dir / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    return meta_path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    quick_data = project / "data" / "processed" / "quick_nyu_like"
    quick_results = project / "results" / "semester2_quick"

    ensure_dir(quick_data)
    ensure_dir(quick_results)

    meta_path = make_synthetic_dataset(quick_data)

    py = sys.executable

    baseline_dir = quick_results / "seg_baseline"
    improved_dir = quick_results / "seg_improved"
    recon_dir = quick_results / "reconstruction"
    ensure_dir(baseline_dir)
    ensure_dir(improved_dir)
    ensure_dir(recon_dir)

    run_cmd(
        [
            py,
            "-m",
            "src.nyu_segmentation",
            "--metadata",
            str(meta_path),
            "--results",
            str(baseline_dir),
            "--epochs",
            "4",
            "--batch",
            "8",
            "--size",
            "96",
            "72",
            "--train-limit",
            "80",
            "--test-limit",
            "24",
            "--skip-db",
            "--seed",
            "42",
        ],
        project,
    )

    run_cmd(
        [
            py,
            "-m",
            "src.nyu_segmentation",
            "--metadata",
            str(meta_path),
            "--results",
            str(improved_dir),
            "--epochs",
            "8",
            "--batch",
            "8",
            "--size",
            "96",
            "72",
            "--train-limit",
            "80",
            "--test-limit",
            "24",
            "--skip-db",
            "--seed",
            "42",
            "--use-aug",
            "--weighted-loss",
            "--use-scheduler",
        ],
        project,
    )

    run_cmd(
        [
            py,
            "-m",
            "src.nyu_reconstruction",
            "--metadata",
            str(meta_path),
            "--results",
            str(recon_dir),
            "--limit",
            "16",
            "--skip-db",
        ],
        project,
    )

    seg_base = pd.read_csv(baseline_dir / "metrics.csv").iloc[0].to_dict()
    seg_improved = pd.read_csv(improved_dir / "metrics.csv").iloc[0].to_dict()
    recon = pd.read_csv(recon_dir / "metrics.csv").iloc[0].to_dict()

    summary = pd.DataFrame(
        [
            {
                "experiment": "seg_baseline",
                **seg_base,
            },
            {
                "experiment": "seg_improved",
                **seg_improved,
            },
            {
                "experiment": "reconstruction_tsdf",
                **recon,
            },
        ]
    )
    summary_path = quick_results / "summary.csv"
    summary.to_csv(summary_path, index=False)

    md_path = quick_results / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Semester 2 Quick Results\n\n")
        f.write(df_to_markdown_table(summary))
        f.write("\n")

    print(f"Saved: {summary_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
