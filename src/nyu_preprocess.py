from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    from .db_setup import connect_db, init_db, load_config
except ImportError:  # pragma: no cover
    from db_setup import connect_db, init_db, load_config

DEFAULT_INPUT = "data/raw/nyu_depth_v2_labeled.mat"
DEFAULT_OUTPUT_DIR = "data/processed/nyu_depth_v2"
DEFAULT_META = "data/processed/nyu_depth_v2/metadata.csv"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_mat(path: str) -> Dict[str, np.ndarray]:
    # NYU Depth V2 labeled is often v7.3 (HDF5). Try h5py first, then scipy.
    try:
        import h5py

        with h5py.File(path, "r") as f:
            images = np.array(f["images"]).transpose(0, 3, 2, 1)
            depths = np.array(f["depths"]).transpose(0, 2, 1)
            labels = np.array(f["labels"]).transpose(0, 2, 1)
            data = {"images": images, "depths": depths, "labels": labels}
            if "trainNdxs" in f and "testNdxs" in f:
                data["trainNdxs"] = np.array(f["trainNdxs"]).astype(int).ravel() - 1
                data["testNdxs"] = np.array(f["testNdxs"]).astype(int).ravel() - 1
            return data
    except Exception:
        pass

    import scipy.io

    mat = scipy.io.loadmat(path)
    images = mat["images"].transpose(3, 2, 1, 0)
    depths = mat["depths"].transpose(2, 1, 0)
    labels = mat["labels"].transpose(2, 1, 0)
    data = {"images": images, "depths": depths, "labels": labels}
    if "trainNdxs" in mat and "testNdxs" in mat:
        data["trainNdxs"] = mat["trainNdxs"].astype(int).ravel() - 1
        data["testNdxs"] = mat["testNdxs"].astype(int).ravel() - 1
    return data


def split_indices(n: int, train_ratio: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng.shuffle(indices)
    train_size = int(n * train_ratio)
    return indices[:train_size], indices[train_size:]


def save_sample(
    index: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    label: np.ndarray,
    output_dir: str,
) -> Tuple[str, str, str]:
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    depth_vis_dir = os.path.join(output_dir, "depth_vis")
    label_dir = os.path.join(output_dir, "labels")
    ensure_dir(rgb_dir)
    ensure_dir(depth_dir)
    ensure_dir(depth_vis_dir)
    ensure_dir(label_dir)

    name = f"{index:05d}"
    rgb_path = os.path.join(rgb_dir, f"{name}.png")
    depth_path = os.path.join(depth_dir, f"{name}.npy")
    label_path = os.path.join(label_dir, f"{name}.png")

    Image.fromarray(rgb.astype(np.uint8)).save(rgb_path)

    np.save(depth_path, depth.astype(np.float32))
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(os.path.join(depth_vis_dir, f"{name}.png"))

    Image.fromarray(label.astype(np.uint8)).save(label_path)

    return rgb_path, depth_path, label_path


def load_to_db(meta: pd.DataFrame, config_path: str = "config/db_config.json") -> None:
    config = load_config(config_path)
    db_type = config.get("db_type", "sqlite")

    init_db(config_path)
    conn = connect_db(config_path)
    try:
        rows = meta[
            ["rgb_path", "depth_path", "label_path", "split", "width", "height"]
        ].values.tolist()

        if db_type == "sqlite":
            conn.executemany(
                """
                INSERT INTO nyu_samples (rgb_path, depth_path, label_path, split, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return

        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO nyu_samples (rgb_path, depth_path, label_path, split, width, height)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess NYU Depth V2 labeled dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to nyu_depth_v2_labeled.mat.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV output path.")
    parser.add_argument("--limit", type=int, default=400, help="Max samples to export.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--skip-db", action="store_true", help="Skip loading into DB.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            "nyu_depth_v2_labeled.mat not found. Run nyu_download.py or place it in data/raw/."
        )

    data = load_mat(args.input)
    images = data["images"]
    depths = data["depths"]
    labels = data["labels"]

    n = images.shape[0]
    limit = min(args.limit, n)

    rng = np.random.default_rng(args.seed)
    if "trainNdxs" in data and "testNdxs" in data:
        train_idx = data["trainNdxs"]
        test_idx = data["testNdxs"]
        train_limit = max(1, int(limit * args.train_ratio))
        test_limit = max(1, limit - train_limit)
        selected = np.concatenate([train_idx[:train_limit], test_idx[:test_limit]])
    else:
        train_idx, test_idx = split_indices(n, args.train_ratio, rng)
        train_limit = max(1, int(limit * args.train_ratio))
        test_limit = max(1, limit - train_limit)
        selected = np.concatenate([train_idx[:train_limit], test_idx[:test_limit]])

    train_set = set(train_idx.tolist())
    records = []
    for out_idx, idx in enumerate(selected):
        rgb = images[idx]
        depth = depths[idx]
        label = labels[idx]

        split = "train" if idx in train_set else "test"
        rgb_path, depth_path, label_path = save_sample(out_idx, rgb, depth, label, args.output_dir)

        records.append(
            {
                "id": out_idx,
                "split": split,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "label_path": label_path,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
            }
        )

    meta = pd.DataFrame.from_records(records)
    ensure_dir(os.path.dirname(args.metadata))
    meta.to_csv(args.metadata, index=False)

    if not args.skip_db:
        load_to_db(meta)

    print(f"Saved {len(meta)} samples to {args.output_dir}")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
