from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import open3d as o3d

try:
    from .db_setup import connect_db, init_db, load_config
    from .metrics import depth_metrics
except ImportError:  # pragma: no cover
    from db_setup import connect_db, init_db, load_config
    from metrics import depth_metrics

DEFAULT_META = "data/processed/nyu_depth_v2/metadata.csv"
DEFAULT_RESULTS = "results/reconstruction"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def depth_to_pointcloud(depth: np.ndarray, intrinsics: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.PointCloud:
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsics,
        depth_scale=1.0,
        depth_trunc=10.0,
        stride=1,
    )
    return pcd


def run_tsdf(depth: np.ndarray, intrinsics: o3d.camera.PinholeCameraIntrinsic) -> o3d.geometry.TriangleMesh:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    rgb_dummy = o3d.geometry.Image(np.zeros((*depth.shape, 3), dtype=np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_dummy,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )
    volume.integrate(rgbd, intrinsics, np.eye(4))
    return volume.extract_triangle_mesh()


def filter_depth(depth: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    filtered = gaussian_filter(depth, sigma=sigma)
    filtered[depth <= 0] = 0
    return filtered


def save_depth_vis(depth: np.ndarray, out_path: str) -> None:
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(out_path)


def save_depth_colormap(depth: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(depth, cmap="viridis")
    ax.set_title("Depth (m)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def render_mesh_png(mesh: o3d.geometry.TriangleMesh, out_path: str) -> None:
    import matplotlib.pyplot as plt

    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        return

    if vertices.shape[0] > 20000:
        idx = np.random.choice(vertices.shape[0], 20000, replace=False)
        vertices = vertices[idx]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=1,
        c=vertices[:, 2],
        cmap="viridis",
    )
    ax.view_init(elev=25, azim=35)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_db_results(metrics: dict, config_path: str = "config/db_config.json") -> None:
    config = load_config(config_path)
    db_type = config.get("db_type", "sqlite")
    init_db(config_path)
    conn = connect_db(config_path)
    try:
        if db_type == "sqlite":
            conn.execute(
                """
                INSERT INTO reconstruction_results
                (sample_id, method, rmse, absrel, delta1, delta2, delta3)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    None,
                    "tsdf-gaussian",
                    metrics["rmse"],
                    metrics["absrel"],
                    metrics["delta1"],
                    metrics["delta2"],
                    metrics["delta3"],
                ),
            )
            conn.commit()
            return

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO reconstruction_results
                (sample_id, method, rmse, absrel, delta1, delta2, delta3)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    None,
                    "tsdf-gaussian",
                    metrics["rmse"],
                    metrics["absrel"],
                    metrics["delta1"],
                    metrics["delta2"],
                    metrics["delta3"],
                ),
            )
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline 3D reconstruction using TSDF.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV path.")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Results directory.")
    parser.add_argument("--limit", type=int, default=30, help="Max samples to process.")
    parser.add_argument("--skip-db", action="store_true", help="Skip writing DB metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    meta = pd.read_csv(args.metadata)
    test_meta = meta[meta["split"] == "test"].head(args.limit)

    ensure_dir(args.results)
    samples_dir = os.path.join(args.results, "samples")
    ensure_dir(samples_dir)

    metrics_list = []
    for _, row in test_meta.iterrows():
        depth = np.load(row["depth_path"]).astype(np.float32)
        h, w = depth.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, 525.0, 525.0, w / 2.0, h / 2.0)

        filtered = filter_depth(depth)
        metrics = depth_metrics(filtered, depth)
        metrics_list.append(metrics)

        mesh = run_tsdf(filtered, intrinsics)
        mesh.compute_vertex_normals()

        base = os.path.splitext(os.path.basename(row["depth_path"]))[0]
        mesh_path = os.path.join(samples_dir, f"mesh_{base}.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        render_mesh_png(mesh, os.path.join(samples_dir, f"mesh_{base}.png"))

        save_depth_vis(depth, os.path.join(samples_dir, f"depth_{base}.png"))
        save_depth_vis(filtered, os.path.join(samples_dir, f"depth_filtered_{base}.png"))
        save_depth_colormap(filtered, os.path.join(samples_dir, f"depth_filtered_{base}_cmap.png"))

    if metrics_list:
        avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    else:
        avg_metrics = {"rmse": 0, "absrel": 0, "delta1": 0, "delta2": 0, "delta3": 0}

    pd.DataFrame([avg_metrics]).to_csv(os.path.join(args.results, "metrics.csv"), index=False)

    if not args.skip_db:
        write_db_results(avg_metrics)

    print("Reconstruction metrics:", avg_metrics)


if __name__ == "__main__":
    main()
