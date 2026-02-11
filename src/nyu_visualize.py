import argparse
import os

import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_SAMPLE = "00240"
DEFAULT_OUT_DIR = "results/figures"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path: str) -> Image.Image:
    return Image.open(path)


def make_reconstruction_panel(sample: str, out_dir: str) -> str:
    depth_path = f"results/reconstruction/samples/depth_{sample}.png"
    filtered_path = f"results/reconstruction/samples/depth_filtered_{sample}.png"
    cmap_path = f"results/reconstruction/samples/depth_filtered_{sample}_cmap.png"
    mesh_path = f"results/reconstruction/samples/mesh_{sample}.png"

    images = [
        ("Depth (raw)", depth_path),
        ("Depth (filtered)", filtered_path),
        ("Depth (cmap)", cmap_path),
        ("Mesh render", mesh_path),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (title, path) in zip(axes.flat, images):
        img = load_image(path)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"reconstruction_compare_{sample}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def make_segmentation_panel(sample: str, out_dir: str) -> str:
    seg_path = f"results/segmentation/samples/seg_{sample}.png"
    fig, ax = plt.subplots(figsize=(7, 3))
    img = load_image(seg_path)
    ax.imshow(img)
    ax.set_title("Segmentation: RGB / GT / Pred")
    ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"segmentation_compare_{sample}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build comparison panels for report.")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE, help="Sample id, e.g. 00240.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    seg_out = make_segmentation_panel(args.sample, args.out_dir)
    rec_out = make_reconstruction_panel(args.sample, args.out_dir)
    print(f"Saved: {seg_out}")
    print(f"Saved: {rec_out}")


if __name__ == "__main__":
    main()
