import os

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_seg_loss(history_csv: str, out_path: str) -> None:
    if not os.path.exists(history_csv):
        return
    df = pd.read_csv(history_csv)
    if "epoch" not in df.columns or "loss" not in df.columns:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Segmentation Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metrics_bar(seg_csv: str, rec_csv: str, out_path: str) -> None:
    if not (os.path.exists(seg_csv) and os.path.exists(rec_csv)):
        return

    seg = pd.read_csv(seg_csv).iloc[0].to_dict()
    rec = pd.read_csv(rec_csv).iloc[0].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    seg_keys = ["miou", "pixel_acc", "mean_acc", "fw_iou", "dice"]
    seg_vals = [seg.get(k, 0) for k in seg_keys]
    axes[0].bar(seg_keys, seg_vals, color="#4C72B0")
    axes[0].set_title("Segmentation Metrics")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)

    rec_keys = ["rmse", "absrel", "delta1", "delta2", "delta3"]
    rec_vals = [rec.get(k, 0) for k in rec_keys]
    axes[1].bar(rec_keys, rec_vals, color="#55A868")
    axes[1].set_title("Reconstruction Metrics")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    out_dir = "results/figures"
    ensure_dir(out_dir)

    plot_seg_loss(
        "results/segmentation/train_history.csv",
        os.path.join(out_dir, "seg_loss_curve.png"),
    )

    plot_metrics_bar(
        "results/segmentation/metrics.csv",
        "results/reconstruction/metrics.csv",
        os.path.join(out_dir, "metrics_bar.png"),
    )


if __name__ == "__main__":
    main()
