from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from .db_setup import connect_db, init_db, load_config
    from .metrics import segmentation_metrics
except ImportError:  # pragma: no cover
    from db_setup import connect_db, init_db, load_config
    from metrics import segmentation_metrics

DEFAULT_META = "data/processed/nyu_depth_v2/metadata.csv"
DEFAULT_RESULTS = "results/segmentation"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class Sample:
    sample_id: int
    rgb_path: str
    label_path: str


class NYUSegDataset(Dataset):
    def __init__(self, samples: List[Sample], size: Tuple[int, int]) -> None:
        self.samples = samples
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        rgb = Image.open(sample.rgb_path).convert("RGB")
        label = Image.open(sample.label_path)

        rgb = rgb.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        rgb = torch.from_numpy(np.array(rgb)).float().permute(2, 0, 1) / 255.0
        label = torch.from_numpy(np.array(label)).long()
        return sample.sample_id, rgb, label


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def build_samples(meta: pd.DataFrame, split: str, limit: int) -> List[Sample]:
    filtered = meta[meta["split"] == split]
    if limit:
        filtered = filtered.head(limit)
    return [
        Sample(int(row["id"]), row["rgb_path"], row["label_path"])
        for _, row in filtered.iterrows()
    ]


def infer_num_classes(meta: pd.DataFrame, sample_count: int = 50) -> int:
    labels = []
    for _, row in meta.head(sample_count).iterrows():
        label = np.array(Image.open(row["label_path"]))
        labels.append(label.max())
    return int(max(labels)) + 1


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for _, rgb, label in loader:
        rgb = rgb.to(device)
        label = label.to(device)
        logits = model(rgb)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device, num_classes: int) -> dict:
    model.eval()
    metrics_list = []
    with torch.no_grad():
        for _, rgb, label in loader:
            rgb = rgb.to(device)
            label = label.to(device)
            logits = model(rgb)
            pred = torch.argmax(logits, dim=1)
            metrics = segmentation_metrics(
                pred.cpu().numpy(),
                label.cpu().numpy(),
                num_classes=num_classes,
            )
            metrics_list.append(metrics)

    if not metrics_list:
        return {"miou": 0, "pixel_acc": 0, "mean_acc": 0, "fw_iou": 0, "dice": 0}

    agg = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    return agg


def save_overlays(model, loader, device, out_dir: str, max_items: int = 6) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    model.eval()
    count = 0
    with torch.no_grad():
        for sample_id, rgb, label in loader:
            rgb = rgb.to(device)
            logits = model(rgb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

            rgb_img = rgb.cpu().numpy()[0].transpose(1, 2, 0)
            label_img = label.numpy()[0]

            plt.figure(figsize=(9, 3))
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_img)
            plt.title("RGB")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(label_img, cmap="tab20")
            plt.title("GT")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred, cmap="tab20")
            plt.title("Pred")
            plt.axis("off")

            out_path = os.path.join(out_dir, f"seg_{int(sample_id[0]):05d}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

            count += 1
            if count >= max_items:
                return


def write_db_results(meta: pd.DataFrame, metrics: dict, config_path: str = "config/db_config.json") -> None:
    config = load_config(config_path)
    db_type = config.get("db_type", "sqlite")
    init_db(config_path)
    conn = connect_db(config_path)
    try:
        if db_type == "sqlite":
            conn.execute(
                """
                INSERT INTO segmentation_results
                (sample_id, model, miou, pixel_acc, mean_acc, fw_iou, dice)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    None,
                    "tiny-unet",
                    metrics["miou"],
                    metrics["pixel_acc"],
                    metrics["mean_acc"],
                    metrics["fw_iou"],
                    metrics["dice"],
                ),
            )
            conn.commit()
            return

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO segmentation_results
                (sample_id, model, miou, pixel_acc, mean_acc, fw_iou, dice)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    None,
                    "tiny-unet",
                    metrics["miou"],
                    metrics["pixel_acc"],
                    metrics["mean_acc"],
                    metrics["fw_iou"],
                    metrics["dice"],
                ),
            )
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny UNet on NYU Depth V2 labels.")
    parser.add_argument("--metadata", default=DEFAULT_META, help="Metadata CSV path.")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Results directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--size", type=int, nargs=2, default=[160, 120], help="Resize (width height).")
    parser.add_argument("--train-limit", type=int, default=240, help="Train samples limit.")
    parser.add_argument("--test-limit", type=int, default=120, help="Test samples limit.")
    parser.add_argument("--skip-db", action="store_true", help="Skip writing DB metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    meta = pd.read_csv(args.metadata)

    num_classes = infer_num_classes(meta)
    train_samples = build_samples(meta, "train", args.train_limit)
    test_samples = build_samples(meta, "test", args.test_limit)

    train_ds = NYUSegDataset(train_samples, tuple(args.size))
    test_ds = NYUSegDataset(test_samples, tuple(args.size))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    device = torch.device("cpu")
    model = TinyUNet(3, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ensure_dir(args.results)

    history = []
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        history.append({"epoch": epoch + 1, "loss": loss})

    metrics = evaluate(model, test_loader, device, num_classes)

    pd.DataFrame(history).to_csv(os.path.join(args.results, "train_history.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.results, "metrics.csv"), index=False)

    save_overlays(model, test_loader, device, os.path.join(args.results, "samples"))

    if not args.skip_db:
        write_db_results(meta, metrics)

    print("Segmentation metrics:", metrics)


if __name__ == "__main__":
    main()
