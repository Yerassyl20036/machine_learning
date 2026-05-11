from __future__ import annotations

import numpy as np


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict:
    if pred.shape != target.shape:
        raise ValueError("Pred and target must have the same shape.")

    pred = pred.astype(np.int64)
    target = target.astype(np.int64)

    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target.flatten(), pred.flatten()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf[t, p] += 1

    true_pos = np.diag(conf)
    false_pos = conf.sum(axis=0) - true_pos
    false_neg = conf.sum(axis=1) - true_pos

    union = true_pos + false_pos + false_neg
    iou = np.divide(true_pos, union, out=np.zeros_like(true_pos, dtype=float), where=union != 0)
    miou = float(np.mean(iou))

    pixel_acc = float(true_pos.sum() / max(conf.sum(), 1))
    class_acc = np.divide(true_pos, conf.sum(axis=1), out=np.zeros_like(true_pos, dtype=float), where=conf.sum(axis=1) != 0)
    mean_acc = float(np.mean(class_acc))

    freq = conf.sum(axis=1) / max(conf.sum(), 1)
    fw_iou = float((freq * iou).sum())

    denom = 2 * true_pos + false_pos + false_neg
    dice = np.divide(2 * true_pos, denom, out=np.zeros_like(true_pos, dtype=float), where=denom != 0)
    dice_score = float(np.mean(dice))

    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "mean_acc": mean_acc,
        "fw_iou": fw_iou,
        "dice": dice_score,
    }
