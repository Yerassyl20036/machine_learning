"""
Evaluation metrics for iris recognition:
  - Accuracy (1:N identification)
  - EER, FAR, FRR (1:1 verification)
  - ROC curve / AUC
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def identification_accuracy(y_true: list | np.ndarray,
                              y_pred: list | np.ndarray) -> float:
    """Top-1 accuracy for closed-set identification."""
    return float(accuracy_score(y_true, y_pred))


def compute_eer(genuine_scores: np.ndarray,
                impostor_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate.

    Args:
        genuine_scores:  similarity scores for same-person pairs (higher = more similar)
        impostor_scores: similarity scores for different-person pairs

    Returns:
        (eer, threshold_at_eer)
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    thresholds = np.linspace(all_scores.min(), all_scores.max(), 500)
    best_diff = float("inf")
    eer, eer_thresh = 1.0, 0.5

    for t in thresholds:
        far = float((impostor_scores >= t).sum()) / (len(impostor_scores) + 1e-10)
        frr = float((genuine_scores < t).sum()) / (len(genuine_scores) + 1e-10)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            eer = (far + frr) / 2.0
            eer_thresh = t

    return eer, eer_thresh


def far_frr_at_threshold(genuine_scores: np.ndarray,
                          impostor_scores: np.ndarray,
                          threshold: float) -> tuple[float, float]:
    """FAR and FRR at a given threshold."""
    far = float((impostor_scores >= threshold).sum()) / (len(impostor_scores) + 1e-10)
    frr = float((genuine_scores < threshold).sum()) / (len(genuine_scores) + 1e-10)
    return far, frr


def auc_score(genuine_scores: np.ndarray,
              impostor_scores: np.ndarray) -> float:
    """ROC-AUC for the verification task."""
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    return float(roc_auc_score(labels, scores))


def full_report(name: str,
                genuine_scores: np.ndarray,
                impostor_scores: np.ndarray,
                y_true: np.ndarray | None = None,
                y_pred: np.ndarray | None = None) -> dict:
    """
    Return a dict with all metrics for one algorithm.
    """
    eer, eer_t = compute_eer(genuine_scores, impostor_scores)
    far, frr = far_frr_at_threshold(genuine_scores, impostor_scores, eer_t)
    auc = auc_score(genuine_scores, impostor_scores)
    acc = identification_accuracy(y_true, y_pred) if y_true is not None else None

    report = {
        "algorithm": name,
        "EER": round(eer, 4),
        "FAR_at_EER": round(far, 4),
        "FRR_at_EER": round(frr, 4),
        "AUC": round(auc, 4),
    }
    if acc is not None:
        report["Accuracy"] = round(acc, 4)
    return report
