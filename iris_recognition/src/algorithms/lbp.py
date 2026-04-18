"""
Algorithm 2 – LBP (Local Binary Patterns)
Features: uniform LBP histogram of the normalized iris image.
Matching: Chi-squared distance between histograms.
"""

import numpy as np
from skimage.feature import local_binary_pattern


_LBP_RADIUS = 2
_LBP_POINTS = 16   # 8 * radius
_LBP_METHOD = "uniform"


def extract(norm_img: np.ndarray,
            radius: int = _LBP_RADIUS,
            n_points: int = _LBP_POINTS,
            n_bins: int = 64) -> np.ndarray:
    """
    Compute LBP histogram feature vector.

    Args:
        norm_img: 2-D uint8 normalized iris image
        radius:   LBP neighbourhood radius
        n_points: number of circularly symmetric neighbour points
        n_bins:   number of histogram bins

    Returns:
        1-D float32 normalized histogram
    """
    lbp = local_binary_pattern(norm_img, n_points, radius, method=_LBP_METHOD)
    # n_points + 2 uniform patterns
    hist, _ = np.histogram(lbp.ravel(),
                            bins=n_bins,
                            range=(0, n_points + 2),
                            density=False)
    hist = hist.astype(np.float32)
    # L1 normalize
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def chi_squared_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Chi-squared distance between two histograms."""
    denom = h1 + h2 + 1e-10
    return float(0.5 * np.sum((h1 - h2) ** 2 / denom))


def similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Similarity score in [0, 1] (1 = identical)."""
    d = chi_squared_distance(feat_a, feat_b)
    # Convert distance to similarity via exponential decay
    return float(np.exp(-d))
