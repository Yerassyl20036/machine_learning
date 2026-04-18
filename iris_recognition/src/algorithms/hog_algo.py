"""
Algorithm 3 – HOG (Histogram of Oriented Gradients)
Features: HOG descriptor computed on the normalized iris image.
Matching: cosine similarity between HOG vectors.
"""

import numpy as np
from skimage.feature import hog


def extract(norm_img: np.ndarray,
            orientations: int = 9,
            pixels_per_cell: tuple = (8, 8),
            cells_per_block: tuple = (2, 2)) -> np.ndarray:
    """
    Compute HOG feature vector from normalized iris image.

    Args:
        norm_img:        2-D uint8 normalized iris image
        orientations:    number of gradient orientation bins
        pixels_per_cell: cell size in pixels
        cells_per_block: block size in cells

    Returns:
        1-D float32 HOG feature vector
    """
    feat = hog(
        norm_img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity in [0, 1]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Similarity score in [0, 1] (1 = identical). Clip to [0,1]."""
    s = cosine_similarity(feat_a, feat_b)
    return float(np.clip(s, 0.0, 1.0))
