"""
Algorithm 4 – ORB (Oriented FAST and Rotated BRIEF) keypoint matching
Features: ORB descriptors aggregated into a fixed-size histogram.
Matching: descriptor match ratio (Hamming distance, ratio test).
"""

import numpy as np
import cv2


_N_KEYPOINTS = 500
_HIST_BINS = 64


def extract(norm_img: np.ndarray,
            n_keypoints: int = _N_KEYPOINTS,
            hist_bins: int = _HIST_BINS) -> np.ndarray:
    """
    Detect ORB keypoints and aggregate descriptors into a fixed-length vector.

    Uses descriptor statistics (mean of each descriptor dimension) to produce
    a fixed-size feature vector regardless of detected keypoint count.

    Args:
        norm_img:    2-D uint8 normalized iris image
        n_keypoints: max ORB keypoints to detect
        hist_bins:   number of bins for the aggregated histogram

    Returns:
        1-D float32 feature vector of length 32 (mean of ORB descriptors)
        or zeros if no keypoints detected.
    """
    # Enhance contrast so ORB can detect keypoints on flat iris texture
    enhanced = cv2.equalizeHist(norm_img)
    orb = cv2.ORB_create(nfeatures=n_keypoints, fastThreshold=5, edgeThreshold=5)
    kps, descs = orb.detectAndCompute(enhanced, None)

    if descs is None or len(descs) == 0:
        # Fallback: FAST keypoints with lower threshold
        fast = cv2.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True)
        kps = fast.detect(enhanced, None)
        if not kps:
            return np.zeros(32, dtype=np.float32)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() if hasattr(cv2, 'xfeatures2d') else None
        if brief is None:
            # Use pixel intensities at keypoint locations as fallback
            coords = np.array([[int(k.pt[0]), int(k.pt[1])] for k in kps[:32]])
            feat = enhanced[coords[:, 1].clip(0, enhanced.shape[0]-1),
                            coords[:, 0].clip(0, enhanced.shape[1]-1)].astype(np.float32)
            feat = np.pad(feat, (0, max(0, 32 - len(feat))))[:32]
        else:
            kps, descs = brief.compute(enhanced, kps)
            if descs is None or len(descs) == 0:
                return np.zeros(32, dtype=np.float32)
            feat = descs.astype(np.float32).mean(axis=0)
    else:
        feat = descs.astype(np.float32).mean(axis=0)

    # Mean descriptor across all keypoints → fixed 32-dim vector
    feat = descs.astype(np.float32).mean(axis=0)

    # L2 normalize
    norm_val = np.linalg.norm(feat)
    if norm_val > 1e-10:
        feat /= norm_val

    return feat


def similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Cosine similarity between aggregated ORB descriptors."""
    n1 = np.linalg.norm(feat_a)
    n2 = np.linalg.norm(feat_b)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.clip(np.dot(feat_a, feat_b) / (n1 * n2), 0.0, 1.0))
