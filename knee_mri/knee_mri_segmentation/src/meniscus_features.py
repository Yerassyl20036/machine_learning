#!/usr/bin/env python3
"""
Feature extraction for meniscus tear binary classification.

20 features total:
  - 15 generic texture/statistical features (work on any grayscale MRI)
  - 5 meniscus-specific features targeting signal voids, regional asymmetry,
    center vs peripheral brightness, and local inhomogeneity

Importable by both train_meniscus.py and app.py.
"""

import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

TARGET_SIZE = 256  # all images resized to this before feature extraction

FEATURE_NAMES = [
    # generic
    "mean_intensity",
    "std_intensity",
    "skew_intensity",
    "kurt_intensity",
    "laplacian_var",
    "fft_mean",
    "edge_density",
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
    "entropy",
    "mean_gradient",
    "hist_peak_pos",
    "hist_spread",
    # meniscus-specific
    "signal_void_ratio",
    "regional_asymmetry",
    "center_brightness",
    "peripheral_brightness",
    "local_inhomogeneity",
]


def load_gray(path: str) -> np.ndarray:
    """Load image as uint8 grayscale, resized to TARGET_SIZE x TARGET_SIZE."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extract 20 features from a uint8 grayscale image (any size).
    Returns a 1-D float32 array of length 20.

    If img is not already 256x256, it is resized internally so the
    regional features are consistent across images.
    """
    if img.shape != (TARGET_SIZE, TARGET_SIZE):
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

    img_f = img.astype(np.float32)
    flat = img_f.ravel()

    # ── 1-4: Intensity statistics ──────────────────────────────────────────
    mean_int = float(np.mean(flat))
    std_int = float(np.std(flat))
    skew_int = float(stats.skew(flat))
    kurt_int = float(stats.kurtosis(flat))

    # ── 5: Laplacian variance (sharpness / structural detail) ──────────────
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap_var = float(lap.var())

    # ── 6: FFT mean (frequency content) ────────────────────────────────────
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(img_f)))
    fft_mean = float(np.mean(np.log1p(fft_mag)))

    # ── 7: Edge density (Canny) ─────────────────────────────────────────────
    edges = cv2.Canny(img, 30, 100)
    edge_density = float(edges.mean())

    # ── 8-11: GLCM texture (contrast, homogeneity, energy, correlation) ────
    # Quantize to 64 levels to keep GLCM tractable
    img_q = (img // 4).astype(np.uint8)
    glcm = graycomatrix(img_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2],
                        levels=64, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, "contrast").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    correlation = float(graycoprops(glcm, "correlation").mean())

    # ── 12: Entropy ─────────────────────────────────────────────────────────
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))

    # ── 13: Mean gradient magnitude ─────────────────────────────────────────
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mean_grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    # ── 14-15: Histogram peak and spread ────────────────────────────────────
    hist_peak_pos = float(np.argmax(hist)) / 255.0
    q25, q75 = float(np.percentile(flat, 25)), float(np.percentile(flat, 75))
    hist_spread = (q75 - q25) / 255.0

    # ── 16: Signal void ratio (dark pixels < 30 → tears appear dark in T2) ─
    signal_void_ratio = float((img < 30).sum()) / flat.size

    # ── 17: Regional asymmetry (left vs right half) ─────────────────────────
    left = img_f[:, : TARGET_SIZE // 2].mean()
    right = img_f[:, TARGET_SIZE // 2 :].mean()
    regional_asymmetry = float(abs(left - right) / (mean_int + 1e-6))

    # ── 18: Center brightness (central 50% crop) ────────────────────────────
    q = TARGET_SIZE // 4
    center_brightness = float(img_f[q : TARGET_SIZE - q, q : TARGET_SIZE - q].mean())

    # ── 19: Peripheral brightness (outer 12% border) ────────────────────────
    border = TARGET_SIZE // 8
    mask = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=bool)
    mask[border : TARGET_SIZE - border, border : TARGET_SIZE - border] = False
    peripheral_brightness = float(img_f[mask].mean())

    # ── 20: Local inhomogeneity (patch std / global std) ────────────────────
    patch = 32
    local_stds = []
    for r in range(0, TARGET_SIZE - patch + 1, patch):
        for c in range(0, TARGET_SIZE - patch + 1, patch):
            local_stds.append(img_f[r : r + patch, c : c + patch].std())
    local_inhomogeneity = float(np.mean(local_stds) / (std_int + 1e-6))

    return np.array([
        mean_int, std_int, skew_int, kurt_int,
        lap_var, fft_mean, edge_density,
        contrast, homogeneity, energy, correlation, entropy,
        mean_grad, hist_peak_pos, hist_spread,
        signal_void_ratio, regional_asymmetry,
        center_brightness, peripheral_brightness, local_inhomogeneity,
    ], dtype=np.float32)
