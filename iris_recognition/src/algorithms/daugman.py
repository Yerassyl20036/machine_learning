"""
Algorithm 1 – Daugman IrisCode
Features: 1D Gabor wavelet applied to each row of the normalized iris strip.
Phase is quantized to 2 bits → binary IrisCode.
Matching: Hamming distance between two IrisCodes.
"""

import numpy as np
import cv2


def _gabor_response(signal: np.ndarray, freq: float, sigma: float) -> np.ndarray:
    """Apply 1-D Gabor filter and return binary phase code."""
    n = len(signal)
    x = np.arange(n) - n / 2
    # Gabor kernel (complex)
    kernel_real = np.exp(-x ** 2 / (2 * sigma ** 2)) * np.cos(2 * np.pi * freq * x)
    kernel_imag = np.exp(-x ** 2 / (2 * sigma ** 2)) * np.sin(2 * np.pi * freq * x)

    real_part = np.convolve(signal.astype(float), kernel_real, mode="same")
    imag_part = np.convolve(signal.astype(float), kernel_imag, mode="same")

    # Phase quantization: 2 bits per sample
    code = np.zeros(2 * n, dtype=np.uint8)
    code[0::2] = (real_part >= 0).astype(np.uint8)
    code[1::2] = (imag_part >= 0).astype(np.uint8)
    return code


def extract(norm_img: np.ndarray,
            freqs: tuple = (0.1, 0.2, 0.3),
            sigma: float = 3.0) -> np.ndarray:
    """
    Extract IrisCode from a normalized iris image.

    Args:
        norm_img: 2-D uint8 array (H × W), output of rubber-sheet normalization
        freqs:    Gabor frequencies to use
        sigma:    Gaussian envelope width

    Returns:
        1-D binary numpy array (IrisCode)
    """
    h, w = norm_img.shape
    codes = []
    for row in range(h):
        signal = norm_img[row].astype(np.float64)
        # Normalize row
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        for freq in freqs:
            codes.append(_gabor_response(signal, freq, sigma))
    return np.concatenate(codes).astype(np.uint8)


def hamming_distance(code_a: np.ndarray, code_b: np.ndarray) -> float:
    """Normalized Hamming distance in [0, 1]. Codes must be same length."""
    min_len = min(len(code_a), len(code_b))
    diff = np.count_nonzero(code_a[:min_len] != code_b[:min_len])
    return diff / min_len


def similarity(code_a: np.ndarray, code_b: np.ndarray) -> float:
    """Similarity score in [0, 1] (1 = identical)."""
    return 1.0 - hamming_distance(code_a, code_b)
