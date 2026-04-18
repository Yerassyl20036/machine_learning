"""
Iris preprocessing pipeline:
  1. Load & grayscale
  2. Segment iris using Hough circle detection (pupil + iris boundaries)
  3. Normalize via Daugman rubber-sheet model → fixed-size polar strip
  4. Resize normalized image to target square size
"""

import cv2
import numpy as np


# ── constants ────────────────────────────────────────────────────────────────
NORM_WIDTH = 512   # angular samples (columns)
NORM_HEIGHT = 64   # radial samples (rows)
TARGET_SIZE = 128  # final square resize (128 or 256)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _detect_circle(gray: np.ndarray, min_r: int, max_r: int,
                   param2: int = 30) -> tuple[int, int, int] | None:
    """Return (cx, cy, r) or None."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=gray.shape[0] // 4,
        param1=50,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    c = np.round(circles[0, 0]).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def segment_iris(gray: np.ndarray) -> tuple[tuple, tuple] | None:
    """
    Detect pupil and iris circles.
    Returns ((px,py,pr), (ix,iy,ir)) or None if detection fails.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Pupil: small, dark circle
    pupil = _detect_circle(blurred,
                            min_r=max(5, h // 20),
                            max_r=h // 5,
                            param2=25)
    if pupil is None:
        # Fallback: use image center with estimated radii
        px, py = w // 2, h // 2
        pr = h // 8
        pupil = (px, py, pr)

    # Iris: larger circle, search around pupil center
    iris = _detect_circle(blurred,
                           min_r=pupil[2] + 5,
                           max_r=min(h, w) // 2,
                           param2=20)
    if iris is None:
        # Fallback: iris ≈ 3× pupil radius
        iris = (pupil[0], pupil[1], pupil[2] * 3)

    return pupil, iris


def rubber_sheet_normalize(gray: np.ndarray,
                            pupil: tuple,
                            iris: tuple,
                            norm_w: int = NORM_WIDTH,
                            norm_h: int = NORM_HEIGHT) -> np.ndarray:
    """
    Daugman rubber-sheet model.
    Maps the annular iris region to a rectangular strip of shape (norm_h, norm_w).
    """
    px, py, pr = pupil
    ix, iy, ir = iris

    norm = np.zeros((norm_h, norm_w), dtype=np.uint8)
    thetas = np.linspace(0, 2 * np.pi, norm_w, endpoint=False)
    rs = np.linspace(0, 1, norm_h)

    for row_idx, r in enumerate(rs):
        # Interpolate between pupil boundary and iris boundary
        xp = px + pr * np.cos(thetas)
        yp = py + pr * np.sin(thetas)
        xi = ix + ir * np.cos(thetas)
        yi = iy + ir * np.sin(thetas)

        xs = ((1 - r) * xp + r * xi).astype(int)
        ys = ((1 - r) * yp + r * yi).astype(int)

        # Clamp to image bounds
        xs = np.clip(xs, 0, gray.shape[1] - 1)
        ys = np.clip(ys, 0, gray.shape[0] - 1)

        norm[row_idx, :] = gray[ys, xs]

    return norm


def preprocess(path: str,
               target_size: int = TARGET_SIZE,
               return_debug: bool = False):
    """
    Full pipeline: raw image → normalized iris strip → resized square.

    Returns:
        np.ndarray of shape (target_size, target_size) uint8
        (optionally also returns gray + circles for debug display)
    """
    gray = load_gray(path)
    pupil, iris_circle = segment_iris(gray)

    norm = rubber_sheet_normalize(gray, pupil, iris_circle)

    # Resize to square
    resized = cv2.resize(norm, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)

    if return_debug:
        return resized, gray, pupil, iris_circle

    return resized


def preprocess_batch(paths: list[str],
                     target_size: int = TARGET_SIZE) -> list[np.ndarray]:
    results = []
    for p in paths:
        try:
            results.append(preprocess(p, target_size=target_size))
        except Exception as e:
            print(f"[preprocess] skipping {p}: {e}")
    return results
