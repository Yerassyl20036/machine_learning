#!/usr/bin/env python3
"""
FastAPI web app — Knee MRI Segmentation Demo.

Endpoints:
  GET  /          → serve index.html
  POST /predict   → upload MRI image → returns segmentation overlay + stats

Start:
  cd knee_mri_segmentation
  uvicorn app:app --reload --port 8000
"""

import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ─── Make src importable ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.segment import segment_improved  # noqa: E402

app = FastAPI(title="Knee MRI Segmentation")

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ─── Colors (BGR for OpenCV) ──────────────────────────────────────────────────
BONE_COLOR = (255, 140, 80)      # blue-ish   (BGR)
CART_COLOR = (120, 220, 50)      # green-ish  (BGR)
ALPHA = 0.50


def _encode_png(arr: np.ndarray) -> str:
    """Encode BGR/gray ndarray to base64 PNG data-URI."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _build_overlay(gray: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Return BGR overlay image: bone=blue, cartilage=green."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    color_layer = bgr.copy()
    color_layer[seg == 1] = BONE_COLOR
    color_layer[seg == 2] = CART_COLOR
    result = np.clip(bgr * (1 - ALPHA) + color_layer * ALPHA, 0, 255).astype(np.uint8)
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html not found in static/</h1>"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and decode uploaded image
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Cannot decode image"})

    # Resize if very large (keep aspect ratio, max 512px)
    h, w = img.shape
    if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Segmentation with step images
    seg, steps = segment_improved(img, return_steps=True)

    # Stats
    total_px = seg.size
    bone_px = int((seg == 1).sum())
    cart_px = int((seg == 2).sum())
    bg_px   = int((seg == 0).sum())

    bone_pct = round(100 * bone_px / total_px, 1)
    cart_pct = round(100 * cart_px / total_px, 1)
    bg_pct   = round(100 * bg_px   / total_px, 1)

    # Count connected components (excluding background)
    bin_mask = (seg > 0).astype(np.uint8)
    n_components, _ = cv2.connectedComponents(bin_mask)
    n_components = max(0, n_components - 1)  # subtract background label

    # Step visualizations (BGR color)
    def _step_tissue(binary):
        """White tissue on dark bg."""
        out = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        out[binary > 0] = (200, 200, 200)
        return out

    def _step_raw(raw_vis):
        """Gray=bone raw, White=cartilage raw."""
        out = np.zeros((raw_vis.shape[0], raw_vis.shape[1], 3), dtype=np.uint8)
        out[raw_vis == 128] = (180, 120, 60)    # bone candidate → blue-ish
        out[raw_vis == 255] = (60, 210, 120)    # cart candidate → green-ish
        return out

    step1_img = cv2.cvtColor(steps["normalized"], cv2.COLOR_GRAY2BGR)
    step2_img = _step_tissue(steps["tissue_binary"])
    step3_img = _step_raw(steps["bone_cart_raw"])

    # Final overlay
    overlay = _build_overlay(img, seg)
    # Add legend strip at bottom
    legend_h = 28
    legend = np.zeros((legend_h, overlay.shape[1], 3), dtype=np.uint8)
    legend[:, :overlay.shape[1]//2] = BONE_COLOR
    legend[:, overlay.shape[1]//2:] = CART_COLOR
    overlay_with_legend = np.vstack([overlay, legend])

    return JSONResponse({
        "original_b64":  _encode_png(img),
        "overlay_b64":   _encode_png(overlay_with_legend),
        "step1_b64":     _encode_png(step1_img),
        "step2_b64":     _encode_png(step2_img),
        "step3_b64":     _encode_png(step3_img),
        "t1":            steps["t1"],
        "t2":            steps["t2"],
        "bone_pct":      bone_pct,
        "cart_pct":      cart_pct,
        "bg_pct":        bg_pct,
        "n_components":  n_components,
        "dice_range":    "0.68\u20130.93",
        "cart_dice":     "0.05\u20130.75",
        "image_size":    f"{img.shape[1]}×{img.shape[0]}",
    })
