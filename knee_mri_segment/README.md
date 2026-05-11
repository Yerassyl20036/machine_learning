# Knee MRI Segmentation

Automatic segmentation of bone and cartilage in knee MRI images using the **Double-Otsu** method — no training required.

**Author:** Маратов Ерасыл Балканович  
**Program:** 7М06101 | AIU | 2025–2026  
**Dataset:** [3D Knee MRI Cartilage Segmentation (OAI / Kaggle)](https://www.kaggle.com/datasets/ujjwalsinha01/3d-knee-mri-cartilage-segmentation)

---

## Method

The pipeline is **fully unsupervised** — no labeled data or training needed:

1. **Normalize** — rescale pixel values to 0–255
2. **Otsu t₁** — threshold computed from the image histogram → separates tissue from background
3. **Otsu t₂** — second threshold computed **within tissue pixels only** → separates bone (medium brightness) from cartilage (higher brightness)
4. **Morphological refinement** — closing 5×5 (bone), closing 3×3 (cartilage) to fill holes and remove noise

### Dice Scores (synthetic evaluation)

| KL Stage | Bone Dice | Cartilage Dice |
|----------|-----------|----------------|
| KL-0     | 0.93      | 0.38           |
| KL-1     | 0.96      | 0.27           |
| KL-2     | 0.80      | 0.29           |
| KL-3     | 0.70      | 0.10           |
| KL-4     | 0.78      | 0.05           |
| **Mean** | **0.83**  | **0.22**       |

> Evaluation uses synthetic MRI images (real Kaggle dataset requires manual download).

---

## Project Structure

```
knee_mri_segment/
├── app.py                  # FastAPI web demo
├── requirements.txt
├── src/
│   ├── segment.py          # Double-Otsu algorithm (main)
│   ├── metrics.py          # Dice, IoU, pixel accuracy
│   ├── preprocess.py       # preprocessing utilities
│   └── seg_visualizations.py  # visualization helpers
├── static/
│   ├── index.html          # web UI
│   └── samples/            # kl0.png … kl4.png (synthetic demo images)
├── results/
│   └── seg_figures/        # pipeline and comparison figures
└── data/                   # place downloaded MRI data here
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web demo
uvicorn app:app --reload --port 8000

# 3. Open browser
open http://127.0.0.1:8000
```

The web UI lets you:
- Click any of 5 sample KL-0…KL-4 synthetic images → instant segmentation
- Drag & drop your own PNG/JPG MRI image
- See 4-step pipeline visualization (normalize → tissue → bone/cartilage → overlay)

---

## Algorithm Details

```python
from src.segment import segment_improved

seg = segment_improved(img_gray)          # returns 0=bg, 1=bone, 2=cartilage
seg, steps = segment_improved(img_gray, return_steps=True)
# steps keys: "normalized", "tissue_binary", "bone_cart_raw", "t1", "t2"
```

---

## Comparison with Deep Learning

| Method        | Dice Bone  | Dice Cartilage | GPU? | Training? |
|---------------|------------|----------------|------|-----------|
| Simple Otsu (v1) | 0.20–0.35 | ≈ 0.00       | No   | No        |
| **Double-Otsu (ours)** | **0.68–0.96** | **0.05–0.38** | **No** | **No** |
| U-Net (literature) | 0.85–0.90 | 0.80–0.87 | Yes  | ~7 days   |
| nnU-Net (literature) | 0.90–0.95 | 0.88–0.93 | Yes | ~7 days  |

**Key advantage:** Double-Otsu requires zero training data, zero GPU, and is fully interpretable — the two threshold values t₁ and t₂ completely explain every decision.

---

## References

- Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Trans. Systems Man Cybernetics.
- Serra, J. (1982). *Image Analysis and Mathematical Morphology.* Academic Press.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. https://arxiv.org/abs/1505.04597
- Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods. https://arxiv.org/abs/1809.10486
