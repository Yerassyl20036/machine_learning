# Quick Start: Knee MRI Cartilage & Bone Segmentation

## Overview

This project implements a baseline computer vision pipeline for automatic segmentation of knee joint structures (bone, cartilage, defects) from MRI scans, based on the Osteoarthritis Initiative dataset from Kaggle.

**Report**: [Методы автоматической сегментации костных и хрящевых тканей коленного сустава по данным МРТ.md](Методы%20автоматической%20сегментации%20костных%20и%20хрящевых%20тканей%20коленного%20сустава%20по%20данным%20МРТ.md)

**Dataset**: [3D Knee MRI Cartilage Segmentation (Kaggle)](https://www.kaggle.com/datasets/ujjwalsinha01/3d-knee-mri-cartilage-segmentation)

---

## Quick Commands

### Full Pipeline (Recommended)

Run all steps in one command:

```bash
python main.py --mode knee-full
```

This will:
1. Download the dataset (or show manual instructions)
2. Preprocess and extract metadata
3. Run baseline segmentation
4. Generate analysis figures

---

## Step-by-Step

### 1. Download Dataset

```bash
python main.py --mode knee-download
```

**If Kaggle CLI is not configured**:
1. Visit https://www.kaggle.com/datasets/ujjwalsinha01/3d-knee-mri-cartilage-segmentation
2. Download the ZIP file
3. Extract to `data/raw/knee_mri_cartilage/`

### 2. Preprocess Data

```bash
python main.py --mode knee-preprocess
```

Creates:
- `data/processed/knee_mri_cartilage/images/` — normalized MRI slices
- `data/processed/knee_mri_cartilage/masks/` — ground truth masks
- `data/processed/knee_mri_cartilage/metadata.csv` — sample metadata

### 3. Run Baseline Segmentation

```bash
python main.py --mode knee-segment
```

Creates:
- `results/knee_segmentation/metrics.csv` — Dice, mIoU, pixel accuracy, etc.
- `results/knee_segmentation/samples/` — example visualizations (MRI / GT / Prediction)
- `results/knee_segmentation/confusion_matrix.png` — class confusion matrix

### 4. Generate Analysis Figures

```bash
python main.py --mode knee-report
```

Creates:
- `results/knee_segmentation/figures/class_distribution.png` — class counts by split
- `results/knee_segmentation/figures/metrics_bar.png` — bar chart of metrics

---

## Results Structure

```
results/knee_segmentation/
├── metrics.csv                    # Segmentation metrics (Dice, mIoU, etc.)
├── confusion_matrix.png           # Confusion matrix heatmap
├── samples/                       # Example predictions
│   ├── knee_seg_000.png
│   ├── knee_seg_001.png
│   └── ...
└── figures/                       # Analysis charts
    ├── class_distribution.png
    └── metrics_bar.png
```

---

## Expected Metrics

Baseline (Otsu thresholding + morphology):
- **mIoU**: 0.12 - 0.25
- **Dice**: 0.18 - 0.35
- **Pixel Accuracy**: 0.65 - 0.75

⚠️ These are **baseline** results. Deep learning methods (U-Net, nnU-Net) achieve Dice >0.85.

---

## Dataset Classes

| Class ID | Description |
|----------|-------------|
| 0 | Background (non-tissue) |
| 1 | Femoral cartilage |
| 2 | Tibial cartilage |
| 3 | Bone (femur, tibia, patella) |
| 4 | Cartilage defects |

---

## Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
```

Key packages:
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `opencv-python` (for morphology)
- `Pillow`, `tqdm`

---

## Full Report

See the detailed Russian-language report with literature review, methodology, and analysis:

[Методы автоматической сегментации костных и хрящевых тканей коленного сустава по данным МРТ.md](Методы%20автоматической%20сегментации%20костных%20и%20хрящевых%20тканей%20коленного%20сустава%20по%20данным%20МРТ.md)

---

## Troubleshooting

**Dataset not found after download?**
- Ensure extraction to `data/raw/knee_mri_cartilage/Osteoarthritis dataset/`

**Import errors?**
- Install dependencies: `pip install -r requirements.txt`

**Low metrics?**
- This is a baseline method. Upgrade to U-Net or nnU-Net for clinical-grade accuracy.

---

## Next Steps

1. Implement U-Net or nnU-Net for improved segmentation
2. Use 3D volumetric segmentation instead of 2D slices
3. Fine-tune on domain-specific augmentations
4. Explore multimodal MRI sequences (T1, T2, DESS)

---

**Author**: Yerassyl  
**Date**: February 2026
