# Knee MRI Cartilage & Bone Segmentation

**Automatic Segmentation of Bone and Cartilage Tissues in Knee Joint from MRI Scans**

---

## Overview

This is a standalone computer vision project for automatic segmentation of knee joint structures (bone, cartilage, defects) from MRI scans using the Osteoarthritis Initiative (OAI) dataset from Kaggle.

**Dataset**: [3D Knee MRI Cartilage Segmentation](https://www.kaggle.com/datasets/ujjwalsinha01/3d-knee-mri-cartilage-segmentation)

**Full Russian Report**: [REPORT.md](REPORT.md)  
**Quick Start Guide**: [KNEE_MRI_QUICKSTART.md](KNEE_MRI_QUICKSTART.md)  
**Project Summary**: [PROJECT_SUMMARY_KNEE_MRI.md](PROJECT_SUMMARY_KNEE_MRI.md)

---

## Quick Start

### Full Pipeline (Recommended)

```bash
python main.py --mode full
```

### Step-by-Step

```bash
# 1. Download dataset (or manual download)
python main.py --mode download

# 2. Preprocess data
python main.py --mode preprocess

# 3. Run baseline segmentation
python main.py --mode segment

# 4. Generate analysis figures
python main.py --mode report
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `opencv-python`, `pillow`, `tqdm`

---

## Dataset

**5 Segmentation Classes**:
- 0: Background (non-tissue)
- 1: Femoral cartilage
- 2: Tibial cartilage
- 3: Bone (femur, tibia, patella)
- 4: Cartilage defects

**Splits**: train, val, test

---

## Methodology

**Baseline Algorithm**: Classical computer vision pipeline
1. Normalize intensity to [0, 255]
2. Otsu automatic thresholding
3. Morphological closing (fill gaps)
4. Connected component analysis
5. Class assignment by component size

**Evaluation Metrics**: Dice, mIoU, Pixel Accuracy, Mean Accuracy, F-weighted IoU

---

## Expected Results

Baseline (Otsu + Morphology):
- **mIoU**: 0.12 - 0.25
- **Dice**: 0.18 - 0.35
- **Pixel Accuracy**: 0.65 - 0.75

⚠️ This is a **baseline** for demonstration. Deep learning methods (U-Net, nnU-Net) achieve clinical-grade accuracy (Dice >0.85).

---

## Results Structure

```
results/
├── metrics.csv                    # Segmentation metrics
├── confusion_matrix.png           # Confusion matrix
├── samples/                       # Example predictions
│   ├── knee_seg_000.png
│   └── ...
└── figures/
    ├── class_distribution.png
    └── metrics_bar.png
```

---

## Project Structure

```
.
├── main.py                        # Pipeline CLI
├── README.md                      # This file
├── REPORT.md                      # Full Russian academic report
├── requirements.txt
├── src/
│   ├── download.py               # Dataset downloader
│   ├── preprocess.py             # Data preprocessing
│   ├── segment.py                # Baseline segmentation
│   ├── report_assets.py          # Figure generation
│   └── metrics.py                # Evaluation metrics
├── data/
│   ├── raw/
│   └── processed/
└── results/
```

---

## Manual Dataset Download

If Kaggle CLI is not configured:

1. Visit https://www.kaggle.com/datasets/ujjwalsinha01/3d-knee-mri-cartilage-segmentation
2. Download the ZIP file
3. Extract to `data/raw/knee_mri_cartilage/`
4. Run preprocessing: `python main.py --mode preprocess`

---

## Next Steps (Improvements)

1. Implement U-Net or nnU-Net for higher accuracy
2. Use 3D volumetric segmentation instead of 2D slices
3. Add data augmentation (rotation, elastic deformation)
4. Fine-tune on domain-specific protocols
5. Explore multimodal MRI sequences (T1, T2, DESS)

---

## References

1. Ronneberger O., et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015.
2. Isensee F., et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 2021.
3. Osteoarthritis Initiative (OAI) Dataset: https://nda.nih.gov/oai/

---

**Author**: Yerassyl  
**Date**: February 2026  
**Purpose**: ML Masters Degree Project (Computer Vision - Medical Imaging)
