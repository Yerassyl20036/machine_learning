# NYU Depth V2: Scene Segmentation and 3D Reconstruction

**Analysis and Development of Scene Segmentation and 3D Reconstruction Methods from Depth Data**

---

## Overview

This project implements scene segmentation and 3D reconstruction pipelines on the NYU Depth Dataset V2, combining RGB-D data for indoor scene understanding.

**Full Russian Report**: [README.md](README.md)  
**Technical Report**: [ТЕХНИЧЕСКИЙ ОТЧЕТ.md](ТЕХНИЧЕСКИЙ%20ОТЧЕТ.md)

---

## Quick Start

### Full Pipeline

```bash
python main.py --mode full
```

### Step-by-Step

```bash
# 1. Download dataset
python main.py --mode download

# 2. Preprocess data
python main.py --mode preprocess

# 3. Run segmentation baseline
python main.py --mode segment

# 4. Run 3D reconstruction baseline
python main.py --mode reconstruct
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `pandas`, `matplotlib`, `torch`, `torchvision`, `open3d`, `scikit-learn`

---

## Results

- Segmentation metrics: `results/segmentation/metrics.csv`
- Reconstruction metrics: `results/reconstruction/metrics.csv`
- Summary: `results/summary_metrics.md`
- Visualizations: `results/segmentation/samples/`, `results/reconstruction/samples/`

---

## Project Structure

```
.
├── main.py                        # Pipeline CLI
├── README.md                      # Full Russian report
├── requirements.txt
├── src/
│   ├── nyu_download.py           # Dataset downloader
│   ├── nyu_preprocess.py         # Data preprocessing
│   ├── nyu_segmentation.py       # Segmentation baseline
│   ├── nyu_reconstruction.py     # 3D reconstruction
│   ├── nyu_report_assets.py      # Report figure generation
│   └── metrics.py                # Evaluation metrics
├── data/
│   ├── raw/
│   └── processed/
└── results/
    ├── segmentation/
    ├── reconstruction/
    └── figures/
```

---

## Separate Projects

**This repository contains two independent ML projects:**

1. **NYU Depth V2** (this directory) - RGB-D scene segmentation & 3D reconstruction
2. **Knee MRI Segmentation** ([knee_mri_segmentation/](knee_mri_segmentation/)) - Medical image segmentation

Each project is completely standalone with its own pipeline, dependencies, and documentation.

---

## Dataset

**NYU Depth Dataset V2**
- RGB-D indoor scenes
- 795 training images, 654 test images
- 40 semantic classes
- Download: Run `python main.py --mode download` or manual download from http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

---

## Methodology

**Segmentation**: Tiny U-Net baseline (CPU-friendly)
- Encoder-decoder architecture
- 5 epochs on subset
- Metrics: mIoU, Dice, Pixel Accuracy

**3D Reconstruction**: TSDF Fusion (Open3D)
- Voxel size: 0.02m
- Integration volume: 4m
- Metrics: RMSE, AbsRel, delta accuracy

---

**Author**: Yerassyl  
**Date**: February 2026  
**Purpose**: ML Masters Degree Project (Depth-based Scene Understanding)
