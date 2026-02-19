# ML Masters Degree Projects

This workspace contains two independent computer vision research projects:

---

## 📂 Project Structure

```
ML_masters_degree/
├── nyu_depth_v2/                    # NYU Depth Dataset V2 Project
│   └── Scene Segmentation & 3D Reconstruction
│
└── knee_mri_segmentation/           # Knee MRI Cartilage Segmentation Project
    └── Medical Image Analysis for Osteoarthritis
```

---

## 1️⃣ NYU Depth V2: Scene Segmentation and 3D Reconstruction

**Directory**: [`nyu_depth_v2/`](nyu_depth_v2/)

**Topic**: Анализ и разработка методов сегментации сцен и 3D реконструкции по данным глубины

**Dataset**: NYU Depth Dataset V2 (RGB-D indoor scenes, 795 train / 654 test, 40 classes)

**Methods**:
- Semantic segmentation with Tiny U-Net
- 3D reconstruction using TSDF fusion (Open3D)

**Quick Start**:
```bash
cd nyu_depth_v2
python main.py --mode full
```

**Documentation**:
- [Project README](nyu_depth_v2/PROJECT_README.md) - Quick start guide
- [Full Report (Russian)](nyu_depth_v2/README.md) - Academic report
- [Technical Report (Russian)](nyu_depth_v2/ТЕХНИЧЕСКИЙ%20ОТЧЕТ.md)

---

## 2️⃣ Knee MRI: Cartilage Segmentation

**Directory**: [`knee_mri_segmentation/`](knee_mri_segmentation/)

**Topic**: Методы автоматической сегментации костных и хрящевых тканей коленного сустава по данным МРТ

**Dataset**: Osteoarthritis Initiative 3D Knee MRI (Kaggle, 5 classes: background, femoral/tibial cartilage, bone, defects)

**Methods**:
- Classical computer vision baseline (Otsu thresholding + morphology)
- Multi-class segmentation with size-based classification

**Quick Start**:
```bash
cd knee_mri_segmentation
python main.py --mode full
```

**Documentation**:
- [Project Summary](knee_mri_segmentation/PROJECT_SUMMARY_KNEE_MRI.md)
- [Quickstart Guide](knee_mri_segmentation/KNEE_MRI_QUICKSTART.md)
- [Full Report (Russian)](knee_mri_segmentation/Методы%20автоматической%20сегментации%20костных%20и%20хрящевых%20тканей%20коленного%20сустава%20по%20данным%20МРТ.md)

---

## 🔍 Project Comparison

| Aspect | NYU Depth V2 | Knee MRI |
|--------|--------------|----------|
| **Domain** | RGB-D Scene Understanding | Medical Image Analysis |
| **Data Type** | RGB-D (color + depth) | MRI (grayscale + masks) |
| **Task** | Segmentation + 3D Reconstruction | Cartilage Segmentation |
| **Classes** | 40 (indoor objects) | 5 (tissues) |
| **Method** | Deep Learning (U-Net) + TSDF | Classical CV (Otsu + morphology) |
| **Dependencies** | PyTorch, Open3D | OpenCV, NumPy |
| **Purpose** | General scene understanding | Osteoarthritis diagnosis support |

---

## 🛠️ Technology Stack

### NYU Depth V2
- **Deep Learning**: PyTorch, TorchVision
- **3D Processing**: Open3D
- **Data**: H5Py, SciPy
- **Database**: PostgreSQL (optional)

### Knee MRI
- **Computer Vision**: OpenCV
- **Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Data Source**: Kaggle CLI

---

## 📊 Results

Both projects generate:
- ✅ Quantitative metrics (mIoU, Dice, accuracy)
- ✅ Visual samples (overlays, predictions)
- ✅ Confusion matrices
- ✅ Summary reports (CSV + Markdown)

Results are stored in respective `results/` directories within each project.

---

## 📦 Installation

Each project has independent dependencies:

```bash
# NYU Depth V2
cd nyu_depth_v2
pip install -r requirements.txt

# Knee MRI
cd knee_mri_segmentation
pip install -r requirements.txt
```

---

## 🚀 Running Projects

Both projects share similar CLI structure:

```bash
# Download dataset
python main.py --mode download

# Preprocess data
python main.py --mode preprocess

# Run segmentation
python main.py --mode segment

# Full pipeline
python main.py --mode full
```

*(NYU also includes `--mode reconstruct` for 3D reconstruction)*

---

## 📝 Documentation Structure

```
nyu_depth_v2/
├── PROJECT_README.md              # Quick start (English)
├── README.md                      # Full report (Russian)
└── ТЕХНИЧЕСКИЙ ОТЧЕТ.md           # Technical report (Russian)

knee_mri_segmentation/
├── PROJECT_SUMMARY_KNEE_MRI.md    # Overview (English)
├── KNEE_MRI_QUICKSTART.md         # Quick start (English)
└── Методы автоматической сегментации... .md  # Full report (Russian)
```

---

## 🎯 Project Objectives

### NYU Depth V2
- Implement baseline semantic segmentation on RGB-D data
- Develop 3D reconstruction pipeline using depth maps
- Evaluate classical and learning-based approaches
- Generate academic-quality technical reports

### Knee MRI
- Segment cartilage tissues for osteoarthritis analysis
- Establish classical CV baseline for comparison
- Prepare dataset for future deep learning experiments
- Create reproducible evaluation framework

---

## 👨‍💻 Author

**Yerassyl**  
ML Masters Degree  
February 2025

---

## 📖 Further Reading

- **NYU Depth V2 Project**: See [nyu_depth_v2/README.md](nyu_depth_v2/README.md)
- **Knee MRI Project**: See [knee_mri_segmentation/PROJECT_SUMMARY_KNEE_MRI.md](knee_mri_segmentation/PROJECT_SUMMARY_KNEE_MRI.md)
- **Project Architecture**: See `PROJECTS_OVERVIEW.md` (if available)

---

**Note**: These two projects are completely independent with different datasets, purposes, and methodologies. They can be run separately without any shared dependencies.
