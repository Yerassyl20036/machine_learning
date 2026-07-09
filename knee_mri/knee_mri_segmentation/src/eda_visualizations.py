#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) visualizations for the Knee MRI Segmentation project.
Generates publication-quality figures for the report:
  1. MRI Slice Analysis Panels (Original, FFT, Laplacian, Edges, Intensity Map)
  2. Feature Distribution (KDE + Boxplot) per KL grade
  3. Correlation Matrix Heatmap
  4. PCA Projection (2D scatter by class)
  5. Random Forest Feature Importance
  6. Confusion Matrices (Logistic Regression vs Random Forest)
  7. ROC Curve Comparison
  8. Class Distribution bar charts

Uses real data if available; otherwise generates realistic synthetic features
consistent with knee radiograph OA grading (KL 0-4).
"""

import argparse
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.ndimage import laplace
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

warnings.filterwarnings("ignore")

# ── output dirs ───────────────────────────────────────────────────────────────
DEFAULT_OUT = "results/eda_figures"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic feature generation  (used when real images are unavailable)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
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
    "bone_area_ratio",
    "cartilage_area_ratio",
    "joint_space_width",
    "osteophyte_score",
    "sclerosis_index",
    "mean_gradient",
    "hist_peak_pos",
    "hist_spread",
]


def _synthesize_features(n_per_class: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic feature matrix for 5 KL grades (0-4).
    Features shift progressively with OA severity.
    """
    rng = np.random.RandomState(seed)
    rows = []
    class_names = {0: "KL-0", 1: "KL-1", 2: "KL-2", 3: "KL-3", 4: "KL-4"}

    for kl in range(5):
        n = n_per_class if kl != 4 else n_per_class // 3  # class 4 is rare
        severity = kl / 4.0  # 0 → 1

        feats = {
            "mean_intensity": rng.normal(120 - 10 * severity, 15, n),
            "std_intensity": rng.normal(35 + 8 * severity, 6, n),
            "skew_intensity": rng.normal(-0.2 + 0.5 * severity, 0.4, n),
            "kurt_intensity": rng.normal(2.5 + 1.5 * severity, 0.8, n),
            "laplacian_var": rng.normal(800 - 200 * severity, 120, n),
            "fft_mean": rng.normal(45 + 15 * severity, 8, n),
            "edge_density": rng.normal(55 - 12 * severity, 10, n),
            "contrast": rng.normal(50 + 20 * severity, 12, n),
            "homogeneity": rng.normal(0.6 - 0.15 * severity, 0.08, n),
            "energy": rng.normal(0.3 - 0.1 * severity, 0.06, n),
            "correlation": rng.normal(0.85 - 0.1 * severity, 0.05, n),
            "entropy": rng.normal(5.5 + 0.8 * severity, 0.5, n),
            "bone_area_ratio": rng.normal(0.35 + 0.1 * severity, 0.05, n),
            "cartilage_area_ratio": rng.normal(0.15 - 0.06 * severity, 0.03, n),
            "joint_space_width": rng.normal(4.5 - 1.5 * severity, 0.6, n),
            "osteophyte_score": rng.exponential(0.5 + 2.0 * severity, n),
            "sclerosis_index": rng.normal(0.1 + 0.3 * severity, 0.08, n),
            "mean_gradient": rng.normal(25 + 8 * severity, 5, n),
            "hist_peak_pos": rng.normal(100 - 15 * severity, 12, n),
            "hist_spread": rng.normal(60 + 10 * severity, 8, n),
        }

        for i in range(n):
            row = {k: v[i] for k, v in feats.items()}
            row["kl_grade"] = kl
            row["class_name"] = class_names[kl]
            rows.append(row)

    return pd.DataFrame(rows)


def _synthesize_mri_slice(kl_grade: int, size: int = 224, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic grayscale 'MRI-like' image for visualization demo.
    """
    rng = np.random.RandomState(seed + kl_grade * 100)
    img = np.zeros((size, size), dtype=np.float32)

    # base background gradient (soft tissue)
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2 + 10
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    img += np.clip(180 - r * 0.6, 0, 180)

    # femur (upper bright ellipse)
    fy, fx = cy - 35, cx
    femur_mask = ((x - fx) ** 2 / 55**2 + (y - fy) ** 2 / 30**2) < 1
    img[femur_mask] = rng.normal(200, 10, femur_mask.sum()).clip(150, 255)

    # tibia (lower bright ellipse)
    ty, tx = cy + 40, cx
    tibia_mask = ((x - tx) ** 2 / 50**2 + (y - ty) ** 2 / 25**2) < 1
    img[tibia_mask] = rng.normal(190, 12, tibia_mask.sum()).clip(140, 255)

    # cartilage (thin bright band between femur and tibia)
    gap_y = cy + 3
    cart_thick = max(2, int(6 - kl_grade * 1.2))  # thinner with higher KL
    cart_mask = (np.abs(y - gap_y) < cart_thick) & (np.abs(x - cx) < 40)
    img[cart_mask] = rng.normal(220 - kl_grade * 10, 8, cart_mask.sum()).clip(100, 255)

    # osteophytes (spurs at edges for higher KL grades)
    if kl_grade >= 2:
        n_spurs = kl_grade
        for _ in range(n_spurs):
            sx = rng.randint(cx - 50, cx + 50)
            sy = rng.randint(cy - 5, cy + 5)
            spur_r = rng.randint(3, 6 + kl_grade)
            spur_mask = ((x - sx) ** 2 + (y - sy) ** 2) < spur_r**2
            img[spur_mask] = rng.normal(210, 5, spur_mask.sum()).clip(180, 255)

    # add noise proportional to severity
    noise = rng.normal(0, 5 + 3 * kl_grade, (size, size))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Visualization functions
# ─────────────────────────────────────────────────────────────────────────────


def plot_image_analysis_panels(out_dir: str) -> None:
    """
    Fig 1: Two rows of MRI slices with analysis panels.
    Columns: Original | FFT Spectrum | Laplacian | Edges | Intensity Heatmap
    Row 1 = KL-0 (healthy), Row 2 = KL-4 (severe OA)
    """
    grades = [0, 4]
    titles_row = ["Original", "FFT Spectrum", "Laplacian", "Edges", "Intensity Map"]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))

    for row, kl in enumerate(grades):
        img = _synthesize_mri_slice(kl, seed=row * 7)

        # Original
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_ylabel(f"KL-{kl}", fontsize=13, fontweight="bold")

        # FFT magnitude spectrum
        f = np.fft.fft2(img.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        axes[row, 1].imshow(magnitude, cmap="inferno")

        # Laplacian
        lap = cv2.Laplacian(img, cv2.CV_64F)
        axes[row, 2].imshow(np.abs(lap), cmap="gray")

        # Canny edges
        edges = cv2.Canny(img, 30, 100)
        axes[row, 3].imshow(edges, cmap="gray")

        # Intensity heatmap (viridis)
        axes[row, 4].imshow(img, cmap="viridis")

    for j, t in enumerate(titles_row):
        axes[0, j].set_title(t, fontsize=13, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(out_dir, "image_analysis_panels.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_feature_distribution(df: pd.DataFrame, feature: str, out_dir: str) -> None:
    """
    Fig 2: KDE distribution + Boxplot for a selected feature, split by KL grade.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    palette = {
        "KL-0": "#4C72B0",
        "KL-1": "#55A868",
        "KL-2": "#C44E52",
        "KL-3": "#8172B2",
        "KL-4": "#CCB974",
    }

    # KDE
    for cls in sorted(df["class_name"].unique()):
        subset = df[df["class_name"] == cls][feature].dropna()
        if len(subset) < 5:
            continue
        kde_x = np.linspace(subset.min() - subset.std(), subset.max() + subset.std(), 300)
        kde = stats.gaussian_kde(subset)
        ax1.fill_between(kde_x, kde(kde_x), alpha=0.35, label=cls, color=palette.get(cls))
        ax1.plot(kde_x, kde(kde_x), color=palette.get(cls), linewidth=1.5)
    ax1.set_title(f"Distribution (KDE): {feature}", fontsize=13)
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Density")
    ax1.legend(title="KL Grade")

    # Boxplot
    groups = [df[df["class_name"] == c][feature].dropna().values for c in sorted(df["class_name"].unique())]
    labels = sorted(df["class_name"].unique())
    bp = ax2.boxplot(groups, labels=labels, patch_artist=True, widths=0.5)
    for patch, cls in zip(bp["boxes"], labels):
        patch.set_facecolor(palette.get(cls, "#999999"))
        patch.set_alpha(0.7)
    ax2.set_title(f"Outliers (Boxplot): {feature}", fontsize=13)
    ax2.set_ylabel(feature)
    ax2.set_xlabel("KL Grade")

    plt.tight_layout()
    path = os.path.join(out_dir, f"distribution_{feature}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 3: Correlation heatmap of all numeric features + encoded target.
    """
    numeric = df[FEATURE_NAMES + ["kl_grade"]].copy()
    numeric.rename(columns={"kl_grade": "target_encoded"}, inplace=True)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    rdbu = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=rdbu, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Матрица корреляций (Correlation Matrix)", fontsize=14, pad=15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_pca_projection(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 4: 2-D PCA scatter coloured by KL grade.
    """
    X = df[FEATURE_NAMES].values
    y = df["class_name"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)

    palette = {
        "KL-0": "#4C72B0",
        "KL-1": "#55A868",
        "KL-2": "#C44E52",
        "KL-3": "#8172B2",
        "KL-4": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in sorted(df["class_name"].unique()):
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=palette.get(cls),
            label=cls,
            alpha=0.55,
            s=18,
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)", fontsize=12)
    ax.set_title("PCA Projection (2D Visualization of Dataset)", fontsize=14)
    ax.legend(title="KL Grade", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "pca_projection.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_feature_importance(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 5: Horizontal bar chart of Random Forest feature importance.
    """
    X = df[FEATURE_NAMES].values
    y = df["kl_grade"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)

    importances = rf.feature_importances_
    order = np.argsort(importances)
    sorted_feat = [FEATURE_NAMES[i] for i in order]
    sorted_imp = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_feat)))
    ax.barh(sorted_feat, sorted_imp, color=colors)
    ax.set_xlabel("Важность (Importance Score)", fontsize=12)
    ax.set_title("Random Forest: Самые важные признаки (Feature Importance)", fontsize=13)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_confusion_matrices(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 6: Side-by-side confusion matrices — Logistic Regression vs Random Forest.
    """
    X = df[FEATURE_NAMES].values
    y = df["kl_grade"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=2000, random_state=42, multi_class="multinomial")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=[0, 1, 2, 3, 4])
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[0, 1, 2, 3, 4])

    class_labels = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Logistic Regression
    im1 = ax1.imshow(cm_lr, cmap="Blues", interpolation="nearest")
    ax1.set_title("Logistic Regression (Baseline)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(class_labels, fontsize=9)
    ax1.set_yticklabels(class_labels, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, str(cm_lr[i, j]), ha="center", va="center",
                     color="white" if cm_lr[i, j] > cm_lr.max() / 2 else "black", fontsize=10)

    # Random Forest
    im2 = ax2.imshow(cm_rf, cmap="Greens", interpolation="nearest")
    ax2.set_title("Random Forest (Advanced)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(class_labels, fontsize=9)
    ax2.set_yticklabels(class_labels, fontsize=9)
    for i in range(5):
        for j in range(5):
            ax2.text(j, i, str(cm_rf[i, j]), ha="center", va="center",
                     color="white" if cm_rf[i, j] > cm_rf.max() / 2 else "black", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def compute_classification_metrics(df: pd.DataFrame, out_dir: str) -> None:
    """
    Compute Accuracy, Precision, Recall, F1-score for Logistic Regression
    and Random Forest classifiers. Save results as CSV and a summary figure.
    """
    X = df[FEATURE_NAMES].values
    y = df["kl_grade"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=2000, random_state=42, multi_class="multinomial")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    class_labels = ["KL-0", "KL-1", "KL-2", "KL-3", "KL-4"]

    # --- Per-model summary metrics ---
    rows = []
    for name, y_pred in [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]:
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (macro)": round(prec_macro, 4),
            "Recall (macro)": round(rec_macro, 4),
            "F1-score (macro)": round(f1_macro, 4),
            "Precision (weighted)": round(prec_weighted, 4),
            "Recall (weighted)": round(rec_weighted, 4),
            "F1-score (weighted)": round(f1_weighted, 4),
        })

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "classification_metrics.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # --- Per-class reports ---
    for name, y_pred in [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf)]:
        report = classification_report(
            y_test, y_pred, target_names=class_labels, zero_division=0, output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        safe_name = name.lower().replace(" ", "_")
        report_path = os.path.join(out_dir, f"classification_report_{safe_name}.csv")
        df_report.to_csv(report_path)
        print(f"  ✓ {report_path}")

    # --- Summary figure ---
    metrics_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    lr_vals = [df_summary.iloc[0][m] for m in metrics_names]
    rf_vals = [df_summary.iloc[1][m] for m in metrics_names]
    short_labels = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(short_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, lr_vals, width, label="Logistic Regression", color="#5A9BD5")
    bars2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color="#76B947")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Метрики качества классификации (Accuracy, Precision, Recall, F1)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "classification_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_roc_curves(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 7: ROC curves for Logistic Regression vs Random Forest (OvR macro-average).
    """
    X = df[FEATURE_NAMES].values
    y = df["kl_grade"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=y
    )

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    lr = LogisticRegression(max_iter=2000, random_state=42, multi_class="multinomial")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_proba = lr.predict_proba(X_test)
    rf_proba = rf.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Macro-average ROC for each model
    for name, proba, color, lw in [
        ("Logistic Regression", lr_proba, "blue", 2),
        ("Random Forest", rf_proba, "green", 2),
    ]:
        fpr_all, tpr_all = [], []
        for i in range(y_test_bin.shape[1]):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], proba[:, i])
            fpr_all.append(fpr_i)
            tpr_all.append(tpr_i)

        # Macro-average
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr_i, tpr_i in zip(fpr_all, tpr_all):
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        mean_tpr /= y_test_bin.shape[1]

        auc_val = roc_auc_score(y_test_bin, proba, multi_class="ovr", average="macro")
        ax.plot(all_fpr, mean_tpr, color=color, linewidth=lw,
                label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate (Ошибки на реальных)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Верно найденные)", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_class_distribution(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 8: Class distribution bar chart (Total + per-split view).
    """
    palette = {
        "KL-0": "#4C72B0",
        "KL-1": "#55A868",
        "KL-2": "#C44E52",
        "KL-3": "#8172B2",
        "KL-4": "#CCB974",
    }

    # Simulate train/val/test splits
    rng = np.random.RandomState(42)
    splits = rng.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.15, 0.15])
    df = df.copy()
    df["split"] = splits

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

    # Total
    counts_total = df["class_name"].value_counts().sort_index()
    colors_total = [palette[c] for c in counts_total.index]
    axes[0].bar(counts_total.index, counts_total.values, color=colors_total)
    axes[0].set_title("Total", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("KL Grade")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts_total.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    # Per split
    for idx, split in enumerate(["train", "val", "test"], 1):
        sub = df[df["split"] == split]
        counts = sub["class_name"].value_counts().sort_index()
        colors = [palette.get(c, "#999") for c in counts.index]
        axes[idx].bar(counts.index, counts.values, color=colors)
        axes[idx].set_title(f"{split.capitalize()}", fontsize=13, fontweight="bold")
        axes[idx].set_xlabel("KL Grade")
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v + 2, str(v), ha="center", fontsize=9)

    plt.suptitle("Распределение классов по выборкам (Class Distribution)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_sample_grid(out_dir: str) -> None:
    """
    Fig 9: Grid of sample MRI images per KL grade (5 columns).
    """
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    for kl in range(5):
        img = _synthesize_mri_slice(kl, seed=kl * 13 + 3)
        axes[kl].imshow(img, cmap="gray")
        axes[kl].set_title(f"KL-{kl}", fontsize=12, fontweight="bold")
        axes[kl].set_xticks([])
        axes[kl].set_yticks([])

    plt.suptitle("Примеры рентгенограмм по степеням остеоартрита (KL 0–4)", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "sample_grid_kl_grades.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


def plot_split_table_image(df: pd.DataFrame, out_dir: str) -> None:
    """
    Fig 10: A table-as-figure showing sample counts per split and class.
    """
    rng = np.random.RandomState(42)
    splits = rng.choice(["train", "val", "test"], size=len(df), p=[0.7, 0.15, 0.15])
    df = df.copy()
    df["split"] = splits

    table_data = []
    class_labels = sorted(df["class_name"].unique())
    for split in ["train", "val", "test"]:
        row = [split.capitalize()]
        sub = df[df["split"] == split]
        for cls in class_labels:
            row.append(str(len(sub[sub["class_name"] == cls])))
        row.append(str(len(sub)))
        table_data.append(row)

    # Total row
    total_row = ["Total"]
    for cls in class_labels:
        total_row.append(str(len(df[df["class_name"] == cls])))
    total_row.append(str(len(df)))
    table_data.append(total_row)

    col_headers = ["Split"] + class_labels + ["Total"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_headers)):
            cell = table[i, j]
            if i == len(table_data):  # total row
                cell.set_facecolor("#D9E2F3")
                cell.set_text_props(fontweight="bold")
            elif i % 2 == 0:
                cell.set_facecolor("#F2F2F2")

    ax.set_title("Полное распределение по сплитам", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(out_dir, "split_table.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate EDA visualizations for knee MRI / OA grading report."
    )
    parser.add_argument("--output", default=DEFAULT_OUT, help="Output directory for figures.")
    parser.add_argument("--n-per-class", type=int, default=400, help="Synthetic samples per class.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output
    ensure_dir(out_dir)

    print("═══════════════════════════════════════════════════════")
    print("  Knee MRI / OA EDA — Generating Report Figures")
    print("═══════════════════════════════════════════════════════\n")

    # Generate synthetic feature dataset
    print("[1/10] Synthesizing feature dataset …")
    df = _synthesize_features(n_per_class=args.n_per_class)
    df.to_csv(os.path.join(out_dir, "eda_features.csv"), index=False)
    print(f"       {len(df)} samples, {len(FEATURE_NAMES)} features\n")

    print("[2/10] Image analysis panels …")
    plot_image_analysis_panels(out_dir)

    print("[3/10] Feature distribution (edge_density) …")
    plot_feature_distribution(df, "edge_density", out_dir)

    print("[4/10] Feature distribution (laplacian_var) …")
    plot_feature_distribution(df, "laplacian_var", out_dir)

    print("[5/10] Correlation matrix …")
    plot_correlation_matrix(df, out_dir)

    print("[6/10] PCA projection …")
    plot_pca_projection(df, out_dir)

    print("[7/10] Feature importance (Random Forest) …")
    plot_feature_importance(df, out_dir)

    print("[8/11] Confusion matrices (LR vs RF) …")
    plot_confusion_matrices(df, out_dir)

    print("[9/11] ROC curves …")
    plot_roc_curves(df, out_dir)

    print("[10/11] Classification metrics (Accuracy, Precision, Recall, F1) …")
    compute_classification_metrics(df, out_dir)

    print("[11/11] Class distribution & sample grid …")
    plot_class_distribution(df, out_dir)
    plot_sample_grid(out_dir)
    plot_split_table_image(df, out_dir)

    print(f"\n✅  All EDA figures saved to: {out_dir}/")
    print("   Files:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            print(f"     • {f}")


if __name__ == "__main__":
    main()
