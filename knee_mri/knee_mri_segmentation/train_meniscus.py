#!/usr/bin/env python3
"""
Train a binary Random Forest classifier for meniscus tear detection.

Input data:  KneeMRI-Meniscus-Dataset/
               Original/0/  ← normal meniscus  (class 0)
               Original/1/  ← torn  meniscus   (class 1)
               Augmented/0/ ← augmented normals (used as extra training data)
               Augmented/1/ ← augmented torn    (used as extra training data)

Output:
  models/meniscus_rf.pkl     — trained RandomForestClassifier
  models/meniscus_scaler.pkl — fitted StandardScaler

Usage:
  python train_meniscus.py
  python train_meniscus.py --data /path/to/KneeMRI-Meniscus-Dataset
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.meniscus_features import FEATURE_NAMES, extract_features, load_gray

DEFAULT_DATA = (
    Path(__file__).parent.parent.parent
    / "real data for knee"
    / "KneeMRI-Meniscus-Dataset"
)
MODELS_DIR = ROOT / "models"


def load_dataset(data_root: Path, use_augmented: bool = True):
    """
    Walk Original/{0,1} and (optionally) Augmented/{0,1}, extract features.
    Returns X (n x 20), y (n,), split_tag (n,) = 'original' or 'augmented'.
    """
    X, y, tags = [], [], []

    for split_name, is_aug in [("Original", False), ("Augmented", True)]:
        if is_aug and not use_augmented:
            continue
        for cls in [0, 1]:
            cls_dir = data_root / split_name / str(cls)
            if not cls_dir.exists():
                print(f"  WARNING: {cls_dir} not found, skipping")
                continue
            files = sorted(cls_dir.glob("*.png"))
            print(f"  {split_name}/class{cls}: {len(files)} images")
            for fpath in tqdm(files, desc=f"{split_name}/{cls}", leave=False):
                try:
                    img = load_gray(fpath)
                    feat = extract_features(img)
                    X.append(feat)
                    y.append(cls)
                    tags.append("augmented" if is_aug else "original")
                except Exception as e:
                    print(f"    skip {fpath.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(tags)


def main():
    parser = argparse.ArgumentParser(description="Train meniscus tear binary classifier.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to KneeMRI-Meniscus-Dataset")
    parser.add_argument("--no-augmented", action="store_true", help="Skip augmented images")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data)
    if not data_root.exists():
        print(f"ERROR: Dataset not found at {data_root}")
        print("Pass the correct path via --data")
        sys.exit(1)

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"\nLoading dataset from: {data_root}")
    X, y, tags = load_dataset(data_root, use_augmented=not args.no_augmented)
    print(f"\nTotal: {len(X)} samples | class 0 (normal): {(y==0).sum()} | class 1 (torn): {(y==1).sum()}")

    # ── Split: use Original for train/val/test; Augmented added to train only ──
    orig_mask = tags == "original"
    X_orig, y_orig = X[orig_mask], y[orig_mask]
    X_aug,  y_aug  = X[~orig_mask], y[~orig_mask]

    # Stratified 70/15/15 split on Original
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
    train_idx, tmp_idx = next(sss1.split(X_orig, y_orig))

    X_tmp, y_tmp = X_orig[tmp_idx], y_orig[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
    val_idx, test_idx = next(sss2.split(X_tmp, y_tmp))

    X_val,  y_val  = X_tmp[val_idx],  y_tmp[val_idx]
    X_test, y_test = X_tmp[test_idx], y_tmp[test_idx]

    # Training set = original train + all augmented
    X_train = np.vstack([X_orig[train_idx], X_aug])
    y_train = np.concatenate([y_orig[train_idx], y_aug])

    print(f"\nSplit summary:")
    print(f"  train : {len(X_train)} (original {len(train_idx)} + augmented {len(X_aug)})")
    print(f"  val   : {len(X_val)}")
    print(f"  test  : {len(X_test)}")

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining RandomForest (n_estimators={args.n_estimators}) …")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    for split_name, Xsc, yt in [("val", X_val_sc, y_val), ("test", X_test_sc, y_test)]:
        y_pred = rf.predict(Xsc)
        y_prob = rf.predict_proba(Xsc)[:, 1]
        acc = accuracy_score(yt, y_pred)
        auc = roc_auc_score(yt, y_prob)
        print(f"\n── {split_name.upper()} ──")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(yt, y_pred, target_names=["Normal", "Torn"], digits=4))
        cm = confusion_matrix(yt, y_pred)
        print(f"  Confusion matrix:\n{cm}")

    # ── Feature importance ─────────────────────────────────────────────────────
    imp = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\nTop-10 features:")
    print(imp.head(10).to_string())

    # ── Save ───────────────────────────────────────────────────────────────────
    scaler_path = MODELS_DIR / "meniscus_scaler.pkl"
    model_path  = MODELS_DIR / "meniscus_rf.pkl"
    joblib.dump(scaler, scaler_path)
    joblib.dump(rf, model_path)
    print(f"\nSaved:\n  {model_path}\n  {scaler_path}")


if __name__ == "__main__":
    main()
