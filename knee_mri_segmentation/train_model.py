#!/usr/bin/env python3
"""
Train Random Forest on eda_features.csv and save:
  models/rf_model.pkl  — trained classifier
  models/scaler.pkl    — fitted StandardScaler

Run once before starting the web app:
  python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "results", "eda_figures", "eda_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "mean_intensity", "std_intensity", "skew_intensity", "kurt_intensity",
    "laplacian_var", "fft_mean", "edge_density", "contrast", "homogeneity",
    "energy", "correlation", "entropy", "bone_area_ratio", "cartilage_area_ratio",
    "joint_space_width", "osteophyte_score", "sclerosis_index",
    "mean_gradient", "hist_peak_pos", "hist_spread",
]


def main():
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna()
    print(f"  {len(df)} samples, {df.shape[1]} columns")

    X = df[FEATURE_NAMES].values
    y = df["kl_grade"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Training Random Forest (n_estimators=300)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",   # handles KL-4 imbalance
    )
    rf.fit(X_train_sc, y_train)

    y_pred = rf.predict(X_test_sc)
    print("\nTest-set report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["KL-0","KL-1","KL-2","KL-3","KL-4"]))

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    model_path  = os.path.join(MODELS_DIR, "rf_model.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(rf,     model_path)
    print(f"\nSaved:\n  {scaler_path}\n  {model_path}")


if __name__ == "__main__":
    main()
