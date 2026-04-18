"""
Main evaluation script — runs all parts of the iris recognition project:

  Part 1: Benchmark 4 classical algorithms + ensemble on the custom dataset
  Part 2: Train the multi-branch IrisNet CNN
  Part 3: End-to-end pipeline demo

Usage:
    python main.py --iris_root ../iris --results_dir results --epochs 30
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from src.dataset import load_dataset, train_test_split_by_person
from src.preprocess import preprocess
from src.algorithms import daugman, lbp as lbp_algo, hog_algo, orb_algo
from src.ensemble import (
    WeightedEnsemble, StackingEnsemble, build_verification_pairs
)
from src.metrics import full_report, compute_eer
from src.neural_model import IrisNet, IrisDataset, train


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iris_root",   default="../iris",  help="Path to iris dataset root")
    p.add_argument("--results_dir", default="results",  help="Output directory")
    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--skip_cnn",    action="store_true", help="Skip CNN training")
    return p.parse_args()


# ── Feature extraction ────────────────────────────────────────────────────────

ALGORITHMS = {
    "Daugman": {
        "extract": daugman.extract,
        "similarity": daugman.similarity,
    },
    "LBP": {
        "extract": lbp_algo.extract,
        "similarity": lbp_algo.similarity,
    },
    "HOG": {
        "extract": hog_algo.extract,
        "similarity": hog_algo.similarity,
    },
    "ORB": {
        "extract": orb_algo.extract,
        "similarity": orb_algo.similarity,
    },
}


def extract_all_features(paths, labels, target_size=128):
    """
    Preprocess all images and extract features for each algorithm.
    Returns:
        features: dict { algo_name: list of feature vectors }
        valid_labels: list of labels (some images may fail preprocessing)
        valid_paths:  list of paths
    """
    features = {name: [] for name in ALGORITHMS}
    valid_labels, valid_paths, norm_images = [], [], []

    print("Preprocessing images...")
    for path, label in tqdm(zip(paths, labels), total=len(paths)):
        try:
            norm = preprocess(path, target_size=target_size)
            norm_images.append(norm)
            valid_labels.append(label)
            valid_paths.append(path)
        except Exception as e:
            print(f"  skip {path}: {e}")

    print("Extracting features...")
    for name, algo in ALGORITHMS.items():
        print(f"  [{name}]")
        for norm in tqdm(norm_images):
            features[name].append(algo["extract"](norm))

    return features, valid_labels, valid_paths, norm_images


# ── Identification (1:N) ──────────────────────────────────────────────────────

def nearest_neighbor_identify(train_feats, train_labels,
                               test_feats, similarity_fn):
    """Predict class by nearest-neighbour in similarity space."""
    preds = []
    for q_feat in test_feats:
        best_score, best_label = -1, -1
        for g_feat, g_label in zip(train_feats, train_labels):
            s = similarity_fn(q_feat, g_feat)
            if s > best_score:
                best_score = s
                best_label = g_label
        preds.append(best_label)
    return np.array(preds)


# ── Verification (1:1) ────────────────────────────────────────────────────────

def build_genuine_impostor(feats_by_class, similarity_fn, seed=42):
    """Build genuine and impostor score lists."""
    import random
    rng = random.Random(seed)
    genuine, impostor = [], []
    classes = list(feats_by_class.keys())

    for cls, feats in feats_by_class.items():
        if len(feats) < 2:
            continue
        for i in range(len(feats) - 1):
            genuine.append(similarity_fn(feats[i], feats[i + 1]))
        other_classes = [c for c in classes if c != cls]
        if not other_classes:
            continue
        for _ in range(len(feats) - 1):
            other_cls = rng.choice(other_classes)
            other_feat = rng.choice(feats_by_class[other_cls])
            impostor.append(similarity_fn(
                feats[rng.randint(0, len(feats) - 1)],
                other_feat
            ))

    return np.array(genuine, dtype=np.float32), np.array(impostor, dtype=np.float32)


# ── Comparison plots ──────────────────────────────────────────────────────────

def plot_comparison(reports: list[dict], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(reports)
    df.to_csv(os.path.join(save_dir, "algorithm_comparison.csv"), index=False)
    print("\n" + df.to_string(index=False))

    # Bar chart: EER
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["EER", "AUC"]
    if "Accuracy" in df.columns:
        metrics.append("Accuracy")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    for i, metric in enumerate(["EER", "AUC", "Accuracy"] if "Accuracy" in df.columns else ["EER", "AUC"]):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.bar(df["algorithm"], df[metric],
               color=colors[:len(df)], edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, 1)
        for bar, val in zip(ax.patches, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Algorithm Comparison — Iris Recognition", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "algorithm_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart → {save_dir}/algorithm_comparison.png")


def plot_training_history(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle("IrisNet Training History", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "cnn_training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("IRIS RECOGNITION — Full Pipeline")
    print(f"{'='*60}\n")

    iris_root = os.path.abspath(args.iris_root)
    print(f"Dataset: {iris_root}")

    paths, labels, classes = load_dataset(iris_root)
    print(f"Classes: {len(classes)}, Images: {len(paths)}")

    train_paths, train_labels, test_paths, test_labels = train_test_split_by_person(
        paths, labels, classes, test_ratio=0.2
    )
    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}\n")

    # ── Part 1: Classical algorithms ──────────────────────────────────────────
    print("=" * 60)
    print("PART 1 — Classical Algorithm Benchmark")
    print("=" * 60)

    train_feats, train_labels_valid, _, train_norms = extract_all_features(
        train_paths, train_labels, args.target_size
    )
    test_feats, test_labels_valid, _, test_norms = extract_all_features(
        test_paths, test_labels, args.target_size
    )

    reports = []
    algo_genuine = {}
    algo_impostor = {}
    algo_test_scores = {}   # for ensemble: {algo: (N_test,) scores per test sample's NN}

    for name, algo in ALGORITHMS.items():
        print(f"\n[{name}]")
        sim_fn = algo["similarity"]

        # 1:N identification
        preds = nearest_neighbor_identify(
            train_feats[name], train_labels_valid,
            test_feats[name], sim_fn
        )

        # 1:1 verification (on test set, group by class)
        test_by_class = {}
        for feat, lbl in zip(test_feats[name], test_labels_valid):
            test_by_class.setdefault(lbl, []).append(feat)

        genuine, impostor = build_genuine_impostor(test_by_class, sim_fn)
        algo_genuine[name] = genuine
        algo_impostor[name] = impostor

        report = full_report(
            name, genuine, impostor,
            y_true=np.array(test_labels_valid),
            y_pred=preds
        )
        reports.append(report)
        print(f"  EER={report['EER']:.4f}  AUC={report['AUC']:.4f}  "
              f"Acc={report.get('Accuracy','N/A')}")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    print("\n[Ensemble — Weighted Fusion]")
    # Build per-sample multi-algo score matrix for test verification
    # Use same pairs structure: genuine/impostor per class
    n_algos = len(ALGORITHMS)
    algo_names = list(ALGORITHMS.keys())

    # Build fused genuine/impostor scores
    # Use minimum length across algorithms
    min_gen = min(len(algo_genuine[n]) for n in algo_names)
    min_imp = min(len(algo_impostor[n]) for n in algo_names)

    gen_matrix  = np.stack([algo_genuine[n][:min_gen]  for n in algo_names], axis=1)
    imp_matrix  = np.stack([algo_impostor[n][:min_imp] for n in algo_names], axis=1)
    score_matrix = np.vstack([gen_matrix, imp_matrix])
    ens_labels   = np.concatenate([
        np.ones(min_gen, dtype=np.int32),
        np.zeros(min_imp, dtype=np.int32)
    ])

    # Weighted ensemble
    w_ens = WeightedEnsemble()
    w_ens.optimize_weights(score_matrix, ens_labels)
    ens_scores = score_matrix @ w_ens.weights
    ens_genuine  = ens_scores[ens_labels == 1]
    ens_impostor = ens_scores[ens_labels == 0]
    ens_eer, _ = compute_eer(ens_genuine, ens_impostor)

    # Stacking ensemble
    print("[Ensemble — Stacking]")
    stack_ens = StackingEnsemble()
    stack_ens.fit(score_matrix, ens_labels)
    stack_proba = np.array([stack_ens.predict_proba(r) for r in score_matrix])
    stack_genuine  = stack_proba[ens_labels == 1]
    stack_impostor = stack_proba[ens_labels == 0]
    stack_eer, _ = compute_eer(stack_genuine, stack_impostor)

    from src.metrics import auc_score
    ens_report = {
        "algorithm": "Ensemble (Weighted)",
        "EER": round(float(ens_eer), 4),
        "FAR_at_EER": None,
        "FRR_at_EER": None,
        "AUC": round(auc_score(ens_genuine, ens_impostor), 4),
    }
    stack_report = {
        "algorithm": "Ensemble (Stacking)",
        "EER": round(float(stack_eer), 4),
        "FAR_at_EER": None,
        "FRR_at_EER": None,
        "AUC": round(auc_score(stack_genuine, stack_impostor), 4),
    }
    reports.append(ens_report)
    reports.append(stack_report)
    print(f"  Weighted EER={ens_eer:.4f}  AUC={ens_report['AUC']:.4f}")
    print(f"  Stacking EER={stack_eer:.4f}  AUC={stack_report['AUC']:.4f}")

    # Plot comparison
    plot_comparison(
        [r for r in reports if r.get("Accuracy") is not None or r.get("AUC")],
        os.path.join(args.results_dir, "comparison")
    )

    # ── Part 2 & 3: CNN ───────────────────────────────────────────────────────
    if not args.skip_cnn:
        print(f"\n{'='*60}")
        print("PART 2 & 3 — Multi-Branch IrisNet CNN")
        print(f"{'='*60}")

        import torch

        train_dataset = IrisDataset(train_norms, train_labels_valid, augment=True,
                                     target_size=args.target_size)
        val_dataset   = IrisDataset(test_norms, test_labels_valid, augment=False,
                                     target_size=args.target_size)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0
        )

        model = IrisNet(n_classes=len(classes), branch_dim=128, dropout=0.4)
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        history = train(model, train_loader, val_loader,
                        n_epochs=args.epochs, lr=1e-3)

        # Save model
        model_path = os.path.join(args.results_dir, "irisnet.pt")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved → {model_path}")

        # Save history
        history_path = os.path.join(args.results_dir, "cnn_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        plot_training_history(history, os.path.join(args.results_dir, "figures"))
        print(f"Training plots saved → results/figures/cnn_training_history.png")

        final_val_acc = history["val_acc"][-1]
        reports.append({
            "algorithm": "IrisNet CNN",
            "EER": None,
            "AUC": None,
            "Accuracy": round(final_val_acc, 4),
        })

    # ── Save full report ──────────────────────────────────────────────────────
    report_df = pd.DataFrame(reports)
    report_path = os.path.join(args.results_dir, "full_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nFull report saved → {report_path}")
    print("\n" + report_df.to_string(index=False))


if __name__ == "__main__":
    main()
