"""
Iris Recognition Demo
Usage:
    python iris_demo.py compare <img1> <img2>
    python iris_demo.py batch  <folder>
    python iris_demo.py results
"""

import sys
import os
import numpy as np
from pathlib import Path

# Allow running from any directory
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.preprocess import preprocess
from src.algorithms import lbp as lbp_algo

# ── Feature extraction via LBP pipeline ──────────────────────────────────────

def extract_features(img_path: str) -> np.ndarray:
    norm = preprocess(str(img_path), target_size=128)
    feat = lbp_algo.extract(norm)
    return feat.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(lbp_algo.similarity(a, b))


# Threshold calibrated on the dataset at EER operating point
# similarity >= THRESHOLD  →  MATCH
THRESHOLD = 0.980

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_compare(img1: str, img2: str):
    feat1 = extract_features(img1)
    feat2 = extract_features(img2)
    sim   = cosine_similarity(feat1, feat2)
    match = sim >= THRESHOLD
    print(f"\nCosine Similarity: {sim:.4f}")
    result = "✅ MATCH" if match else "❌ NO MATCH"
    print(f"Result: {result}")
    print()


def cmd_batch(folder: str):
    folder = Path(folder)
    imgs = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    if not imgs:
        print(f"No images found in {folder}")
        sys.exit(1)

    print(f"\n📷 Processing {len(imgs)} images...\n")

    features = {}
    for p in imgs:
        try:
            feat = extract_features(str(p))
            features[p.name] = feat
            print(f"  ✅ {p.name:<20} →  Feature vector: {len(feat)} values")
        except Exception as e:
            print(f"  ❌ {p.name}: {e}")

    names = list(features.keys())
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i+1, len(names))]

    total = len(pairs)
    matches = 0
    no_match = 0

    sep = "-" * 50
    header = "=" * 50
    print(f"\n{header}")
    print(f"  COMPARISON RESULTS (LBP Cosine Similarity)")
    print(f"  Threshold: {THRESHOLD}  (sim >= {THRESHOLD} = MATCH)")
    print(f"{header}")

    for a, b in pairs:
        sim = cosine_similarity(features[a], features[b])
        is_match = sim >= THRESHOLD
        icon = "✅ MATCH" if is_match else "❌ NO MATCH"
        if is_match:
            matches += 1
        else:
            no_match += 1
        print(f"\n{a} vs {b}")
        print(f"  sim = {sim:.4f}  →  {icon}")
        print(sep)

    print(f"\n📊 Total: {total} pairs | Matches: {matches} | No match: {no_match}")
    print()


def cmd_results():
    """Print saved algorithm benchmark results."""
    base = Path(__file__).parent
    csv_path = base / "results" / "comparison" / "algorithm_comparison.csv"
    history_path = base / "results" / "cnn_history.json"

    print("\n" + "=" * 62)
    print("  IRIS RECOGNITION — ALGORITHM BENCHMARK RESULTS")
    print("  Dataset: Custom iris dataset (16 subjects × 80 images)")
    print("  Method:  LBP (Local Binary Pattern) + Cosine Similarity")
    print("=" * 62)

    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        col_w = [28, 8, 8, 8, 8, 10]
        headers = ["Algorithm", "EER", "FAR", "FRR", "AUC", "Accuracy"]
        row_fmt = "  {:<28}{:<8}{:<8}{:<8}{:<8}{:<10}"
        print()
        print(row_fmt.format(*headers))
        print("  " + "-" * 58)
        for r in rows:
            alg = r["algorithm"]
            eer  = f"{float(r['EER']):.4f}"  if r.get("EER") else "–"
            far  = f"{float(r['FAR_at_EER']):.4f}" if r.get("FAR_at_EER") else "–"
            frr  = f"{float(r['FRR_at_EER']):.4f}" if r.get("FRR_at_EER") else "–"
            auc  = f"{float(r['AUC']):.4f}"  if r.get("AUC") else "–"
            acc  = f"{float(r['Accuracy']):.4f}" if r.get("Accuracy") else "–"
            print(row_fmt.format(alg, eer, far, frr, auc, acc))
    else:
        print("  [!] Run main.py first to generate benchmark results")

    if history_path.exists():
        import json
        h = json.loads(history_path.read_text())
        n_epochs = len(h.get("train_acc", []))
        best_val_acc = max(h.get("val_acc", [0]))
        last_val_acc = h.get("val_acc", [0])[-1]
        print()
        print(f"  IrisNet CNN  (trained {n_epochs} epochs)")
        print(f"  Best Val Accuracy : {best_val_acc:.4f}  ({best_val_acc*100:.2f}%)")
        print(f"  Last Val Accuracy : {last_val_acc:.4f}  ({last_val_acc*100:.2f}%)")

    print()
    print("=" * 62)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "compare":
        if len(sys.argv) < 4:
            print("Usage: python iris_demo.py compare <img1> <img2>")
            sys.exit(1)
        cmd_compare(sys.argv[2], sys.argv[3])

    elif cmd == "batch":
        if len(sys.argv) < 3:
            print("Usage: python iris_demo.py batch <folder>")
            sys.exit(1)
        cmd_batch(sys.argv[2])

    elif cmd == "results":
        cmd_results()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
