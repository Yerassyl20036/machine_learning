"""
Ensemble: combines similarity scores from multiple algorithms via
weighted score-level fusion. Weights are optimized on training data.
Also supports a stacking classifier (LogisticRegression meta-learner).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Default equal weights (one per algorithm)
DEFAULT_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)


class WeightedEnsemble:
    """
    Score-level fusion: final_score = sum(w_i * score_i).
    Decision threshold: predict same person if score >= threshold.
    """

    def __init__(self, weights: np.ndarray | None = None, threshold: float = 0.5):
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()
        self.threshold = threshold

    def fuse_scores(self, scores: np.ndarray) -> float:
        """
        Args:
            scores: 1-D array of similarity scores [s_daugman, s_lbp, s_hog, s_orb]
        Returns:
            fused similarity score in [0, 1]
        """
        w = self.weights[:len(scores)]
        w = w / w.sum()
        return float(np.dot(w, scores[:len(w)]))

    def predict(self, scores: np.ndarray) -> int:
        """Returns 1 (same person) or 0 (different person)."""
        return int(self.fuse_scores(scores) >= self.threshold)

    def optimize_weights(self,
                          score_matrix: np.ndarray,
                          labels: np.ndarray,
                          n_trials: int = 1000,
                          seed: int = 42) -> None:
        """
        Grid-search optimal weights on verification pairs using EER minimization.

        Args:
            score_matrix: (N, n_algos) array of per-algorithm similarity scores
            labels:       (N,) binary array: 1 = genuine pair, 0 = impostor pair
            n_trials:     number of random weight vectors to try
        """
        rng = np.random.default_rng(seed)
        best_eer = float("inf")
        best_weights = self.weights.copy()

        n_algos = score_matrix.shape[1]
        for _ in range(n_trials):
            w = rng.dirichlet(np.ones(n_algos)).astype(np.float32)
            fused = score_matrix @ w
            eer = _compute_eer(fused, labels)
            if eer < best_eer:
                best_eer = eer
                best_weights = w

        self.weights = best_weights
        print(f"[Ensemble] Optimized weights={self.weights.round(3)}, EER={best_eer:.4f}")


class StackingEnsemble:
    """
    Stacking (meta-learning): train a Logistic Regression on the concatenated
    feature vectors from all algorithms, plus their pairwise similarity scores.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self._fitted = False

    def fit(self, score_matrix: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            score_matrix: (N, n_algos) pairwise similarity scores
            labels:       (N,) binary: 1 = genuine, 0 = impostor
        """
        X = self.scaler.fit_transform(score_matrix)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict_proba(self, scores: np.ndarray) -> float:
        """
        Args:
            scores: (n_algos,) or (1, n_algos) similarity scores
        Returns:
            probability of being a genuine pair
        """
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fitted first.")
        X = self.scaler.transform(scores.reshape(1, -1))
        return float(self.clf.predict_proba(X)[0, 1])

    def predict(self, scores: np.ndarray) -> int:
        return int(self.predict_proba(scores) >= 0.5)


# ── utility ───────────────────────────────────────────────────────────────────

def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER) from similarity scores and binary labels.
    EER = threshold where FAR == FRR.
    """
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    best_eer = 1.0
    for t in thresholds:
        far = float((impostor >= t).sum()) / (len(impostor) + 1e-10)
        frr = float((genuine < t).sum()) / (len(genuine) + 1e-10)
        eer = abs(far - frr)
        if eer < abs(best_eer - 0.5) * 2 + 1e-10:
            best_eer = (far + frr) / 2
    return best_eer


def build_verification_pairs(features_by_class: dict[int, list],
                              extractor_fn,
                              n_impostors_per_genuine: int = 1,
                              seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (score, label) pairs for EER evaluation.
    Genuine pairs: same class. Impostor pairs: different classes.

    Args:
        features_by_class: {class_idx: [feat1, feat2, ...]}
        extractor_fn:      function(feat_a, feat_b) -> float similarity score

    Returns:
        scores (N,), labels (N,) binary
    """
    import random
    rng = random.Random(seed)

    genuine_scores, impostor_scores = [], []
    classes = list(features_by_class.keys())

    for cls, feats in features_by_class.items():
        if len(feats) < 2:
            continue
        # Genuine pairs from this class
        for i in range(len(feats) - 1):
            s = extractor_fn(feats[i], feats[i + 1])
            genuine_scores.append(s)

        # Impostor pairs: pick a random different class
        other_classes = [c for c in classes if c != cls]
        if not other_classes:
            continue
        for _ in range(len(feats) - 1):
            other_cls = rng.choice(other_classes)
            other_feat = rng.choice(features_by_class[other_cls])
            s = extractor_fn(feats[rng.randint(0, len(feats) - 1)], other_feat)
            impostor_scores.append(s)

    scores = np.array(genuine_scores + impostor_scores, dtype=np.float32)
    labels = np.array(
        [1] * len(genuine_scores) + [0] * len(impostor_scores), dtype=np.int32
    )
    return scores, labels
