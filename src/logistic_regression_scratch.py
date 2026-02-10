import numpy as np


class LogisticRegressionScratch:
    """Logistic Regression with batch gradient descent.

    Uses sigmoid activation and minimizes binary cross-entropy (BCE).

    Sigmoid: p = 1 / (1 + exp(-z))
    Loss: L = -(1 / n) * sum(y * log(p) + (1 - y) * log(1 - p))
    Gradients:
        dL/dw = (1 / n) * X^T (p - y)
        dL/db = (1 / n) * sum(p - y)
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.acc_history: list[float] = []
        self.grad_history: list[float] = []
        self.coef_history: list[np.ndarray] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Fit the model using batch gradient descent.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Binary targets with shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        self.acc_history = []
        self.grad_history = []
        self.coef_history = []

        for _ in range(self.n_iters):
            logits = X @ self.weights + self.bias
            probs = self._sigmoid(logits)

            # Analytical gradients for BCE loss.
            error = probs - y
            grad_w = (1.0 / n_samples) * (X.T @ error)
            grad_b = (1.0 / n_samples) * float(np.sum(error))

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            self.loss_history.append(self._bce(y, probs))
            preds = (probs >= 0.5).astype(float)
            self.acc_history.append(self._accuracy(y, preds))
            self.grad_history.append(float(np.linalg.norm(grad_w)))
            self.coef_history.append(self.weights.copy())

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        logits = X @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
