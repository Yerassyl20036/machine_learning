import numpy as np


class LinearRegressionScratch:
    """Linear Regression with batch gradient descent.

    Minimizes mean squared error using analytical gradients.

    Loss: L = (1 / n) * sum((y_hat - y)^2)
    Gradients:
        dL/dw = (2 / n) * X^T (y_hat - y)
        dL/db = (2 / n) * sum(y_hat - y)
    """

    def __init__(self, lr: float = 0.01, n_iters: int = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
        self.grad_history: list[float] = []
        self.coef_history: list[np.ndarray] = []

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error."""
        return float(np.mean((y_pred - y_true) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """Fit the model using batch gradient descent.

        Args:
            X: Feature matrix with shape (n_samples, n_features).
            y: Target vector with shape (n_samples,).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        self.grad_history = []
        self.coef_history = []

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Analytical gradients for MSE loss.
            grad_w = (2.0 / n_samples) * (X.T @ error)
            grad_b = (2.0 / n_samples) * float(np.sum(error))

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            self.loss_history.append(self._mse(y, y_pred))
            self.grad_history.append(float(np.linalg.norm(grad_w)))
            self.coef_history.append(self.weights.copy())

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for input features."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias
