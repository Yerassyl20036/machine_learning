import argparse
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from .linear_regression_scratch import LinearRegressionScratch
    from .logistic_regression_scratch import LogisticRegressionScratch
except ImportError:  # pragma: no cover
    from linear_regression_scratch import LinearRegressionScratch
    from logistic_regression_scratch import LogisticRegressionScratch


DEFAULT_INPUT = "data/processed/cleaned_variants.csv"
DEFAULT_RESULTS_DIR = "results/figures"
LEARNING_RATES = [0.001, 0.01, 0.1]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str, drop_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number])
    return X, y


def plot_loss_curves(loss_map: Dict[float, List[float]], out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for lr, losses in loss_map.items():
        plt.plot(losses, label=f"lr={lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_grad_curves(grad_map: Dict[float, List[float]], out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for lr, grads in grad_map.items():
        plt.plot(grads, label=f"lr={lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient L2 Norm")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_coef_evolution(coef_history: List[np.ndarray], feature_names: List[str], out_path: str, title: str, max_features: int) -> None:
    if not coef_history:
        return
    coef_matrix = np.vstack(coef_history)
    n_features = min(max_features, coef_matrix.shape[1])
    plt.figure(figsize=(9, 5))
    for idx in range(n_features):
        plt.plot(coef_matrix[:, idx], label=feature_names[idx])
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient Value")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_speed_comparison(times: Dict[str, float], losses: Dict[str, float], out_path: str, title: str) -> None:
    labels = list(times.keys())
    time_vals = [times[label] for label in labels]
    loss_vals = [losses[label] for label in labels]

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.bar(labels, time_vals, color=["#4c72b0", "#55a868"])
    plt.ylabel("Fit Time (s)")
    plt.title("Convergence Speed")

    plt.subplot(1, 2, 2)
    plt.bar(labels, loss_vals, color=["#4c72b0", "#55a868"])
    plt.ylabel("Final Loss")
    plt.title("Final Loss")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_linear_analysis(df: pd.DataFrame, results_dir: str, n_iters: int, coef_max: int) -> None:
    if "depth" not in df.columns:
        raise ValueError("Column 'depth' not found for linear regression.")

    drop_cols = ["depth"]
    if "label" in df.columns:
        drop_cols.append("label")

    X, y = prepare_features(df, "depth", drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    loss_map: Dict[float, List[float]] = {}
    grad_map: Dict[float, List[float]] = {}

    for lr in LEARNING_RATES:
        model = LinearRegressionScratch(lr=lr, n_iters=n_iters)
        model.fit(X_train, y_train)
        loss_map[lr] = model.loss_history
        grad_map[lr] = model.grad_history

        if lr == 0.01:
            plot_coef_evolution(
                model.coef_history,
                list(X_train.columns),
                os.path.join(results_dir, "linear_coef_evolution.png"),
                "Linear Coefficient Evolution (lr=0.01)",
                coef_max,
            )

    plot_loss_curves(
        loss_map,
        os.path.join(results_dir, "linear_loss_curves.png"),
        "Linear Loss Curves",
    )
    plot_grad_curves(
        grad_map,
        os.path.join(results_dir, "linear_gradients.png"),
        "Linear Gradient Magnitudes",
    )

    custom = LinearRegressionScratch(lr=0.01, n_iters=n_iters)
    start = time.perf_counter()
    custom.fit(X_train, y_train)
    custom_time = time.perf_counter() - start
    custom_pred = custom.predict(X_test)
    custom_loss = mean_squared_error(y_test, custom_pred)

    sk_model = LinearRegression()
    start = time.perf_counter()
    sk_model.fit(X_train, y_train)
    sk_time = time.perf_counter() - start
    sk_pred = sk_model.predict(X_test)
    sk_loss = mean_squared_error(y_test, sk_pred)

    plot_speed_comparison(
        {"custom": custom_time, "sklearn": sk_time},
        {"custom": custom_loss, "sklearn": sk_loss},
        os.path.join(results_dir, "linear_speed_comparison.png"),
        "Linear Regression Speed Comparison",
    )


def run_logistic_analysis(df: pd.DataFrame, results_dir: str, n_iters: int, coef_max: int) -> None:
    if "label" not in df.columns:
        raise ValueError("Label column not found. Run preprocessing first.")

    X, y = prepare_features(df, "label", ["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    loss_map: Dict[float, List[float]] = {}
    grad_map: Dict[float, List[float]] = {}

    for lr in LEARNING_RATES:
        model = LogisticRegressionScratch(lr=lr, n_iters=n_iters)
        model.fit(X_train, y_train)
        loss_map[lr] = model.loss_history
        grad_map[lr] = model.grad_history

        if lr == 0.01:
            plot_coef_evolution(
                model.coef_history,
                list(X_train.columns),
                os.path.join(results_dir, "logistic_coef_evolution.png"),
                "Logistic Coefficient Evolution (lr=0.01)",
                coef_max,
            )

    plot_loss_curves(
        loss_map,
        os.path.join(results_dir, "logistic_loss_curves.png"),
        "Logistic Loss Curves",
    )
    plot_grad_curves(
        grad_map,
        os.path.join(results_dir, "logistic_gradients.png"),
        "Logistic Gradient Magnitudes",
    )

    custom = LogisticRegressionScratch(lr=0.01, n_iters=n_iters)
    start = time.perf_counter()
    custom.fit(X_train, y_train)
    custom_time = time.perf_counter() - start
    custom_proba = custom.predict_proba(X_test)
    custom_loss = log_loss(y_test, custom_proba)

    sk_model = LogisticRegression(max_iter=2000)
    start = time.perf_counter()
    sk_model.fit(X_train, y_train)
    sk_time = time.perf_counter() - start
    sk_proba = sk_model.predict_proba(X_test)[:, 1]
    sk_loss = log_loss(y_test, sk_proba)

    plot_speed_comparison(
        {"custom": custom_time, "sklearn": sk_time},
        {"custom": custom_loss, "sklearn": sk_loss},
        os.path.join(results_dir, "logistic_speed_comparison.png"),
        "Logistic Regression Speed Comparison",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convergence analysis for custom models.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Processed CSV input.")
    parser.add_argument(
        "--task",
        choices=["linear", "logistic"],
        default="logistic",
        help="Task type for analysis.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help="Number of gradient descent iterations.",
    )
    parser.add_argument(
        "--coef-max",
        type=int,
        default=8,
        help="Max number of coefficient curves to plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.results_dir)

    df = load_dataset(args.input)

    if args.task == "linear":
        run_linear_analysis(df, args.results_dir, args.n_iters, args.coef_max)
    else:
        run_logistic_analysis(df, args.results_dir, args.n_iters, args.coef_max)


if __name__ == "__main__":
    main()
