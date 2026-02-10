import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    from .linear_regression_scratch import LinearRegressionScratch
    from .logistic_regression_scratch import LogisticRegressionScratch
except ImportError:  # pragma: no cover
    from linear_regression_scratch import LinearRegressionScratch
    from logistic_regression_scratch import LogisticRegressionScratch


DEFAULT_INPUT = "data/processed/cleaned_variants.csv"
DEFAULT_RESULTS_DIR = "results"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str, drop_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    y = df[target]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number])
    return X, y


def compare_linear(X_train, X_test, y_train, y_test):
    custom = LinearRegressionScratch(lr=0.01, n_iters=2000).fit(X_train, y_train)
    sk_model = LinearRegression().fit(X_train, y_train)

    custom_pred = custom.predict(X_test)
    sk_pred = sk_model.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "model": "custom",
                "mse": mean_squared_error(y_test, custom_pred),
                "r2": r2_score(y_test, custom_pred),
            },
            {
                "model": "sklearn",
                "mse": mean_squared_error(y_test, sk_pred),
                "r2": r2_score(y_test, sk_pred),
            },
        ]
    )

    coef = pd.DataFrame(
        {
            "feature": X_train.columns,
            "custom_coef": custom.weights,
            "sklearn_coef": sk_model.coef_,
        }
    )

    return custom, sk_model, custom_pred, sk_pred, metrics, coef


def compare_logistic(X_train, X_test, y_train, y_test):
    custom = LogisticRegressionScratch(lr=0.1, n_iters=2000).fit(X_train, y_train)
    sk_model = LogisticRegression(max_iter=2000).fit(X_train, y_train)

    custom_proba = custom.predict_proba(X_test)
    sk_proba = sk_model.predict_proba(X_test)[:, 1]

    custom_pred = (custom_proba >= 0.5).astype(int)
    sk_pred = (sk_proba >= 0.5).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "model": "custom",
                "accuracy": accuracy_score(y_test, custom_pred),
                "precision": precision_score(y_test, custom_pred, zero_division=0),
                "recall": recall_score(y_test, custom_pred, zero_division=0),
                "f1": f1_score(y_test, custom_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, custom_proba),
            },
            {
                "model": "sklearn",
                "accuracy": accuracy_score(y_test, sk_pred),
                "precision": precision_score(y_test, sk_pred, zero_division=0),
                "recall": recall_score(y_test, sk_pred, zero_division=0),
                "f1": f1_score(y_test, sk_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, sk_proba),
            },
        ]
    )

    coef = pd.DataFrame(
        {
            "feature": X_train.columns,
            "custom_coef": custom.weights,
            "sklearn_coef": sk_model.coef_.ravel(),
        }
    )

    return custom, sk_model, custom_pred, sk_pred, custom_proba, sk_proba, metrics, coef


def plot_linear(y_test, custom_pred, sk_pred, out_dir: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, custom_pred, s=10, alpha=0.7)
    plt.xlabel("True")
    plt.ylabel("Custom Pred")
    plt.title("Custom Linear")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, sk_pred, s=10, alpha=0.7)
    plt.xlabel("True")
    plt.ylabel("Sklearn Pred")
    plt.title("Sklearn Linear")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "linear_predictions.png"), dpi=150)
    plt.close()


def plot_logistic(y_test, custom_proba, sk_proba, out_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    fpr_c, tpr_c, _ = roc_curve(y_test, custom_proba)
    fpr_s, tpr_s, _ = roc_curve(y_test, sk_proba)
    plt.plot(fpr_c, tpr_c, label="custom")
    plt.plot(fpr_s, tpr_s, label="sklearn")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "logistic_roc.png"), dpi=150)
    plt.close()


def plot_loss(custom_model, out_dir: str, name: str) -> None:
    if not custom_model.loss_history:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(custom_model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{name} Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name.lower()}_loss.png"), dpi=150)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare custom vs sklearn models.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Processed CSV input.")
    parser.add_argument(
        "--task",
        choices=["linear", "logistic"],
        default="logistic",
        help="Task type for comparison.",
    )
    parser.add_argument(
        "--target",
        default="depth",
        help="Target column for linear regression.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory to save outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = load_dataset(args.input)

    ensure_dir(args.results_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.results_dir, args.task, timestamp)
    ensure_dir(out_dir)

    if args.task == "linear":
        if args.target not in df.columns:
            raise ValueError(f"Target column '{args.target}' not found in input.")
        drop_cols = [args.target]
        if "label" in df.columns:
            drop_cols.append("label")
        X, y = prepare_features(df, args.target, drop_cols)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        custom, sk_model, custom_pred, sk_pred, metrics, coef = compare_linear(
            X_train, X_test, y_train, y_test
        )

        metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        coef.to_csv(os.path.join(out_dir, "coefficients.csv"), index=False)

        preds = pd.DataFrame(
            {
                "y_true": y_test.values,
                "custom_pred": custom_pred,
                "sklearn_pred": sk_pred,
            }
        )
        preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

        plot_linear(y_test, custom_pred, sk_pred, out_dir)
        plot_loss(custom, out_dir, "Linear")

        return

    if "label" not in df.columns:
        raise ValueError("Label column not found. Run preprocessing to add labels.")

    X, y = prepare_features(df, "label", ["label"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    (
        custom,
        sk_model,
        custom_pred,
        sk_pred,
        custom_proba,
        sk_proba,
        metrics,
        coef,
    ) = compare_logistic(X_train, X_test, y_train, y_test)

    metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    coef.to_csv(os.path.join(out_dir, "coefficients.csv"), index=False)

    preds = pd.DataFrame(
        {
            "y_true": y_test.values,
            "custom_pred": custom_pred,
            "sklearn_pred": sk_pred,
            "custom_proba": custom_proba,
            "sklearn_proba": sk_proba,
        }
    )
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    plot_logistic(y_test, custom_proba, sk_proba, out_dir)
    plot_loss(custom, out_dir, "Logistic")


if __name__ == "__main__":
    main()
