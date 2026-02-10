import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.compare_models import main as compare_main
from src.convergence_analysis import main as convergence_main
from src.db_setup import connect_db, init_database_if_missing
from src.linear_regression_scratch import LinearRegressionScratch
from src.logistic_regression_scratch import LogisticRegressionScratch
from src.preprocess_genomics import (
    DEFAULT_INPUT as RAW_DEFAULT,
    DEFAULT_OUTPUT as PROCESSED_DEFAULT,
    detect_format,
    load_to_db,
    normalize_dataframe,
    parse_csv,
    parse_vcf,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def connect_database() -> None:
    init_database_if_missing()
    conn = connect_db()
    conn.close()


def preprocess_data(input_path: str, input_format: str, output_path: str, skip_db: bool) -> pd.DataFrame:
    if input_format == "auto":
        input_format = detect_format(input_path)

    if input_format == "vcf":
        df = parse_vcf(input_path)
    else:
        df = parse_csv(input_path)

    df = normalize_dataframe(df)
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)

    if not skip_db:
        load_to_db(df)

    return df


def train_custom_models(df: pd.DataFrame, results_dir: str) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, "train", timestamp)
    ensure_dir(out_dir)

    if "depth" in df.columns:
        X_lin = df.drop(columns=["depth", "label"], errors="ignore").select_dtypes(include="number")
        y_lin = df["depth"]
        linear = LinearRegressionScratch(lr=0.01, n_iters=1500).fit(X_lin, y_lin)
        lin_pred = linear.predict(X_lin)
        lin_metrics = {
            "mse": mean_squared_error(y_lin, lin_pred),
            "r2": r2_score(y_lin, lin_pred),
        }
        pd.DataFrame([lin_metrics]).to_csv(os.path.join(out_dir, "linear_metrics.csv"), index=False)

    if "label" in df.columns:
        X_log = df.drop(columns=["label"], errors="ignore").select_dtypes(include="number")
        y_log = df["label"]
        logistic = LogisticRegressionScratch(lr=0.1, n_iters=1500).fit(X_log, y_log)
        pd.DataFrame({"loss": logistic.loss_history, "accuracy": logistic.acc_history}).to_csv(
            os.path.join(out_dir, "logistic_history.csv"), index=False
        )


def run_compare_script(input_path: str) -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["compare_models", "--input", input_path]
        compare_main()
    finally:
        sys.argv = original_argv


def run_convergence_script(input_path: str, task: str) -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["convergence_analysis", "--input", input_path, "--task", task]
        convergence_main()
    finally:
        sys.argv = original_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the genomics ML pipeline.")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "compare"],
        default="compare",
        help="Pipeline mode.",
    )
    parser.add_argument("--input", default=RAW_DEFAULT, help="Raw input file path.")
    parser.add_argument(
        "--format",
        choices=["vcf", "csv", "auto"],
        default="auto",
        help="Input format for preprocessing.",
    )
    parser.add_argument(
        "--processed",
        default=PROCESSED_DEFAULT,
        help="Processed CSV output path.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip loading data into database.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    connect_database()

    df: pd.DataFrame | None = None
    if args.mode in {"train", "compare"}:
        df = preprocess_data(args.input, args.format, args.processed, args.skip_db)
    elif not os.path.exists(args.processed):
        raise FileNotFoundError("Processed file not found. Run preprocessing first.")

    if args.mode == "train":
        train_custom_models(df, "results")
        return

    if args.mode == "evaluate":
        run_compare_script(args.processed)
        run_convergence_script(args.processed, "logistic")
        return

    run_compare_script(args.processed)
    run_convergence_script(args.processed, "logistic")


if __name__ == "__main__":
    main()
