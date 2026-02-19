import argparse
import os

import pandas as pd

DEFAULT_SEG = "results/segmentation/metrics.csv"
DEFAULT_REC = "results/reconstruction/metrics.csv"
DEFAULT_OUT = "results/summary_metrics.csv"
DEFAULT_MD = "results/summary_metrics.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build summary tables for the report.")
    parser.add_argument("--seg", default=DEFAULT_SEG, help="Segmentation metrics CSV.")
    parser.add_argument("--rec", default=DEFAULT_REC, help="Reconstruction metrics CSV.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    parser.add_argument("--md", default=DEFAULT_MD, help="Output Markdown path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rows = []
    if os.path.exists(args.seg):
        seg = pd.read_csv(args.seg).iloc[0].to_dict()
        for key, value in seg.items():
            rows.append({"group": "segmentation", "metric": key, "value": value})

    if os.path.exists(args.rec):
        rec = pd.read_csv(args.rec).iloc[0].to_dict()
        for key, value in rec.items():
            rows.append({"group": "reconstruction", "metric": key, "value": value})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    if args.md:
        with open(args.md, "w", encoding="utf-8") as handle:
            handle.write("| Group | Metric | Value |\n")
            handle.write("|---|---|---|\n")
            for _, row in df.iterrows():
                handle.write(f"| {row['group']} | {row['metric']} | {row['value']:.4f} |\n")

    print(f"Summary saved: {args.out}")


if __name__ == "__main__":
    main()
