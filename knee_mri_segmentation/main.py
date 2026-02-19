#!/usr/bin/env python3
"""
Knee MRI Cartilage & Bone Segmentation Pipeline
Standalone project for automatic segmentation of knee joint structures from MRI scans.
"""

import argparse
import sys

from src.download import main as download_main
from src.preprocess import main as preprocess_main
from src.segment import main as segment_main
from src.report_assets import main as report_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knee MRI segmentation pipeline (Osteoarthritis Initiative dataset)."
    )
    parser.add_argument(
        "--mode",
        choices=["download", "preprocess", "segment", "report", "full"],
        default="full",
        help="Pipeline mode.",
    )
    return parser


def _run_tool(tool_main, argv: list[str]) -> None:
    original_argv = sys.argv
    try:
        sys.argv = argv
        tool_main()
    finally:
        sys.argv = original_argv


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "download":
        _run_tool(download_main, ["download"])
        return

    if args.mode == "preprocess":
        _run_tool(preprocess_main, ["preprocess"])
        return

    if args.mode == "segment":
        _run_tool(segment_main, ["segment"])
        return

    if args.mode == "report":
        _run_tool(report_main, ["report"])
        return

    # Full pipeline
    _run_tool(download_main, ["download"])
    _run_tool(preprocess_main, ["preprocess"])
    _run_tool(segment_main, ["segment"])
    _run_tool(report_main, ["report"])


if __name__ == "__main__":
    main()
