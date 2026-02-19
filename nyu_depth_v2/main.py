import argparse
import sys

from src.nyu_download import main as download_main
from src.nyu_preprocess import main as preprocess_main
from src.nyu_reconstruction import main as recon_main
from src.nyu_segmentation import main as seg_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NYU Depth V2 pipeline.")
    parser.add_argument(
        "--mode",
        choices=["download", "preprocess", "segment", "reconstruct", "full"],
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
        _run_tool(download_main, ["nyu_download"])
        return

    if args.mode == "preprocess":
        _run_tool(preprocess_main, ["nyu_preprocess"])
        return

    if args.mode == "segment":
        _run_tool(seg_main, ["nyu_segmentation"])
        return

    if args.mode == "reconstruct":
        _run_tool(recon_main, ["nyu_reconstruction"])
        return

    _run_tool(download_main, ["nyu_download"])
    _run_tool(preprocess_main, ["nyu_preprocess"])
    _run_tool(seg_main, ["nyu_segmentation"])
    _run_tool(recon_main, ["nyu_reconstruction"])


if __name__ == "__main__":
    main()
