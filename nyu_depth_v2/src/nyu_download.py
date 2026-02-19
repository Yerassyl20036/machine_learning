import argparse
import os
import urllib.request

DEFAULT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
DEFAULT_OUTPUT = "data/raw/nyu_depth_v2_labeled.mat"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_file(url: str, output_path: str) -> str:
    ensure_dir(os.path.dirname(output_path))
    urllib.request.urlretrieve(url, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download NYU Depth V2 labeled dataset.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset URL.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save nyu_depth_v2_labeled.mat.",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip download (use if file already exists).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.skip and os.path.exists(args.output):
        print(f"File already exists: {args.output}")
        return

    try:
        path = download_file(args.url, args.output)
        print(f"Downloaded: {path}")
    except Exception as exc:
        raise RuntimeError(
            "Download failed. If the NYU server requires manual access, "
            "download nyu_depth_v2_labeled.mat in your browser and place it in data/raw/."
        ) from exc


if __name__ == "__main__":
    main()
