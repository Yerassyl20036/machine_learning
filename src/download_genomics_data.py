import argparse
import csv
import os
import random
import urllib.request
from datetime import datetime


CLINVAR_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
DEFAULT_RAW_DIR = "data/raw"

CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
GENES = [
    "BRCA1",
    "BRCA2",
    "TP53",
    "EGFR",
    "APOE",
    "CFTR",
    "KRAS",
    "PTEN",
    "MYC",
    "VHL",
]
EFFECTS = [
    "missense_variant",
    "synonymous_variant",
    "frameshift_variant",
    "stop_gained",
    "splice_region_variant",
    "intron_variant",
]
CLINSIG = [
    "benign",
    "likely_benign",
    "uncertain_significance",
    "likely_pathogenic",
    "pathogenic",
]
GENOTYPES = ["0/1", "1/1", "0/0", "1/2"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_clinvar_vcf(output_dir: str, url: str = CLINVAR_VCF_URL) -> str:
    ensure_dir(output_dir)
    filename = os.path.join(output_dir, "clinvar.vcf.gz")
    urllib.request.urlretrieve(url, filename)
    return filename


def write_synthetic_variants(output_dir: str, n_rows: int = 2000) -> str:
    ensure_dir(output_dir)
    filename = os.path.join(output_dir, "synthetic_variants.csv")
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "chromosome",
                "position",
                "gene",
                "REF",
                "ALT",
                "genotype",
                "depth",
                "effect_type",
                "clinical_significance",
            ]
        )
        for _ in range(n_rows):
            chrom = random.choice(CHROMS)
            pos = random.randint(1, 250_000_000)
            gene = random.choice(GENES)
            ref = random.choice(["A", "C", "G", "T"])
            alt = random.choice([b for b in ["A", "C", "G", "T"] if b != ref])
            gt = random.choice(GENOTYPES)
            depth = random.randint(5, 500)
            effect = random.choice(EFFECTS)
            clinsig = random.choice(CLINSIG)
            writer.writerow([chrom, pos, gene, ref, alt, gt, depth, effect, clinsig])
    return filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ClinVar VCF or generate a synthetic variant dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["download", "synthetic"],
        default="download",
        help="Choose download or synthetic data generation.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RAW_DIR,
        help="Directory to save raw data.",
    )
    parser.add_argument(
        "--url",
        default=CLINVAR_VCF_URL,
        help="ClinVar VCF URL for download mode.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of synthetic rows to generate (synthetic mode).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "synthetic":
        path = write_synthetic_variants(args.output_dir, args.rows)
        print(f"Synthetic dataset saved: {path}")
        return

    try:
        path = download_clinvar_vcf(args.output_dir, args.url)
        print(f"ClinVar VCF downloaded: {path}")
    except Exception as exc:
        print(f"Download failed ({exc}). Falling back to synthetic dataset.")
        path = write_synthetic_variants(args.output_dir, args.rows)
        print(f"Synthetic dataset saved: {path}")


if __name__ == "__main__":
    random.seed(datetime.utcnow().timestamp())
    main()
