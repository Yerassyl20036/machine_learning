import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

from db_setup import connect_db, init_db, load_config


DEFAULT_INPUT = "data/raw/clinvar.vcf.gz"
DEFAULT_OUTPUT = "data/processed/cleaned_variants.csv"

REQUIRED_COLUMNS = [
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


CLINSIG_MAP = {
    "benign": 0,
    "likely_benign": 0,
    "pathogenic": 1,
    "likely_pathogenic": 1,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_info_field(info: str) -> Dict[str, str]:
    fields = {}
    for item in info.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def parse_vcf(path: str) -> pd.DataFrame:
    records: List[Dict[str, Optional[str]]] = []
    opener = open
    if path.endswith(".gz"):
        import gzip

        opener = gzip.open

    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _vid, ref, alt, _qual, _filter, info = parts[:8]
            fmt = parts[8] if len(parts) > 8 else ""
            sample = parts[9] if len(parts) > 9 else ""

            info_map = parse_info_field(info)
            geneinfo = info_map.get("GENEINFO", "")
            gene = geneinfo.split(":", 1)[0] if geneinfo else ""
            effect = info_map.get("CLNVC", "") or info_map.get("CLNV", "")
            clinsig = info_map.get("CLNSIG", "")
            clinsig = clinsig.split("|")[0].lower() if clinsig else ""

            genotype = ""
            depth = ""
            if fmt and sample:
                fmt_keys = fmt.split(":")
                fmt_vals = sample.split(":")
                fmt_map = {k: v for k, v in zip(fmt_keys, fmt_vals)}
                genotype = fmt_map.get("GT", "")
                depth = fmt_map.get("DP", "")

            records.append(
                {
                    "chromosome": chrom,
                    "position": pos,
                    "gene": gene,
                    "REF": ref,
                    "ALT": alt,
                    "genotype": genotype,
                    "depth": depth,
                    "effect_type": effect.lower(),
                    "clinical_significance": clinsig,
                }
            )

    return pd.DataFrame.from_records(records)


def parse_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    rename_map = {}
    for required in REQUIRED_COLUMNS:
        if required in cols:
            rename_map[cols[required]] = required
    df = df.rename(columns=rename_map)
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[REQUIRED_COLUMNS].copy()
    df["clinical_significance"] = df["clinical_significance"].fillna("")
    df["clinical_significance"] = df["clinical_significance"].astype(str).str.lower()

    df = df[df["clinical_significance"].isin(CLINSIG_MAP.keys())]
    df["label"] = df["clinical_significance"].map(CLINSIG_MAP)

    df["gene"] = df["gene"].fillna("unknown")
    df["genotype"] = df["genotype"].fillna("unknown")
    df["effect_type"] = df["effect_type"].fillna("unknown")

    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    depth_median = int(df["depth"].median()) if df["depth"].notna().any() else 0
    df["depth"] = df["depth"].fillna(depth_median)

    df = df.dropna(subset=["chromosome", "position", "REF", "ALT"])
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df[df["position"].notna()]
    df["position"] = df["position"].astype(int)

    effect_dummies = pd.get_dummies(df["effect_type"], prefix="effect")
    clinsig_dummies = pd.get_dummies(df["clinical_significance"], prefix="clinsig")

    df = pd.concat([df, effect_dummies, clinsig_dummies], axis=1)
    return df


def load_to_db(df: pd.DataFrame, config_path: str = "config/db_config.json") -> None:
    config = load_config(config_path)
    db_type = config.get("db_type", "postgres")

    init_db(config_path)
    conn = connect_db(config_path)
    try:
        rows = df[
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
        ].values.tolist()

        if db_type == "sqlite":
            conn.executemany(
                """
                INSERT INTO genomic_variants
                (chrom, pos, gene, ref, alt, gt, depth, effect, clinsig)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return

        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO genomic_variants
                (chrom, pos, gene, ref, alt, gt, depth, effect, clinsig)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess genomic data.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to raw VCF or CSV input.",
    )
    parser.add_argument(
        "--format",
        choices=["vcf", "csv", "auto"],
        default="auto",
        help="Input format (auto detects by extension).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path for cleaned CSV output.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip loading cleaned data into the database.",
    )
    return parser


def detect_format(path: str) -> str:
    if path.endswith(".vcf") or path.endswith(".vcf.gz"):
        return "vcf"
    return "csv"


def main() -> None:
    args = build_parser().parse_args()
    input_format = args.format
    if input_format == "auto":
        input_format = detect_format(args.input)

    if input_format == "vcf":
        df = parse_vcf(args.input)
    else:
        df = parse_csv(args.input)

    df = normalize_dataframe(df)

    ensure_dir(os.path.dirname(args.output))
    df.to_csv(args.output, index=False)

    if not args.skip_db:
        load_to_db(df)

    print(f"Saved cleaned data to: {args.output}")


if __name__ == "__main__":
    main()
