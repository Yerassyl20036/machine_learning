"""
VCF & Annotation Analyser
Parses the annotated SNV/INDEL Excel and VCF to produce:
  - Clinical variant table (filtered to pathogenic/likely pathogenic/uncertain)
  - Consequence distribution
  - ACMG class breakdown
  - Gene-level burden table
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gzip
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent.parent.parent / "EPJ25-LCRJ"
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ──────────────────────────────────────────────
# 1. Load annotation data
# ──────────────────────────────────────────────
def load_annotation(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Annotation", dtype=str)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    print(f"[Annotation] Loaded {len(df):,} variants")
    return df


# ──────────────────────────────────────────────
# 2. Clinical filter
# ──────────────────────────────────────────────
PATHOGENIC_TERMS = ["pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"]

def filter_clinical(df: pd.DataFrame) -> pd.DataFrame:
    """Keep variants that are PASS-filtered AND have a ClinVar / ACMG classification."""
    mask_pass = df["FILTER"].str.upper().str.contains("PASS", na=False)

    mask_clinvar = df["ClinVar_PATHOGENICITY"].str.lower().isin(PATHOGENIC_TERMS) \
        if "ClinVar_PATHOGENICITY" in df.columns else pd.Series(False, index=df.index)

    mask_acmg = df["ACMG_class"].str.lower().isin(["pathogenic", "likely pathogenic"]) \
        if "ACMG_class" in df.columns else pd.Series(False, index=df.index)

    clinical = df[mask_pass & (mask_clinvar | mask_acmg)].copy()
    print(f"[Filter] Clinical variants (PASS + Pathogenic/LP): {len(clinical):,}")
    return clinical


def filter_vus(df: pd.DataFrame) -> pd.DataFrame:
    """VUS variants – uncertain significance, PASS."""
    mask_pass = df["FILTER"].str.upper().str.contains("PASS", na=False)
    mask_vus = (
        df.get("ClinVar_PATHOGENICITY", pd.Series(dtype=str)).str.lower().str.contains("uncertain", na=False) |
        df.get("ACMG_class", pd.Series(dtype=str)).str.lower().str.contains("vus|uncertain", na=False)
    )
    vus = df[mask_pass & mask_vus].copy()
    print(f"[Filter] VUS variants: {len(vus):,}")
    return vus


# ──────────────────────────────────────────────
# 3. Clinical report table
# ──────────────────────────────────────────────
CLINICAL_COLS = [
    "CHROM", "POS", "REF", "ALT", "GT",
    "GENE_SYMBOL", "CONSEQUENCE", "AMINO_ACIDS",
    "HGVSc", "HGVSp",
    "ACMG_class", "ACMG_bayesian",
    "ClinVar_PATHOGENICITY", "ClinVar_STAR",
    "TITLE", "INHERITANCE", "REVEL_SCORE",
    "gnomAD_v3_WGS_AF_total", "DP", "VAF",
]


def build_clinical_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CLINICAL_COLS if c in df.columns]
    tbl = df[cols].copy()
    tbl = tbl.rename(columns={
        "CHROM": "Chr", "POS": "Position", "REF": "Ref", "ALT": "Alt",
        "GT": "Genotype", "GENE_SYMBOL": "Gene", "CONSEQUENCE": "Effect",
        "AMINO_ACIDS": "AA_Change", "HGVSc": "cDNA", "HGVSp": "Protein",
        "ACMG_class": "ACMG", "ACMG_bayesian": "ACMG_score",
        "ClinVar_PATHOGENICITY": "ClinVar", "ClinVar_STAR": "ClinVar_Stars",
        "TITLE": "Disease", "INHERITANCE": "Inheritance",
        "REVEL_SCORE": "REVEL", "gnomAD_v3_WGS_AF_total": "gnomAD_AF",
        "DP": "Depth", "VAF": "VAF",
    })
    return tbl.reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. Figures
# ──────────────────────────────────────────────
def plot_consequence_distribution(df: pd.DataFrame, out_path: Path):
    if "CONSEQUENCE" not in df.columns:
        return
    counts = df["CONSEQUENCE"].dropna().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(counts)))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of variants", fontsize=11)
    ax.set_title("Variant Consequence Distribution (top 15)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Consequence distribution saved → {out_path.name}")


def plot_acmg_classes(df: pd.DataFrame, out_path: Path):
    if "ACMG_class" not in df.columns:
        return
    counts = df["ACMG_class"].fillna("Unclassified").value_counts()
    color_map = {
        "Pathogenic": "#D32F2F",
        "Likely pathogenic": "#FF5722",
        "VUS": "#FFC107",
        "Likely benign": "#8BC34A",
        "Benign": "#388E3C",
        "Unclassified": "#9E9E9E",
    }
    colors = [color_map.get(k, "#9E9E9E") for k in counts.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p/100*counts.sum()):,})" if p > 1 else "",
        startangle=90, textprops={"fontsize": 9}
    )
    ax.set_title("ACMG Classification Breakdown", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] ACMG classes saved → {out_path.name}")


def plot_top_genes(clinical: pd.DataFrame, out_path: Path):
    if "GENE_SYMBOL" not in clinical.columns or len(clinical) == 0:
        return
    top = clinical["GENE_SYMBOL"].dropna().value_counts().head(20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(top)), top.values, color="#1565C0", alpha=0.85)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pathogenic / LP Variant Count", fontsize=11)
    ax.set_title("Top Genes with Pathogenic/Likely Pathogenic Variants", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Top genes saved → {out_path.name}")


def plot_vaf_depth(df: pd.DataFrame, out_path: Path):
    """VAF vs Depth scatter for PASS variants (sample 5000 for speed)."""
    mask = df["FILTER"].str.upper().str.contains("PASS", na=False)
    sub = df[mask].copy()
    try:
        sub["VAF_f"] = pd.to_numeric(sub["VAF"], errors="coerce")
        sub["DP_f"] = pd.to_numeric(sub["DP"], errors="coerce")
        sub = sub.dropna(subset=["VAF_f", "DP_f"])
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(sub["DP_f"], sub["VAF_f"], alpha=0.15, s=8, c=sub["VAF_f"],
                        cmap="coolwarm", vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="VAF")
        ax.set_xlabel("Read Depth (DP)", fontsize=11)
        ax.set_ylabel("Variant Allele Frequency (VAF)", fontsize=11)
        ax.set_title("VAF vs Depth (PASS variants, n≤5000)", fontsize=12, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] VAF vs Depth saved → {out_path.name}")
    except Exception as e:
        print(f"[Plot] VAF/Depth skipped: {e}")


def plot_chromosome_burden(df: pd.DataFrame, out_path: Path):
    chrom_order = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y", "MT"]]
    mask = df["FILTER"].str.upper().str.contains("PASS", na=False) if "FILTER" in df.columns else pd.Series(True, index=df.index)
    counts = df.loc[mask, "CHROM"].value_counts()
    ordered = [counts.get(c, 0) for c in chrom_order]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(chrom_order, ordered, color="#42A5F5", edgecolor="white")
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("PASS Variant Count", fontsize=10)
    ax.set_title("Variant Distribution by Chromosome", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Chromosome burden saved → {out_path.name}")


# ──────────────────────────────────────────────
# 5. VCF Summary
# ──────────────────────────────────────────────
def vcf_pass_summary(vcf_gz: Path) -> dict:
    total, passed = 0, 0
    filters = Counter()
    try:
        with gzip.open(vcf_gz, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 7:
                    continue
                total += 1
                flt = parts[6].strip()
                filters[flt] += 1
                if flt == "PASS":
                    passed += 1
    except Exception as e:
        print(f"[VCF] Error reading VCF: {e}")
        return {}
    return {"total": total, "passed": passed, "filters": dict(filters)}


def run():
    xlsx = DATA_DIR / "EPJ25-LCRJ.snv.indel.annotated.xlsx"
    vcf_gz = DATA_DIR / "EPJ25-LCRJ.final.vcf.gz"
    fig_dir = RESULTS_DIR / "figures"
    vcf_dir = RESULTS_DIR / "vcf_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    vcf_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_annotation(xlsx)

    # VCF summary
    print("[VCF] Parsing VCF for filter summary...")
    vcf_summary = vcf_pass_summary(vcf_gz)
    print(f"[VCF] Total variants: {vcf_summary.get('total',0):,} | PASS: {vcf_summary.get('passed',0):,}")

    # Save VCF filter summary
    if vcf_summary.get("filters"):
        flt_df = pd.DataFrame(
            [{"Filter": k, "Count": v} for k, v in vcf_summary["filters"].items()]
        ).sort_values("Count", ascending=False)
        flt_df.to_csv(vcf_dir / "vcf_filter_summary.csv", index=False)
        print(f"[VCF] Filter summary saved")

    # Plots on full dataset
    plot_consequence_distribution(df, fig_dir / "consequence_distribution.png")
    plot_acmg_classes(df, fig_dir / "acmg_classes.png")
    plot_vaf_depth(df, fig_dir / "vaf_depth_scatter.png")
    plot_chromosome_burden(df, fig_dir / "chromosome_burden.png")

    # Clinical filter
    clinical = filter_clinical(df)
    vus = filter_vus(df)

    plot_top_genes(clinical, fig_dir / "top_pathogenic_genes.png")

    # Save tables
    clinical_tbl = build_clinical_table(clinical)
    clinical_tbl.to_csv(vcf_dir / "clinical_variants_pathogenic.csv", index=False)
    clinical_tbl.to_excel(vcf_dir / "clinical_variants_pathogenic.xlsx", index=False)
    print(f"[Output] Clinical pathogenic table: {len(clinical_tbl):,} variants")

    vus_tbl = build_clinical_table(vus)
    vus_tbl.to_csv(vcf_dir / "clinical_variants_vus.csv", index=False)
    print(f"[Output] VUS table: {len(vus_tbl):,} variants")

    # Save full PASS annotated table
    pass_mask = df["FILTER"].str.upper().str.contains("PASS", na=False)
    pass_df = df[pass_mask].copy()
    pass_df.to_csv(vcf_dir / "all_pass_variants.csv", index=False)
    print(f"[Output] All PASS variants: {len(pass_df):,}")

    return {
        "total_annotated": len(df),
        "pass_count": int(pass_mask.sum()),
        "pathogenic_count": len(clinical_tbl),
        "vus_count": len(vus_tbl),
        "vcf_summary": vcf_summary,
    }


if __name__ == "__main__":
    run()
