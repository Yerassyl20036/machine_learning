"""
QC Report Generator
Reads QC statistics from the annotation Excel and produces an HTML QC report + figures.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import os

DATA_DIR = Path(__file__).parent.parent.parent / "EPJ25-LCRJ"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_basic_stats(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Basic stat", header=None, names=["Metric", "Value"])
    df = df[df["Metric"].notna() & (df["Metric"] != "Basic stat")].reset_index(drop=True)
    return df


def plot_coverage_profile(stats: dict, out_path: Path):
    thresholds = [1, 2, 5, 10, 20, 30, 40, 50]
    coverages = [
        float(str(stats.get(f"{t}X Coverage (%)", "0")).replace("%", ""))
        for t in thresholds
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, coverages, marker="o", color="#2196F3", linewidth=2)
    ax.fill_between(thresholds, coverages, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Depth Threshold (X)", fontsize=11)
    ax.set_ylabel("% Bases Covered", fontsize=11)
    ax.set_title("Coverage Profile – EPJ25-LCRJ", fontsize=13, fontweight="bold")
    ax.set_ylim(95, 100.5)
    ax.grid(True, alpha=0.3)
    for x, y in zip(thresholds, coverages):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1565C0")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variant_composition(stats: dict, out_path: Path):
    snp = int(stats.get("SNP", 0))
    indel = int(stats.get("INDEL", 0))
    labels = ["SNPs", "INDELs"]
    sizes = [snp, indel]
    colors = ["#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Variant Composition", fontsize=13, fontweight="bold")
    ax.text(0, -1.35, f"Total: {snp+indel:,}  |  Ts/Tv: {stats.get('TsTv', '–')}  |  Het/Hom: {stats.get('Het/Hom', '–')}",
            ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_bars(stats: dict, out_path: Path):
    metrics = {
        "Q20 (%)": float(stats.get("Q20 (%)", 0)),
        "Q30 (%)": float(stats.get("Q30 (%)", 0)),
        "Mappable\nReads (%)": float(stats.get("Initial Mappable Reads (%)", 0)),
        "Non-Redundant\nReads (%)": float(stats.get("Non Redundant Reads (%)", 0)),
        "On-Target\nReads (%)": min(float(stats.get("On-Target Reads (%)", 0)), 100),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=["#4CAF50" if v >= 95 else "#FF9800" if v >= 80 else "#F44336"
                         for v in metrics.values()],
                  edgecolor="white", linewidth=0.5)
    ax.axhline(95, color="#F44336", linestyle="--", linewidth=1, alpha=0.5, label="95% threshold")
    ax.set_ylim(75, 102)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("Sequencing Quality Metrics – EPJ25-LCRJ", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_qc_html(stats: dict, out_html: Path, fig_dir: Path):
    rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
        for k, v in stats.items()
    )
    mean_depth = stats.get("Mean Depth (x)", "–")
    q30 = stats.get("Q30 (%)", "–")
    snp = int(stats.get("SNP", 0))
    indel = int(stats.get("INDEL", 0))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QC Report – EPJ25-LCRJ</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f7fa; color: #222; }}
  .header {{ background: linear-gradient(135deg,#1565C0,#42A5F5); color: white; padding: 30px 40px; }}
  .header h1 {{ margin: 0; font-size: 28px; }}
  .header p {{ margin: 6px 0 0; opacity: 0.85; font-size: 14px; }}
  .container {{ max-width: 1100px; margin: 30px auto; padding: 0 20px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 30px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px 24px; flex: 1; min-width: 180px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #1565C0; }}
  .card .val {{ font-size: 28px; font-weight: 700; color: #1565C0; }}
  .card .lbl {{ font-size: 12px; color: #777; margin-top: 4px; }}
  .section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .section h2 {{ margin-top: 0; color: #1565C0; border-bottom: 2px solid #e3f2fd; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #e3f2fd; color: #1565C0; padding: 10px 14px; text-align: left; }}
  td {{ padding: 8px 14px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafafa; }}
  .figures {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .fig-box {{ flex: 1; min-width: 280px; text-align: center; }}
  .fig-box img {{ width: 100%; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px;
            font-weight: 600; background: #e8f5e9; color: #2e7d32; }}
  footer {{ text-align: center; color: #aaa; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<div class="header">
  <h1>QC Report — EPJ25-LCRJ</h1>
  <p>Whole Exome Sequencing · GRCh38 · IDT 3B v2 Kit · Generated 2026-04-20</p>
</div>
<div class="container">
  <div class="cards">
    <div class="card"><div class="val">{mean_depth}×</div><div class="lbl">Mean Depth</div></div>
    <div class="card"><div class="val">{q30}%</div><div class="lbl">Q30</div></div>
    <div class="card"><div class="val">{snp:,}</div><div class="lbl">SNPs called</div></div>
    <div class="card"><div class="val">{indel:,}</div><div class="lbl">INDELs called</div></div>
    <div class="card"><div class="val">{stats.get('TsTv','–')}</div><div class="lbl">Ts/Tv Ratio</div></div>
    <div class="card"><div class="val">{stats.get('Het/Hom','–')}</div><div class="lbl">Het/Hom Ratio</div></div>
  </div>

  <div class="section">
    <h2>📊 Quality Figures</h2>
    <div class="figures">
      <div class="fig-box">
        <img src="../figures/qc_quality_bars.png" alt="Quality Metrics">
        <p>Sequencing Quality Metrics</p>
      </div>
      <div class="fig-box">
        <img src="../figures/qc_coverage_profile.png" alt="Coverage Profile">
        <p>Coverage Profile</p>
      </div>
      <div class="fig-box">
        <img src="../figures/qc_variant_composition.png" alt="Variant Composition">
        <p>Variant Composition</p>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>📋 Full Statistics Table</h2>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>
<footer>EPJ25-LCRJ · Big Data / Genomics Assignment · Maratov Yerassyl · 2026</footer>
</body>
</html>"""
    out_html.write_text(html, encoding="utf-8")
    print(f"[QC] HTML report saved → {out_html}")


def run():
    xlsx = DATA_DIR / "EPJ25-LCRJ.snv.indel.annotated.xlsx"
    df = load_basic_stats(xlsx)
    stats = dict(zip(df["Metric"], df["Value"]))

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = RESULTS_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    plot_coverage_profile(stats, fig_dir / "qc_coverage_profile.png")
    print("[QC] Coverage profile saved")
    plot_variant_composition(stats, fig_dir / "qc_variant_composition.png")
    print("[QC] Variant composition saved")
    plot_quality_bars(stats, fig_dir / "qc_quality_bars.png")
    print("[QC] Quality bars saved")

    generate_qc_html(stats, qc_dir / "qc_report.html", fig_dir)


if __name__ == "__main__":
    run()
