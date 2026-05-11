"""
Clinical HTML Report Generator
Produces the final clinical summary report with all tables and figures.
"""
import pandas as pd
from pathlib import Path
import base64
import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent.parent / "EPJ25-LCRJ"


def img_b64(path: Path) -> str:
    """Embed image as base64 so HTML is self-contained."""
    if path.exists():
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""


def acmg_badge(cls: str) -> str:
    cls = str(cls).strip().lower() if pd.notna(cls) else ""
    colors = {
        "pathogenic": ("#D32F2F", "#FFCDD2"),
        "likely pathogenic": ("#E64A19", "#FFE0B2"),
        "vus": ("#F57F17", "#FFF9C4"),
        "likely benign": ("#558B2F", "#DCEDC8"),
        "benign": ("#1B5E20", "#C8E6C9"),
    }
    bg, fg_bg = colors.get(cls, ("#616161", "#EEEEEE"))
    label = cls.title() if cls else "–"
    return f'<span style="background:{fg_bg};color:{bg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;border:1px solid {bg}40">{label}</span>'


def render_variant_rows(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or len(df) == 0:
        return "<tr><td colspan='12' style='text-align:center;color:#999'>No variants found</td></tr>"
    rows = []
    for _, r in df.head(max_rows).iterrows():
        gene = r.get("Gene", r.get("GENE_SYMBOL", "–"))
        chrom = r.get("Chr", r.get("CHROM", "–"))
        pos = r.get("Position", r.get("POS", "–"))
        ref = r.get("Ref", r.get("REF", "–"))
        alt = r.get("Alt", r.get("ALT", "–"))
        gt = r.get("Genotype", r.get("GT", "–"))
        eff = r.get("Effect", r.get("CONSEQUENCE", "–"))
        prot = r.get("Protein", r.get("HGVSp", "–"))
        cdna = r.get("cDNA", r.get("HGVSc", "–"))
        acmg = r.get("ACMG", r.get("ACMG_class", "–"))
        clinvar = r.get("ClinVar", r.get("ClinVar_PATHOGENICITY", "–"))
        disease = r.get("Disease", r.get("TITLE", "–"))
        af = r.get("gnomAD_AF", r.get("gnomAD_v3_WGS_AF_total", "–"))
        try:
            af_f = float(af)
            af_str = f"{af_f:.2e}" if af_f < 0.001 else f"{af_f:.4f}"
        except Exception:
            af_str = str(af) if pd.notna(af) else "–"

        rows.append(f"""<tr>
          <td><strong>{gene}</strong></td>
          <td>{chrom}:{pos}</td>
          <td style="font-family:monospace;font-size:11px">{ref}>{alt}</td>
          <td>{gt}</td>
          <td style="font-size:11px">{eff}</td>
          <td style="font-size:10px">{str(cdna)[:35] if pd.notna(cdna) else '–'}</td>
          <td style="font-size:10px">{str(prot)[:35] if pd.notna(prot) else '–'}</td>
          <td>{acmg_badge(acmg)}</td>
          <td style="font-size:11px">{clinvar if pd.notna(clinvar) else '–'}</td>
          <td style="font-size:10px;max-width:160px;overflow:hidden">{str(disease)[:60] if pd.notna(disease) else '–'}</td>
          <td style="font-size:10px">{af_str}</td>
        </tr>""")
    return "\n".join(rows)


def generate_report(stats: dict):
    fig_dir = RESULTS_DIR / "figures"
    vcf_dir = RESULTS_DIR / "vcf_analysis"
    out_dir = RESULTS_DIR / "clinical_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tables
    path_csv = vcf_dir / "clinical_variants_pathogenic.csv"
    vus_csv = vcf_dir / "clinical_variants_vus.csv"
    path_df = pd.read_csv(path_csv) if path_csv.exists() else pd.DataFrame()
    vus_df = pd.read_csv(vus_csv) if vus_csv.exists() else pd.DataFrame()

    # QC stats
    xlsx = DATA_DIR / "EPJ25-LCRJ.snv.indel.annotated.xlsx"
    bs = pd.read_excel(xlsx, sheet_name="Basic stat", header=None, names=["Metric", "Value"])
    bs = bs[bs["Metric"].notna() & (bs["Metric"] != "Basic stat")].reset_index(drop=True)
    qc_rows = "".join(
        f"<tr><td>{r['Metric']}</td><td><strong>{r['Value']}</strong></td></tr>"
        for _, r in bs.iterrows()
    )

    # CNV table
    cnv_df = pd.read_excel(DATA_DIR / "EPJ25-LCRJ.cnv.annotation_report.xlsx")
    cnv_rows = ""
    for _, r in cnv_df.head(50).iterrows():
        cnv_rows += f"""<tr>
          <td>{r.get('Chromosome','–')}</td>
          <td>{r.get('Start','–')}–{r.get('End','–')}</td>
          <td>{r.get('CNV_type','–')}</td>
          <td>{r.get('All_symbols','–')}</td>
          <td>{acmg_badge(r.get('ACMG_class','–'))}</td>
          <td style='font-size:10px'>{r.get('GnomAD_frequency','–')}</td>
          <td style='font-size:10px'>{r.get('Cytoband','–')}</td>
        </tr>"""

    path_rows = render_variant_rows(path_df)
    vus_rows = render_variant_rows(vus_df, max_rows=100)

    n_path = len(path_df)
    n_vus = len(vus_df)
    n_cnv = len(cnv_df)
    total_ann = stats.get("total_annotated", 0)
    n_pass = stats.get("pass_count", 0)
    today = datetime.date.today().strftime("%B %d, %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clinical Genomics Report – EPJ25-LCRJ</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f7fa; color: #222; }}
  .header {{ background: linear-gradient(135deg,#1a237e,#283593,#3949AB); color: white; padding: 36px 50px; }}
  .header h1 {{ font-size: 30px; letter-spacing: -0.5px; }}
  .header .sub {{ margin-top: 8px; opacity: 0.85; font-size: 14px; line-height: 1.6; }}
  .container {{ max-width: 1300px; margin: 30px auto; padding: 0 24px 60px; }}
  .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 10px; padding: 18px 22px; flex: 1; min-width: 150px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07); border-top: 4px solid; }}
  .card.red {{ border-color: #D32F2F; }} .card.orange {{ border-color: #E64A19; }}
  .card.amber {{ border-color: #F57F17; }} .card.blue {{ border-color: #1565C0; }}
  .card.green {{ border-color: #2E7D32; }} .card.purple {{ border-color: #6A1B9A; }}
  .card .val {{ font-size: 30px; font-weight: 700; }}
  .card.red .val {{ color: #D32F2F; }} .card.orange .val {{ color: #E64A19; }}
  .card.amber .val {{ color: #F57F17; }} .card.blue .val {{ color: #1565C0; }}
  .card.green .val {{ color: #2E7D32; }} .card.purple .val {{ color: #6A1B9A; }}
  .card .lbl {{ font-size: 11px; color: #777; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .section {{ background: white; border-radius: 10px; padding: 26px; margin-bottom: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
  .section h2 {{ color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 10px; margin-bottom: 18px; font-size: 18px; }}
  .section h3 {{ color: #283593; margin-bottom: 12px; font-size: 15px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: #e8eaf6; color: #1a237e; padding: 9px 12px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafbff; }}
  .figures {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; }}
  .fig-box {{ flex: 1; min-width: 280px; }}
  .fig-box img {{ width: 100%; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }}
  .fig-box p {{ text-align: center; font-size: 11px; color: #777; margin-top: 6px; }}
  .tbl-wrap {{ overflow-x: auto; }}
  .tag {{ display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 10px;
           font-weight: 600; background: #e3f2fd; color: #1565C0; }}
  footer {{ text-align: center; color: #aaa; font-size: 12px; padding: 24px; border-top: 1px solid #e0e0e0; margin-top: 20px; }}
  .alert {{ background: #FFF3E0; border-left: 4px solid #E64A19; padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #BF360C; margin-bottom: 16px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Clinical Genomics Report</h1>
  <div class="sub">
    <strong>Sample:</strong> EPJ25-LCRJ &nbsp;|&nbsp;
    <strong>Analysis:</strong> Whole Exome Sequencing (WES) &nbsp;|&nbsp;
    <strong>Reference:</strong> GRCh38/hg38 &nbsp;|&nbsp;
    <strong>Kit:</strong> IDT 3B v2 &nbsp;|&nbsp;
    <strong>Date:</strong> {today}<br>
    <strong>Pipeline:</strong> FASTQ → fastp QC → BWA-MEM → GATK HaplotypeCaller → VQSR → VEP Annotation
  </div>
</div>

<div class="container">

  <!-- Summary Cards -->
  <div class="cards">
    <div class="card blue"><div class="val">{total_ann:,}</div><div class="lbl">Total Variants (Ann.)</div></div>
    <div class="card green"><div class="val">{n_pass:,}</div><div class="lbl">PASS Variants</div></div>
    <div class="card red"><div class="val">{n_path}</div><div class="lbl">Pathogenic / Likely Path.</div></div>
    <div class="card amber"><div class="val">{n_vus}</div><div class="lbl">VUS</div></div>
    <div class="card purple"><div class="val">{n_cnv}</div><div class="lbl">CNV Events</div></div>
    <div class="card orange"><div class="val">201×</div><div class="lbl">Mean Depth</div></div>
  </div>

  <!-- QC Section -->
  <div class="section">
    <h2>🧬 Sequencing QC Summary</h2>
    <div class="figures">
      <div class="fig-box">
        <img src="../figures/qc_quality_bars.png" alt="Quality Metrics">
        <p>Sequencing Quality Metrics</p>
      </div>
      <div class="fig-box">
        <img src="../figures/qc_coverage_profile.png" alt="Coverage">
        <p>Coverage Profile</p>
      </div>
      <div class="fig-box">
        <img src="../figures/qc_variant_composition.png" alt="Variant Types">
        <p>Variant Composition</p>
      </div>
    </div>
    <br>
    <div class="tbl-wrap">
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{qc_rows}</tbody></table>
    </div>
  </div>

  <!-- Variant Analysis -->
  <div class="section">
    <h2>📊 Variant Analysis</h2>
    <div class="figures">
      <div class="fig-box">
        <img src="../figures/consequence_distribution.png" alt="Consequences">
        <p>Consequence Distribution</p>
      </div>
      <div class="fig-box">
        <img src="../figures/acmg_classes.png" alt="ACMG">
        <p>ACMG Classification</p>
      </div>
    </div>
    <br>
    <div class="figures">
      <div class="fig-box">
        <img src="../figures/vaf_depth_scatter.png" alt="VAF vs Depth">
        <p>VAF vs Read Depth</p>
      </div>
      <div class="fig-box">
        <img src="../figures/chromosome_burden.png" alt="Chromosomes">
        <p>Variants by Chromosome</p>
      </div>
    </div>
    <br>
    <div class="fig-box" style="max-width:100%">
      <img src="../figures/top_pathogenic_genes.png" alt="Top Genes">
      <p>Top Genes with Pathogenic/LP Variants</p>
    </div>
  </div>

  <!-- Clinical Pathogenic Table -->
  <div class="section">
    <h2>⚠️ Pathogenic / Likely Pathogenic Variants ({n_path})</h2>
    {"<div class='alert'>⚠️ No pathogenic variants found after filtering. Check ClinVar / ACMG columns in annotation file.</div>" if n_path == 0 else ""}
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Gene</th><th>Locus</th><th>Change</th><th>GT</th>
          <th>Consequence</th><th>cDNA</th><th>Protein</th>
          <th>ACMG</th><th>ClinVar</th><th>Disease</th><th>gnomAD AF</th>
        </tr></thead>
        <tbody>{path_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- VUS Table -->
  <div class="section">
    <h2>🔍 Variants of Uncertain Significance – VUS ({n_vus})</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Gene</th><th>Locus</th><th>Change</th><th>GT</th>
          <th>Consequence</th><th>cDNA</th><th>Protein</th>
          <th>ACMG</th><th>ClinVar</th><th>Disease</th><th>gnomAD AF</th>
        </tr></thead>
        <tbody>{vus_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- CNV Table -->
  <div class="section">
    <h2>🧩 Copy Number Variants – CNV ({n_cnv})</h2>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>Chr</th><th>Coordinates</th><th>Type</th><th>Genes</th>
          <th>ACMG</th><th>gnomAD Freq</th><th>Cytoband</th>
        </tr></thead>
        <tbody>{cnv_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Pipeline Description -->
  <div class="section">
    <h2>🔬 Pipeline Description</h2>
    <table>
      <thead><tr><th>Step</th><th>Tool / Version</th><th>Input</th><th>Output</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td><strong>1. QC</strong></td><td>fastp / FastQC</td><td>FASTQ (paired)</td><td>Cleaned FASTQ, HTML report</td><td>Adapter trimming, Q30 filter</td></tr>
        <tr><td><strong>2. Alignment</strong></td><td>BWA-MEM</td><td>Cleaned FASTQ</td><td>BAM + index</td><td>GRCh38.NGSdeadzone_masked reference</td></tr>
        <tr><td><strong>3. Variant Calling</strong></td><td>GATK 4.4.0 HaplotypeCaller + GenotypeGVCFs</td><td>BAM</td><td>Raw VCF (per-chr)</td><td>Germline mode, VQSR filtering</td></tr>
        <tr><td><strong>4. Filtration</strong></td><td>GATK VQSR + hard filters</td><td>Raw VCF</td><td>Filtered VCF</td><td>SNP_FILTER, INDEL_FILTER, STRAND_BIAS_FILTER</td></tr>
        <tr><td><strong>5. Annotation</strong></td><td>VEP + ClinVar + gnomAD v3</td><td>Filtered VCF</td><td>Annotated XLSX</td><td>ACMG classification, disease phenotypes</td></tr>
        <tr><td><strong>6. CNV</strong></td><td>CNV caller (lab internal)</td><td>BAM</td><td>CNV XLSX</td><td>IDT 3B v2 panel-matched</td></tr>
        <tr><td><strong>7. Report</strong></td><td>Python (pandas, matplotlib)</td><td>XLSX, VCF</td><td>HTML report</td><td>This document</td></tr>
      </tbody>
    </table>
  </div>

</div>
<footer>
  Sample: EPJ25-LCRJ &nbsp;·&nbsp; Big Data / Genomics Assignment &nbsp;·&nbsp; Maratov Yerassyl Balkanovich &nbsp;·&nbsp; {today}<br>
  <em>This report is for academic purposes only. All variant classifications should be confirmed by a certified clinical geneticist.</em>
</footer>
</body>
</html>"""

    out_path = out_dir / "clinical_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[Report] Clinical HTML report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    generate_report({
        "total_annotated": 0,
        "pass_count": 0,
        "pathogenic_count": 0,
        "vus_count": 0,
    })
