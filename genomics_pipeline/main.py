"""
Main pipeline entry point.
Runs all steps sequentially:
  1. QC report
  2. VCF / annotation analysis
  3. Clinical HTML report
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import qc_report
import vcf_analysis
import report_generator


def main():
    print("=" * 60)
    print("  EPJ25-LCRJ Genomics Analysis Pipeline")
    print("  Big Data Assignment – Maratov Yerassyl")
    print("=" * 60)

    print("\n[Step 1/3] Generating QC Report...")
    qc_report.run()

    print("\n[Step 2/3] Running VCF & Annotation Analysis...")
    stats = vcf_analysis.run()

    print("\n[Step 3/3] Generating Clinical HTML Report...")
    report_path = report_generator.generate_report(stats)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Clinical report → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
