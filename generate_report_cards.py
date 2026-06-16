"""
generate_report_cards.py
========================
Generates a markdown technical report card for the tested targets in reports/.
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.report_generator import generate_report_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("report_runner")

def main():
    reports_dir = os.path.join(ROOT, "results", "reports")
    output_dir = os.path.join(ROOT, "reports")
    
    if not os.path.exists(reports_dir):
        logger.error(f"Reports directory not found: {reports_dir}")
        return

    report_files = [f for f in os.listdir(reports_dir) if f.endswith("_report.json")]
    if not report_files:
        logger.warning(f"No report JSON files found in {reports_dir}")
        return

    for rf in report_files:
        report_json = os.path.join(reports_dir, rf)
        logger.info(f"Generating report card for {rf}...")
        res = generate_report_card(report_json, output_dir)
        if res["success"]:
            logger.info(f"✅ Report card successfully saved to: {res['output_path']}")
        else:
            logger.error(f"❌ Failed to generate report card for {rf}: {res['error']}")

if __name__ == "__main__":
    main()
