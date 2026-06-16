"""
generate_summary_dashboard.py
=============================
Compiles the statistics from results.csv and runs the dashboard generator to
output a publication-quality results/pipeline_summary.png summary figure.
"""

import os
import sys
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.dashboard import generate_pipeline_dashboard

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard_runner")

def main():
    results_csv = os.path.join(ROOT, "results", "results.csv")
    output_png = os.path.join(ROOT, "results", "pipeline_summary.png")
    
    logger.info("Generating summary dashboard...")
    res = generate_pipeline_dashboard(results_csv, output_png)
    
    if res["success"]:
        logger.info(f"✅ Pipeline summary dashboard successfully saved to: {res['output_path']}")
    else:
        logger.error(f"❌ Failed to generate dashboard: {res['error']}")

if __name__ == "__main__":
    main()
