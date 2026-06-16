"""
src/dashboard.py
================
Visualization dashboard generator for pipeline performance summary.
Compiles all target predictions from results.csv and generates a 4-panel pipeline_summary.png.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set Matplotlib style to seaborn-v0_8-whitegrid
plt.style.use("seaborn-v0_8-whitegrid")

def generate_pipeline_dashboard(results_csv_path: str, output_image_path: str) -> dict:
    """
    Generate a 4-panel summary dashboard figure of the pipeline execution.
    
    WHAT: Reads the batch results and creates statistical plots.
    WHY: Evaluates overall pipeline classification distributions and candidates for ISRO reporting.
    """
    try:
        if not os.path.exists(results_csv_path):
            return {"success": False, "error": f"Results file not found: {results_csv_path}"}
            
        df = pd.read_csv(results_csv_path)
        if len(df) == 0:
            return {"success": False, "error": "Results file is empty"}
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Panel A: Distribution of Combined Confidence scores
        sns.histplot(df["confidence"], bins=20, kde=True, ax=axes[0, 0], color="#4FC3F7")
        axes[0, 0].set_title("Combined Confidence Score Distribution", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Confidence", fontsize=10)
        axes[0, 0].set_ylabel("Count", fontsize=10)
        
        # 2. Panel B: Decision distribution (KEEP / FLAG / DISCARD)
        decision_counts = df["decision"].value_counts()
        colors = ["#F44336" if idx == "DISCARD" else "#FFC107" if idx == "FLAG" else "#4CAF50" for idx in decision_counts.index]
        decision_counts.plot(kind="bar", color=colors, ax=axes[0, 1])
        axes[0, 1].set_title("Three-Tier Decision Counts", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Vetting Decision", fontsize=10)
        axes[0, 1].set_ylabel("Count", fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=0)
        
        # 3. Panel C: Period vs Depth scatter plot for KEEP/FLAG candidates
        candidates = df[df["decision"].isin(["KEEP", "FLAG"])]
        if len(candidates) > 0:
            sns.scatterplot(
                data=candidates,
                x="period",
                y="depth",
                hue="decision",
                palette={"KEEP": "#4CAF50", "FLAG": "#FFC107"},
                ax=axes[1, 0]
            )
            axes[1, 0].set_yscale("log")
            axes[1, 0].set_title("Period vs. Depth for Vetted Candidates", fontsize=12, fontweight="bold")
            axes[1, 0].set_xlabel("Period (days)", fontsize=10)
            axes[1, 0].set_ylabel("Transit Depth (fractional, log)", fontsize=10)
        else:
            axes[1, 0].text(0.5, 0.5, "No vetted candidates (KEEP/FLAG) to plot", ha="center", va="center")
            axes[1, 0].set_title("Period vs. Depth", fontsize=12, fontweight="bold")
            
        # 4. Panel D: SNR vs. Depth comparison
        if len(df) > 0:
            sns.scatterplot(
                data=df,
                x="snr",
                y="depth",
                hue="final_class",
                style="decision",
                ax=axes[1, 1]
            )
            axes[1, 1].set_yscale("log")
            axes[1, 1].set_title("BLS SNR vs. Depth by Model Classification", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("BLS Peak SNR", fontsize=10)
            axes[1, 1].set_ylabel("Transit Depth (fractional, log)", fontsize=10)
            
        plt.suptitle("ISRO Hackathon Challenge 7 - Exoplanet Detection Summary Dashboard", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save as 150 DPI PNG
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        logger.info(f"Summary dashboard saved to: {output_image_path}")
        return {"success": True, "output_path": output_image_path}
        
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
        return {"success": False, "error": str(e)}
