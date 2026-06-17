"""
src/validation/run_validation.py
================================
Validation script to measure exoplanet pipeline metrics (accuracy, TPR, FPR,
precision, recall, F1-score) on a blind test set of known targets.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Benchmark targets
PLANETS = {
    261136679: "WASP-121b",
    307210830: "WASP-39b",
    460205581: "WASP-43b",
    149603524: "TOI-270b",
    259377017: "TOI-700d",
    441420236: "TOI-1231b",
    350618622: "TOI-132b",
    100990000: "L 98-59b"
}

EBS = {
    229742722: "EB-1",
    388857263: "EB-2",
    229055790: "EB-3"
}

FPS = {
    271893367: "FP-1",
    144700903: "FP-2"
}

def run_validation_suite(fast_mode=False):
    """Runs the pipeline blindly on the validation targets and calculates stats."""
    from pipeline import run_pipeline
    
    results = []
    
    logger.info("=" * 60)
    logger.info("STARTING PIPELINE VALIDATION RUN")
    logger.info("=" * 60)
    
    # 1. Run Planets
    for tic, name in PLANETS.items():
        logger.info(f"\nTesting Planet: {name} (TIC {tic})")
        try:
            # Run pipeline
            report = run_pipeline(tic_id=tic, snr_threshold=5.0, force_download=False, fast=fast_mode)
            decision = report.get("decision", {}).get("decision", "DISCARD")
            final_class = report.get("classification", {}).get("label_name", "Unknown")
            confidence = report.get("decision", {}).get("combined_confidence", 0.0)
            period = report.get("transit_parameters", {}).get("period", 0.0)
            depth = report.get("transit_parameters", {}).get("transit_depth", 0.0)
            
            results.append({
                "tic_id": tic, "name": name, "true_type": "planet",
                "predicted_decision": decision, "predicted_class": final_class,
                "confidence": confidence, "period": period, "depth": depth, "status": "success"
            })
        except Exception as e:
            logger.error(f"Failed to validate TIC {tic}: {e}")
            results.append({
                "tic_id": tic, "name": name, "true_type": "planet",
                "predicted_decision": "ERROR", "predicted_class": "ERROR",
                "confidence": 0.0, "period": 0.0, "depth": 0.0, "status": f"failed: {e}"
            })
            
    # 2. Run EBs
    for tic, name in EBS.items():
        logger.info(f"\nTesting EB: {name} (TIC {tic})")
        try:
            report = run_pipeline(tic_id=tic, snr_threshold=5.0, force_download=False, fast=fast_mode)
            decision = report.get("decision", {}).get("decision", "DISCARD")
            final_class = report.get("classification", {}).get("label_name", "Unknown")
            confidence = report.get("decision", {}).get("combined_confidence", 0.0)
            period = report.get("transit_parameters", {}).get("period", 0.0)
            depth = report.get("transit_parameters", {}).get("transit_depth", 0.0)
            
            results.append({
                "tic_id": tic, "name": name, "true_type": "eb",
                "predicted_decision": decision, "predicted_class": final_class,
                "confidence": confidence, "period": period, "depth": depth, "status": "success"
            })
        except Exception as e:
            logger.error(f"Failed to validate TIC {tic}: {e}")
            results.append({
                "tic_id": tic, "name": name, "true_type": "eb",
                "predicted_decision": "ERROR", "predicted_class": "ERROR",
                "confidence": 0.0, "period": 0.0, "depth": 0.0, "status": f"failed: {e}"
            })

    # 3. Run FPs
    for tic, name in FPS.items():
        logger.info(f"\nTesting False Positive: {name} (TIC {tic})")
        try:
            report = run_pipeline(tic_id=tic, snr_threshold=5.0, force_download=False, fast=fast_mode)
            decision = report.get("decision", {}).get("decision", "DISCARD")
            final_class = report.get("classification", {}).get("label_name", "Unknown")
            confidence = report.get("decision", {}).get("combined_confidence", 0.0)
            period = report.get("transit_parameters", {}).get("period", 0.0)
            depth = report.get("transit_parameters", {}).get("transit_depth", 0.0)
            
            results.append({
                "tic_id": tic, "name": name, "true_type": "fp",
                "predicted_decision": decision, "predicted_class": final_class,
                "confidence": confidence, "period": period, "depth": depth, "status": "success"
            })
        except Exception as e:
            logger.error(f"Failed to validate TIC {tic}: {e}")
            results.append({
                "tic_id": tic, "name": name, "true_type": "fp",
                "predicted_decision": "ERROR", "predicted_class": "ERROR",
                "confidence": 0.0, "period": 0.0, "depth": 0.0, "status": f"failed: {e}"
            })
            
    df_res = pd.DataFrame(results)
    
    # Ensure validation output directory exists
    val_dir = Path("validation")
    val_dir.mkdir(exist_ok=True)
    
    # Save CSV report
    df_res.to_csv(val_dir / "validation_report.csv", index=False)
    logger.info(f"Saved validation report to: {val_dir / 'validation_report.csv'}")
    
    # Calculate performance metrics
    # Planet positive class = true_type is 'planet'
    # Predicted positive class = predicted_decision is 'KEEP' and predicted_class is 'Planet Transit'
    
    total_planets = len(PLANETS)
    total_non_planets = len(EBS) + len(FPS)
    
    true_positives = len(df_res[(df_res["true_type"] == "planet") & (df_res["predicted_decision"] == "KEEP")])
    false_negatives = total_planets - true_positives
    
    false_positives = len(df_res[(df_res["true_type"] != "planet") & (df_res["predicted_decision"] == "KEEP")])
    true_negatives = total_non_planets - false_positives
    
    tpr = true_positives / total_planets if total_planets > 0 else 0.0
    fpr = false_positives / total_non_planets if total_non_planets > 0 else 0.0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"True Positive Rate (Recall):     {tpr*100:.1f}% ({true_positives}/{total_planets})")
    logger.info(f"False Positive Rate:             {fpr*100:.1f}% ({false_positives}/{total_non_planets})")
    logger.info(f"Precision:                       {precision*100:.1f}%")
    logger.info(f"F1 Score:                        {f1:.3f}")
    logger.info("-" * 60)
    logger.info(f"Confusion Matrix:")
    logger.info(f"                Predicted Planet   Predicted Non-Planet")
    logger.info(f"Actual Planet       {true_positives:<18} {false_negatives:<20}")
    logger.info(f"Actual Non-Planet   {false_positives:<18} {true_negatives:<20}")
    logger.info("=" * 60)
    
    # Plot simple confusion matrix using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(6, 5))
        matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Planet (KEEP)", "Predicted Non-Planet"],
                    yticklabels=["Actual Planet", "Actual Non-Planet"])
        plt.title("Confusion Matrix - TESS Pipeline v2.0")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(val_dir / "validation_report.png", dpi=150)
        logger.info(f"Saved confusion matrix plot to: {val_dir / 'validation_report.png'}")
    except Exception as e:
        logger.warning(f"Could not generate confusion matrix plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation suite on exoplanet pipeline.")
    parser.add_argument("--fast", action="store_true", help="Skip FPP and pixel-level checks")
    args = parser.parse_args()
    
    run_validation_suite(fast_mode=args.fast)
