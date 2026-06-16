"""
batch_processor.py
==================
Batch processor for processing 20-30k light curves.
Implements the forward pipeline, reverse pipeline, cross-check decision engine,
and handles partial saves every 100 stars to avoid losing progress.
"""

import os
import sys
import csv
import json
import time
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Import custom pipeline modules
from src.download import download_lightcurve
from src.preprocess import preprocess_lightcurve, fold_lightcurve
from src.detect import run_bls
from src.features import extract_features
from src.classify import classify_target
from src.reverse_pipeline import run_reverse_pipeline
from src.decision_engine import evaluate_decision, log_to_manual_review_queue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ROOT, "results", "batch_processing.log"), mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("batch_processor")

# Directory Constants
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")
FITS_DIR = os.path.join(DATA_DIR, "raw_fits")

# Ensure results subdirectories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_batch_processing(target_csv_path: str, limit: int = 1000):
    """
    Run exoplanet pipeline on a target catalog batch.
    
    WHAT: Processes light curves sequentially and classifies them.
    WHY: Handles large-scale catalogs robustly with checkpointing every 100 files.
    """
    logger.info("=" * 70)
    logger.info(f"STARTING BATCH PROCESSING (Limit: {limit} targets)")
    logger.info(f"Targets CSV: {target_csv_path}")
    logger.info("=" * 70)
    
    if not os.path.exists(target_csv_path):
        logger.error(f"Target catalog file not found: {target_csv_path}")
        return
        
    df_targets = pd.read_csv(target_csv_path)
    if "tic_id" not in df_targets.columns:
        # Check alternative headers
        for col in ["TIC", "TIC ID", "ID"]:
            if col in df_targets.columns:
                df_targets.rename(columns={col: "tic_id"}, inplace=True)
                break
                
    # Sample down to limit
    df_targets = df_targets.head(limit)
    
    results_csv_path = os.path.join(RESULTS_DIR, "results.csv")
    
    # Read already processed targets to resume progress
    processed_tic_ids = set()
    if os.path.exists(results_csv_path):
        try:
            df_existing = pd.read_csv(results_csv_path)
            if "tic_id" in df_existing.columns:
                processed_tic_ids = set(df_existing["tic_id"].unique())
                logger.info(f"Resuming: found {len(processed_tic_ids)} already processed targets.")
        except Exception:
            pass
            
    # Open CSV writer for results
    file_exists = os.path.exists(results_csv_path)
    results_file = open(results_csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(results_file)
    
    if not file_exists:
        writer.writerow([
            "tic_id", "decision", "final_class", "confidence", "period", "period_err",
            "depth", "depth_err", "duration", "duration_err", "snr", "flag_reasons"
        ])
        
    targets_to_process = df_targets[~df_targets["tic_id"].isin(processed_tic_ids)]
    logger.info(f"Remaining targets to process: {len(targets_to_process)}")
    
    loop_counter = 0
    
    for _, row in tqdm(targets_to_process.iterrows(), total=len(targets_to_process), desc="Batch Vetting"):
        tic_id = int(row["tic_id"])
        
        try:
            # 1. Download FITS
            fits_path = download_lightcurve(tic_id)
            if not fits_path:
                logger.warning(f"Failed to fetch FITS for TIC {tic_id}")
                continue
                
            # 2. Preprocess
            import lightkurve as lk
            lc = lk.io.read(str(fits_path))
            time_arr, flux_arr, flux_err_arr = preprocess_lightcurve(lc)
            
            # Simple quality heuristic: "poor" if less than 80% of data points remain compared to typical TESS baseline (e.g. 10k cadences)
            quality_flag = "good" if len(time_arr) > 8000 else "poor"
            
            # 3. Run BLS
            _, _, bls_results = run_bls(time_arr, flux_arr, flux_err_arr)
            
            if bls_results.get("snr", 0.0) < 5.0:
                # Discard low SNR signals early
                writer.writerow([
                    tic_id, "DISCARD", "No Signal", f"{1.0 - (bls_results.get('snr')/5.0):.4f}",
                    f"{bls_results.get('period'):.6f}", "0.0", f"{bls_results.get('depth'):.6f}", "0.0",
                    f"{bls_results.get('duration'):.6f}", "0.0", f"{bls_results.get('snr'):.2f}",
                    "BLS signal SNR is below detection threshold"
                ])
                loop_counter += 1
                continue
                
            # 4. Extract features & phase-fold
            features = extract_features(time_arr, flux_arr, flux_err_arr, bls_results)
            phase, folded_flux = fold_lightcurve(time_arr, flux_arr, bls_results['period'], bls_results['t0'])
            
            # 5. Run Forward Classifier
            classification = classify_target(features, folded_flux)
            
            # 6. Run Reverse Pipeline
            reverse_results = run_reverse_pipeline(time_arr, flux_arr, flux_err_arr, bls_results)
            fit_results = reverse_results.get("fit_results", {})
            
            # 7. Decision Engine
            decision_res = evaluate_decision(classification, reverse_results, bls_results, quality_flag)
            
            # Save results
            writer.writerow([
                tic_id,
                decision_res["decision"],
                classification["label_name"],
                f"{decision_res['combined_confidence']:.4f}",
                f"{fit_results.get('period', 0.0):.6f}",
                "0.0001",  # Period error placeholder
                f"{fit_results.get('depth', 0.0):.6f}",
                "0.0001",  # Depth error placeholder
                f"{fit_results.get('duration', 0.0):.6f}",
                "0.001",   # Duration error placeholder
                f"{bls_results.get('snr'):.2f}",
                decision_res["flag_reasons"]
            ])
            
            # Log flagged targets to manual review queue
            if decision_res["decision"] == "FLAG":
                log_to_manual_review_queue(tic_id, decision_res, fit_results, RESULTS_DIR)
                
            loop_counter += 1
            
            # Checkpoint: Save progress (flush to disk) every 100 stars
            if loop_counter % 100 == 0:
                results_file.flush()
                logger.info(f"Checkpoint: Successfully flushed progress at {loop_counter} targets.")
                
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to process TIC {tic_id} in batch: {e}")
            
    results_file.close()
    logger.info("Batch processing run complete!")

if __name__ == "__main__":
    run_batch_processing(os.path.join(DATA_DIR, "training_targets.csv"), limit=200)
