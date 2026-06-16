"""
src/decision_engine.py
======================
Decides if a candidate should be KEEP, FLAG (for manual review), or DISCARD.
Implements the Cross-Check Engine and the Three-Tier Decision System.
"""

import os
import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_decision(classification: dict, reverse_results: dict, bls_params: dict, quality_flag: str = "good") -> dict:
    """
    Perform cross-check between forward ML and reverse physics pipelines, and make a decision.
    
    WHAT: Cross-checks period, depth, and classification agreement to calculate combined confidence.
    WHY: Provides a robust three-tier decision output matching ISRO hackathon specifications.
    """
    forward_class = classification.get("label", 3)
    forward_confidence = classification.get("confidence", 0.0)
    
    reverse_fit = reverse_results.get("fit_results", {})
    reverse_tests_passed = reverse_results.get("tests_passed", 0)
    
    # 1. Period Agreement: BLS period and fitted period agree within 5%
    bls_period = bls_params.get("period", 1.0)
    fit_period = reverse_fit.get("period", 1.0)
    period_diff = np.abs(bls_period - fit_period) / bls_period
    period_agreement = period_diff < 0.05
    
    # 2. Depth Agreement: depths agree within 20%
    bls_depth = bls_params.get("depth", 0.01)
    fit_depth = reverse_fit.get("depth", 0.01)
    depth_diff = np.abs(bls_depth - fit_depth) / max(1e-5, bls_depth)
    depth_agreement = depth_diff < 0.20
    
    # 3. Class Physics Agreement: forward class is consistent with physical tests
    # Planet (1): must pass depth, duration, and secondary eclipse tests
    is_depth_physical = reverse_results.get("is_depth_physical", True)
    is_duration_physical = reverse_results.get("is_duration_physical", True)
    is_secondary_shallow = reverse_results.get("is_secondary_shallow", True)
    
    class_physics_agreement = True
    if forward_class == 1:
        if not (is_depth_physical and is_duration_physical and is_secondary_shallow):
            class_physics_agreement = False
    elif forward_class == 2:
        # Eclipsing Binary should have a deep secondary or deep primary depth
        if is_secondary_shallow and fit_depth < 0.05:
            class_physics_agreement = False
            
    # Calculate Combined Confidence
    combined_confidence = forward_confidence * (reverse_tests_passed / 6.0)
    if not period_agreement:
        combined_confidence *= 0.5
    if not class_physics_agreement:
        combined_confidence *= 0.6
        
    combined_confidence = float(np.clip(combined_confidence, 0.0, 1.0))
    
    # Determine flag reasons (Special cases & borderline criteria)
    flag_reasons = []
    
    # Special case A: Transit shape symmetry < 0.7
    symmetry_score = reverse_fit.get("symmetry_score", 1.0)
    if symmetry_score < 0.7:
        flag_reasons.append(f"Asymmetric transit shape (symmetry score: {symmetry_score:.2f} < 0.70)")
        
    # Special case B: Period ambiguity (BLS has multiple close peaks - handled in pipeline check or if bls power is low)
    # E.g. we add a flag if the BLS SNR is borderline or harmonics check flags it
    if bls_params.get("snr", 0.0) < 6.5:
        flag_reasons.append(f"Borderline BLS signal SNR ({bls_params.get('snr'):.2f} < 6.5)")
        
    # Special case C: Preprocessing quality flag is "poor"
    if quality_flag == "poor":
        flag_reasons.append("Poor data quality after preprocessing")
        
    # Special case D: Forward pipeline predicts False Positive / Blend (Class 3)
    if forward_class == 3:
        flag_reasons.append("ML model predicts False Positive / Blend")
        
    # Special case E: Reduced chi-squared > 5.0 (model fits poorly)
    reduced_chi2 = reverse_fit.get("reduced_chi2", 1.0)
    if reduced_chi2 > 5.0:
        flag_reasons.append(f"Poor model fit (reduced chi-squared: {reduced_chi2:.2f} > 5.0)")
        
    # Check boundaries for Three-Tier Decision
    if combined_confidence < 0.30:
        decision = "DISCARD"
        reason_summary = "Low confidence signal, likely noise or artefact"
    elif len(flag_reasons) > 0 or (0.30 <= combined_confidence < 0.70):
        decision = "FLAG"
        if not flag_reasons:
            flag_reasons.append("Borderline combined confidence score")
        reason_summary = "; ".join(flag_reasons)
    else:
        decision = "KEEP"
        reason_summary = "Plausible candidate matching physical transit shape"
        
    return {
        "decision": decision,
        "combined_confidence": combined_confidence,
        "period_agreement": bool(period_agreement),
        "depth_agreement": bool(depth_agreement),
        "class_physics_agreement": bool(class_physics_agreement),
        "flag_reasons": reason_summary
    }

def log_to_manual_review_queue(tic_id: int, decision_results: dict, fit_results: dict, output_dir: str):
    """
    Append FLAG cases to results/manual_review_queue.csv.
    
    WHAT: Saves borderline or flagged targets for manual human inspection.
    WHY: Prevents missing edge cases or hard false positive/blend signatures.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "manual_review_queue.csv")
        file_exists = os.path.exists(file_path)
        
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["tic_id", "combined_confidence", "period", "depth", "flag_reasons"])
            writer.writerow([
                tic_id,
                f"{decision_results['combined_confidence']:.4f}",
                f"{fit_results.get('period', 0.0):.6f}",
                f"{fit_results.get('depth', 0.0):.6f}",
                decision_results["flag_reasons"]
            ])
    except Exception as e:
        logger.error(f"Failed to log to manual review queue: {e}")
