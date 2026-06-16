"""
src/report_generator.py
========================
Generates a detailed technical report card for a single candidate.
Reads the TIC_{tic_id}_report.json and outputs a formatted Markdown file report_TIC_{tic_id}.md.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

def generate_report_card(report_json_path: str, output_dir: str) -> dict:
    """
    Generate a formatted technical report card markdown file.
    
    WHAT: Parses candidate parameters and formats them as a clean report card.
    WHY: Satisfies ISRO challenge specifications for detailed planetary report cards.
    """
    try:
        if not os.path.exists(report_json_path):
            return {"success": False, "error": f"JSON report not found: {report_json_path}"}
            
        with open(report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tic_id = data.get("tic_id")
        decision_data = data.get("decision", {})
        classification = data.get("classification", {})
        transit_params = data.get("transit_parameters", {})
        reverse_results = data.get("reverse_results", {})
        bls_params = data.get("bls_parameters", {})
        
        output_path = os.path.join(output_dir, f"report_TIC_{tic_id}.md")
        os.makedirs(output_dir, exist_ok=True)
        
        report_md = f"""# 🪐 Exoplanet Vetting Report Card: TIC {tic_id}
**ISRO Hackathon Challenge 7 - AI-enabled Exoplanet Detection Pipeline**

---

## 1. Executive Summary
- **Target Star**: TIC {tic_id}
- **Vetting Decision**: **{decision_data.get('decision', 'FLAG')}**
- **Combined Pipeline Confidence**: **{decision_data.get('combined_confidence', 0.0)*100:.2f}%**
- **Forward ML Classification**: {classification.get('label_name', 'Unknown')}
- **Vetting Flag Reasons**: *{decision_data.get('flag_reasons', 'None')}*

---

## 2. Orbital & Physical Parameters (Mandel-Agol Fit)
The physical parameters below were estimated by fitting a Mandel-Agol transit model using `batman`. Uncertainties were computed using 25 bootstrap iterations.

| Parameter | Value | Uncertainty | Unit | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Orbital Period** | {transit_params.get('period', 0.0):.6f} | {transit_params.get('period_err', 0.0):.6f} | days | Time between consecutive transits |
| **Transit Depth** | {transit_params.get('transit_depth_pct', 0.0):.4f} | {transit_params.get('transit_depth_err', 0.0)*100:.4f} | % | Fractional flux drop |
| **Transit Duration** | {transit_params.get('transit_duration_hr', 0.0):.3f} | {transit_params.get('transit_duration_hr_err', 0.0):.3f} | hours | Total transit event width |
| **Radius Ratio ($R_p/R_*$)** | {transit_params.get('rp_over_rs', 0.0):.4f} | - | - | Ratio of planet-to-stellar radius |
| **Planet Radius ($R_e$)** | {transit_params.get('rp_earth', 0.0):.2f} | - | $R_\\oplus$ | Calculated planet radius (assuming solar-type host) |
| **Reduced $\\chi^2$** | {transit_params.get('chi2_reduced', 0.0):.3f} | - | - | Mandel-Agol fit goodness |

---

## 3. Dual-Pipeline Cross-Check Diagnostics

### A. Forward Pipeline (ML Classification Ensemble)
- **Random Forest Probabilities**: {['{:.1f}%'.format(p*100) for p in classification.get('rf_proba', [0.25]*4)]}
- **CNN Probabilities**: {['{:.1f}%'.format(p*100) for p in classification.get('cnn_proba', [0.25]*4)]}
- **Ensemble Result**: {classification.get('label_name', 'Unknown')}

### B. Reverse Pipeline (Trapezoidal Physics Checks)
- **Trapezoidal Fit reduced $\\chi^2$**: {reverse_results.get('fit_results', {}).get('reduced_chi2', 0.0):.3f}
- **Transit Symmetry Score**: {reverse_results.get('fit_results', {}).get('symmetry_score', 0.0):.3f}
- **Physical Tests Passed**: **{reverse_results.get('tests_passed', 0)}/6**
  - Is period consistent (BLS vs Fit): `{reverse_results.get('is_period_consistent', False)}`
  - Is transit depth physical: `{reverse_results.get('is_depth_physical', False)}`
  - Is transit duration physical: `{reverse_results.get('is_duration_physical', False)}`
  - Is secondary eclipse shallow (non-EB): `{reverse_results.get('is_secondary_shallow', False)}`
  - Is transit shape symmetric: `{reverse_results.get('is_shape_symmetric', False)}`
  - Is reduced chi-squared good: `{reverse_results.get('is_fit_good', False)}`

---

## 4. Visual Diagnostics
The 4-panel diagnostic plot for this target shows the detrended curve, the BLS periodogram power, the phase-folded light curve compared to the Mandel-Agol batman model, and the ML prediction bar charts.

![TIC {tic_id} Diagnostics](../results/figures/TIC_{tic_id}_diagnostic.png)
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_md)
            
        logger.info(f"Report card generated at: {output_path}")
        return {"success": True, "output_path": output_path}
    except Exception as e:
        logger.error(f"Failed to generate report card: {e}")
        return {"success": False, "error": str(e)}
