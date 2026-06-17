#!/usr/bin/env python
"""
pipeline.py
===========
Main entry point for the Exoplanet Detection Pipeline.
Assembles downloading, preprocessing, transit search (BLS),
model classification, physical fitting (batman), and visualization
into a single command-line interface.

Example usage:
    python pipeline.py --tic_id 261136679
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightkurve")

import argparse
import json
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("pipeline")

# Import custom modules
from src.download import download_lightcurve, load_xctl
from src.preprocess import preprocess_lightcurve, fold_lightcurve
from src.detect import run_bls, compute_snr, check_secondary_eclipse
from src.features import extract_features, features_to_array
from src.classify import classify_target
from src.fit_transit import fit_batman_transit
from src.visualize import plot_diagnostic, LABEL_NAMES

ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT / "results" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Import custom decision engines
from src.reverse_pipeline import run_reverse_pipeline
from src.decision_engine import evaluate_decision, log_to_manual_review_queue

def run_pipeline(tic_id: int, sector: int = None, snr_threshold: float = 5.0, force_download: bool = False, fits_path: str = None, fast: bool = False) -> dict:
    """Run end-to-end pipeline for a single TIC target star."""
    logger.info(f"============================================================")
    logger.info(f"   STARTING PIPELINE FOR TARGET: TIC {tic_id}")
    logger.info(f"============================================================")

    # 1. Obtain light curve FITS file
    if fits_path:
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"Specified FITS file does not exist: {fits_path}")
        logger.info(f"Step 1: Using local FITS file: {fits_path}")
    else:
        logger.info(f"Step 1: Fetching light curve from MAST/cache...")
        fits_path = download_lightcurve(tic_id, sector=sector)
        if not fits_path or not Path(fits_path).exists():
            raise FileNotFoundError(f"Failed to obtain light curve FITS for TIC {tic_id}")
        logger.info(f"✅ FITS file: {fits_path}")

    # 2. Preprocess light curve (sigma-clip, detrend, normalise)
    logger.info(f"Step 2: Preprocessing and cleaning time-series flux...")
    import lightkurve as lk
    lc = lk.io.read(str(fits_path))
    time, flux, flux_err = preprocess_lightcurve(lc)
    quality_flag = "good" if len(time) > 8000 else "poor"
    logger.info(f"✅ Cleaned data: {len(time)} valid cadences (baseline flux ~1.0) | Quality: {quality_flag}")

    # 3. Run Transit Least Squares (TLS) or Box Least Squares (BLS) periodic search
    logger.info(f"Step 3: Running periodic transit search...")
    if fast:
        logger.info("Fast mode enabled: running Box Least Squares (BLS)...")
        from src.detect import run_bls
        periods, power, bls_params = run_bls(time, flux, flux_err)
    else:
        logger.info("Full analysis mode: running Transit Least Squares (TLS)...")
        from src.detect import run_tls
        periods, power, bls_params = run_tls(time, flux, flux_err)

    # 3a. Alias rejection early-exit
    # If run_bls could only find periods matching known TESS systematics, stop here.
    if bls_params.get("alias_rejected"):
        rejected_period = bls_params["period"]
        rejected_snr    = bls_params["snr"]
        logger.warning(
            f"ALIAS DISCARD: TIC {tic_id} top BLS period {rejected_period:.4f} d "
            f"(SNR={rejected_snr:.2f}) is a TESS systematic alias with no clean "
            f"alternative above SNR floor. Discarding immediately."
        )
        alias_reason = (
            f"alias_discard: top period {rejected_period:.4f} d matches TESS systematic "
            f"(aliases: {[0.5, 1.0, 2.0, 13.5]} d +-0.01 d), no clean alternative found"
        )
        # Minimal structs for the CSV + JSON report
        classification = {
            "label": 0, "label_name": "No Signal",
            "confidence": 0.0,
            "combined_proba": [1.0, 0.0, 0.0, 0.0],
            "rf_proba":       [1.0, 0.0, 0.0, 0.0],
            "cnn_proba":      [1.0, 0.0, 0.0, 0.0],
        }
        transit_params = {
            "period": rejected_period, "period_err": 0.0, "t0": bls_params["t0"],
            "transit_depth": 0.0,    "transit_depth_err": 0.0,
            "transit_depth_pct": 0.0, "transit_duration_hr": 0.0,
            "transit_duration_hr_err": 0.0, "rp_over_rs": 0.0,
            "a_over_rs": 10.0, "inclination": 90.0,
            "chi2_reduced": 1.0, "rp_earth": 0.0,
        }
        reverse_results = {
            "fit_results": {"period": rejected_period, "t0": bls_params["t0"],
                            "depth": 0.0, "duration": 0.1,
                            "reduced_chi2": 1.0, "symmetry_score": 1.0},
            "tests_passed": 0, "reverse_confidence": 0.0,
        }
        decision_res = {
            "decision": "DISCARD",
            "combined_confidence": 0.0,
            "flag_reasons": alias_reason,
        }
        # Append to results.csv with alias_discard flag_reason
        import csv as _csv
        results_csv_path = ROOT / "results" / "results.csv"
        file_exists = results_csv_path.exists()
        with open(results_csv_path, "a", newline="", encoding="utf-8") as _f:
            _w = _csv.writer(_f)
            if not file_exists:
                _w.writerow(["tic_id", "decision", "final_class", "confidence",
                             "period", "period_err", "depth", "depth_err",
                             "duration", "duration_err", "snr", "flag_reasons"])
            _w.writerow([
                tic_id, "DISCARD", "No Signal", "0.0000",
                f"{rejected_period:.6f}", "0.0001",
                "0.000000", "0.0001", "0.000000", "0.001",
                f"{rejected_snr:.2f}",
                alias_reason,
            ])
        # Save minimal JSON report
        report = {
            "tic_id": tic_id,
            "decision": decision_res,
            "classification": classification,
            "transit_parameters": transit_params,
            "reverse_results": reverse_results,
            "bls_parameters": {
                "period": rejected_period, "depth": 0.0, "snr": rejected_snr,
                "duration": bls_params["duration"], "t0": bls_params["t0"],
                "alias_rejected": True,
            },
        }
        report_path = REPORTS_DIR / f"TIC_{tic_id}_report.json"
        with open(report_path, "w", encoding="utf-8") as _f:
            json.dump(report, _f, indent=4)
        logger.info(f"Alias-discard report saved -> {report_path}")
        logger.info("============================================================")
        logger.info(f"   PIPELINE COMPLETE (ALIAS DISCARD) FOR TIC {tic_id}")
        logger.info("============================================================")
        return report

    logger.info(f"Strongest candidate period found: {bls_params['period']:.5f} days (SNR: {bls_params['snr']:.2f})")

    # 4. Check if signal is significant
    if bls_params['snr'] < snr_threshold:
        logger.warning(f"⚠️ Signal SNR ({bls_params['snr']:.2f}) is below threshold ({snr_threshold}).")
        logger.warning(f"Classification will fallback to 'No Signal'.")
        classification = {
            "label": 0,
            "label_name": "No Signal",
            "confidence": 1.0 - (bls_params['snr'] / snr_threshold),
            "combined_proba": [1.0, 0.0, 0.0, 0.0],
            "rf_proba": [1.0, 0.0, 0.0, 0.0],
            "cnn_proba": [1.0, 0.0, 0.0, 0.0]
        }
        depth_non_neg = max(0.0, bls_params['depth'])
        transit_params = {
            "period": bls_params['period'], "period_err": 0.0, "t0": bls_params['t0'],
            "transit_depth": bls_params['depth'], "transit_depth_err": 0.0,
            "transit_depth_pct": bls_params['depth'] * 100, "transit_duration_hr": bls_params['duration'] * 24,
            "transit_duration_hr_err": 0.0, "rp_over_rs": np.sqrt(depth_non_neg),
            "a_over_rs": 10.0, "inclination": 90.0, "chi2_reduced": 1.0, "rp_earth": np.sqrt(depth_non_neg) * 109.2
        }
        reverse_results = {
            "fit_results": {
                "period": bls_params['period'], "t0": bls_params['t0'], "depth": 0.0, "duration": 0.1, "reduced_chi2": 1.0, "symmetry_score": 1.0
            },
            "tests_passed": 0,
            "reverse_confidence": 0.0
        }
        decision_res = {
            "decision": "DISCARD",
            "combined_confidence": 0.0,
            "flag_reasons": "SNR below threshold"
        }
        phase_folded, folded_model = None, None
        contamination_res = {"contaminated": False, "contamination_ratio": None, "n_nearby_gaia_stars": 0}
        fpp_res = {"fpp": None, "combined_fpp": None, "fpp_status": "skipped"}
        n_sectors_consistent = 1
    else:
        # 5. Extract features for classical ML
        logger.info(f"Step 4: Extracting signal parameters and diagnostic features...")
        features = extract_features(time, flux, flux_err, bls_params)

        # 6. Fold light curve at best period for CNN/fitting
        phase, folded_flux = fold_lightcurve(time, flux, bls_params['period'], bls_params['t0'])

        # 7. Run Classifier (Random Forest + CNN Ensemble)
        logger.info(f"Step 5: Running ML classification ensemble...")
        classification = classify_target(features, folded_flux)
        logger.info(f"✅ Classification Result: {classification['label_name']} (Confidence: {classification['confidence']*100:.1f}%)")

        # 8. Fit Mandel & Agol transit model (batman)
        logger.info(f"Step 6: Fitting physical transit model & running bootstrap errors...")
        
        # Get sector from light curve metadata
        sector_val = sector
        try:
            if sector_val is None:
                if hasattr(lc, "sector"):
                    sector_val = lc.sector
                elif hasattr(lc, "meta") and "SECTOR" in lc.meta:
                    sector_val = lc.meta["SECTOR"]
            
            # Plain English: Safe check to convert list/array sector values to a single integer
            if sector_val is not None:
                if hasattr(sector_val, "__iter__") and not isinstance(sector_val, (str, bytes)):
                    sector_val = int(sector_val[0])
                else:
                    sector_val = int(sector_val)
        except Exception as se_err:
            logger.warning(f"Failed to resolve sector value: {se_err}")
            sector_val = None

        # 8_pre. Pixel contamination check (Test 9)
        contamination_res = {"contaminated": False, "contamination_ratio": None, "n_nearby_gaia_stars": 0}
        if not fast:
            logger.info("Running pixel-level contamination check...")
            try:
                from flag_analyzer import check_pixel_contamination
                contamination_res = check_pixel_contamination(
                    tic_id=tic_id,
                    sector=sector_val,
                    period=bls_params["period"],
                    epoch=bls_params["t0"],
                    duration=bls_params["duration"]
                )
            except Exception as e:
                logger.warning(f"Pixel contamination check failed: {e}")

        # Add tic_id to bls_params for stellar parameter lookup inside fit_batman_transit
        bls_params["tic_id"] = tic_id
        transit_params = fit_batman_transit(time, flux, flux_err, bls_params, n_bootstrap=25)
        rp_earth = transit_params['rp_earth']
        logger.info(f"Physical Fit: Radius: {rp_earth:.2f} R_earth | Depth: {transit_params['transit_depth_pct']:.4f}%")

        # 8a. Physical radius plausibility check
        # ─────────────────────────────────────────────────────────────────────
        # Fitted planet radius is the most direct indicator of what kind of
        # object we are looking at. True planets span ~0.5–25 R⊕. Anything
        # larger is almost certainly a stellar-sized body (eclipsing binary
        # or blended EB), not a planet.
        #
        # Thresholds (Earth radii):
        #   < 0.5         → too_small_flag      (below detection sensitivity)
        #   0.5 – 25      → plausible planet     (no override)
        #   25 – 100      → giant_radius_eb_suspect
        #   > 100         → almost_certainly_eb  (stellar radius range)
        # ─────────────────────────────────────────────────────────────────────
        radius_flag  = None   # one of: too_small_flag | giant_radius_eb_suspect | almost_certainly_eb
        radius_note  = None

        if rp_earth < 0.5:
            radius_flag = "too_small_flag"
            radius_note = (
                f"Fitted radius {rp_earth:.2f} R_earth is below the ~0.5 R_earth "
                f"TESS detection limit. Signal may be instrumental noise or a very "
                f"grazing geometry. Treat with caution."
            )
            logger.warning(f"Radius plausibility: {rp_earth:.2f} R_earth < 0.5 -> too_small_flag")

        elif rp_earth > 100.0:
            radius_flag = "almost_certainly_eb"
            radius_note = (
                f"Fitted radius {rp_earth:.1f} R_earth exceeds 100 R_earth (stellar "
                f"radius range). Almost certainly an eclipsing binary or contaminating "
                f"star. Recommend EB catalog cross-check and spectroscopic follow-up."
            )
            logger.warning(f"Radius plausibility: {rp_earth:.1f} R_earth > 100 -> almost_certainly_eb")

        elif rp_earth > 25.0:
            radius_flag = "giant_radius_eb_suspect"
            radius_note = (
                f"Fitted radius {rp_earth:.1f} R_earth exceeds maximum planet size "
                f"(~25 R_earth). Almost certainly an eclipsing binary or blended "
                f"eclipsing binary. Recommend EB catalog cross-check."
            )
            logger.warning(f"Radius plausibility: {rp_earth:.1f} R_earth > 25 -> giant_radius_eb_suspect")

        else:
            logger.info(f"Radius plausibility: {rp_earth:.2f} R_earth is in the plausible planet range (0.5-25 R_earth). OK.")

        # Run Reverse Pipeline
        logger.info(f"Step 6b: Running reverse pipeline (trapezoidal fit & physical checks)...")
        reverse_results = run_reverse_pipeline(time, flux, flux_err, bls_params)
        logger.info(f"✅ Reverse tests passed: {reverse_results['tests_passed']}/6 | Confidence: {reverse_results['reverse_confidence']*100:.1f}%")

        # Determine consistent sector count using lightkurve search
        n_sectors_consistent = 1
        try:
            import lightkurve as lk
            search_sectors = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            if len(search_sectors) > 0:
                sectors = search_sectors.table["sequence_number"]
                n_sectors_consistent = int(len(np.unique(sectors)))
        except Exception:
            pass

        # Decision Engine
        logger.info(f"Step 6c: Running cross-check engine & three-tier decision system...")
        decision_res = evaluate_decision(
            classification, reverse_results, bls_params, quality_flag,
            forward_fit=transit_params, n_sectors_consistent=n_sectors_consistent
        )

        # 8b. Radius plausibility overrides
        # Apply AFTER evaluate_decision so radius evidence always wins.
        if radius_flag in ("giant_radius_eb_suspect", "almost_certainly_eb"):
            # Downgrade classification label to eclipsing_binary
            original_label = classification.get("label_name", "Unknown")
            classification["label"]      = 3          # EB label index (consistent with LABEL_NAMES)
            classification["label_name"] = "Eclipsing Binary"
            classification["confidence"] = 0.0        # reset confidence — this is a veto
            # Force DISCARD regardless of what the ML ensemble said
            decision_res["decision"]           = "DISCARD"
            decision_res["combined_confidence"] = 0.0
            decision_res["flag_reasons"] = (
                f"[{radius_flag}] {radius_note} "
                f"(original ML label was: {original_label})"
            )
            logger.warning(
                f"Radius override: classification downgraded from '{original_label}' "
                f"to 'Eclipsing Binary'. Decision forced to DISCARD. "
                f"Flag: {radius_flag}."
            )

        elif radius_flag == "too_small_flag":
            # Add a caution note but don't force DISCARD — could still be a real small planet
            existing = decision_res.get("flag_reasons", "")
            decision_res["flag_reasons"] = (
                f"[too_small_flag] {radius_note}"
                + (f"; {existing}" if existing else "")
            )
            logger.warning(
                f"Radius caution: {rp_earth:.2f} R_earth flagged as too_small. "
                f"Decision kept as-is ({decision_res['decision']})."
            )

        # Apply pixel contamination overrides
        if contamination_res.get("contaminated") is True:
            is_borderline = (decision_res["decision"] == "FLAG") or (decision_res["decision"] == "KEEP" and decision_res["combined_confidence"] < 0.75)
            if rp_earth > 25.0:
                classification["label"] = 3
                classification["label_name"] = "Eclipsing Binary"
                decision_res["decision"] = "DISCARD"
                decision_res["combined_confidence"] = 0.0
                decision_res["flag_reasons"] = "giant_radius_plus_contamination — almost certainly blend"
                logger.warning("Contamination override: giant radius + pixel contamination. Forced to DISCARD.")
            elif is_borderline:
                decision_res["decision"] = "FLAG"
                decision_res["flag_reasons"] = "pixel_contamination_detected"
                logger.warning("Contamination override: borderline signal + pixel contamination. Forced to FLAG.")

        # False Positive Probability (FPP) calculation (only for KEEP decisions)
        fpp_res = {"fpp": None, "combined_fpp": None, "fpp_status": "skipped"}
        if decision_res["decision"] == "KEEP" and not fast:
            logger.info("Running TRICERATOPS False Positive Probability calculation...")
            try:
                from src.fpp_calculator import calculate_fpp
                fpp_res = calculate_fpp(
                    tic_id=tic_id,
                    period=transit_params["period"],
                    epoch=transit_params["t0"],
                    depth=transit_params["transit_depth"],
                    duration=transit_params["transit_duration_hr"] / 24.0,
                    sector=sector_val,
                    time=time,
                    flux=flux
                )
                fpp = fpp_res.get("fpp")
                combined_fpp = fpp_res.get("combined_fpp")
                fpp_status = fpp_res.get("fpp_status")
                
                if combined_fpp is not None:
                    print(f"   FPP: {fpp:.3f} — {fpp_status}")
                    
                    if combined_fpp > 0.5:
                        logger.warning(f"High false positive probability ({combined_fpp:.2f} > 0.5). Downgrading KEEP to FLAG.")
                        decision_res["decision"] = "FLAG"
                        decision_res["flag_reasons"] = f"High false positive probability: {fpp:.2f}"
            except Exception as e:
                logger.warning(f"FPP calculation failed: {e}")

        logger.info(f"DECISION: {decision_res['decision'].upper()}")
        logger.info(f"   Reason: {decision_res['flag_reasons']}")

        classification["vetting_status"] = decision_res["decision"]
        classification["vetting_reason"] = decision_res["flag_reasons"]
        classification["confidence"]     = decision_res["combined_confidence"]


        # Generate binned model curves for plotting
        import batman
        params = batman.TransitParams()
        params.t0 = transit_params['t0']
        params.per = transit_params['period']
        params.rp = transit_params['rp_over_rs']
        params.a = transit_params['a_over_rs']
        params.inc = transit_params['inclination']
        params.ecc = 0.0
        params.w = 90.0
        params.u = [0.4, 0.3]
        params.limb_dark = "quadratic"

        # Generate model curve at phase positions
        phases_grid = np.linspace(-0.5, 0.5, len(folded_flux))
        time_grid = phases_grid * transit_params['period'] + transit_params['t0']
        m_model = batman.TransitModel(params, time_grid)
        folded_model = m_model.light_curve(params)
        phase_folded = folded_flux

    # Log to Manual Review Queue if flagged
    if decision_res["decision"] == "FLAG":
        log_to_manual_review_queue(tic_id, decision_res, reverse_results.get("fit_results", {}), str(ROOT / "results"))

    # 9. Create diagnostic 4-panel plot
    logger.info(f"Step 7: Plotting and saving diagnostic visualization...")
    fig = plot_diagnostic(
        tic_id=tic_id,
        time_raw=lc.time.value,
        flux_raw=lc.flux.value,
        time_clean=time,
        flux_clean=flux,
        bls_periods=periods,
        bls_power=power,
        bls_params=bls_params,
        transit_params=transit_params,
        classification=classification,
        phase_folded_flux=phase_folded,
        model_flux=folded_model,
        save=True
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info(f"✅ Plot saved to results/figures/TIC_{tic_id}_diagnostic.png")

    # 10. Save JSON report
    report = {
        "tic_id": tic_id,
        "decision": decision_res,
        "classification": classification,
        "transit_parameters": transit_params,
        "reverse_results": reverse_results,
        "bls_parameters": {
            "period": bls_params["period"],
            "depth": bls_params["depth"],
            "snr": bls_params["snr"],
            "duration": bls_params["duration"],
            "t0": bls_params["t0"]
        }
    }
    
    report_path = REPORTS_DIR / f"TIC_{tic_id}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    logger.info(f"✅ Report saved to {report_path}")

    # Append to results.csv
    import csv
    results_csv_path = ROOT / "results" / "results.csv"
    file_exists = results_csv_path.exists()
    with open(results_csv_path, "a", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        if not file_exists:
            csv_writer.writerow([
                "tic_id", "decision", "final_class", "confidence", "period", "period_err",
                "depth", "depth_err", "duration", "duration_err", "snr", "flag_reasons",
                "fpp", "combined_fpp", "fpp_status",
                "contamination_ratio", "n_nearby_gaia_stars", "n_sectors_consistent"
            ])
        fit_res = reverse_results.get("fit_results", {})
        csv_writer.writerow([
            tic_id,
            decision_res["decision"],
            classification["label_name"],
            f"{decision_res['combined_confidence']:.4f}",
            f"{fit_res.get('period', 0.0):.6f}",
            "0.0001",
            f"{fit_res.get('depth', 0.0):.6f}",
            "0.0001",
            f"{fit_res.get('duration', 0.0):.6f}",
            "0.001",
            f"{bls_params.get('snr'):.2f}",
            decision_res["flag_reasons"],
            fpp_res.get("fpp"),
            fpp_res.get("combined_fpp"),
            fpp_res.get("fpp_status"),
            contamination_res.get("contamination_ratio"),
            contamination_res.get("n_nearby_gaia_stars"),
            n_sectors_consistent
        ])
    logger.info(f"✅ Appended to results.csv")
    
    # Update results/metadata.json with pipeline version
    try:
        meta_path = ROOT / "results" / "metadata.json"
        meta_data = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        meta_data["pipeline_version"] = "2.0"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to update metadata.json: {e}")
    
    logger.info(f"============================================================")
    logger.info(f"   PIPELINE RUN SUCCESSFULLY COMPLETED FOR TIC {tic_id}")
    logger.info(f"============================================================")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TESS Exoplanet Detection Pipeline")

    # --- core target arguments ---
    parser.add_argument("--tic_id", type=int, default=None, help="TIC ID of the star (e.g. 261136679)")
    parser.add_argument("--sector", type=int, default=None, help="TESS Sector to download (optional)")
    parser.add_argument("--snr", type=float, default=5.0, help="BLS detection SNR threshold (default: 5.0)")
    parser.add_argument("--force", action="store_true", help="Force re-download of light curve FITS")
    parser.add_argument("--fits_path", type=str, default=None, help="Path to local TESS FITS file (bypasses MAST download)")

    # --- FLAG deep-analysis mode ---
    parser.add_argument(
        "--analyze-flags", action="store_true",
        help="Run the FLAG deep-analysis layer on all entries in manual_review_queue.csv "
             "(no --tic_id required when used alone)."
    )

    # --- speed / analysis depth flags ---
    parser.add_argument("--fast", action="store_true", help="Skip FPP and pixel-level checks for speed, use BLS instead of TLS")
    parser.add_argument("--full-analysis", action="store_true", default=True, help="Enable all physical checks (TLS, FPP, and pixel checks)")

    args = parser.parse_args()

    # ── Mode A: single-star pipeline ─────────────────────────────────────────
    if args.tic_id is not None:
        try:
            report = run_pipeline(
                args.tic_id,
                sector=args.sector,
                snr_threshold=args.snr,
                force_download=args.force,
                fits_path=args.fits_path,
                fast=args.fast
            )
            # Auto-trigger deep analysis if this star was flagged
            if getattr(args, "analyze_flags", False) or \
               report.get("decision", {}).get("decision") == "FLAG":
                logger.info("Star was FLAG — running deep-analysis layer...")
                from flag_analyzer import run_flag_analysis
                run_flag_analysis(tic_ids=[args.tic_id])
        except Exception as e:
            logger.exception(f"Pipeline crashed for TIC {args.tic_id}: {e}")
            exit(1)

    # ── Mode B: batch flag analysis on the whole review queue ────────────────
    elif getattr(args, "analyze_flags", False):
        logger.info("Running FLAG deep-analysis on all entries in manual_review_queue.csv ...")
        from flag_analyzer import run_flag_analysis
        run_flag_analysis()

    else:
        parser.print_help()
        logger.error("Please provide --tic_id or --analyze-flags.")
        exit(1)
