"""
kaggle_discovery_runner.py
===========================
Companion module for tess_weekly_runner.ipynb.

Provides three public functions called by the notebook cells:
    build_target_list()        — Cell 3: query TIC, filter, prioritise
    run_discovery_session()    — Cell 4: full 13-step pipeline loop
    generate_session_summary() — Cell 5: statistics and report

All network/file operations have try/except so one bad star never
crashes the whole session. Results are saved incrementally every
save_every_n stars so a Kaggle timeout loses at most ~50 stars.
"""

import csv
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger("kaggle_runner")

# ─── Known TESS systematic alias periods (copied from detect.py) ─────────────
TESS_ALIAS_PERIODS = [0.5, 1.0, 2.0, 13.5]
ALIAS_TOLERANCE    = 0.01


def _is_alias(period: float) -> bool:
    return any(abs(period - a) < ALIAS_TOLERANCE for a in TESS_ALIAS_PERIODS)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — BUILD TARGET LIST
# ═══════════════════════════════════════════════════════════════════════════════

def build_target_list(
    n_targets: int = 800,
    already_processed: set = None,
    mag_range: tuple = (8, 13),
    teff_range: tuple = (3500, 7000),
    radius_range: tuple = (0.5, 2.0),
    exclude_giants: bool = True,
    exclude_known_contaminated: bool = True,
    prioritise_multi_sector: bool = True,
    prioritise_not_in_toi: bool = True,
) -> pd.DataFrame:
    """
    WHAT: Query the TESS Input Catalog (TIC) via astroquery to get a list of
          candidate stars for this week's discovery run.
    WHY:  We apply physical filters to focus on sun-like stars where TESS is
          likely to detect a planet, and skip stars we already processed to
          make the weekly run truly cumulative.

    Args:
        n_targets:              How many stars to return.
        already_processed:      Set of TIC IDs to skip.
        mag_range:              TESS magnitude limits (bright enough for photometry).
        teff_range:             Effective temperature range (planet-hosting range).
        radius_range:           Stellar radius in solar radii (exclude giants).
        exclude_giants:         Remove log g < 3.5 (luminosity class III+).
        exclude_known_contaminated: Remove crowded fields with contamination ratio > 0.1.
        prioritise_multi_sector: Prefer stars observed in 2+ TESS sectors.
        prioritise_not_in_toi:   Prefer stars not yet in the TOI candidate catalog.

    Returns:
        DataFrame with columns: tic_id, ra, dec, tmag, teff, radius, n_sectors, priority_score
    """
    if already_processed is None:
        already_processed = set()

    logger.info(f"Querying TESS Input Catalog for up to {n_targets} targets...")

    # ── Query TIC via astroquery ──────────────────────────────────────────────
    try:
        from astroquery.mast import Catalogs
        catalog = Catalogs.query_criteria(
            catalog    = "TIC",
            Tmag       = list(mag_range),
            Teff       = list(teff_range),
            rad        = list(radius_range),
            objType    = "STAR",
            columns    = ["ID", "ra", "dec", "Tmag", "Teff", "rad", "logg",
                          "contratio", "priority", "wdflag"],
            sortby     = "priority",
        )
        df = catalog.to_pandas()
        df = df.rename(columns={
            "ID": "tic_id", "Tmag": "tmag", "Teff": "teff",
            "rad": "radius", "logg": "logg", "contratio": "contratio",
            "priority": "tic_priority",
        })
        df["tic_id"] = df["tic_id"].astype(int)
        logger.info(f"TIC query returned {len(df)} raw candidates.")
    except Exception as e:
        logger.error(f"TIC query failed: {e}. Falling back to cached list.")
        # Graceful fallback: return an empty dataframe with correct columns
        return pd.DataFrame(columns=["tic_id", "ra", "dec", "tmag", "teff",
                                     "radius", "n_sectors", "priority_score"])

    # ── Remove already-processed stars ───────────────────────────────────────
    before = len(df)
    df = df[~df["tic_id"].isin(already_processed)].copy()
    logger.info(f"Removed {before - len(df)} already-processed stars.")

    # ── Apply physical filters ────────────────────────────────────────────────
    if exclude_giants:
        # log g < 3.5 indicates a giant star (less suitable for planet searches)
        mask = df["logg"].isna() | (df["logg"] >= 3.5)
        df = df[mask].copy()

    if exclude_known_contaminated:
        # contratio > 0.1 means >10% of flux is from nearby contaminating stars
        mask = df["contratio"].isna() | (df["contratio"] <= 0.1)
        df = df[mask].copy()

    # White dwarfs are not good planet hosts
    if "wdflag" in df.columns:
        df = df[df["wdflag"] != 1].copy()

    logger.info(f"After physical filters: {len(df)} candidates remain.")

    # ── Compute sector counts via MAST (optional, improves prioritisation) ────
    df["n_sectors"] = 1   # default; updated below if possible
    if prioritise_multi_sector:
        try:
            from astroquery.mast import Observations
            # Sample up to 2000 stars for sector counting (API rate limit friendly)
            sample_ids = df["tic_id"].head(2000).tolist()
            obs = Observations.query_criteria(
                target_name = [f"TIC {t}" for t in sample_ids],
                obs_collection = "TESS",
                dataproduct_type = "timeseries",
            )
            if obs is not None and len(obs) > 0:
                obs_df = obs.to_pandas()
                obs_df["tic_id"] = obs_df["target_name"].str.extract(r"TIC (\d+)").astype(float)
                sector_counts = obs_df.groupby("tic_id").size().reset_index(name="n_sectors")
                sector_counts["tic_id"] = sector_counts["tic_id"].astype(int)
                df = df.merge(sector_counts, on="tic_id", how="left", suffixes=("", "_new"))
                if "n_sectors_new" in df.columns:
                    df["n_sectors"] = df["n_sectors_new"].fillna(1).astype(int)
                    df = df.drop(columns=["n_sectors_new"])
        except Exception as e:
            logger.warning(f"Sector count query failed (will use n_sectors=1): {e}")

    # ── Load TOI catalog for cross-check ─────────────────────────────────────
    toi_tic_ids = set()
    if prioritise_not_in_toi:
        try:
            from astroquery.mast import Catalogs as MastCat
            toi = MastCat.query_criteria(catalog="Exo.Mast", columns=["tid"])
            if toi is not None and len(toi) > 0:
                toi_tic_ids = set(toi["tid"].astype(int).tolist())
                logger.info(f"TOI catalog loaded: {len(toi_tic_ids)} known targets.")
        except Exception as e:
            logger.warning(f"TOI catalog query failed: {e}")

    df["in_toi"] = df["tic_id"].isin(toi_tic_ids)

    # ── Compute priority score ────────────────────────────────────────────────
    # Higher score = run first. Formula balances data quality, stellar suitability,
    # and discovery potential.
    df["priority_score"] = 0.0

    # Prefer bright stars (more photons = less noise)
    df["priority_score"] += (13.0 - df["tmag"].clip(8, 13)) / 5.0 * 0.3

    # Prefer sun-like temperatures (5000-6000 K is the sweet spot)
    df["priority_score"] += (
        1.0 - np.abs(df["teff"].fillna(5500) - 5500) / 1500.0
    ).clip(0, 1) * 0.3

    # Prefer stars with more TESS sectors (more transits = better statistics)
    df["priority_score"] += (df["n_sectors"].clip(1, 5) - 1) / 4.0 * 0.2

    # Strong bonus for stars NOT already in the TOI catalog
    df["priority_score"] += (~df["in_toi"]).astype(float) * 0.2

    # ── Sort and truncate to n_targets ────────────────────────────────────────
    df = df.sort_values("priority_score", ascending=False).head(n_targets)
    df = df.reset_index(drop=True)

    # Keep only the columns the notebook needs
    keep_cols = ["tic_id", "ra", "dec", "tmag", "teff", "radius",
                 "n_sectors", "in_toi", "priority_score"]
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"Final target list: {len(df)} stars (priority-sorted).")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DISCOVERY SESSION (MAIN LOOP)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_disk_used_gb(path: str) -> float:
    """Return GB used under path (not total disk — used by our output)."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1024**3


def _log_star(log_file: str, tic_id: int, verdict: str, detail: str):
    """Append one line to the plain-text discovery log."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] TIC {tic_id} -> {verdict} ({detail})\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _write_csv_row(csv_path: str, row: dict):
    """Append one row to results.csv, writing the header if the file is new."""
    fieldnames = [
        "tic_id", "session_label", "decision", "final_class", "confidence",
        "period", "period_err", "depth", "depth_err", "duration",
        "duration_err", "snr", "flag_reasons", "rp_earth", "is_new_discovery",
        "alias_rejected",
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _copy_to_verdict_folder(output_dir: str, tic_id: int, verdict: str,
                             plot_path: Optional[str], report_path: Optional[str]):
    """Copy plot and report into results/KEEP/, FLAG/, or DISCARD/."""
    dest = os.path.join(output_dir, verdict)
    os.makedirs(dest, exist_ok=True)
    for src in [plot_path, report_path]:
        if src and os.path.isfile(src):
            try:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
            except Exception:
                pass


def _toi_crosscheck(tic_id: int) -> bool:
    """
    WHAT: Check whether this TIC ID is in the TESS Object of Interest (TOI) catalog.
    WHY:  If it's NOT in the catalog, it's potentially a new discovery worth highlighting.
    Returns True if the star is NOT in the TOI catalog (= potentially new).
    """
    try:
        from astroquery.mast import Catalogs
        result = Catalogs.query_criteria(
            catalog="Exo.Mast",
            tid=int(tic_id),
            columns=["tid"],
        )
        return result is None or len(result) == 0
    except Exception:
        return False   # conservative: assume it is known if query fails


def _run_single_star(
    tic_id: int,
    output_dir: str,
    models_dir: str,
    session_label: str,
    run_flag_analyzer: bool,
    run_toi_crosscheck: bool,
    alias_rejection: bool,
    max_planet_radius_earth: float,
    min_transit_snr: float,
    min_depth_ppm: float,
    log_file: str,
) -> dict:
    """
    WHAT: Run the complete 13-step discovery pipeline for ONE star.
    WHY:  Encapsulated in a function so try/except in the main loop
          can catch any error and continue to the next star safely.

    Returns a result dict with keys matching results.csv columns.
    """
    result = {
        "tic_id": tic_id, "session_label": session_label,
        "decision": "DISCARD", "final_class": "No Signal",
        "confidence": 0.0, "period": 0.0, "period_err": 0.0,
        "depth": 0.0, "depth_err": 0.0, "duration": 0.0,
        "duration_err": 0.0, "snr": 0.0, "flag_reasons": "",
        "rp_earth": 0.0, "is_new_discovery": False, "alias_rejected": False,
    }
    fits_path = None

    try:
        import lightkurve as lk
        from src.preprocess import preprocess_lightcurve, fold_lightcurve
        from src.detect    import run_bls, compute_snr
        from src.features  import extract_features
        from src.classify  import classify_target
        from src.fit_transit import fit_batman_transit
        from src.reverse_pipeline import run_reverse_pipeline
        from src.decision_engine  import evaluate_decision, log_to_manual_review_queue

        # ── Step 1: Download light curve ─────────────────────────────────────
        raw_dir = os.path.join(output_dir, "raw_fits")
        os.makedirs(raw_dir, exist_ok=True)

        # Try 2-minute cadence first, fall back to 30-minute
        fits_path = None
        for cadence in ["short", "long"]:
            try:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}", mission="TESS", cadence=cadence, limit=1
                )
                if search is not None and len(search) > 0:
                    lc_col = search.download_all(download_dir=raw_dir)
                    if lc_col is not None and len(lc_col) > 0:
                        lc_obj = lc_col[0]
                        fits_path = "in_memory"   # we have the lc object
                        break
            except Exception:
                continue

        if fits_path is None:
            result["flag_reasons"] = "no_data: could not download from MAST"
            return result

        # ── Step 2: Preprocess ───────────────────────────────────────────────
        time_arr, flux_arr, flux_err = preprocess_lightcurve(lc_obj)
        quality_flag = "good" if len(time_arr) > 8000 else "poor"

        # ── Step 3: BLS period search + alias rejection ──────────────────────
        periods, power, bls_params = run_bls(time_arr, flux_arr, flux_err)

        if alias_rejection and bls_params.get("alias_rejected"):
            reason = (
                f"alias_discard: top period {bls_params['period']:.4f} d "
                f"matches TESS systematic alias"
            )
            result["flag_reasons"]    = reason
            result["period"]          = bls_params["period"]
            result["snr"]             = bls_params["snr"]
            result["alias_rejected"]  = True
            _log_star(log_file, tic_id, "alias_discard",
                      f"P={bls_params['period']:.3f}d")
            return result

        snr = bls_params.get("snr", 0.0)
        result["period"] = bls_params.get("period", 0.0)
        result["snr"]    = snr
        result["depth"]  = bls_params.get("depth", 0.0)

        # SNR gate
        if snr < min_transit_snr:
            result["flag_reasons"] = f"low_snr: {snr:.2f} < {min_transit_snr}"
            _log_star(log_file, tic_id, "DISCARD", f"SNR={snr:.1f}")
            return result

        # Depth gate (ppm)
        depth_ppm = abs(bls_params.get("depth", 0.0)) * 1e6
        if depth_ppm < min_depth_ppm:
            result["flag_reasons"] = f"shallow_transit: {depth_ppm:.0f} ppm < {min_depth_ppm} ppm"
            _log_star(log_file, tic_id, "DISCARD", f"depth={depth_ppm:.0f}ppm")
            return result

        # ── Step 4: Feature extraction ────────────────────────────────────────
        features = extract_features(time_arr, flux_arr, flux_err, bls_params)

        # ── Step 5: Forward pipeline (RF + CNN) ───────────────────────────────
        phase_arr, folded_flux = fold_lightcurve(
            time_arr, flux_arr, bls_params["period"], bls_params["t0"]
        )
        classification = classify_target(features, folded_flux)
        result["final_class"] = classification.get("label_name", "Unknown")
        result["confidence"]  = classification.get("confidence", 0.0)

        # ── Step 6: Batman fitting + radius plausibility ──────────────────────
        transit_params = fit_batman_transit(
            time_arr, flux_arr, flux_err, bls_params, n_bootstrap=15
        )
        rp_earth = transit_params.get("rp_earth", 0.0)
        result["rp_earth"]       = rp_earth
        result["period"]         = transit_params.get("period", result["period"])
        result["depth"]          = transit_params.get("transit_depth", result["depth"])
        result["duration"]       = transit_params.get("transit_duration_hr", 0.0) / 24.0

        # Radius plausibility check
        radius_flag = None
        radius_note = None
        if rp_earth < 0.5:
            radius_flag = "too_small_flag"
            radius_note = f"Fitted radius {rp_earth:.2f} R_earth below TESS sensitivity limit."
        elif rp_earth > 100.0:
            radius_flag = "almost_certainly_eb"
            radius_note = (f"Fitted radius {rp_earth:.1f} R_earth in stellar range (>100). "
                           f"Almost certainly an eclipsing binary.")
        elif rp_earth > max_planet_radius_earth:
            radius_flag = "giant_radius_eb_suspect"
            radius_note = (f"Fitted radius {rp_earth:.1f} R_earth exceeds {max_planet_radius_earth} R_earth. "
                           f"Likely eclipsing binary or blended EB. Recommend EB catalog cross-check.")

        # ── Step 7: Reverse pipeline ──────────────────────────────────────────
        reverse_results = run_reverse_pipeline(
            time_arr, flux_arr, flux_err, bls_params
        )

        # ── Step 8: Cross-check + three-tier decision ─────────────────────────
        decision_res = evaluate_decision(
            classification, reverse_results, bls_params, quality_flag
        )

        # Apply radius overrides AFTER decision (radius evidence always wins)
        if radius_flag in ("giant_radius_eb_suspect", "almost_certainly_eb"):
            classification["label_name"] = "Eclipsing Binary"
            decision_res["decision"]            = "DISCARD"
            decision_res["combined_confidence"] = 0.0
            decision_res["flag_reasons"] = f"[{radius_flag}] {radius_note}"
        elif radius_flag == "too_small_flag":
            existing = decision_res.get("flag_reasons", "")
            decision_res["flag_reasons"] = f"[too_small_flag] {radius_note}; {existing}"

        verdict    = decision_res["decision"]
        confidence = decision_res["combined_confidence"]
        flag_reasons = decision_res.get("flag_reasons", "")

        result["decision"]     = verdict
        result["confidence"]   = confidence
        result["final_class"]  = classification.get("label_name", result["final_class"])
        result["flag_reasons"] = flag_reasons

        # ── Step 9 (optional): Flag deep-analysis ─────────────────────────────
        if verdict == "FLAG" and run_flag_analyzer:
            try:
                from flag_analyzer import run_flag_analysis
                fa_result = run_flag_analysis(tic_ids=[tic_id])
                # If upgraded/downgraded, update verdict
                if fa_result.get("upgraded", 0) > 0:
                    verdict = "KEEP"
                elif fa_result.get("downgraded", 0) > 0:
                    verdict = "DISCARD"
                result["decision"] = verdict
            except Exception as e:
                logger.warning(f"TIC {tic_id}: flag_analyzer failed ({e}) — keeping FLAG")

            # Log to manual review queue for remaining FLAGs
            if verdict == "FLAG":
                try:
                    log_to_manual_review_queue(
                        tic_id, decision_res,
                        reverse_results.get("fit_results", {}),
                        output_dir
                    )
                except Exception:
                    pass

        # ── Step 10 (optional): TOI cross-check ──────────────────────────────
        is_new = False
        if run_toi_crosscheck and verdict == "KEEP":
            try:
                is_new = _toi_crosscheck(tic_id)
                result["is_new_discovery"] = is_new
            except Exception:
                pass

        if is_new:
            new_file = os.path.join(output_dir, "new_discoveries.txt")
            try:
                with open(new_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"TIC {tic_id} | P={result['period']:.4f}d | "
                        f"depth={result['depth']*100:.4f}% | "
                        f"Rp={rp_earth:.2f}R_earth\n"
                    )
            except Exception:
                pass

        # ── Step 11: Save plots for KEEP and FLAG ─────────────────────────────
        plot_path = None
        if verdict in ("KEEP", "FLAG"):
            try:
                from src.visualize import plot_diagnostic
                import matplotlib.pyplot as plt
                fig = plot_diagnostic(
                    tic_id=tic_id,
                    time_raw=lc_obj.time.value,
                    flux_raw=lc_obj.flux.value,
                    time_clean=time_arr,
                    flux_clean=flux_arr,
                    bls_periods=periods,
                    bls_power=power,
                    bls_params=bls_params,
                    transit_params=transit_params,
                    classification=classification,
                    phase_folded_flux=folded_flux,
                    model_flux=None,
                    save=True,
                    save_dir=os.path.join(output_dir, "figures"),
                )
                plt.close(fig)
                plot_path = os.path.join(
                    output_dir, "figures", f"TIC_{tic_id}_diagnostic.png"
                )
            except Exception as e:
                logger.warning(f"TIC {tic_id}: plot failed ({e})")

        # Save report JSON
        report_path = None
        try:
            report = {
                "tic_id": tic_id, "session_label": session_label,
                "decision": verdict, "classification": classification,
                "transit_parameters": transit_params,
                "bls_parameters": bls_params,
                "reverse_results": reverse_results,
            }
            report_dir = os.path.join(output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"TIC_{tic_id}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.warning(f"TIC {tic_id}: report save failed ({e})")

        # Copy to verdict folder
        try:
            _copy_to_verdict_folder(output_dir, tic_id, verdict, plot_path, report_path)
        except Exception:
            pass

        # Log one line to discovery_log.txt
        detail = (f"{result['final_class']}, {confidence*100:.0f}%, "
                  f"P={result['period']:.3f}d")
        _log_star(log_file, tic_id, verdict, detail)

    except Exception as e:
        logger.error(f"TIC {tic_id}: pipeline crashed — {e}")
        result["flag_reasons"] = f"pipeline_error: {e}"
        _log_star(log_file, tic_id, "ERROR", str(e)[:60])

    finally:
        # Always delete raw FITS files to conserve disk space
        try:
            if fits_path and fits_path != "in_memory":
                if os.path.isfile(fits_path):
                    os.remove(fits_path)
        except Exception:
            pass

        # Also purge raw_fits directory if it is getting large
        try:
            raw_dir = os.path.join(output_dir, "raw_fits")
            if os.path.isdir(raw_dir):
                for f in os.listdir(raw_dir):
                    fp = os.path.join(raw_dir, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
        except Exception:
            pass

    return result


def run_discovery_session(
    targets: pd.DataFrame,
    output_dir: str,
    models_dir: str,
    time_limit_hours: float = 8.5,
    disk_limit_gb: float = 18.0,
    save_every_n: int = 50,
    session_label: str = "",
    run_flag_analyzer: bool = True,
    run_candidate_export: bool = True,
    run_toi_crosscheck: bool = True,
    alias_rejection: bool = True,
    max_planet_radius_earth: float = 25.0,
    min_transit_snr: float = 5.0,
    min_depth_ppm: float = 100.0,
    log_file: str = "",
) -> dict:
    """
    WHAT: Main discovery loop. Runs the 13-step pipeline on every star
          in the target DataFrame until the time or disk limit is hit.
    WHY:  Kaggle sessions time out after ~9 hours. This function respects
          that limit, saves progress incrementally, and never loses results
          from stars already processed.

    Returns a summary dict with counts for the session.
    """
    if not session_label:
        session_label = datetime.now().strftime("%Y-%m-%d")
    if not log_file:
        log_file = os.path.join(output_dir, "discovery_log.txt")

    results_csv = os.path.join(output_dir, "results.csv")
    start_time  = time.time()
    time_limit_sec = time_limit_hours * 3600

    os.makedirs(output_dir, exist_ok=True)

    n_attempted  = 0
    n_processed  = 0
    n_skipped    = 0
    n_alias      = 0
    n_keep       = 0
    n_flag       = 0
    n_discard    = 0
    n_new        = 0
    pending_rows = []   # buffer for incremental CSV saves

    target_list = targets["tic_id"].tolist()
    keep_tic_ids = []   # for smart re-prioritisation

    logger.info(f"Starting discovery session: {len(target_list)} stars, "
                f"limit {time_limit_hours} hrs / {disk_limit_gb} GB")

    pbar = tqdm(total=len(target_list), desc="Discovery", unit="star")

    for i, tic_id in enumerate(target_list):
        tic_id = int(tic_id)
        n_attempted += 1

        # ── Time check (every star) ───────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            logger.info(f"Time limit reached ({elapsed/3600:.1f} hrs). Stopping.")
            pbar.write(f"Time limit reached after {n_processed} stars.")
            break

        # ── Disk check (every 100 stars) ──────────────────────────────────────
        if i % 100 == 0 and i > 0:
            disk_used = _get_disk_used_gb(output_dir)
            if disk_used > disk_limit_gb:
                logger.warning(f"Disk limit reached ({disk_used:.1f} GB). "
                               f"Finishing current queue.")
                pbar.write(f"Disk limit ({disk_limit_gb} GB) reached. Stopping downloads.")
                break

        # ── Smart re-prioritisation (every 500 stars) ─────────────────────────
        if i > 0 and i % 500 == 0 and keep_tic_ids:
            # Move high-SNR stars in the same sky region to the front
            try:
                remaining = target_list[i+1:]
                keep_rows = targets[targets["tic_id"].isin(keep_tic_ids)]
                if len(keep_rows) > 0:
                    mean_ra  = keep_rows["ra"].mean() if "ra"  in keep_rows.columns else None
                    mean_dec = keep_rows["dec"].mean() if "dec" in keep_rows.columns else None
                    if mean_ra is not None and mean_dec is not None:
                        rem_df = targets[targets["tic_id"].isin(remaining)].copy()
                        rem_df["sky_dist"] = np.sqrt(
                            (rem_df.get("ra",  mean_ra)  - mean_ra)**2 +
                            (rem_df.get("dec", mean_dec) - mean_dec)**2
                        )
                        rem_df = rem_df.sort_values("sky_dist")
                        target_list[i+1:] = rem_df["tic_id"].tolist()
                        pbar.write(f"Re-prioritised remaining targets around {mean_ra:.2f}, {mean_dec:.2f}")
            except Exception:
                pass   # re-prioritisation is optional

        # ── Run the pipeline for this star ────────────────────────────────────
        pbar.set_postfix({"tic": tic_id})
        try:
            row = _run_single_star(
                tic_id               = tic_id,
                output_dir           = output_dir,
                models_dir           = models_dir,
                session_label        = session_label,
                run_flag_analyzer    = run_flag_analyzer,
                run_toi_crosscheck   = run_toi_crosscheck,
                alias_rejection      = alias_rejection,
                max_planet_radius_earth = max_planet_radius_earth,
                min_transit_snr      = min_transit_snr,
                min_depth_ppm        = min_depth_ppm,
                log_file             = log_file,
            )
        except Exception as e:
            logger.error(f"Unexpected error for TIC {tic_id}: {e}")
            n_skipped += 1
            pbar.update(1)
            continue

        n_processed += 1
        verdict = row.get("decision", "DISCARD")

        if row.get("alias_rejected"):
            n_alias += 1
        if "no_data" in row.get("flag_reasons", ""):
            n_skipped += 1
        if verdict == "KEEP":
            n_keep += 1
            keep_tic_ids.append(tic_id)
        elif verdict == "FLAG":
            n_flag += 1
        else:
            n_discard += 1

        if row.get("is_new_discovery"):
            n_new += 1

        pending_rows.append(row)

        # ── Incremental save every save_every_n stars ─────────────────────────
        if len(pending_rows) >= save_every_n:
            try:
                for r in pending_rows:
                    _write_csv_row(results_csv, r)
                pending_rows = []
                logger.info(f"Checkpoint: saved {n_processed} stars so far.")
            except Exception as e:
                logger.warning(f"Incremental save failed: {e}")

        pbar.update(1)

    pbar.close()

    # ── Final save for any remaining rows ─────────────────────────────────────
    try:
        for r in pending_rows:
            _write_csv_row(results_csv, r)
        pending_rows = []
    except Exception as e:
        logger.error(f"Final save failed: {e}")

    # ── Export candidate submission CSV ───────────────────────────────────────
    if run_candidate_export:
        try:
            if os.path.isfile(results_csv):
                df = pd.read_csv(results_csv)
                candidates = df[df["decision"] == "KEEP"].copy()
                candidates.to_csv(
                    os.path.join(output_dir, "candidates_submission.csv"),
                    index=False
                )
                logger.info(f"Candidates CSV: {len(candidates)} KEEP targets exported.")
        except Exception as e:
            logger.warning(f"Candidate export failed: {e}")

    elapsed_hrs = (time.time() - start_time) / 3600

    summary = {
        "stars_attempted":  n_attempted,
        "stars_processed":  n_processed,
        "stars_skipped":    n_skipped,
        "alias_discards":   n_alias,
        "elapsed_hours":    round(elapsed_hrs, 2),
        "keep":             n_keep,
        "flag":             n_flag,
        "discard":          n_discard,
        "new_discoveries":  n_new,
    }
    logger.info(f"Session complete: {summary}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SESSION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_session_summary(
    results_dir: str,
    session_label: str,
    session_data: Optional[dict] = None,
) -> dict:
    """
    WHAT: Read results.csv and compute cumulative statistics for this run.
    WHY:  Gives the user an at-a-glance report of what was found, how
          many stars were processed all-time, and whether any new
          discoveries are waiting for follow-up.

    Args:
        results_dir:   Path to the results/ directory.
        session_label: YYYY-MM-DD string for this session.
        session_data:  Dict returned by run_discovery_session() (optional).

    Returns a summary dict.
    """
    if session_data is None:
        session_data = {}

    results_csv = os.path.join(results_dir, "results.csv")

    # ── Load cumulative results ───────────────────────────────────────────────
    df_all = None
    try:
        if os.path.isfile(results_csv):
            df_all = pd.read_csv(results_csv)
    except Exception as e:
        logger.warning(f"Could not load results.csv: {e}")

    n_cumulative = len(df_all) if df_all is not None else 0

    # ── Counts from this session ──────────────────────────────────────────────
    n_keep    = session_data.get("keep", 0)
    n_flag    = session_data.get("flag", 0)
    n_discard = session_data.get("discard", 0)
    n_new     = session_data.get("new_discoveries", 0)

    # Refine from CSV if this session's rows are tagged
    if df_all is not None and "session_label" in df_all.columns:
        this_session = df_all[df_all["session_label"] == session_label]
        if len(this_session) > 0:
            from collections import Counter
            counts = Counter(this_session["decision"].fillna("DISCARD"))
            n_keep    = counts.get("KEEP", n_keep)
            n_flag    = counts.get("FLAG", n_flag)
            n_discard = counts.get("DISCARD", n_discard)
            if "is_new_discovery" in this_session.columns:
                n_new = int(this_session["is_new_discovery"].sum())

    summary = {
        "stars_attempted":  session_data.get("stars_attempted", 0),
        "stars_processed":  session_data.get("stars_processed", 0),
        "stars_skipped":    session_data.get("stars_skipped", 0),
        "alias_discards":   session_data.get("alias_discards", 0),
        "elapsed_hours":    session_data.get("elapsed_hours", 0.0),
        "keep":             n_keep,
        "flag":             n_flag,
        "discard":          n_discard,
        "new_discoveries":  n_new,
        "cumulative_total": n_cumulative,
    }
    return summary
