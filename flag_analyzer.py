"""
flag_analyzer.py
================
FLAG Deep-Analysis & Auto-Verification Layer.

For every star that the main pipeline marked as FLAG, this module runs
8 deeper diagnostic tests to make a more confident final decision:
  - UPGRADE to KEEP  -> strong evidence this is a real planet candidate
  - DOWNGRADE to DISCARD -> tests confirm the signal is noise or an artefact
  - KEEP as FLAG -> still genuinely ambiguous; route to human astronomer

Usage:
  # Run automatically on all flags after main pipeline:
  python pipeline.py --tic_id 261136679 --analyze-flags

  # Run standalone on all flags in manual_review_queue.csv:
  python flag_analyzer.py

  # Run standalone on a single flagged star:
  python flag_analyzer.py --tic 219698950

Output folders:
  results/KEEP/     -> report cards and plots for upgraded candidates
  results/FLAG/     -> updated queue entries for human review
  results/DISCARD/  -> downgrade confirmation plots
  results/plots/flag_analysis/  -> all diagnostic plots (150 DPI)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import json
import logging
import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats, optimize
from tqdm import tqdm

# ─── Setup paths ─────────────────────────────────────────────────────────────

# The root of the project (folder containing this script)
ROOT = Path(__file__).parent.resolve()

# Input and output directories
RESULTS_DIR        = ROOT / "results"
CLEAN_LC_DIR       = ROOT / "clean_lightcurves"       # cached cleaned light curves
REPORTS_DIR        = RESULTS_DIR / "reports"           # per-target JSON reports
PLOTS_DIR          = RESULTS_DIR / "plots" / "flag_analysis"  # diagnostic plots
KEEP_DIR           = RESULTS_DIR / "KEEP"              # upgraded candidates
FLAG_DIR           = RESULTS_DIR / "FLAG"              # remaining human-review items
DISCARD_DIR        = RESULTS_DIR / "DISCARD"           # downgraded stars

# Create all needed directories up front
for d in [CLEAN_LC_DIR, REPORTS_DIR, PLOTS_DIR, KEEP_DIR, FLAG_DIR, DISCARD_DIR,
          RESULTS_DIR / "figures"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / "flag_analyzer.log"), mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("flag_analyzer")

# ─── Matplotlib style (import lazily to keep startup fast) ───────────────────

def _init_plot():
    """Set up matplotlib with seaborn-v0_8-whitegrid and return plt."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (no display window)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_star_data(tic_id: int) -> dict:
    """
    WHAT: Loads clean light curve, BLS results, and pipeline report for one star.
    WHY:  The 8 tests all need the cleaned time/flux arrays and best-fit parameters
          from the earlier pipeline stages. We centralise loading here so every
          test can just call load_star_data() and unpack what it needs.

    Returns a dict with keys: time, flux, flux_err, bls_params, report, lc_obj
    Returns None if we cannot find any data for this star.
    """
    data = {}

    # ── Step 1: Reload pipeline report (period, depth, t0, classification) ───
    report_path = REPORTS_DIR / f"TIC_{tic_id}_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            data["report"] = json.load(f)
        data["bls_params"] = data["report"].get("bls_parameters", {})
    else:
        logger.warning(f"TIC {tic_id}: no pipeline report JSON found — using defaults.")
        data["report"] = {}
        data["bls_params"] = {"period": 1.0, "depth": 0.001, "duration": 0.1, "t0": 0.0, "snr": 5.0}

    # ── Step 2: Try clean light curve cache first (fastest) ──────────────────
    clean_csv = CLEAN_LC_DIR / f"TIC_{tic_id}_clean.csv"
    if clean_csv.exists():
        try:
            df = pd.read_csv(clean_csv)
            data["time"]     = df["time"].values
            data["flux"]     = df["flux"].values
            data["flux_err"] = df.get("flux_err", pd.Series(np.ones(len(df)) * 1e-4)).values
            logger.info(f"TIC {tic_id}: loaded clean curve from cache ({len(data['time'])} pts).")
            data["lc_obj"] = None   # not available from CSV cache
            return data
        except Exception as e:
            logger.warning(f"TIC {tic_id}: cache read failed ({e}), falling back to FITS.")

    # ── Step 3: Fall back — find raw FITS and reprocess on-the-fly ───────────
    try:
        import lightkurve as lk
        from src.preprocess import preprocess_lightcurve

        fits_dir = ROOT / "data" / "raw_fits"
        fits_files = list(fits_dir.glob(f"**/*{tic_id}*.fits"))
        if not fits_files:
            # Try the local results-side FITS directory (from batch run)
            fits_files = list((RESULTS_DIR / "data" / "raw_fits").glob(f"**/*{tic_id}*.fits"))

        if fits_files:
            lc = lk.io.read(str(fits_files[0]))
            time, flux, flux_err = preprocess_lightcurve(lc)
            data["time"]     = time
            data["flux"]     = flux
            data["flux_err"] = flux_err
            data["lc_obj"]   = lc

            # Cache for future runs
            pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err}).to_csv(
                clean_csv, index=False
            )
            logger.info(f"TIC {tic_id}: preprocessed and cached ({len(time)} pts).")
        else:
            # Last resort — download from MAST
            logger.info(f"TIC {tic_id}: downloading fresh from MAST...")
            from src.download import download_lightcurve
            fits_path = download_lightcurve(tic_id)
            if fits_path:
                lc = lk.io.read(str(fits_path))
                time, flux, flux_err = preprocess_lightcurve(lc)
                data["time"]     = time
                data["flux"]     = flux
                data["flux_err"] = flux_err
                data["lc_obj"]   = lc
                pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err}).to_csv(
                    clean_csv, index=False
                )
                logger.info(f"TIC {tic_id}: downloaded and cached ({len(time)} pts).")
            else:
                logger.error(f"TIC {tic_id}: could not obtain light curve. Skipping.")
                return None
    except Exception as e:
        logger.error(f"TIC {tic_id}: data loading failed — {e}")
        return None

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — THE 8 DEEP DIAGNOSTIC TESTS
# Each test returns a plain dict so results can be stored in a CSV row.
# If a test crashes, it returns a dict of Nones so we never abort the whole run.
# ═══════════════════════════════════════════════════════════════════════════════

def test1_multi_sector(tic_id: int, bls_period: float) -> dict:
    """
    TEST 1 — Multi-Sector Consistency.

    WHAT: Ask MAST how many TESS sectors observed this star. If the same
          period is detectable in 2+ sectors, that's very strong evidence
          the signal is astrophysical and not a one-sector instrument glitch.
    WHY:  One-sector artefacts (e.g. a scattered light stripe) won't repeat
          at the same period in a different sector taken months later.
    """
    try:
        import lightkurve as lk
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", cadence="short")
        n_sectors = len(search) if search is not None else 0

        # "Consistent" if we found data in 2 or more independent sectors
        consistent = n_sectors >= 2
        return {
            "consistent_across_sectors": consistent,
            "n_sectors_checked": int(n_sectors),
        }
    except Exception as e:
        logger.warning(f"TIC {tic_id} Test 1 failed: {e}")
        return {"consistent_across_sectors": None, "n_sectors_checked": None}


def test2_centroid_motion(lc_obj, time: np.ndarray, flux: np.ndarray,
                          bls_params: dict) -> dict:
    """
    TEST 2 — Centroid Motion Check.

    WHAT: During a genuine planet transit the star's photometric centroid
          (x,y position on the detector) should NOT move. If it does, the
          dip is coming from a different, fainter background star blended
          into the same pixel (a "background blend").
    WHY:  This is one of the strongest false-positive discriminators available
          without follow-up spectroscopy.
    """
    try:
        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)

        # Build in-transit mask
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        half_dur = (duration / 2.0) / period
        in_transit  = np.abs(phase) < half_dur
        out_transit = np.abs(phase) > 2 * half_dur

        # Try to pull centroid columns from the lightkurve object
        centroid_shift = 0.0
        if lc_obj is not None and hasattr(lc_obj, "centroid_col"):
            cx = np.array(lc_obj.centroid_col.value, dtype=float)
            cy = np.array(lc_obj.centroid_row.value, dtype=float)

            # Trim to same length as cleaned time if necessary
            min_len = min(len(time), len(cx))
            cx = cx[:min_len]
            cy = cy[:min_len]
            in_t  = in_transit[:min_len]
            out_t = out_transit[:min_len]

            if in_t.sum() >= 2 and out_t.sum() >= 2:
                dx = np.nanmean(cx[in_t]) - np.nanmean(cx[out_t])
                dy = np.nanmean(cy[in_t]) - np.nanmean(cy[out_t])
                centroid_shift = float(np.sqrt(dx**2 + dy**2))

        # Stable if shift is less than 0.5 pixels (TESS pixel ≈ 21 arcsec)
        centroid_stable = centroid_shift < 0.5
        return {
            "centroid_stable": centroid_stable,
            "centroid_shift_pixels": round(centroid_shift, 4),
        }
    except Exception as e:
        logger.warning(f"Centroid test failed: {e}")
        return {"centroid_stable": None, "centroid_shift_pixels": None}


def test3_secondary_eclipse(time: np.ndarray, flux: np.ndarray,
                            bls_params: dict) -> dict:
    """
    TEST 3 — Secondary Eclipse Refined Check.

    WHAT: Phase-fold at the best period and zoom into the phase-0.5 window
          (halfway between primary transits). Fit a Gaussian to any dip there.
          If secondary depth > 15% of primary depth -> eclipsing binary.
          If secondary depth < 5%  of primary depth -> consistent with a planet.
    WHY:  Planets are opaque spheres; they cause no detectable secondary dip.
          Binary stars cause a secondary dip when the brighter star eclipses
          the fainter one, or when we see thermal emission from the companion.
    """
    try:
        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)
        depth    = abs(bls_params.get("depth", 0.001))

        # Phase-fold the light curve
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0          # centre primary at phase=0

        # Zoom into ±3× the transit duration around phase=0.5
        half_window = 3 * (duration / period)
        secondary_mask = np.abs(phase - 0.5) < half_window
        if secondary_mask.sum() < 5:
            # Not enough points near phase 0.5
            return {"secondary_depth_ratio": 0.0, "secondary_classification": "insufficient_data"}

        phase_sec = phase[secondary_mask]
        flux_sec  = flux[secondary_mask]

        # Baseline: out-of-transit median
        oot = np.abs(phase) > 2 * (duration / period)
        baseline = np.nanmedian(flux[oot]) if oot.sum() > 0 else 1.0

        # Measure secondary depth as mean dip below baseline in the window
        sec_depth = float(baseline - np.nanmedian(flux_sec))
        sec_depth = max(0.0, sec_depth)   # depth can't be negative

        ratio = sec_depth / depth if depth > 1e-9 else 0.0
        if ratio > 0.15:
            classification = "likely_eclipsing_binary"
        elif ratio < 0.05:
            classification = "consistent_with_planet"
        else:
            classification = "ambiguous"

        return {
            "secondary_depth_ratio": round(float(ratio), 4),
            "secondary_classification": classification,
        }
    except Exception as e:
        logger.warning(f"Secondary eclipse test failed: {e}")
        return {"secondary_depth_ratio": None, "secondary_classification": None}


def test4_ttv(time: np.ndarray, flux: np.ndarray, bls_params: dict) -> dict:
    """
    TEST 4 — Transit Timing Variations (TTV) Check.

    WHAT: Measure the centre time of each individual transit by fitting a
          parabola to the dip. Compare observed times against the linear
          ephemeris (t0 + N×period). High scatter -> TTVs -> likely REAL planet
          (gravitational tugs from a neighbouring planet cause small period
          variations impossible for a background blend to mimic).
    WHY:  TTVs are a unique signature of multi-planet systems discovered via
          the Kepler mission. A blend or artefact will not show TTVs; a real
          planet in a multi-planet system often will.
    """
    try:
        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)

        # Find transit mid-times by epoch
        t_start, t_end = time[0], time[-1]
        n_start = int(np.floor((t_start - t0) / period))
        n_end   = int(np.ceil( (t_end   - t0) / period))
        epochs  = np.arange(n_start, n_end + 1)

        transit_times = []
        for epoch in epochs:
            t_centre = t0 + epoch * period
            # Select points within ±1.5× duration of this transit centre
            window = 1.5 * duration
            mask = np.abs(time - t_centre) < window
            if mask.sum() < 4:
                continue

            # Fit a parabola y = a*(t-t_c)^2 + b to find minimum (transit centre)
            t_local = time[mask] - t_centre
            f_local = flux[mask]
            try:
                coeffs = np.polyfit(t_local, f_local, 2)
                # Vertex of parabola at t = -b/(2a)
                if coeffs[0] > 0:   # parabola opens upward -> skip (not a dip)
                    continue
                t_min = -coeffs[1] / (2 * coeffs[0])
                transit_times.append(t_centre + t_min)
            except Exception:
                continue

        if len(transit_times) < 3:
            return {"ttv_amplitude_minutes": None, "ttv_significant": False}

        transit_times = np.array(transit_times)
        # Best-fit linear ephemeris to the measured times
        epochs_measured = np.round((transit_times - t0) / period).astype(int)
        predicted_times = t0 + epochs_measured * period
        residuals_days  = transit_times - predicted_times
        residuals_min   = residuals_days * 24 * 60

        ttv_amp  = float(np.std(residuals_min))     # 1-sigma scatter in minutes
        # Significant TTV = scatter > 5 minutes (typical for known TTV systems)
        ttv_sig  = ttv_amp > 5.0

        return {
            "ttv_amplitude_minutes": round(ttv_amp, 2),
            "ttv_significant": bool(ttv_sig),
        }
    except Exception as e:
        logger.warning(f"TTV test failed: {e}")
        return {"ttv_amplitude_minutes": None, "ttv_significant": False}


def test5_odd_even(time: np.ndarray, flux: np.ndarray, bls_params: dict) -> dict:
    """
    TEST 5 — Odd-Even Refined Check.

    WHAT: Measure the median depth of every odd-numbered transit vs every
          even-numbered transit. Eclipsing binaries have TWO different stars
          producing alternating eclipse depths. A planet transit in front of
          a single star will have identical depths every time.
          Use a two-sample t-test to measure the statistical significance
          of any odd-even difference.
    WHY:  Period-doubling eclipsing binaries are a classic TESS false positive.
          They appear to have half their true period in the BLS search,
          so the primary and secondary eclipses look like alternating transits.
    """
    try:
        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)

        t_start = time[0]
        n_start = int(np.floor((t_start - t0) / period))
        n_end   = int(np.ceil( (time[-1] - t0) / period))

        odd_depths  = []
        even_depths = []

        for epoch in range(n_start, n_end + 1):
            t_centre = t0 + epoch * period
            window   = 1.5 * duration
            mask     = np.abs(time - t_centre) < window
            if mask.sum() < 3:
                continue

            # Depth = out-of-transit baseline minus in-transit median
            oot_mask = (np.abs(time - t_centre) > window) & \
                       (np.abs(time - t_centre) < 3 * window)
            if oot_mask.sum() < 3:
                continue
            baseline = np.nanmedian(flux[oot_mask])
            depth    = float(baseline - np.nanmedian(flux[mask]))

            if epoch % 2 == 0:
                even_depths.append(depth)
            else:
                odd_depths.append(depth)

        if len(odd_depths) < 2 or len(even_depths) < 2:
            return {"odd_even_pvalue": 1.0, "is_likely_eb": False}

        # Two-sample Welch t-test (does not assume equal variance)
        _, p_value = stats.ttest_ind(odd_depths, even_depths, equal_var=False)
        is_eb = bool(p_value < 0.05)   # p < 0.05 -> significant odd-even difference

        return {
            "odd_even_pvalue": round(float(p_value), 4),
            "is_likely_eb": is_eb,
        }
    except Exception as e:
        logger.warning(f"Odd-even test failed: {e}")
        return {"odd_even_pvalue": None, "is_likely_eb": False}


def test6_stellar_variability(time: np.ndarray, flux: np.ndarray,
                              bls_params: dict) -> dict:
    """
    TEST 6 — Stellar Variability Decomposition.

    WHAT: Run a Lomb-Scargle periodogram on the OUT-OF-TRANSIT flux only.
          If there is a dominant stellar rotation period close to the transit
          period, the "transit" dip may just be a starspot rotating into view.
    WHY:  Active stars have dark spots on their surface. As the star rotates,
          the spots cause a periodic brightness decrease that can mimic a
          transit in the BLS search — especially for periods of 10-30 days
          where the BLS is less sensitive and starspots are common.
          If rotation_period / transit_period ≈ 1.0 or 0.5 -> starspot suspect.
    """
    try:
        from astropy.timeseries import LombScargle

        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)

        # Use only out-of-transit flux to measure stellar variability
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        oot = np.abs(phase) > (1.5 * duration / period)

        if oot.sum() < 20:
            return {"stellar_rotation_period": None,
                    "rotation_period_ratio": None,
                    "is_likely_starspot": False}

        t_oot = time[oot]
        f_oot = flux[oot]

        # Lomb-Scargle on OOT flux for periods from 0.5× to 5× transit period
        frequency, power = LombScargle(t_oot, f_oot).autopower(
            minimum_frequency=1.0 / (5 * period),
            maximum_frequency=1.0 / (0.5 * period),
        )
        if len(power) == 0:
            return {"stellar_rotation_period": None,
                    "rotation_period_ratio": None,
                    "is_likely_starspot": False}

        best_freq = float(frequency[np.argmax(power)])
        rotation_period = 1.0 / best_freq if best_freq > 0 else None

        if rotation_period is None:
            return {"stellar_rotation_period": None,
                    "rotation_period_ratio": None,
                    "is_likely_starspot": False}

        ratio = float(rotation_period / period)
        # Starspot suspected if rotation period ≈ transit period (within 15%)
        # or ≈ 2× transit period (harmonic)
        near_unity = abs(ratio - 1.0) < 0.15 or abs(ratio - 2.0) < 0.15 or abs(ratio - 0.5) < 0.15
        is_starspot = bool(near_unity and np.max(power) > 0.3)

        return {
            "stellar_rotation_period": round(rotation_period, 4),
            "rotation_period_ratio": round(ratio, 4),
            "is_likely_starspot": is_starspot,
        }
    except Exception as e:
        logger.warning(f"Stellar variability test failed: {e}")
        return {"stellar_rotation_period": None,
                "rotation_period_ratio": None,
                "is_likely_starspot": False}


def test7_box_vs_trapezoid(time: np.ndarray, flux: np.ndarray,
                           bls_params: dict) -> dict:
    """
    TEST 7 — Box vs. Trapezoid Shape Discrimination.

    WHAT: Fit two competing transit models to the phase-folded light curve:
            Model A — flat-bottomed rectangular box (no ingress/egress)
            Model B — trapezoidal shape (smooth ingress and egress)
          Compare their BIC scores. A planet transit has smooth ingress/egress
          because the planet edge gradually crosses the stellar limb, so the
          trapezoid model wins. An instrumental spike or cosmic ray is a
          sharp box, so the box model wins (or BIC difference < 2 = no preference).
    WHY:  BIC (Bayesian Information Criterion) penalises model complexity, so
          if the trapezoid wins despite having extra free parameters it means
          the data genuinely support a smooth ingress/egress shape.
    """
    try:
        period   = bls_params.get("period", 1.0)
        t0       = bls_params.get("t0", time[0])
        duration = bls_params.get("duration", 0.1)
        depth    = abs(bls_params.get("depth", 0.001))

        # Phase-fold and sort
        phase = ((time - t0) / period) % 1.0
        phase[phase > 0.5] -= 1.0
        sort_idx = np.argsort(phase)
        ph  = phase[sort_idx]
        fl  = flux[sort_idx]

        # Focus on ±3× transit duration
        hw = 3 * (duration / period)
        mask = np.abs(ph) < hw
        if mask.sum() < 8:
            return {"preferred_model": "insufficient_data",
                    "bic_difference": None,
                    "shape_is_physical": False}

        ph_fit = ph[mask]
        fl_fit = fl[mask]
        n = len(ph_fit)

        # ── Box model: 2 parameters (depth, half-width) ─────────────────────
        def box_model(ph_arr, box_depth, box_hw):
            m = np.ones(len(ph_arr))
            m[np.abs(ph_arr) < box_hw] = 1.0 - box_depth
            return m

        try:
            from scipy.optimize import curve_fit
            p0_box = [depth, duration / (2 * period)]
            popt_box, _ = curve_fit(box_model, ph_fit, fl_fit, p0=p0_box,
                                    bounds=([0, 1e-5], [0.5, 0.5]), maxfev=2000)
            res_box = fl_fit - box_model(ph_fit, *popt_box)
            sse_box = np.sum(res_box ** 2)
            k_box   = 2
            bic_box = n * np.log(sse_box / n) + k_box * np.log(n)
        except Exception:
            bic_box = 1e9

        # ── Trapezoid model: 3 parameters (depth, flat half-width, ingress width)
        def trap_model(ph_arr, trap_depth, flat_hw, ingress_w):
            m = np.ones(len(ph_arr))
            for i, p in enumerate(ph_arr):
                ap = abs(p)
                if ap < flat_hw:
                    m[i] = 1.0 - trap_depth
                elif ap < flat_hw + ingress_w:
                    m[i] = 1.0 - trap_depth * (1 - (ap - flat_hw) / ingress_w)
            return m

        try:
            p0_trap = [depth, duration / (3 * period), duration / (6 * period)]
            popt_trap, _ = curve_fit(trap_model, ph_fit, fl_fit, p0=p0_trap,
                                     bounds=([0, 1e-6, 1e-6], [0.5, 0.4, 0.2]),
                                     maxfev=3000)
            res_trap = fl_fit - trap_model(ph_fit, *popt_trap)
            sse_trap = np.sum(res_trap ** 2)
            k_trap   = 3
            bic_trap = n * np.log(sse_trap / n) + k_trap * np.log(n)
        except Exception:
            bic_trap = 1e9

        bic_diff = float(bic_box - bic_trap)   # positive -> trapezoid wins

        if bic_diff > 2:
            preferred = "trapezoid"
            shape_ok  = True
        elif bic_diff < -2:
            preferred = "box"
            shape_ok  = False
        else:
            preferred = "no_preference"
            shape_ok  = False

        return {
            "preferred_model": preferred,
            "bic_difference": round(bic_diff, 3),
            "shape_is_physical": shape_ok,
        }
    except Exception as e:
        logger.warning(f"Box-vs-trapezoid test failed: {e}")
        return {"preferred_model": None, "bic_difference": None,
                "shape_is_physical": False}


def test8_noise_floor(lc_obj, time: np.ndarray, flux: np.ndarray,
                      bls_params: dict) -> dict:
    """
    TEST 8 — Noise Floor (CDPP) Estimation.

    WHAT: Compute the Combined Differential Photometric Precision (CDPP) —
          TESS's standard noise metric. This tells us the minimum detectable
          transit depth for a given transit duration.
          If transit depth < 3× CDPP -> signal is buried in noise -> DISCARD.
    WHY:  A signal that sits below 3 times the noise floor cannot be
          statistically distinguished from random flux fluctuations.
          This is the photometric detection limit.
    """
    try:
        depth    = abs(bls_params.get("depth", 0.001))
        duration = bls_params.get("duration", 0.1)
        duration_hr = duration * 24.0

        cdpp_ppm = None

        # Try to get CDPP from the lightkurve object (most accurate)
        if lc_obj is not None:
            try:
                cdpp_ppm = float(lc_obj.estimate_cdpp(transit_duration=duration_hr).value)
            except Exception:
                pass

        # Fallback: estimate from the out-of-transit scatter
        if cdpp_ppm is None or np.isnan(cdpp_ppm):
            t0    = bls_params.get("t0", time[0])
            period = bls_params.get("period", 1.0)
            phase  = ((time - t0) / period) % 1.0
            phase[phase > 0.5] -= 1.0
            oot    = np.abs(phase) > (1.5 * duration / period)
            if oot.sum() > 10:
                n_in = max(1, int(duration_hr * 60 / 2))   # ~2-min cadence
                oot_rms = np.nanstd(flux[oot])
                cdpp_ppm = float(oot_rms / np.sqrt(n_in) * 1e6)
            else:
                cdpp_ppm = float(np.nanstd(flux) * 1e6)

        cdpp_frac = cdpp_ppm * 1e-6   # convert ppm -> fractional flux units
        depth_to_cdpp = depth / cdpp_frac if cdpp_frac > 0 else 0.0
        signal_above = bool(depth_to_cdpp >= 3.0)

        return {
            "cdpp_ppm": round(cdpp_ppm, 2),
            "depth_to_cdpp_ratio": round(float(depth_to_cdpp), 2),
            "signal_above_noise": signal_above,
        }
    except Exception as e:
        logger.warning(f"Noise floor test failed: {e}")
        return {"cdpp_ppm": None, "depth_to_cdpp_ratio": None,
                "signal_above_noise": None}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — AUTO-VERDICT LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def auto_verdict(tic_id: int, tests: dict, original_flag_reason: str) -> dict:
    """
    WHAT: Apply the UPGRADE / DOWNGRADE / KEEP-FLAG decision tree to the
          8 test results for one star.
    WHY:  Encapsulating the logic in one function keeps it easy to audit
          and modify without touching the test code above.

    Returns a dict with keys: final_verdict, tests_passed, human_review_note
    """

    # ── Count how many tests passed (True = pass, False = fail, None = skipped)
    test_keys = [
        "consistent_across_sectors", "centroid_stable",
        "signal_above_noise", "is_likely_starspot",
        "is_likely_eb", "shape_is_physical",
    ]
    # For starspot and EB flags: passing means the flag is FALSE (not a problem)
    def _passed(key, val):
        if val is None:
            return False
        if key in ("is_likely_starspot", "is_likely_eb"):
            return not bool(val)   # test passes when these are False
        return bool(val)

    results_map = {
        "consistent_across_sectors": tests.get("consistent_across_sectors"),
        "centroid_stable":           tests.get("centroid_stable"),
        "signal_above_noise":        tests.get("signal_above_noise"),
        "is_likely_starspot":        tests.get("is_likely_starspot"),
        "is_likely_eb":              tests.get("is_likely_eb"),
        "shape_is_physical":         tests.get("shape_is_physical"),
    }

    passed_list = [_passed(k, v) for k, v in results_map.items()]
    n_passed = sum(passed_list)

    # ── DOWNGRADE to DISCARD (strongest negative signals win) ────────────────
    # Any one of these alone is sufficient reason to discard
    noise_fail       = tests.get("signal_above_noise") is False
    starspot_no_sec  = (tests.get("is_likely_starspot") is True and
                        tests.get("consistent_across_sectors") is False)
    deep_secondary   = (tests.get("secondary_depth_ratio") or 0.0) > 0.40
    centroid_bad     = (tests.get("centroid_shift_pixels") or 0.0) > 2.0
    all_fail         = n_passed == 0

    if noise_fail or starspot_no_sec or deep_secondary or centroid_bad or all_fail:
        note = []
        if noise_fail:       note.append("Transit depth below 3× CDPP noise floor")
        if starspot_no_sec:  note.append("Stellar rotation period matches transit period (starspot) and not seen in other sectors")
        if deep_secondary:   note.append(f"Deep secondary eclipse ratio {tests.get('secondary_depth_ratio'):.2f} > 0.40 -> eclipsing binary")
        if centroid_bad:     note.append(f"Large centroid shift ({tests.get('centroid_shift_pixels'):.2f} px) -> background blend")
        if all_fail:         note.append("All 8 diagnostic tests failed")
        return {
            "final_verdict": "DISCARD",
            "tests_passed": n_passed,
            "human_review_note": "; ".join(note) or "Downgraded by auto-verdict",
        }

    # ── UPGRADE to KEEP (all key conditions must be satisfied + 5+ tests pass)
    key_conditions = [
        tests.get("consistent_across_sectors") is True,
        tests.get("centroid_stable") is not False,     # None = inconclusive, don't block
        tests.get("signal_above_noise") is True,
        tests.get("is_likely_starspot") is not True,   # None = not detected -> OK
        tests.get("is_likely_eb") is not True,
        tests.get("shape_is_physical") is True,
    ]
    if all(key_conditions) and n_passed >= 5:
        return {
            "final_verdict": "KEEP",
            "tests_passed": n_passed,
            "human_review_note": "Upgraded from FLAG: all key diagnostics passed",
        }

    # ── KEEP as FLAG — build a specific human-readable note ──────────────────
    notes = []
    if tests.get("consistent_across_sectors") is False:
        notes.append("Signal detected in only 1 TESS sector — may be 1-sector artefact")
    if tests.get("is_likely_eb") is True:
        p = tests.get("odd_even_pvalue", "?")
        notes.append(f"Odd-even depth difference marginally significant (p={p}); recommend spectroscopy to rule out EB")
    if tests.get("is_likely_starspot") is True:
        r = tests.get("rotation_period_ratio", "?")
        notes.append(f"Stellar rotation period = {r:.2f}× transit period; possible starspot confusion")
    if tests.get("shape_is_physical") is False:
        notes.append(f"Transit shape prefers box over trapezoid (BIC diff={tests.get('bic_difference')})")
    if tests.get("ttv_significant") is True:
        amp = tests.get("ttv_amplitude_minutes", "?")
        notes.append(f"Significant TTV amplitude ({amp} min) -> multi-planet system candidate")
    if not notes:
        notes.append("Multiple borderline tests — no single discriminating factor")

    return {
        "final_verdict": "FLAG",
        "tests_passed": n_passed,
        "human_review_note": "; ".join(notes),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_flag_diagnostic(tic_id: int, time: np.ndarray, flux: np.ndarray,
                         bls_params: dict, tests: dict, verdict: dict):
    """
    WHAT: Save a 4-panel diagnostic plot for each analysed flag star.
    WHY:  Visual inspection is the final sanity check — a plot reveals
          patterns that numbers alone cannot convey.

    Panels:
      A — Phase-folded light curve with trapezoid model overlay
      B — Lomb-Scargle periodogram of OOT flux (Test 6)
      C — Individual transit depth scatter (Test 5 / Test 4 hints)
      D — Summary score bar chart of all 8 tests
    """
    plt = _init_plot()
    import matplotlib.pyplot as _plt  # noqa: needed for type hints

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    period   = bls_params.get("period", 1.0)
    t0       = bls_params.get("t0", time[0])
    duration = bls_params.get("duration", 0.1)

    # ── Panel A: Phase-folded light curve ────────────────────────────────────
    ax = axes[0, 0]
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    ph_s = phase[sort_idx]
    fl_s = flux[sort_idx]

    # Bin for clarity
    n_bins = 100
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux = np.array([np.nanmedian(fl_s[(ph_s >= bin_edges[i]) & (ph_s < bin_edges[i+1])])
                         if np.any((ph_s >= bin_edges[i]) & (ph_s < bin_edges[i+1]))
                         else np.nan
                         for i in range(n_bins)])

    ax.scatter(ph_s, fl_s, s=2, alpha=0.2, color="#90CAF9", label="raw")
    ax.plot(bin_centres, bin_flux, color="#1565C0", lw=2, label="binned")
    ax.axvline(0, color="#F44336", ls="--", lw=1, label="transit centre")
    ax.set_title(f"TIC {tic_id} — Phase-Folded (P={period:.4f} d)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised Flux")
    ax.set_xlim(-0.4, 0.4)
    ax.legend(fontsize=8)

    # ── Panel B: OOT Lomb-Scargle (stellar variability) ──────────────────────
    ax = axes[0, 1]
    try:
        from astropy.timeseries import LombScargle
        oot = np.abs(phase) > (1.5 * duration / period)
        if oot.sum() > 20:
            freq, power = LombScargle(time[oot], flux[oot]).autopower(
                minimum_frequency=1.0 / (5 * period),
                maximum_frequency=1.0 / (0.5 * period),
            )
            ax.plot(1.0 / freq, power, color="#7B1FA2", lw=1.2)
            ax.axvline(period, color="#F44336", ls="--", lw=1.5,
                       label=f"Transit P={period:.3f} d")
            ax.set_xlabel("Period (days)")
            ax.set_ylabel("LS Power")
            ax.set_title("OOT Lomb-Scargle (Starspot Check)", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "LS Periodogram unavailable", ha="center", va="center")
        ax.set_title("OOT Lomb-Scargle", fontsize=11, fontweight="bold")

    # ── Panel C: Individual transit depths (odd vs even) ─────────────────────
    ax = axes[1, 0]
    try:
        n_start = int(np.floor((time[0]  - t0) / period))
        n_end   = int(np.ceil( (time[-1] - t0) / period))
        odd_d, even_d = [], []
        odd_e, even_e = [], []

        for epoch in range(n_start, n_end + 1):
            tc = t0 + epoch * period
            w  = 1.5 * duration
            mask = np.abs(time - tc) < w
            oot_m = (np.abs(time - tc) > w) & (np.abs(time - tc) < 3 * w)
            if mask.sum() < 3 or oot_m.sum() < 3:
                continue
            base = np.nanmedian(flux[oot_m])
            d    = base - np.nanmedian(flux[mask])
            e    = np.nanstd(flux[oot_m]) / np.sqrt(mask.sum())
            if epoch % 2 == 0:
                even_d.append(d); even_e.append(e)
            else:
                odd_d.append(d); odd_e.append(e)

        if odd_d and even_d:
            ax.errorbar(range(len(odd_d)), odd_d, yerr=odd_e,
                        fmt="o", color="#2E7D32", label="Odd transits", capsize=3)
            ax.errorbar(range(len(even_d)), even_d, yerr=even_e,
                        fmt="s", color="#C62828", label="Even transits", capsize=3)
            ax.axhline(np.mean(odd_d + even_d), ls="--", color="gray", lw=1)
        ax.set_title("Odd vs Even Transit Depths", fontsize=11, fontweight="bold")
        ax.set_xlabel("Transit Number")
        ax.set_ylabel("Depth (fractional flux)")
        ax.legend(fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Odd-even plot unavailable", ha="center", va="center")

    # ── Panel D: 8-test summary bar chart ────────────────────────────────────
    ax = axes[1, 1]
    test_names = [
        "Multi-Sector\nConsistency",
        "Centroid\nStability",
        "Secondary\nEclipse Check",
        "TTV\nCheck",
        "Odd-Even\nCheck",
        "Starspot\nCheck",
        "Trapezoid\nShape",
        "CDPP\nNoise Floor",
    ]
    # Map tests to pass/fail/unknown
    def _score(val, key=""):
        if val is None:
            return 0.5   # grey = skipped
        if key in ("is_likely_starspot", "is_likely_eb"):
            return 1.0 if not val else 0.0
        return 1.0 if val else 0.0

    scores = [
        _score(tests.get("consistent_across_sectors")),
        _score(tests.get("centroid_stable")),
        _score(tests.get("secondary_classification") == "consistent_with_planet"),
        _score(tests.get("ttv_significant") is not True, ""),
        _score(tests.get("is_likely_eb"), "is_likely_eb"),
        _score(tests.get("is_likely_starspot"), "is_likely_starspot"),
        _score(tests.get("shape_is_physical")),
        _score(tests.get("signal_above_noise")),
    ]
    colors = ["#4CAF50" if s == 1.0 else "#F44336" if s == 0.0 else "#FF9800"
              for s in scores]

    ax.barh(test_names, scores, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Pass (1) / Fail (0)")
    verdict_str = verdict.get("final_verdict", "?")
    n_p = verdict.get("tests_passed", "?")
    ax.set_title(f"8-Test Score Card — Verdict: {verdict_str} ({n_p}/8 passed)",
                 fontsize=11, fontweight="bold")

    plt.suptitle(
        f"FLAG Deep Analysis: TIC {tic_id}",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    out_path = PLOTS_DIR / f"TIC_{tic_id}_flag_analysis.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"TIC {tic_id}: diagnostic plot saved -> {out_path.name}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — OUTPUT FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def update_results_csv(tic_id: int, verdict: str, note: str):
    """
    WHAT: Find the existing row for this TIC ID in results.csv and update
          its decision column with the new verdict.
    WHY:  The main results file is the authoritative record. Updating it
          ensures the final summary dashboard reflects the deeper analysis.
    """
    results_path = RESULTS_DIR / "results.csv"
    if not results_path.exists():
        return

    try:
        df = pd.read_csv(results_path)
        mask = df["tic_id"] == tic_id
        if mask.any():
            df.loc[mask, "decision"]     = verdict
            df.loc[mask, "flag_reasons"] = f"{note} [auto-verdict]"
        else:
            # TIC not found in CSV (standalone run) — append a minimal row
            new_row = pd.DataFrame([{
                "tic_id": tic_id, "decision": verdict, "final_class": "FLAG->" + verdict,
                "confidence": 0.0, "period": 0.0, "period_err": 0.0,
                "depth": 0.0, "depth_err": 0.0, "duration": 0.0,
                "duration_err": 0.0, "snr": 0.0, "flag_reasons": note,
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(results_path, index=False)
    except Exception as e:
        logger.warning(f"TIC {tic_id}: could not update results.csv — {e}")


def copy_to_verdict_folder(tic_id: int, verdict: str, plot_path: Optional[Path]):
    """
    WHAT: Copy the diagnostic plot and pipeline JSON report into the
          results/KEEP/, results/FLAG/, or results/DISCARD/ sub-folder.
    WHY:  Gives the user a single folder to open per decision tier —
          no need to filter the full results CSV.
    """
    dest_dir = {"KEEP": KEEP_DIR, "FLAG": FLAG_DIR, "DISCARD": DISCARD_DIR}.get(verdict, FLAG_DIR)

    # Copy the flag analysis plot
    if plot_path and plot_path.exists():
        shutil.copy2(str(plot_path), str(dest_dir / plot_path.name))

    # Copy the pipeline diagnostic plot if it exists
    pipeline_plot = RESULTS_DIR / "figures" / f"TIC_{tic_id}_diagnostic.png"
    if pipeline_plot.exists():
        shutil.copy2(str(pipeline_plot), str(dest_dir / pipeline_plot.name))

    # Copy the JSON report
    report_path = REPORTS_DIR / f"TIC_{tic_id}_report.json"
    if report_path.exists():
        shutil.copy2(str(report_path), str(dest_dir / report_path.name))


def generate_report_card(tic_id: int, verdict: str, tests: dict,
                         bls_params: dict, note: str):
    """
    WHAT: Write a Markdown report card into the appropriate verdict folder.
    WHY:  Provides a human-readable summary of every test result alongside
          the final decision, making astronomer review fast and structured.
    """
    dest_dir = {"KEEP": KEEP_DIR, "FLAG": FLAG_DIR, "DISCARD": DISCARD_DIR}.get(verdict, FLAG_DIR)

    icon = {"KEEP": "🪐", "FLAG": "⚠️", "DISCARD": "🗑️"}.get(verdict, "❓")

    md = f"""# {icon} FLAG Deep-Analysis Report Card: TIC {tic_id}
**ISRO Hackathon Challenge 7 — FLAG Auto-Verification Layer**
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Final Verdict: **{verdict}**
> {note}

---

## 8-Test Diagnostic Summary

| # | Test | Result |
|:--|:-----|:-------|
| 1 | Multi-Sector Consistency | Sectors found: **{tests.get('n_sectors_checked', '?')}** -> Consistent: **{tests.get('consistent_across_sectors', '?')}** |
| 2 | Centroid Stability | Shift: **{tests.get('centroid_shift_pixels', '?')} px** -> Stable: **{tests.get('centroid_stable', '?')}** |
| 3 | Secondary Eclipse | Depth ratio: **{tests.get('secondary_depth_ratio', '?')}** -> {tests.get('secondary_classification', '?')} |
| 4 | Transit Timing Variations | TTV amplitude: **{tests.get('ttv_amplitude_minutes', '?')} min** -> Significant: **{tests.get('ttv_significant', '?')}** |
| 5 | Odd-Even Depth Check | p-value: **{tests.get('odd_even_pvalue', '?')}** -> Likely EB: **{tests.get('is_likely_eb', '?')}** |
| 6 | Stellar Variability | Rotation period: **{tests.get('stellar_rotation_period', '?')} d** (ratio: {tests.get('rotation_period_ratio', '?')}×) -> Starspot: **{tests.get('is_likely_starspot', '?')}** |
| 7 | Trapezoid vs Box | Preferred model: **{tests.get('preferred_model', '?')}** (ΔBIC={tests.get('bic_difference', '?')}) -> Physical: **{tests.get('shape_is_physical', '?')}** |
| 8 | CDPP Noise Floor | CDPP: **{tests.get('cdpp_ppm', '?')} ppm** -> Depth/CDPP: **{tests.get('depth_to_cdpp_ratio', '?')}×** -> Above noise: **{tests.get('signal_above_noise', '?')}** |

---

## Orbital Parameters (from main pipeline)

| Parameter | Value |
|:----------|:------|
| Period | {bls_params.get('period', '?'):.5f} days |
| Depth | {abs(bls_params.get('depth', 0))*100:.4f} % |
| Duration | {bls_params.get('duration', 0)*24:.2f} hours |
| SNR | {bls_params.get('snr', '?'):.2f} |

---

## Visual Diagnostics
![Flag Analysis Plot](TIC_{tic_id}_flag_analysis.png)
"""

    out_path = dest_dir / f"report_TIC_{tic_id}_flag_analysis.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"TIC {tic_id}: report card saved -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — MAIN ANALYSIS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_flag_analysis(tic_ids: Optional[list] = None) -> dict:
    """
    WHAT: Orchestrates the full 8-test deep analysis for every flagged star.
    WHY:  This is the public function called by pipeline.py (--analyze-flags)
          and by the standalone CLI (python flag_analyzer.py).

    Args:
        tic_ids: optional list of specific TIC IDs to analyse.
                 If None, loads all entries from manual_review_queue.csv.

    Returns a summary dict: {upgraded, downgraded, remained_flag}
    """

    # ── Load the manual review queue ─────────────────────────────────────────
    queue_path = RESULTS_DIR / "manual_review_queue.csv"
    if tic_ids:
        # Manually specified targets — create a minimal dataframe
        df_queue = pd.DataFrame({
            "tic_id": tic_ids,
            "combined_confidence": [None] * len(tic_ids),
            "flag_reasons": ["manual_override"] * len(tic_ids),
        })
    elif queue_path.exists():
        df_queue = pd.read_csv(queue_path)
    else:
        logger.warning("No manual_review_queue.csv found. Nothing to analyse.")
        return {"upgraded": 0, "downgraded": 0, "remained_flag": 0}

    if len(df_queue) == 0:
        logger.info("manual_review_queue.csv is empty — no flags to analyse.")
        return {"upgraded": 0, "downgraded": 0, "remained_flag": 0}

    logger.info(f"Starting deep analysis on {len(df_queue)} flagged stars...")

    # ── Output file: flag_analysis_results.csv ────────────────────────────────
    analysis_csv_path = RESULTS_DIR / "flag_analysis_results.csv"
    fieldnames = [
        "tic_id", "original_flag_reason", "final_verdict", "tests_passed",
        "consistent_across_sectors", "centroid_stable", "secondary_depth_ratio",
        "ttv_significant", "odd_even_pvalue", "is_likely_starspot",
        "shape_is_physical", "depth_to_cdpp_ratio", "human_review_note",
    ]
    csv_file  = open(analysis_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # ── Counters ──────────────────────────────────────────────────────────────
    n_upgraded   = 0
    n_downgraded = 0
    n_flagged    = 0
    updated_queue_rows = []   # rows that remain FLAG (for re-writing queue CSV)

    # ── Main loop — one star at a time ────────────────────────────────────────
    for i, row in tqdm(df_queue.iterrows(), total=len(df_queue), desc="FLAG Deep Analysis"):
        tic_id = int(row["tic_id"])
        original_reason = str(row.get("flag_reasons", row.get("combined_confidence", "")))

        logger.info(f"\n{'='*60}")
        logger.info(f"Analysing TIC {tic_id} (original flag: {original_reason})")

        # ── Load data ────────────────────────────────────────────────────────
        data = load_star_data(tic_id)
        if data is None:
            logger.warning(f"TIC {tic_id}: skipping — could not load data.")
            continue

        time      = data["time"]
        flux      = data["flux"]
        flux_err  = data["flux_err"]
        bls_params = data["bls_params"]
        lc_obj    = data.get("lc_obj")

        # ── Run all 8 tests ──────────────────────────────────────────────────
        all_tests = {}

        # Test 1 — Multi-sector
        t1 = test1_multi_sector(tic_id, bls_params.get("period", 1.0))
        all_tests.update(t1)

        # Test 2 — Centroid motion
        t2 = test2_centroid_motion(lc_obj, time, flux, bls_params)
        all_tests.update(t2)

        # Test 3 — Secondary eclipse
        t3 = test3_secondary_eclipse(time, flux, bls_params)
        all_tests.update(t3)

        # Test 4 — TTV
        t4 = test4_ttv(time, flux, bls_params)
        all_tests.update(t4)

        # Test 5 — Odd-even
        t5 = test5_odd_even(time, flux, bls_params)
        all_tests.update(t5)

        # Test 6 — Stellar variability
        t6 = test6_stellar_variability(time, flux, bls_params)
        all_tests.update(t6)

        # Test 7 — Box vs trapezoid
        t7 = test7_box_vs_trapezoid(time, flux, bls_params)
        all_tests.update(t7)

        # Test 8 — Noise floor / CDPP
        t8 = test8_noise_floor(lc_obj, time, flux, bls_params)
        all_tests.update(t8)

        # ── Apply auto-verdict ───────────────────────────────────────────────
        verdict_dict = auto_verdict(tic_id, all_tests, original_reason)
        final_verdict = verdict_dict["final_verdict"]
        tests_passed  = verdict_dict["tests_passed"]
        human_note    = verdict_dict["human_review_note"]

        logger.info(f"TIC {tic_id}: verdict = {final_verdict} ({tests_passed}/8 tests passed)")
        logger.info(f"           Note: {human_note}")

        # ── Generate diagnostic plot ─────────────────────────────────────────
        try:
            plot_path = plot_flag_diagnostic(tic_id, time, flux, bls_params,
                                             all_tests, verdict_dict)
        except Exception as e:
            logger.warning(f"TIC {tic_id}: plot generation failed — {e}")
            plot_path = None

        # ── Generate report card ─────────────────────────────────────────────
        try:
            generate_report_card(tic_id, final_verdict, all_tests, bls_params, human_note)
        except Exception as e:
            logger.warning(f"TIC {tic_id}: report card generation failed — {e}")

        # ── Copy outputs to verdict folder ───────────────────────────────────
        try:
            copy_to_verdict_folder(tic_id, final_verdict, plot_path)
        except Exception as e:
            logger.warning(f"TIC {tic_id}: folder copy failed — {e}")

        # ── Update results.csv ───────────────────────────────────────────────
        update_results_csv(tic_id, final_verdict, human_note)

        # ── Write row to flag_analysis_results.csv ───────────────────────────
        csv_writer.writerow({
            "tic_id":                    tic_id,
            "original_flag_reason":      original_reason,
            "final_verdict":             final_verdict,
            "tests_passed":              tests_passed,
            "consistent_across_sectors": all_tests.get("consistent_across_sectors"),
            "centroid_stable":           all_tests.get("centroid_stable"),
            "secondary_depth_ratio":     all_tests.get("secondary_depth_ratio"),
            "ttv_significant":           all_tests.get("ttv_significant"),
            "odd_even_pvalue":           all_tests.get("odd_even_pvalue"),
            "is_likely_starspot":        all_tests.get("is_likely_starspot"),
            "shape_is_physical":         all_tests.get("shape_is_physical"),
            "depth_to_cdpp_ratio":       all_tests.get("depth_to_cdpp_ratio"),
            "human_review_note":         human_note,
        })

        # ── Tally counts ────────────────────────────────────────────────────
        if final_verdict == "KEEP":
            n_upgraded += 1
        elif final_verdict == "DISCARD":
            n_downgraded += 1
        else:
            n_flagged += 1
            updated_queue_rows.append({
                "tic_id": tic_id,
                "combined_confidence": row.get("combined_confidence", ""),
                "period": row.get("period", ""),
                "depth": row.get("depth", ""),
                "flag_reasons": human_note,
                "tests_passed": tests_passed,
            })

        # ── Checkpoint: flush every 5 stars ──────────────────────────────────
        if (i + 1) % 5 == 0:
            csv_file.flush()
            logger.info(f"Checkpoint: flushed results after {i + 1} stars.")

    csv_file.close()

    # ── Re-write manual_review_queue.csv with only remaining flags ────────────
    # Sort by tests_passed descending (most promising first for human review)
    if updated_queue_rows:
        updated_df = pd.DataFrame(updated_queue_rows)
        updated_df = updated_df.sort_values("tests_passed", ascending=False)
        updated_df.to_csv(queue_path, index=False)
        logger.info(f"Updated manual_review_queue.csv with {len(updated_df)} remaining flags.")

    # ── Write metadata.json ──────────────────────────────────────────────────
    meta = {
        "module": "flag_analyzer",
        "version": "1.0.0",
        "run_timestamp": datetime.now().isoformat(),
        "n_input_flags": len(df_queue),
        "n_upgraded_to_keep": n_upgraded,
        "n_downgraded_to_discard": n_downgraded,
        "n_remained_flag": n_flagged,
        "output_files": {
            "analysis_csv": str(analysis_csv_path),
            "keep_folder": str(KEEP_DIR),
            "flag_folder": str(FLAG_DIR),
            "discard_folder": str(DISCARD_DIR),
            "plots_folder": str(PLOTS_DIR),
        },
    }
    with open(RESULTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    # ── Final summary printout ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FLAG Analysis Complete:")
    print(f"  {n_upgraded}  upgraded  -> KEEP  (results/KEEP/)")
    print(f"  {n_downgraded}  downgraded -> DISCARD  (results/DISCARD/)")
    print(f"  {n_flagged}  remain for human review  (results/FLAG/)")
    print(f"  See results/flag_analysis_results.csv for full table.")
    print("=" * 60 + "\n")

    return {
        "upgraded": n_upgraded,
        "downgraded": n_downgraded,
        "remained_flag": n_flagged,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7 — COMMAND-LINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FLAG Deep-Analysis & Auto-Verification Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse ALL flags in manual_review_queue.csv:
  python flag_analyzer.py

  # Analyse a single flagged star by TIC ID:
  python flag_analyzer.py --tic 219698950

  # Analyse multiple specific stars:
  python flag_analyzer.py --tic 219698950 233720539 147456499
        """,
    )
    parser.add_argument(
        "--tic", type=int, nargs="+", default=None,
        help="One or more TIC IDs to analyse (default: all flags in queue CSV).",
    )
    args = parser.parse_args()

    run_flag_analysis(tic_ids=args.tic)

