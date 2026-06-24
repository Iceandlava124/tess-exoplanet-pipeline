"""
src/preprocess.py
=================
Light curve preprocessing: cleaning, detrending, and normalising raw TESS data.

📚 LEARNING NOTE:
    Raw telescope data is messy. Before we can do any ML or signal
    detection, we need to:
    1. Remove outliers (cosmic rays, satellite crossings)
    2. Remove slow trends (stellar variability, instrument drift)
    3. Normalise so all light curves are on the same scale
    4. Filter out bad data using quality flags

    This is called "preprocessing" and it's crucial in ML —
    "garbage in, garbage out."
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.signal import savgol_filter
import lightkurve as lk

logger = logging.getLogger(__name__)


def load_fits(fits_path: str | Path) -> Optional[lk.LightCurve]:
    """
    Load a TESS FITS file into a lightkurve LightCurve object.

    Args:
        fits_path: Path to .fits file

    Returns:
        LightCurve object, or None if loading fails.

    📚 LEARNING NOTE:
        A LightCurve object is like a fancy pandas DataFrame with
        columns: time (in BTJD days), flux (electrons/sec), flux_err,
        and quality flags. lightkurve handles all the FITS parsing for us.
    """
    try:
        lc = lk.io.read(str(fits_path))
        return lc
    except Exception as e:
        logger.error(f"Failed to load {fits_path}: {e}")
        return None


def apply_quality_mask(lc: lk.LightCurve, bitmask: int = 175) -> lk.LightCurve:
    """
    Remove cadences flagged as bad quality by TESS.

    Args:
        lc:      Input LightCurve
        bitmask: Quality bitmask (175 = default TESS recommended)

    Returns:
        Cleaned LightCurve with bad cadences removed.

    📚 LEARNING NOTE:
        TESS records a "quality" bitmask for each data point.
        Each bit (0/1) flags a different problem:
          Bit 1  = Attitude tweak (satellite pointing changed)
          Bit 2  = Safe mode
          Bit 3  = Coarse pointing
          Bit 5  = Argabrightening event (cosmic ray hitting sensor)
          ...etc.
        By applying a bitmask, we zero out points where any of these
        issues occurred. Bitmask 175 removes the most common issues.
    """
    try:
        # Apply TESS quality bitmask to remove bad cadences
        # (attitude tweaks, safe mode, cosmic rays, coarse pointing, etc.)
        if hasattr(lc, 'quality') and lc.quality is not None:
            quality_mask = (lc.quality & bitmask) == 0
            if quality_mask.sum() > 50:   # keep if enough good points remain
                lc = lc[quality_mask]
                logger.debug(f"Quality mask (bitmask={bitmask}): {quality_mask.sum()}/{len(quality_mask)} cadences kept")
            else:
                logger.warning(f"Quality mask too aggressive ({quality_mask.sum()} points). Skipping bitmask.")
    except Exception as qm_err:
        logger.warning(f"Quality bitmask application failed: {qm_err}. Proceeding without.")
    return lc.remove_outliers(sigma=5).select_flux("pdcsap_flux")



def sigma_clip(
    flux: np.ndarray,
    sigma: float = 5.0,
    n_iter: int = 5,
) -> np.ndarray:
    """
    Iterative sigma-clipping using robust Median Absolute Deviation (MAD)
    to remove outliers from flux array without scale inflation.
    """
    flux = flux.copy().astype(float)

    for _ in range(n_iter):
        median = np.nanmedian(flux)
        
        # Use MAD as a robust estimator of standard deviation (rescaled by 1.4826)
        mad = np.nanmedian(np.abs(flux - median))
        if mad < 1e-10:
            scale = np.nanstd(flux)
        else:
            scale = 1.4826 * mad
            
        if scale < 1e-10:
            break
            
        mask = np.abs(flux - median) > sigma * scale
        if mask.sum() == 0:
            break
        flux[mask] = np.nan

    return flux

# ==========================================
# STAGE 1 DETRENDING: WOTan Biweight (Robust Rough Pass)
# ==========================================
def detrend_wotan_biweight(
    time: np.ndarray,
    flux: np.ndarray,
    window_length: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 1 of two-stage detrending: robust outlier-resistant smoothing.
    Wotan's biweight method down-weights outliers automatically, so it
    removes large-scale trends without being distorted by remaining
    cosmic ray hits or uncleaned bad points. This runs BEFORE the
    adaptive Savitzky-Golay pass, which then fine-tunes using the
    star's own measured variability period.

    window_length is in DAYS, not cadences (wotan's native unit).
    Returns the flattened flux and the trend that was removed,
    so the trend can be inspected/plotted for QA if needed.
    """
    from wotan import flatten

    flattened_flux, trend = flatten(
        time,
        flux,
        method="biweight",
        window_length=window_length,
        return_trend=True
    )
    return flattened_flux, trend


def detrend_savgol(
    flux: np.ndarray,
    window_length: int = 401,
    polyorder: int = 3,
    time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remove long-term stellar variability using a Savitzky-Golay filter.

    A S-G filter fits a polynomial to a sliding window of data points,
    producing a smooth "trend" curve. Dividing by this trend removes
    slow variations while preserving sharp transit dips.
    
    If time is provided, detects gaps (>0.5 days) and detrends continuous
    segments individually to avoid boundary artifacts.

    Args:
        flux:          Flux array (with NaNs replaced by interpolation first)
        window_length: Window size in data points. Must be odd.
                       For 2-min cadence, 401 points ≈ 13.4 hours.
        polyorder:     Polynomial degree (3 = cubic)
        time:          Optional time array to check for gaps (e.g. stitched sectors)

    Returns:
        Detrended flux (flux / trend), normalised around 1.0
    """
    # Plain English: If time is provided, split the light curve by sector gaps (>0.5d) and detrend each individually
    if time is not None:
        dt = np.diff(time)
        gap_indices = np.where(dt > 0.5)[0] + 1
        segments = np.split(np.arange(len(flux)), gap_indices)
        
        detrended = np.zeros_like(flux)
        for seg in segments:
            if len(seg) == 0:
                continue
            seg_flux = flux[seg]
            
            # Adjust window length if segment is shorter than the window_length
            if len(seg) < window_length:
                seg_window = len(seg)
                if seg_window % 2 == 0:
                    seg_window -= 1
                if seg_window > polyorder:
                    detrended[seg] = detrend_savgol(seg_flux, window_length=seg_window, polyorder=polyorder, time=None)
                else:
                    median_val = np.nanmedian(seg_flux)
                    detrended[seg] = seg_flux / (median_val if abs(median_val) > 1e-10 else 1.0)
            else:
                detrended[seg] = detrend_savgol(seg_flux, window_length=window_length, polyorder=polyorder, time=None)
        return detrended

    # Replace NaNs with linear interpolation before filtering
    nans = np.isnan(np.asarray(flux, dtype=float))
    valid_mask = ~nans
    if not valid_mask.any():   # no finite points at all → nothing to interpolate
        return flux

    x = np.arange(len(flux))
    flux_interp = flux.copy()
    flux_interp[nans] = np.interp(x[nans], x[valid_mask], flux[valid_mask])

    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Compute the smooth trend
    trend = savgol_filter(flux_interp, window_length=window_length, polyorder=polyorder)

    # Divide by trend (avoid division by zero)
    trend = np.where(np.abs(trend) < 1e-10, 1.0, trend)
    detrended = flux_interp / trend

    # Restore NaN positions
    detrended[nans] = np.nan

    return detrended


def normalise(flux: np.ndarray) -> np.ndarray:
    """
    Normalise flux so the median = 1.0.

    This ensures all light curves are on the same scale,
    regardless of how bright the star is.

    📚 LEARNING NOTE:
        Normalisation is one of the most important preprocessing steps
        in ML. If one feature (e.g., raw flux) has values in the
        millions while another has values in the range 0-1, the ML
        model will be dominated by the large-valued feature.

        By dividing by the median, we make every star's "quiet" flux
        equal to 1.0. A transit dip of 1% becomes visible as a dip
        from 1.0 to 0.99.
    """
    median = np.nanmedian(flux)
    if abs(median) < 1e-10:
        return flux
    return flux / median


def preprocess_lightcurve(
    lc: lk.LightCurve,
    sigma_clip_threshold: float = 5.0,
    window_length: int = 401,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single light curve.

    Steps:
        1. Extract flux and time arrays
        2. Apply quality mask (remove flagged cadences)
        3. Sigma-clip outliers
        4. Detrend with Savitzky-Golay filter
        5. Normalise to median = 1.0

    Args:
        lc:                    lightkurve LightCurve object
        sigma_clip_threshold:  Sigma for outlier removal (default 5)
        window_length:         Savitzky-Golay window in cadences

    Returns:
        Tuple of (time, flux, flux_err) as numpy arrays.
        All three arrays are aligned and have the same length.
        NaN positions are consistent across all three.
    """
    # Try to get PDCSAP flux (pre-corrected systematics)
    is_ppm = False
    # Try to get PDCSAP flux (pre-corrected systematics)
    is_ppm = False
    try:
        if hasattr(lc, "pdcsap_flux"):
            flux = lc.pdcsap_flux.value
            flux_err = lc.pdcsap_flux_err.value
            is_ppm = str(getattr(lc.pdcsap_flux, "unit", "")) == "ppm"
        else:
            flux = lc.flux.value
            flux_err = lc.flux_err.value
            is_ppm = str(getattr(lc.flux, "unit", "")) == "ppm"
        time = lc.time.value
    except Exception:
        flux = np.array(lc.flux)
        flux_err = np.ones_like(flux) * np.nanstd(flux) * 0.01
        time = np.arange(len(flux), dtype=float)
        is_ppm = False

    # ── Sanitise: cast MaskedArrays to plain float ndarrays ─────────────────
    # lightkurve can return astropy MaskedArrays for heavily quality-flagged
    # light curves. numpy.ma.MaskedArray.all() returns masked constant "--",
    # not True/False — so bool(nans.all()) returns False even when all-NaN,
    # breaking our NaN guard. Calling .filled(np.nan) converts masked elements
    # to ordinary NaN and strips the mask wrapper entirely.
    if isinstance(flux, np.ma.MaskedArray):
        flux = np.asarray(flux.filled(np.nan), dtype=float)
    if isinstance(flux_err, np.ma.MaskedArray):
        flux_err = np.asarray(flux_err.filled(np.nan), dtype=float)
    if isinstance(time, np.ma.MaskedArray):
        time = np.asarray(time.filled(np.nan), dtype=float)

    if is_ppm:
        logger.info("Detecting flux unit is 'ppm' (TASOC data product). Converting to normalized scale around 1.0.")
        flux = 1.0 + (flux / 1e6)
        flux_err = flux_err / 1e6

    # Calculate median flux before normalisation to normalise errors consistently
    median_flux = np.nanmedian(flux)
    if abs(median_flux) < 1e-10:
        median_flux = 1.0

    # Step 1: Replace inf/nan in flux_err
    flux_err = np.where(np.isfinite(flux_err), flux_err, np.nanmedian(flux_err))

    # Step 2: Sigma clip
    flux = sigma_clip(flux, sigma=sigma_clip_threshold)

    # ==========================================
    # LOMB-SCARGLE VARIABILITY ANALYSIS (First pass on raw flux)
    # ==========================================
    # We run Lomb-Scargle first to determine if the star has significant variability,
    # and size the detrending windows dynamically for both Stage 1 and Stage 2.
    p_var = np.nan
    w_days = 3.5  # Raised default — protects transits up to ~7 hours duration
    fap = 1.0
    try:
        from astropy.timeseries import LombScargle
        ls_mask = np.isfinite(time) & np.isfinite(flux)
        t_ls = time[ls_mask]
        f_ls = flux[ls_mask] - np.nanmedian(flux[ls_mask])
        if len(t_ls) > 100:
            frequency, power = LombScargle(t_ls, f_ls).autopower(
                minimum_frequency=1.0/10.0, 
                maximum_frequency=1.0/0.1
            )
            max_power = np.max(power)
            try:
                fap = float(LombScargle(t_ls, f_ls).false_alarm_probability(max_power))
            except Exception:
                fap = 1.0
                
            if fap < 0.01:
                best_freq = frequency[np.argmax(power)]
                p_var = 1.0 / best_freq
                # Optimal window is 3x the variability period, capped between 0.5 and 3.5 days
                w_days = np.clip(3 * p_var, 0.5, 3.5)
                logger.info(f"Stellar variability detected (FAP={fap:.2e}, period={p_var:.2f}d) -> using adaptive window={w_days:.2f}d")
            else:
                w_days = 3.5
                logger.info(f"Quiet star (FAP={fap:.3f} >= 0.01) -> using safe default window of {w_days:.2f}d")
    except Exception as e:
        logger.warning(f"Lomb-Scargle periodogram calculation failed: {e}. Defaulting to {w_days:.2f}d window.")

    # ==========================================
    # STAGE 1: Wotan biweight — robust rough detrend
    # ==========================================
    # Removes large-scale stellar trends, resistant to remaining outliers.
    # Uses the dynamically sized w_days window instead of a hardcoded 0.5d window.
    wotan_window = w_days
    wotan_trend_removed = False
    flux_stage1 = flux.copy()
    try:
        flux_stage1, wotan_trend = detrend_wotan_biweight(
            time, flux, window_length=wotan_window
        )
        wotan_trend_removed = True
        logger.info(f"Stage 1 Detrending: wotan biweight (window={wotan_window:.2f}d) successful.")
    except Exception as e:
        logger.warning(f"Stage 1 Wotan detrending failed: {e}. Skipping straight to Stage 2.")

    # ==========================================
    # STAGE 2: Adaptive Savitzky-Golay — fine-tuned per-star pass
    # ==========================================
    # Apply Savgol smoothing on the Stage 1 output using the same window length
    dt = np.nanmedian(np.diff(time))
    if dt > 0:
        adaptive_window = int(w_days / dt)
        if adaptive_window % 2 == 0:
            adaptive_window += 1
        if adaptive_window < 5:
            adaptive_window = 5
        window_length = adaptive_window
        logger.info(f"Stage 2 Adaptive Detrending points: {window_length} points based on {w_days:.2f}d window")
    else:
        logger.warning("Could not calculate dt for adaptive window. Using default window_length.")

    flux_final = detrend_savgol(flux_stage1, window_length=window_length, time=time)

    # Add these new fields to the preprocessing output/quality report
    if hasattr(lc, "meta"):
        lc.meta["wotan_window_days"] = wotan_window
        lc.meta["wotan_trend_removed"] = wotan_trend_removed
        lc.meta["savgol_adaptive_window_days"] = w_days
        lc.meta["savgol_stellar_variability_period"] = p_var
        lc.meta["detrend_method"] = "wotan_biweight+adaptive_savgol"
        
        # Calculate Stage 1 RMS (wotan-cleaned but before savgol)
        if len(flux_stage1) > 0 and np.nanmedian(flux_stage1) != 0:
            norm_stage1 = flux_stage1 / np.nanmedian(flux_stage1)
            valid_s1 = np.isfinite(norm_stage1)
            if valid_s1.any():
                lc.meta["stage1_rms"] = float(np.nanstd(norm_stage1[valid_s1]))
            else:
                lc.meta["stage1_rms"] = np.nan
        else:
            lc.meta["stage1_rms"] = np.nan

    # ==========================================
    # STAGE 4: Sanity check for over-flattening
    # ==========================================
    try:
        # Calculate relative depths to be invariant to normalisation scale
        pre_p50 = np.nanpercentile(flux, 50)
        pre_p1 = np.nanpercentile(flux, 1)
        pre_stage1_depth_rel = (pre_p50 - pre_p1) / pre_p50 if pre_p50 != 0 else 0
        
        post_p50 = np.nanpercentile(flux_final, 50)
        post_p1 = np.nanpercentile(flux_final, 1)
        post_stage2_depth_rel = (post_p50 - post_p1) / post_p50 if post_p50 != 0 else 0
        
        if pre_stage1_depth_rel > 0:
            depth_retention_ratio = post_stage2_depth_rel / pre_stage1_depth_rel
        else:
            depth_retention_ratio = 1.0
            
        possible_overflattening = depth_retention_ratio < 0.5
        
        if hasattr(lc, "meta"):
            lc.meta["depth_retention_ratio"] = float(depth_retention_ratio)
            lc.meta["possible_overflattening"] = bool(possible_overflattening)
            
        if possible_overflattening:
            logger.warning(f"Possible over-flattening — more than 50% of apparent transit depth was removed during detrending (retention={depth_retention_ratio:.2f}). Consider widening wotan_window_days for this star.")
    except Exception as e:
        logger.warning(f"Failed to calculate over-flattening check: {e}")

    # Set flux to flux_final for the rest of the pipeline
    flux = flux_final

    # Step 4: Normalise
    flux = normalise(flux)

    # Step 5: Remove remaining NaNs (mask consistently)
    valid = np.isfinite(flux) & np.isfinite(time) & np.isfinite(flux_err)
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid] / median_flux  # normalise errors using the same factor as flux

    return time, flux, flux_err


def fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve at a given period and epoch.

    Phase-folding "stacks" all transits on top of each other:
    - Compute phase = (time - t0) / period  (modulo 1)
    - Sort by phase
    - Optionally bin to reduce noise

    Args:
        time:    Time array (days)
        flux:    Normalised flux array
        period:  Orbital period (days)
        t0:      Reference transit epoch (BJD)
        n_bins:  Number of phase bins for binned output

    Returns:
        Tuple of (phase, flux) both sorted by phase.

    📚 LEARNING NOTE:
        A single transit might be too noisy to see clearly.
        But if we have 10 transits, we can "fold" the light curve
        so all 10 line up. The signal-to-noise improves by ~sqrt(10) ≈ 3x.

        Phase goes from -0.5 to 0.5:
          Phase = 0 → transit centre
          Phase = ±0.5 → halfway between transits (out-of-transit)
    """
    # Compute phase: range from -0.5 to +0.5
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0  # centre on transit

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]
