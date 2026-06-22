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
    nans = np.isnan(flux)
    if nans.all():
        return flux

    x = np.arange(len(flux))
    flux_interp = flux.copy()
    flux_interp[nans] = np.interp(x[nans], x[~nans], flux[~nans])

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
    try:
        if hasattr(lc, "pdcsap_flux"):
            flux = lc.pdcsap_flux.value
            flux_err = lc.pdcsap_flux_err.value
        else:
            flux = lc.flux.value
            flux_err = lc.flux_err.value
        time = lc.time.value
    except Exception:
        flux = np.array(lc.flux)
        flux_err = np.ones_like(flux) * np.nanstd(flux) * 0.01
        time = np.arange(len(flux), dtype=float)

    # Calculate median flux before normalisation to normalise errors consistently
    median_flux = np.nanmedian(flux)
    if abs(median_flux) < 1e-10:
        median_flux = 1.0

    # Step 1: Replace inf/nan in flux_err
    flux_err = np.where(np.isfinite(flux_err), flux_err, np.nanmedian(flux_err))

    # Step 2: Sigma clip
    flux = sigma_clip(flux, sigma=sigma_clip_threshold)

    # ==========================================
    # STAGE 1: Wotan biweight — robust rough detrend
    # ==========================================
    # Removes large-scale stellar trends, resistant to remaining outliers
    wotan_window = 0.5
    wotan_trend_removed = False
    flux_stage1 = flux.copy()
    try:
        flux_stage1, wotan_trend = detrend_wotan_biweight(
            time, flux, window_length=wotan_window
        )
        wotan_trend_removed = True
        logger.info(f"Stage 1 Detrending: wotan biweight (window={wotan_window}d) successful.")
    except Exception as e:
        logger.warning(f"Stage 1 Wotan detrending failed: {e}. Skipping straight to Stage 2.")

    # ==========================================
    # STAGE 2: Adaptive Savitzky-Golay — fine-tuned per-star pass
    # ==========================================
    # Uses the EXISTING Lomb-Scargle logic to measure this star's own
    # variability period, then sizes the Savgol window accordingly.
    # This now runs on the wotan-cleaned flux, not the raw flux.
    p_var = np.nan
    w_days = np.nan
    try:
        from astropy.timeseries import LombScargle
        # Filter out NaNs from time and flux for LS periodogram calculation
        ls_mask = np.isfinite(time) & np.isfinite(flux_stage1)
        t_ls = time[ls_mask]
        f_ls = flux_stage1[ls_mask]
        if len(t_ls) > 100:
            # Search frequency range corresponding to periods between 0.1 and 10 days
            frequency, power = LombScargle(t_ls, f_ls).autopower(
                minimum_frequency=1.0/10.0, 
                maximum_frequency=1.0/0.1
            )
            best_freq = frequency[np.argmax(power)]
            p_var = 1.0 / best_freq
            
            # Optimal window is 3x the variability timescale, capped between 0.5 and 2.0 days
            # Increased minimum from 0.1 to 0.5 days to prevent the "Detrending Trap" from erasing transits.
            w_days = np.clip(3 * p_var, 0.5, 2.0)
            dt = np.nanmedian(np.diff(time))
            if dt > 0:
                adaptive_window = int(w_days / dt)
                if adaptive_window % 2 == 0:
                    adaptive_window += 1
                # Savitzky-Golay window must be greater than polyorder (default 3)
                if adaptive_window < 5:
                    adaptive_window = 5
                
                logger.info(f"Stage 2 Adaptive Detrending: Stellar variability period={p_var:.2f}d -> Window={w_days:.2f}d ({adaptive_window} points)")
                window_length = adaptive_window
    except Exception as e:
        logger.warning(f"Failed to calculate adaptive detrending window: {e}. Falling back to default window of {window_length} points.")

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
