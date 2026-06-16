"""
src/features.py
===============
Extract machine-learning features from detected transit signals.

📚 LEARNING NOTE:
    Raw light curves are time series with ~20,000 data points.
    We could feed this directly into a neural network (we will in Phase 6!),
    but for classical ML (Random Forest), we need to extract
    a fixed set of meaningful numbers — called FEATURES.

    Good features are:
    ✅ Informative (actually different between classes)
    ✅ Invariant (same value regardless of star brightness)
    ✅ Robust (not sensitive to noise)

    We extract ~30 features covering:
    - Transit shape (depth, duration, shape asymmetry)
    - Periodicity (BLS power, period, secondary eclipse)
    - Statistical (out-of-transit scatter, χ², skewness)
    - Stellar (temperature, radius from TIC catalog)
"""

import logging
from typing import Dict, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    bls_params: Dict,
    stellar_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Extract all features from a preprocessed light curve + BLS result.

    Args:
        time:           Time array (days)
        flux:           Normalised, detrended flux
        flux_err:       Flux uncertainty
        bls_params:     Output from detect.run_bls() — best-fit parameters
        stellar_params: Optional dict with Teff, Rstar, Mstar from TIC

    Returns:
        Dict of feature_name → float value

    📚 LEARNING NOTE:
        Think of features like this:
        Instead of describing a person by 1000 photos (raw data),
        you describe them with 30 numbers: height, weight, eye color code,
        hair length, etc. (features). The Random Forest then learns
        which numbers separate "planet" from "not planet."
    """
    features = {}
    period = bls_params.get("period", 1.0)
    t0 = bls_params.get("t0", time[0])
    duration = bls_params.get("duration", 0.1)

    # Ensure duration is in same units as time (days)
    if duration == 0:
        duration = 0.1  # fallback

    # ── Phase masks ──────────────────────────────────────────────────────────
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    half_dur_phase = (duration / 2.0) / period
    in_transit = np.abs(phase) < half_dur_phase
    out_of_transit = np.abs(phase) > 2 * half_dur_phase

    flux_in = flux[in_transit]
    flux_out = flux[out_of_transit]

    # ── Feature 1-5: BLS direct outputs ──────────────────────────────────────
    features["bls_period"] = float(period)
    features["bls_power"] = float(bls_params.get("power", 0.0))
    features["bls_snr"] = float(bls_params.get("snr", 0.0))
    features["transit_depth"] = float(bls_params.get("depth", 0.0))
    features["transit_duration_hrs"] = float(duration * 24.0)

    # ── Feature 6-10: Out-of-transit statistics ───────────────────────────────
    features["oot_rms"] = float(np.nanstd(flux_out)) if len(flux_out) > 5 else 1.0
    features["oot_skewness"] = float(stats.skew(flux_out)) if len(flux_out) > 5 else 0.0
    features["oot_kurtosis"] = float(stats.kurtosis(flux_out)) if len(flux_out) > 5 else 0.0
    features["oot_median"] = float(np.nanmedian(flux_out)) if len(flux_out) > 5 else 1.0
    features["n_transits"] = float(np.ceil((time[-1] - time[0]) / period)) if period > 0 else 0

    # ── Feature 11-15: Transit shape ─────────────────────────────────────────
    if len(flux_in) > 2:
        features["transit_mean_depth"] = float(
            np.nanmedian(flux_out) - np.nanmedian(flux_in)
        )
        features["transit_min"] = float(np.nanmin(flux_in))
        features["transit_scatter"] = float(np.nanstd(flux_in))

        # Shape: V-shaped (binary) vs U-shaped (planet transit)
        # Sort in-transit flux by phase distance from centre
        in_phase = np.abs(phase[in_transit])
        if len(in_phase) > 4:
            sort_idx = np.argsort(in_phase)
            flux_sorted = flux_in[sort_idx]
            # Curvature: if centre < edges → U-shaped (planet)
            #            if centre > edges → V-shaped (binary)
            mid = len(flux_sorted) // 2
            edge_mean = np.mean(flux_sorted[:max(1, mid//2)])
            centre_mean = np.mean(flux_sorted[mid:])
            features["shape_curvature"] = float(edge_mean - centre_mean)
        else:
            features["shape_curvature"] = 0.0
    else:
        features["transit_mean_depth"] = 0.0
        features["transit_min"] = 1.0
        features["transit_scatter"] = 0.0
        features["shape_curvature"] = 0.0

    # ── Feature 16-20: Secondary eclipse test (EB discriminator) ────────────
    # Check dip at phase = 0.5 (secondary eclipse position for EB)
    phase_sec = ((time - (t0 + period / 2)) / period) % 1.0
    phase_sec[phase_sec > 0.5] -= 1.0
    in_secondary = np.abs(phase_sec) < half_dur_phase

    if in_secondary.sum() > 2 and len(flux_out) > 5:
        sec_depth = np.nanmedian(flux_out) - np.nanmedian(flux[in_secondary])
        features["secondary_depth"] = float(sec_depth)
        features["secondary_to_primary_ratio"] = float(
            sec_depth / max(1e-6, features["transit_mean_depth"])
        )
    else:
        features["secondary_depth"] = 0.0
        features["secondary_to_primary_ratio"] = 0.0

    # ── Feature 21-25: Odd-even transit depth difference ─────────────────────
    # In eclipsing binaries, alternating transits have different depths.
    # This test splits transits into "odd" and "even" numbered ones.
    transit_epochs = np.round((time[in_transit] - t0) / period).astype(int)
    odd_mask = in_transit & (np.isin(np.round((time - t0) / period).astype(int),
                                      transit_epochs[transit_epochs % 2 == 1]))
    even_mask = in_transit & (np.isin(np.round((time - t0) / period).astype(int),
                                       transit_epochs[transit_epochs % 2 == 0]))

    odd_depth = (np.nanmedian(flux_out) - np.nanmedian(flux[odd_mask])
                 if odd_mask.sum() > 2 else 0.0)
    even_depth = (np.nanmedian(flux_out) - np.nanmedian(flux[even_mask])
                  if even_mask.sum() > 2 else 0.0)

    features["odd_depth"] = float(odd_depth)
    features["even_depth"] = float(even_depth)
    features["odd_even_diff"] = float(abs(odd_depth - even_depth))
    features["odd_even_ratio"] = float(
        abs(odd_depth - even_depth) / max(1e-6, max(odd_depth, even_depth))
    )

    # ── Feature 26-30: Stellar parameters (from TIC catalog) ─────────────────
    if stellar_params:
        features["teff"] = float(stellar_params.get("Teff", 5500.0))
        features["rstar"] = float(stellar_params.get("rad", 1.0))
        features["mstar"] = float(stellar_params.get("mass", 1.0))
        features["tmag"] = float(stellar_params.get("Tmag", 10.0))
        features["contamination"] = float(stellar_params.get("contratio", 0.0))
    else:
        features["teff"] = 5500.0
        features["rstar"] = 1.0
        features["mstar"] = 1.0
        features["tmag"] = 10.0
        features["contamination"] = 0.0

    return features


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dict to a numpy array (for sklearn/tensorflow).

    The ORDER of features matters — it must be consistent between
    training and inference. We use sorted keys to ensure this.
    """
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float32)


FEATURE_NAMES = [
    "bls_period", "bls_power", "bls_snr", "transit_depth",
    "transit_duration_hrs", "oot_rms", "oot_skewness", "oot_kurtosis",
    "oot_median", "n_transits", "transit_mean_depth", "transit_min",
    "transit_scatter", "shape_curvature", "secondary_depth",
    "secondary_to_primary_ratio", "odd_depth", "even_depth",
    "odd_even_diff", "odd_even_ratio", "teff", "rstar", "mstar",
    "tmag", "contamination",
]
