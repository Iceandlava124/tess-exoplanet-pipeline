"""
src/detect.py
=============
Transit detection using Box Least Squares (BLS) periodogram.

📚 LEARNING NOTE:
    After preprocessing, we need to FIND the transits automatically.
    We can't look at 20,000 light curves by eye!

    The main algorithm we use is BLS (Box Least Squares), which was
    invented specifically for finding planetary transits.
    It works by trying every possible combination of:
      - Period P (e.g., 1.0 days, 1.1 days, ..., 30 days)
      - Transit duration T (e.g., 1 hour, 2 hours, ..., 12 hours)
      - Transit epoch t0 (when does the first transit occur?)
    and finding the combination that best explains the dips in the data.

    The result is called a "periodogram" — a plot of "how well does
    this period explain the data?" as a function of period.
"""

import logging
from typing import Tuple, Optional, Dict

import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u

logger = logging.getLogger(__name__)

# Known TESS systematic alias periods (days).
# Signals peaking within ALIAS_TOLERANCE of these values are almost certainly
# instrumental artefacts caused by the 13.5-day orbital period, momentum
# dumps at 1-day multiples, or half-day beating with the spacecraft clock.
TESS_ALIAS_PERIODS = [0.5, 1.0, 2.0, 3.125, 6.25, 13.5, 27.0]
ALIAS_TOLERANCE    = 0.01   # days

try:
    # Try importing transitleastsquares
    from transitleastsquares import transitleastsquares, cleaned_array, transit_mask
    TLS_AVAILABLE = True
except ImportError:
    TLS_AVAILABLE = False


def _is_alias(period: float) -> bool:
    """
    Check if a period is close to a known TESS systematic alias or its harmonics.
    Rejects period if it's within tolerance of alias or simple fractions/multiples.
    """
    for alias in TESS_ALIAS_PERIODS:
        if abs(period - alias) < ALIAS_TOLERANCE:
            return True
        # Check harmonics: 1/2, 1/3, 2/3, 2/1, 3/1
        for factor in [0.5, 1.0/3.0, 2.0/3.0, 2.0, 3.0]:
            if abs(period - alias * factor) < ALIAS_TOLERANCE:
                return True
    return False


def run_tls(time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    TLS is specifically designed for planet-shaped transit signals.
    It uses a physical transit model instead of a box,
    finding 10-20% more small planets than BLS.
    Drop-in replacement for run_bls() — same inputs, same output format.
    """
    # Fallback to BLS if transitleastsquares is not available
    if not TLS_AVAILABLE:
        logger.warning("transitleastsquares not available. Falling back to run_bls.")
        return run_bls(time, flux, flux_err)

    try:
        # Clean the arrays first — TLS requires no NaN values
        time_clean, flux_clean = cleaned_array(time, flux)
        
        # Run TLS with automatic period grid
        model = transitleastsquares(time_clean, flux_clean)
        results = model.power(
            minimum_period=0.5,
            maximum_period=20.0,
            show_progress_bar=False
        )
        
        # Get best period — skip if alias
        best_period = float(results.period)
        
        # Reject known systematic aliases and harmonics
        if _is_alias(best_period):
            # Mask out the alias period and find next peak
            mask = transit_mask(time_clean, best_period, results.duration, results.T0)
            
            # Re-run on masked data if alias detected
            model2 = transitleastsquares(time_clean[~mask], flux_clean[~mask])
            results2 = model2.power(
                minimum_period=0.5,
                maximum_period=20.0,
                show_progress_bar=False
            )
            
            if results2.SDE < 5.0 or _is_alias(float(results2.period)):
                # No clean period found — discard target
                results_dict = {
                    "status": "alias_discard",
                    "reason": f"Best period {best_period:.4f}d is a known alias",
                    "period": best_period,
                    "snr": float(results.SDE),
                    "alias_rejected": True,
                    "epoch": float(results.T0),
                    "t0": float(results.T0),
                    "duration": float(results.duration),
                    "depth": 1.0 - float(results.depth),
                }
                return results.periods, results.power, results_dict
                
            results = results2
            best_period = float(results.period)
        
        # Check SDE (Signal Detection Efficiency) — TLS equivalent of SNR
        if results.SDE < 5.0:
            results_dict = {
                "status": "below_threshold",
                "period": best_period,
                "snr": float(results.SDE),
                "alias_rejected": False,
                "epoch": float(results.T0),
                "t0": float(results.T0),
                "duration": float(results.duration),
                "depth": 1.0 - float(results.depth),
            }
            return results.periods, results.power, results_dict
        
        # Check for secondary eclipse
        sec_check = check_secondary_eclipse(time_clean, flux_clean, best_period, float(results.T0), float(results.duration))
        secondary_depth = sec_check.get("secondary_depth", 0.0)

        # Return in same format as old run_bls() so nothing else breaks
        results_dict = {
            "status": "ok",
            "period": float(results.period),
            "epoch": float(results.T0),
            "t0": float(results.T0),
            "duration": float(results.duration),
            "depth": 1.0 - float(results.depth),
            "snr": float(results.SDE),
            "power_spectrum": {
                "periods": results.periods.tolist(),
                "power": results.power.tolist()
            },
            "odd_even_mismatch": float(results.odd_even_mismatch) if hasattr(results, "odd_even_mismatch") and results.odd_even_mismatch is not None else 0.0,
            "secondary_depth": float(secondary_depth),
            "transit_count": int(results.transit_count) if hasattr(results, "transit_count") and results.transit_count is not None else 0,
            "distinct_transit_count": int(results.distinct_transit_count) if hasattr(results, "distinct_transit_count") and results.distinct_transit_count is not None else 0,
            "alias_rejected": False,
            "raw_results": results
        }
        return results.periods, results.power, results_dict
        
    except Exception as e:
        logger.error(f"run_tls failed: {e}. Falling back to run_bls.")
        return run_bls(time, flux, flux_err)


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.5,
    period_max: float = 27.0,
    n_periods: int = 50_000,
    duration_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run the Box Least Squares algorithm to search for periodic transit signals.

    Args:
        time:          Time array (days)
        flux:          Normalised flux (should be ~1.0 out-of-transit)
        flux_err:      Flux uncertainty array (optional)
        period_min:    Minimum period to search (days)
        period_max:    Maximum period to search (days)
        n_periods:     Number of periods to test
        duration_grid: Transit durations to test (hours). Default: 1-12 hours.

    Returns:
        periods:     Array of tested periods (days)
        power:       BLS power at each period (higher = better fit)
        best_params: Dict with best-fit period, duration, depth, t0

    📚 LEARNING NOTE — How BLS works:
        Imagine you have a light curve with a dip every 3 days.
        BLS tries to fit a "box" (rectangle) model to the data:

              Flux
              1.0 ──────┐     ┌────────┐     ┌───
                        │     │        │     │
              0.99       └─────┘        └─────┘
                         Transit        Transit
                        (period = 3 days)

        For each period P and duration T it tries, it computes how
        much the data is IMPROVED by adding a box dip.
        The period with the HIGHEST improvement is the best candidate.

        BLS power ≈ SNR² of the transit signal.
        A power > 9 (SNR > 3) is typically a "detection".
    """
    if flux_err is None:
        flux_err = np.ones_like(flux) * np.std(flux) * 0.01

    if duration_grid is None:
        # Dynamically limit max duration to be strictly less than period_min to prevent astropy errors
        max_duration_hr = min(12.0, (period_min * 24.0) - 0.1)
        duration_candidates = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        duration_grid = duration_candidates[duration_candidates < max_duration_hr]
        if len(duration_grid) == 0:
            duration_grid = np.array([max_duration_hr / 2.0])
        duration_grid = duration_grid / 24.0  # in days

    # Create a log-spaced period grid (more resolution at short periods)
    periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), n_periods))

    # Astropy BLS implementation
    bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)

    try:
        periodogram = bls.power(
            periods * u.day,
            duration_grid * u.day,
            method="fast",
            objective="snr",
        )
        power = np.array(periodogram.power)
    except Exception as e:
        logger.error(f"BLS failed: {e}")
        return periods, np.zeros_like(periods), {
            "period": float(period_min),
            "power": 0.0,
            "snr": 0.0,
            "depth": 0.0,
            "duration": 0.1,
            "t0": float(time[0]),
        }

    # ── Helper: extract detailed stats at a given period index ──────────────
    def get_stat(stats, key, default_val):
        if isinstance(stats, dict):
            val = stats.get(key, default_val)
        elif hasattr(stats, "get"):
            val = stats.get(key, default_val)
        elif hasattr(stats, key):
            val = getattr(stats, key)
        else:
            try:
                val = stats[key]
            except Exception:
                val = default_val
        if isinstance(val, (list, np.ndarray, tuple)):
            return val[0] if len(val) > 0 else default_val
        return val

    def _params_at_idx(idx: int) -> dict:
        """Build a best_params dict for the peak at *idx* in the power array."""
        p = float(periods[idx])
        try:
            stats = bls.compute_stats(
                p * u.day,
                duration_grid[0] * u.day,
                0.1 * u.day,
            )
            return {
                "period":   p,
                "power":    float(power[idx]),
                "snr":      float(np.sqrt(max(0, power[idx]))),
                "depth":    float(get_stat(stats, "depth", 0.0)),
                "duration": float(get_stat(stats, "duration", 0.1)),
                "t0":       float(get_stat(stats, "transit_time", time[0])),
            }
        except Exception as e:
            logger.warning(f"Failed to compute BLS stats at period {p:.4f} d: {e}")
            return {
                "period":   p,
                "power":    float(power[idx]),
                "snr":      float(np.sqrt(max(0, power[idx]))),
                "depth":    0.0,
                "duration": 0.1,
                "t0":       float(time[0]),
            }

    # ── TESS Alias Rejection ─────────────────────────────────────────────────
    # Walk through periods sorted by BLS power (strongest first). Skip any
    # period that falls within ALIAS_TOLERANCE of a known TESS systematic.
    # Accept the first clean period whose SNR is still above snr_min (= 3.0).
    # If none survives, flag the star for immediate DISCARD.
    sorted_idx    = np.argsort(power)[::-1]   # indices: highest power first
    snr_min       = 3.0                        # minimum acceptable SNR for any alternative

    best_params   = None
    alias_hit     = False      # did the top period hit an alias?

    for rank, idx in enumerate(sorted_idx):
        candidate_period = float(periods[idx])
        candidate_snr    = float(np.sqrt(max(0, power[idx])))

        if _is_alias(candidate_period):
            if rank == 0:
                alias_hit = True
                logger.warning(
                    f"BLS top period {candidate_period:.4f} d is a known TESS alias "
                    f"(within {ALIAS_TOLERANCE} d of {TESS_ALIAS_PERIODS}) -- rejected."
                )
            continue   # skip this alias peak

        if candidate_snr < snr_min:
            # Once SNR drops below floor, nothing useful remains
            break

        # First clean peak found
        best_params = _params_at_idx(idx)
        break

    # No clean peak above SNR floor survived alias rejection
    if best_params is None:
        logger.warning(
            "Alias rejection: no clean period above SNR floor survived. "
            "Returning alias_rejected=True for immediate DISCARD."
        )
        # Return the (rejected) top period so the caller has something to log
        fallback = _params_at_idx(int(sorted_idx[0]))
        fallback["alias_rejected"] = True
        return periods, power, fallback

    best_params["alias_rejected"] = alias_hit   # True if we had to skip the top hit

    if alias_hit:
        logger.info(
            f"Alias rejection: using next-best clean period {best_params['period']:.4f} d "
            f"(SNR: {best_params['snr']:.2f})."
        )
    else:
        logger.info(
            f"BLS best period: {best_params['period']:.4f} d | "
            f"SNR: {best_params['snr']:.2f} | "
            f"Depth: {best_params['depth']*100:.3f}%"
        )

    return periods, power, best_params


def compute_snr(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """
    Compute the Signal-to-Noise Ratio of a transit signal.

    SNR = (transit depth) / (scatter in out-of-transit flux)
        = depth / (RMS noise / sqrt(n_in_transit))

    📚 LEARNING NOTE:
        SNR is one of the most important concepts in signal detection.

        Signal = how deep is the transit?
        Noise  = how much does the light curve fluctuate normally?

        SNR = Signal / Noise

        SNR < 3  → probably just noise, ignore it
        SNR 3-7  → marginal detection, needs further investigation
        SNR > 7  → strong detection, likely real
        SNR > 15 → very strong signal

        In real TESS data, we typically require SNR > 7.1 (the
        standard 7.1-sigma threshold used by the TESS pipeline).

        The key insight: SNR improves with MORE transits.
        If one transit has SNR=3, then 9 transits gives SNR ≈ 3*sqrt(9) = 9.
        This is why long-baseline missions (more transits) find more planets!
    """
    # In-transit mask
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    half_dur = duration / (2.0 * period)
    in_transit = np.abs(phase) < half_dur
    out_of_transit = ~in_transit

    if in_transit.sum() < 2 or out_of_transit.sum() < 10:
        return 0.0

    depth = np.nanmedian(flux[out_of_transit]) - np.nanmedian(flux[in_transit])
    noise = np.nanstd(flux[out_of_transit]) / np.sqrt(in_transit.sum())

    return float(depth / noise) if noise > 0 else 0.0


def check_secondary_eclipse(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> Dict:
    """
    Check for a secondary eclipse at phase = 0.5 (halfway between transits).

    Eclipsing binaries produce TWO dips per orbit:
    - Primary eclipse at phase = 0.0 (deeper)
    - Secondary eclipse at phase = 0.5 (shallower)

    True planet transits only show ONE dip (planets don't emit enough
    light to cause a secondary eclipse detectable by TESS).

    📚 LEARNING NOTE:
        This is one of our most powerful FALSE POSITIVE discriminators.

        If we see a dip at BOTH phase=0 and phase=0.5, it's almost
        certainly an ECLIPSING BINARY (two stars orbiting each other),
        not a planet. The "odd-even depth difference" is another
        related test: if alternating dips have different depths,
        it's a binary with unequal-brightness eclipses.

        This shows how DOMAIN KNOWLEDGE makes ML features powerful.
        A generic ML model might not know to look for this,
        but if WE compute the secondary eclipse depth as a FEATURE,
        the model can use it for discrimination.
    """
    # Phase at secondary eclipse position (phase = 0.5)
    t0_secondary = t0 + period / 2.0
    half_dur = duration / 2.0

    phase = ((time - t0_secondary) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    in_secondary = np.abs(phase) < (half_dur / period)
    out_of_transit = np.abs(phase) > (2 * half_dur / period)

    if in_secondary.sum() < 2 or out_of_transit.sum() < 10:
        return {"secondary_depth": 0.0, "secondary_snr": 0.0, "is_eb_candidate": False}

    primary_depth = 1.0 - np.nanmedian(
        flux[np.abs(((time - t0) / period) % 1.0 - 0.0) < (half_dur / period)]
    )
    secondary_depth = np.nanmedian(flux[out_of_transit]) - np.nanmedian(flux[in_secondary])
    noise = np.nanstd(flux[out_of_transit])

    secondary_snr = secondary_depth / noise if noise > 0 else 0.0

    # EB candidate if secondary depth is > 10% of primary depth AND SNR > 3
    is_eb = (secondary_depth > 0.1 * primary_depth) and (secondary_snr > 3.0)

    return {
        "secondary_depth": float(secondary_depth),
        "secondary_snr": float(secondary_snr),
        "is_eb_candidate": bool(is_eb),
    }
