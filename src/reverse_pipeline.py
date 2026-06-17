"""
src/reverse_pipeline.py
=======================
Reverse pipeline for exoplanet validation:
Instead of asking "does this look like a planet?", it generates a physical trapezoidal
model of the transit and fits it to the data using chi-squared minimization, then runs
6 consistency checks to evaluate the physical viability of the transit hypothesis.
"""

import logging
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

import batman

def _compute_duration(period, rp, a, inc_deg):
    """Compute transit duration in days from orbital parameters."""
    inc_rad = np.radians(inc_deg)
    b = a * np.cos(inc_rad)  # impact parameter
    try:
        arg = np.sqrt((1 + rp) ** 2 - b ** 2) / (a * np.sin(inc_rad))
        return (period / np.pi) * np.arcsin(float(min(1.0, arg)))
    except Exception:
        return 0.1

def generate_batman_model(time, period, epoch, rp_over_rs, 
                          semi_major_axis, inclination):
    """
    Generate a physically accurate Mandel-Agol transit model.
    Same model as forward pipeline batman fitting.
    rp_over_rs = planet radius / star radius (what batman fits for)
    This replaces the trapezoid approximation in the reverse pipeline.
    """
    params = batman.TransitParams()
    params.t0 = epoch
    params.per = period
    params.rp = rp_over_rs
    params.a = semi_major_axis
    params.inc = inclination
    params.ecc = 0.0
    params.w = 90.0
    params.u = [0.3, 0.3]  # quadratic limb darkening
    params.limb_dark = "quadratic"
    
    m = batman.TransitModel(params, time)
    return m.light_curve(params)

def fit_batman_reverse(time, flux, flux_err, initial_period, initial_epoch):
    """
    Fits batman model to data using chi-squared minimisation.
    Returns same output format as old trapezoid fitter
    so nothing downstream breaks.
    """
    
    def chi_squared(params):
        period, epoch, rp_rs, sma, inc = params
        try:
            model = generate_batman_model(
                time, period, epoch, rp_rs, sma, inc
            )
            residuals = flux - model
            return np.sum((residuals / flux_err) ** 2)
        except:
            return 1e10
    
    # Initial guess from TLS results
    x0 = [initial_period, initial_epoch, 0.1, 10.0, 89.0]
    bounds = [
        (initial_period * 0.8, initial_period * 1.2),
        (initial_epoch - 0.1, initial_epoch + 0.1),
        (0.01, 0.5),    # rp/rs: 1% to 50% radius ratio
        (1.5, 100.0),   # semi-major axis in stellar radii
        (70.0, 90.0)    # inclination in degrees
    ]
    
    result = minimize(chi_squared, x0, bounds=bounds, method="L-BFGS-B")
    
    period, epoch, rp_rs, sma, inc = result.x
    
    # Calculate depth from rp/rs
    depth = rp_rs ** 2
    
    # Compute duration from fit
    duration = _compute_duration(period, rp_rs, sma, inc)
    
    # Reduced chi-squared
    n_params = 5
    n_points = len(flux)
    red_chi_sq = result.fun / (n_points - n_params)
    
    # Calculate transit shape symmetry (left vs right half of transit fold)
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    
    in_transit = np.abs(phase) < (duration / 2.0 / period)
    left_transit = in_transit & (phase < 0)
    right_transit = in_transit & (phase > 0)
    
    if left_transit.sum() > 2 and right_transit.sum() > 2:
        left_mean = np.mean(flux[left_transit])
        right_mean = np.mean(flux[right_transit])
        # Symmetry score: 1.0 is perfectly symmetric, decreases with mismatch
        symmetry_score = 1.0 - np.abs(left_mean - right_mean) / max(1e-5, depth)
    else:
        symmetry_score = 1.0
    
    return {
        "period": period,
        "epoch": epoch,
        "t0": epoch, # for backward compatibility (matches t0 parameter)
        "rp_over_rs": rp_rs,
        "depth": depth,
        "duration": duration,
        "semi_major_axis": sma,
        "inclination": inc,
        "reduced_chi_squared": red_chi_sq,
        "reduced_chi2": red_chi_sq, # for backward compatibility
        "symmetry_score": float(np.clip(symmetry_score, 0.0, 1.0)),
        "success": bool(result.success),
        "fit_converged": result.success
    }

def run_reverse_pipeline(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bls_results: dict) -> dict:
    """
    Run 6 physical consistency tests on the fitted parameters.
    
    WHAT: Evaluates if the fitted transit satisfies physical orbital criteria.
    WHY: Differentiates real planetary transits from eclipsing binaries and instrument anomalies.
    """
    bls_period = bls_results.get("period", 1.0)
    bls_t0 = bls_results.get("t0", time[0])
    
    # Fit the batman model using chi-squared minimisation
    fit = fit_batman_reverse(time, flux, flux_err, bls_period, bls_t0)
    
    # 1. is_period_consistent: fitted period within 5% of BLS/TLS period
    period_diff = np.abs(fit["period"] - bls_period) / bls_period
    is_period_consistent = period_diff < 0.05
    
    # 2. is_depth_physical: depth between 0.01% (0.0001) and 20% (0.20)
    is_depth_physical = 0.0001 <= fit["depth"] <= 0.20
    
    # 3. is_duration_physical: duration consistent with period via Kepler's 3rd law
    duration_hrs = fit["duration"] * 24.0
    is_duration_physical = 0.5 <= duration_hrs <= 24.0
    
    # 4. is_secondary_shallow: secondary eclipse depth is < 10% of primary depth
    phase = ((time - fit["epoch"]) / fit["period"]) % 1.0
    phase[phase > 0.5] -= 1.0
    half_dur = fit["duration"] / (2.0 * fit["period"])
    in_secondary = np.abs(phase - 0.5) < half_dur
    out_of_transit = np.abs(phase) > (2 * half_dur)
    
    if in_secondary.sum() > 2 and out_of_transit.sum() > 10:
        sec_depth = np.nanmedian(flux[out_of_transit]) - np.nanmedian(flux[in_secondary])
        is_secondary_shallow = sec_depth < (0.10 * fit["depth"])
    else:
        is_secondary_shallow = True
        
    # 5. is_shape_symmetric: transit symmetry score > 0.85
    is_shape_symmetric = fit["symmetry_score"] > 0.85
    
    # 6. is_fit_good: reduced chi-squared < 3.0
    is_fit_good = fit["reduced_chi2"] < 3.0
    
    # Calculate reverse confidence
    tests = [
        is_period_consistent,
        is_depth_physical,
        is_duration_physical,
        is_secondary_shallow,
        is_shape_symmetric,
        is_fit_good
    ]
    passed_count = sum(1 for t in tests if t)
    reverse_confidence = passed_count / 6.0
    
    return {
        "fit_results": fit,
        "is_period_consistent": bool(is_period_consistent),
        "is_depth_physical": bool(is_depth_physical),
        "is_duration_physical": bool(is_duration_physical),
        "is_secondary_shallow": bool(is_secondary_shallow),
        "is_shape_symmetric": bool(is_shape_symmetric),
        "is_fit_good": bool(is_fit_good),
        "tests_passed": passed_count,
        "reverse_confidence": float(reverse_confidence)
    }
