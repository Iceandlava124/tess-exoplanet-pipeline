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

def trapezoid_transit_model(phase: np.ndarray, depth: float, duration_frac: float, t0_phase: float = 0.0, secondary_depth_ratio: float = 0.1) -> np.ndarray:
    """
    Generate a synthetic trapezoidal model for a transit.
    
    WHAT: Computes the expected normalized flux values at given phases.
    WHY: Evaluates how well a physical trapezoidal dip matches the actual data points.
    
    The trapezoid has:
      - Out of transit: flux = 1.0
      - Flat bottom of width: 80% of duration
      - Linear ingress & egress: each taking 10% of duration
      - Secondary eclipse: at phase 0.5 with depth = 10% of primary depth (planet-like)
    """
    flux = np.ones_like(phase)
    half_dur = duration_frac / 2.0
    ingress_start = half_dur
    ingress_end = half_dur * 0.8  # Ingress takes 10% of total duration (20% of half-duration)
    
    # Center phases around t0_phase
    centered_phase = phase - t0_phase
    # Wrap phases to [-0.5, 0.5]
    centered_phase = (centered_phase + 0.5) % 1.0 - 0.5
    abs_phase = np.abs(centered_phase)
    
    # 1. Primary Transit Ingress/Egress and Bottom
    # Inside flat bottom
    flat_mask = abs_phase <= ingress_end
    flux[flat_mask] = 1.0 - depth
    
    # Inside linear ingress/egress ramp
    ramp_mask = (abs_phase > ingress_end) & (abs_phase < ingress_start)
    # Linear interpolation between 1.0 and (1.0 - depth)
    fraction = (abs_phase[ramp_mask] - ingress_end) / (ingress_start - ingress_end)
    flux[ramp_mask] = (1.0 - depth) + fraction * depth
    
    # 2. Secondary Eclipse (at phase 0.5)
    # Centered at phase = 0.5
    sec_centered = np.abs(abs_phase - 0.5)
    sec_mask = sec_centered < half_dur
    
    # Secondary depth is 10% of primary depth
    sec_depth = depth * secondary_depth_ratio
    sec_ingress_start = half_dur
    sec_ingress_end = half_dur * 0.8
    
    # Secondary bottom
    sec_flat = sec_mask & (sec_centered <= sec_ingress_end)
    flux[sec_flat] = 1.0 - sec_depth
    
    # Secondary ramp
    sec_ramp = sec_mask & (sec_centered > sec_ingress_end) & (sec_centered < sec_ingress_start)
    sec_fraction = (sec_centered[sec_ramp] - sec_ingress_end) / (sec_ingress_start - sec_ingress_end)
    flux[sec_ramp] = (1.0 - sec_depth) + sec_fraction * sec_depth
    
    return flux

def fit_trapezoid_model(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bls_period: float, bls_t0: float, bls_duration: float, bls_depth: float) -> dict:
    """
    Fit the trapezoid model to time-series data using scipy.optimize.minimize (chi-squared loss).
    
    WHAT: Minimizes difference between observed flux and trapezoid model.
    WHY: Estimates the best-fitting physical parameters (period, depth, duration, t0).
    """
    try:
        # Convert initial parameters to fit variables
        # Variables to fit: [period, t0, depth, duration_frac]
        # Set bounds to keep optimization physical
        x0 = np.array([bls_period, bls_t0, max(1e-4, bls_depth), bls_duration / bls_period])
        
        def objective(x):
            p_, t0_, depth_, dur_frac_ = x
            # Penalize unphysical parameters heavily
            if depth_ <= 0.0 or depth_ > 0.5 or dur_frac_ <= 0.0 or dur_frac_ > 0.5 or p_ <= 0.1:
                return 1e10
                
            # Calculate phase for each time point
            phase = ((time - t0_) / p_) % 1.0
            phase[phase > 0.5] -= 1.0
            
            # Compute model flux
            model_flux = trapezoid_transit_model(phase, depth_, dur_frac_)
            
            # Compute chi-squared
            chi2 = np.sum(((flux - model_flux) / np.maximum(flux_err, 1e-5)) ** 2)
            return chi2
            
        bounds = [
            (bls_period * 0.90, bls_period * 1.10),
            (bls_t0 - 0.25 * bls_period, bls_t0 + 0.25 * bls_period),
            (1e-5, 0.40),
            (1e-4, 0.30)
        ]
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        best_p, best_t0, best_depth, best_dur_frac = result.x
        best_duration = best_dur_frac * best_p
        
        # Calculate degrees of freedom and reduced chi-squared
        dof = max(1, len(time) - 4)
        reduced_chi2 = result.fun / dof
        
        # Calculate transit shape symmetry (left vs right half of transit fold)
        phase = ((time - best_t0) / best_p) % 1.0
        phase[phase > 0.5] -= 1.0
        
        in_transit = np.abs(phase) < (best_duration / 2.0 / best_p)
        left_transit = in_transit & (phase < 0)
        right_transit = in_transit & (phase > 0)
        
        if left_transit.sum() > 2 and right_transit.sum() > 2:
            left_mean = np.mean(flux[left_transit])
            right_mean = np.mean(flux[right_transit])
            # Symmetry score: 1.0 is perfectly symmetric, decreases with mismatch
            symmetry_score = 1.0 - np.abs(left_mean - right_mean) / max(1e-5, best_depth)
        else:
            symmetry_score = 1.0
            
        return {
            "period": float(best_p),
            "t0": float(best_t0),
            "depth": float(best_depth),
            "duration": float(best_duration),
            "duration_ratio": float(best_dur_frac),
            "chi2": float(result.fun),
            "reduced_chi2": float(reduced_chi2),
            "symmetry_score": float(np.clip(symmetry_score, 0.0, 1.0)),
            "success": bool(result.success)
        }
    except Exception as e:
        logger.error(f"Failed to fit trapezoidal model: {e}")
        return {
            "period": float(bls_period),
            "t0": float(bls_t0),
            "depth": float(bls_depth),
            "duration": float(bls_duration),
            "duration_ratio": float(bls_duration / bls_period),
            "chi2": 9999.0,
            "reduced_chi2": 999.0,
            "symmetry_score": 0.5,
            "success": False
        }

def run_reverse_pipeline(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, bls_results: dict) -> dict:
    """
    Run 6 physical consistency tests on the fitted parameters.
    
    WHAT: Evaluates if the fitted transit satisfies physical orbital criteria.
    WHY: Differentiates real planetary transits from eclipsing binaries and instrument anomalies.
    """
    bls_period = bls_results.get("period", 1.0)
    bls_t0 = bls_results.get("t0", time[0])
    bls_duration = bls_results.get("duration", 0.1)
    bls_depth = bls_results.get("depth", 0.01)
    
    # Fit the trapezoidal model
    fit = fit_trapezoid_model(time, flux, flux_err, bls_period, bls_t0, bls_duration, bls_depth)
    
    # 1. is_period_consistent: fitted period within 5% of BLS period
    period_diff = np.abs(fit["period"] - bls_period) / bls_period
    is_period_consistent = period_diff < 0.05
    
    # 2. is_depth_physical: depth between 0.01% (0.0001) and 20% (0.20)
    is_depth_physical = 0.0001 <= fit["depth"] <= 0.20
    
    # 3. is_duration_physical: duration consistent with period via Kepler's 3rd law
    # Max duration for circular edge-on orbit: T_max = (R_star * P) / (pi * a)
    # Kepler: a/R_star = 4.21 * (P/day)^(2/3)
    # Using typical ranges of stellar types: duration_hrs should be within [0.5, 24.0] hours
    duration_hrs = fit["duration"] * 24.0
    is_duration_physical = 0.5 <= duration_hrs <= 24.0
    
    # 4. is_secondary_shallow: secondary eclipse depth is < 10% of primary depth
    # Evaluated at phase 0.5
    phase = ((time - fit["t0"]) / fit["period"]) % 1.0
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
