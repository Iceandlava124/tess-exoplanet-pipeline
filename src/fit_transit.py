"""
src/fit_transit.py
==================
Fit a physical transit model to a detected signal and estimate orbital parameters
with uncertainties.

📚 LEARNING NOTE:
    After we've DETECTED a transit (Phase 3) and CLASSIFIED it (Phase 5/6),
    the final step is to MEASURE its properties precisely.

    We use the batman package (Kreidberg 2015) which implements the exact
    Mandel & Agol (2002) transit light curve model — the same model used
    in published exoplanet papers!

    Parameters we estimate:
    ┌─────────────────────────────────────────────────────────┐
    │  P   — Orbital period (days)         ← from BLS          │
    │  t0  — Transit epoch (BJD)           ← from BLS          │
    │  Rp/Rs — Planet-to-star radius ratio ← from depth        │
    │  a/Rs — Orbital distance / star rad  ← from Kepler's 3rd │
    │  b   — Impact parameter (0=centre)   ← from shape        │
    │  u1,u2 — Limb darkening coefficients ← from stellar Teff │
    └─────────────────────────────────────────────────────────┘
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# Plain English: Query cache or MAST catalog for Teff and logg of the star.
def query_stellar_params(tic_id):
    """Query MAST catalog for Teff and logg of the star, with local SQLite caching."""
    try:
        # Try checking local SQLite cache first to avoid slow network queries
        from src.cache_manager import get_stellar_params, save_stellar_params
        cached_res = get_stellar_params(tic_id)
        if cached_res is not None:
            logger.info(f"Loaded cached stellar parameters for TIC {tic_id}: Teff={cached_res[0]}, logg={cached_res[1]}")
            return cached_res[0], cached_res[1]
    except Exception as ce:
        logger.warning(f"Failed to read stellar parameters from cache: {ce}")

    try:
        from astroquery.mast import Catalogs
        result = Catalogs.query_criteria(catalog="TIC", ID=int(tic_id))
        if result is not None and len(result) > 0:
            teff = result['Teff'][0]
            logg = result['logg'][0]
            # Handle possible masked/NaN values
            if teff is not None and not np.isnan(teff):
                teff = float(teff)
            else:
                teff = None
            if logg is not None and not np.isnan(logg):
                logg = float(logg)
            else:
                logg = None
                
            # Save the retrieved values to local cache
            try:
                save_stellar_params(tic_id, teff, logg)
            except Exception as ce:
                logger.warning(f"Failed to write stellar parameters to cache: {ce}")
                
            return teff, logg
    except Exception as e:
        logger.warning(f"Failed to query TIC stellar parameters for TIC {tic_id}: {e}")
    return None, None


# Plain English: Calculate limb darkening coefficients u1, u2 from Teff and logg using LDTK.
def get_limb_darkening_coefficients(teff, logg, metallicity=0.0):
    """
    Calculates quadratic limb darkening coefficients u1, u2
    from stellar parameters using the Claret (2017) tables.
    This significantly improves planet radius accuracy.
    """
    if teff is None or logg is None:
        return {
            "u1": 0.4,
            "u2": 0.26,
            "source": "fallback_solar"
        }
    try:
        from ldtk import LDPSetCreator, BoxcarFilter
        
        # Define TESS bandpass filter
        filters = [BoxcarFilter("TESS", 600, 1000)]
        
        # Create profile set with stellar parameters
        sc = LDPSetCreator(
            teff=(teff, 100),      # Teff ± 100K
            logg=(logg, 0.1),      # logg ± 0.1
            z=(metallicity, 0.1),  # metallicity ± 0.1
            filters=filters
        )
        
        ps = sc.create_profiles(nsamples=500)
        u1, u2 = ps.coeffs_qd(do_mc=False)
        
        return {
            "u1": float(u1[0][0]),
            "u2": float(u2[0][0]),
            "source": "ldtk_claret2017"
        }
        
    except Exception as e:
        # Fall back to solar-like values if ldtk fails
        logger.warning(f"ldtk failed: {e}. Using fallback solar values.")
        return {
            "u1": 0.4,
            "u2": 0.26,
            "source": "fallback_solar"
        }


def fit_batman_transit(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    init_params: Dict,
    n_bootstrap: int = 200,
) -> Dict:
    """
    Fit a batman transit model using scipy optimisation + bootstrap uncertainties.

    Args:
        time:        Time array (days)
        flux:        Normalised, detrended flux
        flux_err:    Flux uncertainty array
        init_params: Initial guess: {'period', 't0', 'depth', 'duration'}
        n_bootstrap: Number of bootstrap iterations for uncertainty

    Returns:
        Dict with best-fit parameters and 1-sigma uncertainties.

    📚 LEARNING NOTE — What is chi-squared minimisation?

        We want to find the model parameters (P, t0, Rp/Rs, etc.) that
        make the model match the data as closely as possible.

        We measure "closeness" using chi-squared (χ²):

            χ² = Σᵢ [ (data_i - model_i)² / error_i² ]

        When χ² is small, the model fits the data well.
        We use scipy's Nelder-Mead algorithm to MINIMISE χ² by
        adjusting the parameters.

        📚 LEARNING NOTE — What is bootstrapping?

        To get UNCERTAINTIES on our parameters, we use bootstrapping:
        1. Take our N data points
        2. Randomly resample N points WITH replacement → some appear twice,
           some not at all (this simulates "what if we ran the observation again?")
        3. Refit the model
        4. Repeat 200 times
        5. The standard deviation of the 200 results = our uncertainty!

        This is a non-parametric way to estimate uncertainties that doesn't
        require any assumptions about the error distribution.
    """
    try:
        import batman
    except ImportError:
        logger.error("batman-package not installed. Run: pip install batman-package")
        return _fallback_params(init_params)

    period = float(init_params.get("period", 1.0))
    t0 = float(init_params.get("t0", time[0]))
    depth = float(init_params.get("depth", 0.01))
    duration = float(init_params.get("duration", 0.1))

    # Convert depth to Rp/Rs ratio (depth = (Rp/Rs)²)
    rp_over_rs = float(np.sqrt(max(depth, 1e-6)))

    # Estimate a/Rs from period using Kepler's 3rd law (assuming solar-type star)
    # a/Rs ≈ (P/day)^(2/3) × 4.21 for solar parameters
    a_over_rs = max(1.5, 4.21 * (period ** (2.0 / 3.0)))

    # Query stellar parameters from TIC
    teff, logg = None, None
    tic_id = init_params.get("tic_id")
    if tic_id is not None:
        try:
            teff, logg = query_stellar_params(tic_id)
        except Exception as e:
            logger.warning(f"Failed to query stellar params: {e}")
            
    # Calculate limb darkening coefficients
    ld_res = get_limb_darkening_coefficients(teff, logg)
    u1 = ld_res["u1"]
    u2 = ld_res["u2"]
    ld_source = ld_res["source"]
    
    if ld_source == "fallback_solar":
        logger.warning(f"Using fallback solar limb darkening coefficients [0.4, 0.26] for TIC {tic_id}")

    # Set up batman parameter object
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_over_rs
    params.a = a_over_rs
    params.inc = 90.0          # inclination (degrees); 90 = edge-on
    params.ecc = 0.0           # eccentricity (circular orbit)
    params.w = 90.0            # argument of periapsis
    params.u = [u1, u2]        # quadratic limb darkening coefficients
    params.limb_dark = "quadratic"

    # Initial parameter vector for optimiser: [t0, rp, a, inc]
    x0 = np.array([t0, rp_over_rs, a_over_rs, 90.0])

    def chi_squared(x):
        """Compute χ² for a given parameter vector."""
        t0_, rp_, a_, inc_ = x
        if rp_ <= 0 or a_ <= 1 or not (60 < inc_ <= 90):
            return 1e10  # penalty for unphysical parameters

        params.t0 = t0_
        params.rp = rp_
        params.a = a_
        params.inc = inc_

        try:
            m = batman.TransitModel(params, time)
            model_flux = m.light_curve(params)
            residuals = (flux - model_flux) / np.maximum(flux_err, 1e-6)
            return float(np.sum(residuals ** 2))
        except Exception:
            return 1e10

    # Minimise chi-squared
    logger.info("Fitting batman transit model...")
    result = minimize(chi_squared, x0, method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})

    best_t0, best_rp, best_a, best_inc = result.x
    best_depth = best_rp ** 2
    best_dur = _compute_duration(period, best_rp, best_a, best_inc)
    chi2_dof = result.fun / max(1, len(time) - 4)

    # Bootstrap uncertainty estimation
    logger.info(f"Running {n_bootstrap} bootstrap iterations for uncertainties...")
    depths_bs, periods_bs, durs_bs = [], [], []

    for _ in range(n_bootstrap):
        # Sort indices to keep the time array strictly increasing (strictly required by batman model)
        idx = np.sort(np.random.choice(len(time), size=len(time), replace=True))
        t_bs, f_bs, e_bs = time[idx], flux[idx], flux_err[idx]

        def chi2_bs(x):
            t0_, rp_, a_, inc_ = x
            if rp_ <= 0 or a_ <= 1 or not (60 < inc_ <= 90):
                return 1e10
            params.t0 = t0_
            params.rp = rp_
            params.a = a_
            params.inc = inc_
            try:
                m = batman.TransitModel(params, t_bs)
                model_flux = m.light_curve(params)
                residuals = (f_bs - model_flux) / np.maximum(e_bs, 1e-6)
                return float(np.sum(residuals ** 2))
            except Exception:
                return 1e10

        res_bs = minimize(chi2_bs, result.x, method="Nelder-Mead",
                          options={"maxiter": 1000})
        t0_bs, rp_bs, a_bs, inc_bs = res_bs.x
        depths_bs.append(rp_bs ** 2)
        periods_bs.append(period)  # period fixed from BLS
        durs_bs.append(_compute_duration(period, rp_bs, a_bs, inc_bs))

    return {
        # Best-fit values
        "period": float(period),
        "period_err": float(np.std(periods_bs)),
        "t0": float(best_t0),
        "transit_depth": float(best_depth),
        "transit_depth_err": float(np.std(depths_bs)),
        "transit_depth_pct": float(best_depth * 100),
        "transit_duration_hr": float(best_dur * 24),
        "transit_duration_hr_err": float(np.std(durs_bs) * 24),
        "rp_over_rs": float(best_rp),
        "a_over_rs": float(best_a),
        "inclination": float(best_inc),
        # Goodness of fit
        "chi2_reduced": float(chi2_dof),
        # Derived
        "rp_earth": float(best_rp * 109.2),   # if Rstar ~ 1 Rsun
        "n_bootstrap": n_bootstrap,
        "u1": float(u1),
        "u2": float(u2),
        "limb_darkening_source": ld_source,
    }


def _compute_duration(period, rp, a, inc_deg):
    """Compute transit duration in days from orbital parameters."""
    inc_rad = np.radians(inc_deg)
    b = a * np.cos(inc_rad)  # impact parameter
    try:
        arg = np.sqrt((1 + rp) ** 2 - b ** 2) / (a * np.sin(inc_rad))
        return (period / np.pi) * np.arcsin(float(min(1.0, arg)))
    except Exception:
        return 0.1


def _fallback_params(init_params: Dict) -> Dict:
    """Return parameter estimates from BLS when batman fitting fails."""
    depth = float(init_params.get("depth", 0.0))
    return {
        "period": float(init_params.get("period", 0.0)),
        "period_err": 0.0,
        "t0": float(init_params.get("t0", 0.0)),
        "transit_depth": depth,
        "transit_depth_err": 0.0,
        "transit_depth_pct": depth * 100,
        "transit_duration_hr": float(init_params.get("duration", 0.0)) * 24,
        "transit_duration_hr_err": 0.0,
        "rp_over_rs": float(np.sqrt(depth)),
        "a_over_rs": 10.0,
        "inclination": 90.0,
        "chi2_reduced": 999.0,
        "rp_earth": float(np.sqrt(depth)) * 109.2,
        "n_bootstrap": 0,
        "u1": 0.4,
        "u2": 0.3,
        "limb_darkening_source": "fallback_failsafe",
    }
