import os
import json
import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
import astropy.constants as const
from astroquery.mast import Catalogs
import warnings
warnings.filterwarnings('ignore')

def run_kinematic_duration_check(period_days, r_star_sun, m_star_sun, obs_duration_hours):
    """
    Calculates the maximum theoretical transit duration for a circular orbit
    and compares it to the observed duration.
    """
    P = period_days * u.day
    # Fallbacks for missing stellar parameters
    if r_star_sun is None or np.isnan(r_star_sun):
        r_star_sun = 1.0
    R_s = r_star_sun * u.Rsun
    
    m_val = m_star_sun if (m_star_sun is not None and not np.isnan(m_star_sun)) else r_star_sun
    M_s = m_val * u.Msun
    
    # Kepler's Third Law to find semi-major axis (a)
    a = (((const.G * M_s * P**2) / (4 * np.pi**2))**(1/3)).to(u.Rsun)
    
    # Velocity of the planet assuming a circular orbit
    v = (2 * np.pi * a / P).to(u.Rsun / u.hour)
    
    # Max duration occurs at an impact parameter of b = 0
    max_duration = (2 * R_s / v).value 
    
    verdict = "PASS"
    if obs_duration_hours > (max_duration * 1.1): # 10% buffer
        verdict = "FAIL (Duration Unphysical)"
        
    return {
        "Stellar Radius (R_sun)": r_star_sun,
        "Stellar Mass (M_sun)": m_star_sun,
        "Theoretical Max Duration (hr)": max_duration,
        "Observed Duration (hr)": obs_duration_hours,
        "Kinematic Verdict": verdict
    }

def run_sweet_test(tic_id, period_days, epoch_btjd, transit_duration_hours):
    """
    Masks out the transits and runs Lomb-Scargle to check if the star 
    pulsates at the same frequency or harmonics.
    """
    # Download FFI Light Curve
    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="QLP")
    
    if len(search) == 0:
        return {
            "Dominant Period (d)": np.nan,
            "SWEET Verdict": "ERROR: No Light Curve Found"
        }

    try:
        lc = search[-1].download().normalize().remove_nans().remove_outliers()
        
        # Create transit mask
        duration_days = (transit_duration_hours / 24.0) * 1.2
        transit_mask = np.abs((lc.time.value - epoch_btjd + period_days/2) % period_days - period_days/2) < duration_days
        
        oot_lc = lc[~transit_mask]
        pg = oot_lc.to_periodogram(method='lombscargle', minimum_period=0.1, maximum_period=30)
        dominant_period = pg.period_at_max_power.value
        
        ratio = dominant_period / period_days
        verdict = "PASS"
        if np.isclose(ratio, 1.0, atol=0.01) or np.isclose(ratio, 0.5, atol=0.01) or np.isclose(ratio, 2.0, atol=0.01):
            verdict = "FAIL (Phase-locked)"
            
        return {
            "Dominant Period (d)": dominant_period,
            "SWEET Verdict": verdict
        }
    except Exception as e:
        return {
            "Dominant Period (d)": np.nan,
            "SWEET Verdict": f"ERROR: {str(e)}"
        }

if __name__ == "__main__":
    report_dir = r"C:\Users\gudae\Desktop\Learn_ml\results_output\results\FLAG"
    targets = [258285711, 319431206, 420914536, 432121978]
    
    # Standard query-based details if JSON is missing or incomplete
    backup_params = {
        258285711: {"period": 7.510148, "t0": 2800.28, "depth": 0.010217, "duration": 4.08},
        319431206: {"period": 7.000774, "t0": 2800.0, "depth": 0.024885, "duration": 4.14},
        420914536: {"period": 0.736353, "t0": 1625.50, "depth": 0.005198, "duration": 0.67},
        432121978: {"period": 0.712305, "t0": 1491.95, "depth": 0.007122, "duration": 0.58}
    }
    
    results = []
    
    print("=" * 80)
    print("  RUNNING SECONDARY DIAGNOSTICS SUITE (SWEET + KINEMATIC DURATION CHECKS)")
    print("=" * 80)
    
    for tic_id in targets:
        print(f"\nProcessing TIC {tic_id}...")
        
        # Load transit params from json report if available
        json_path = os.path.join(report_dir, f"TIC_{tic_id}_report.json")
        period = backup_params[tic_id]["period"]
        t0 = backup_params[tic_id]["t0"]
        duration = backup_params[tic_id]["duration"]
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    tp = data.get("transit_parameters", {})
                    period = tp.get("period", period)
                    t0 = tp.get("t0", t0)
                    duration = tp.get("transit_duration_hr", duration)
                    print(f"  --> Loaded parameters from JSON report.")
            except Exception as e:
                print(f"  [Warning] Failed to load JSON for TIC {tic_id}: {e}")
        else:
            print(f"  --> Using backup catalog parameters.")
            
        # Query MAST/TIC for stellar parameters
        r_star = None
        m_star = None
        try:
            print(f"  --> Querying MAST/TIC for stellar parameters...")
            tic_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
            if len(tic_data) > 0:
                r_star = tic_data['rad'][0]
                m_star = tic_data['mass'][0]
                print(f"  --> Retrieved: R_star = {r_star} Rsun, M_star = {m_star} Msun")
        except Exception as e:
            print(f"  [Warning] Failed to query TIC catalog: {e}")
            
        # Kinematic duration check
        kin_res = run_kinematic_duration_check(period, r_star, m_star, duration)
        
        # SWEET check
        sweet_res = run_sweet_test(tic_id, period, t0, duration)
        
        row = {
            "TIC ID": tic_id,
            "Period (d)": period,
            "Epoch (BTJD)": t0,
            "Obs Dur (hr)": duration,
            "R_star": r_star,
            "M_star": m_star,
            "Max Dur (hr)": kin_res["Theoretical Max Duration (hr)"],
            "Kinematic Verdict": kin_res["Kinematic Verdict"],
            "Stellar Period (d)": sweet_res["Dominant Period (d)"],
            "SWEET Verdict": sweet_res["SWEET Verdict"]
        }
        results.append(row)
        print(f"  [RESULT] Kinematic: {row['Kinematic Verdict']} | SWEET: {row['SWEET Verdict']}")
        
    print("\n" + "=" * 80)
    print("                      SECONDARY VETTING SUMMARY TABLE")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 80)
    df.to_csv("secondary_vetting_verdicts.csv", index=False)
