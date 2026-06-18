import numpy as np
import lightkurve as lk
import astropy.units as u
import astropy.constants as const
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def calculate_odd_even_mismatch(time, flux, period, epoch, duration_hours):
    """
    Compares the transit depths of odd-numbered and even-numbered transits.
    A significant difference indicates an eclipsing binary (EB).
    """
    duration_days = duration_hours / 24.0
    
    # Identify epoch offsets for each transit index
    phase = (time - epoch) / period
    transit_indices = np.round(phase)
    
    # Mask to isolate in-transit data
    in_transit = np.abs(phase - transit_indices) < (duration_days / (2 * period))
    
    odd_mask = (transit_indices % 2 == 1) & in_transit
    even_mask = (transit_indices % 2 == 0) & in_transit
    oot_mask = ~in_transit
    
    if not np.any(odd_mask) or not np.any(even_mask):
        return "N/A (Insufficient Transits)", 0.0
        
    oot_level = np.median(flux[oot_mask])
    odd_depth = oot_level - np.median(flux[odd_mask])
    even_depth = oot_level - np.median(flux[even_mask])
    
    # Calculate depth difference ratio
    avg_depth = (odd_depth + even_depth) / 2
    if avg_depth <= 0:
        return "N/A (No Dips)", 0.0
        
    diff = np.abs(odd_depth - even_depth) / avg_depth
    
    verdict = "PASS"
    if diff > 0.3:  # Depth difference greater than 30% indicates an EB
        verdict = f"FAIL (Odd depth: {odd_depth:.4f}, Even depth: {even_depth:.4f})"
        
    return verdict, diff

def check_uv_shape_vshape(time, flux, period, epoch, duration_hours):
    """
    Evaluates whether the transit is U-shaped (planetary) or V-shaped (binary).
    Uses the ratio of the transit core duration to the overall duration.
    """
    duration_days = duration_hours / 24.0
    phase = ((time - epoch + period/2) % period) - period/2
    
    # Extract only the window containing the transit
    window = np.abs(phase) < (duration_days * 1.5)
    t_win = phase[window]
    f_win = flux[window]
    
    if len(t_win) < 10:
        return "N/A (No Data)", 1.0
        
    # Standardize dip
    f_normalized = (f_win - np.median(f_win)) / (np.min(f_win) - np.median(f_win))
    
    # Measure the fraction of points in the flat bottom (value < 0.2 of max depth)
    # vs points in the ingress/egress boundaries
    in_transit_points = np.abs(t_win) < (duration_days / 2)
    bottom_points = f_normalized[in_transit_points] > 0.8
    
    if len(bottom_points) == 0:
        return "N/A", 1.0
        
    flatness_ratio = np.sum(bottom_points) / len(bottom_points)
    
    verdict = "PASS (U-Shaped)"
    if flatness_ratio < 0.25:  # Sharp, narrow bottom
        verdict = "FAIL (V-Shaped Grazing/EB)"
        
    return verdict, flatness_ratio

def check_centroid_offset(tic_id, sector):
    """
    Downloads the Target Pixel File (TPF) and checks if the photocenter (centroid)
    moves during the transit.
    """
    print(f"  --> Downloading TPF for Sector {sector} centroid check...")
    tpf_search = lk.search_targetpixelfile(f"TIC {tic_id}", mission="TESS", sector=sector)
    
    if len(tpf_search) == 0:
        return "N/A (No TPF Available)", 0.0
        
    try:
        tpf = tpf_search[0].download()
        
        # Estimate centroids using the standard lightkurve moments method
        col_centroids, row_centroids = tpf.estimate_centroids()
        
        # Calculate standard deviation of photocenter movement
        std_col = np.std(col_centroids.value - np.nanmedian(col_centroids.value))
        std_row = np.std(row_centroids.value - np.nanmedian(row_centroids.value))
        total_drift = np.sqrt(std_col**2 + std_row**2)
        
        verdict = "PASS"
        if total_drift > 0.15:  # Drift greater than 0.15 pixels indicates background contamination
            verdict = "FAIL (High Centroid Drift)"
            
        return verdict, total_drift
    except Exception as e:
        return f"ERROR ({str(e)})", 0.0

if __name__ == "__main__":
    print("=" * 80)
    print("  ADVANCED TRANSIT VETTING SUITE: ODD-EVEN, U/V SHAPE & CENTROID DIAGNOSTICS")
    print("=" * 80)
    
    candidates_info = {
        258285711: {"period": 7.510148, "epoch": 2800.28, "duration": 4.08, "sector": 82},
        319431206: {"period": 7.000774, "epoch": 2800.00, "duration": 4.14, "sector": 91},
        420914536: {"period": 0.736353, "epoch": 1625.50, "duration": 0.67, "sector": 12},
        432121978: {"period": 0.712305, "epoch": 1491.95, "duration": 0.58, "sector": 34}
    }
    
    for tic_id, params in candidates_info.items():
        print(f"\n[TARGET] Analyzing TIC {tic_id}...")
        
        # Download lightcurve
        print("  --> Fetching light curve...")
        lc_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
        if len(lc_search) == 0:
            lc_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="QLP")
            
        if len(lc_search) == 0:
            print("  [ERROR] No light curve found for target. Skipping.")
            continue
            
        try:
            lc = lc_search[-1].download().normalize().remove_nans().remove_outliers()
            time = lc.time.value
            flux = lc.flux.value
            
            # 1. Odd-Even Depth check
            oe_verdict, oe_diff = calculate_odd_even_mismatch(time, flux, params["period"], params["epoch"], params["duration"])
            print(f"  [1] Odd-Even Check: {oe_verdict} (Mismatch Diff: {oe_diff:.4f})")
            
            # 2. U-shape vs V-shape check
            uv_verdict, flatness = check_uv_shape_vshape(time, flux, params["period"], params["epoch"], params["duration"])
            print(f"  [2] U/V Shape Check: {uv_verdict} (Flatness ratio: {flatness:.4f})")
            
            # 3. Centroid Drift check
            centroid_verdict, drift = check_centroid_offset(tic_id, params["sector"])
            print(f"  [3] Centroid Shift Check: {centroid_verdict} (Mean Drift: {drift:.4f} px)")
            
        except Exception as e:
            print(f"  [ERROR] Failed during analysis: {e}")

