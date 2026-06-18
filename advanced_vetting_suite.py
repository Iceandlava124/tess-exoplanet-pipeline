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
    1. Uses a median-based out-of-transit normalization (robust against outliers).
    2. Performs a classic model comparison by fitting a Trapezoid (U-shape) 
       and a Triangle (V-shape) to compare the Sum of Squared Residuals (SSR).
    """
    duration_days = duration_hours / 24.0
    phase = ((time - epoch + period/2) % period) - period/2
    
    # Extract only the window containing the transit
    window = np.abs(phase) < (duration_days * 1.5)
    t_win = phase[window]
    f_win = flux[window]
    
    if len(t_win) < 10:
        return "N/A (No Data)", 1.0, "N/A"
        
    # 1. Median-based standardization (robust against detrending outliers)
    oot_median = np.median(f_win)
    f_normalized_med = (f_win - oot_median) / (np.min(f_win) - oot_median)
    
    # 2. Mean-based standardization (sensitive to outliers)
    oot_mean = np.mean(f_win)
    f_normalized_mean = (f_win - oot_mean) / (np.min(f_win) - oot_mean)
    
    # --- Check 1: Core Flatness Ratio (using robust median-normalized data) ---
    in_transit_points = np.abs(t_win) < (duration_days / 2)
    bottom_points = f_normalized_med[in_transit_points] > 0.8
    flatness_ratio = np.sum(bottom_points) / len(bottom_points) if len(bottom_points) > 0 else 0.0
    
    # --- Check 2: Model Comparison (Trapezoid vs Triangle) ---
    # Define models normalized to [0, 1] dip depth
    def trapezoid_model(t, width, ingress):
        t_abs = np.abs(t)
        y = np.zeros_like(t)
        y[t_abs > width/2] = 0.0
        slope_mask = (t_abs <= width/2) & (t_abs > (width/2 - ingress))
        y[slope_mask] = (width/2 - t_abs[slope_mask]) / ingress
        y[t_abs <= (width/2 - ingress)] = 1.0
        return y
        
    def triangle_model(t, width):
        t_abs = np.abs(t)
        y = np.zeros_like(t)
        y[t_abs > width/2] = 0.0
        slope_mask = t_abs <= width/2
        y[slope_mask] = 1.0 - (t_abs[slope_mask] / (width/2))
        return y

    # Fit under Median Normalization
    try:
        popt_u_med, _ = curve_fit(trapezoid_model, t_win, f_normalized_med, p0=[duration_days, duration_days*0.2], bounds=([0, 0], [duration_days*3, duration_days]))
        popt_v_med, _ = curve_fit(triangle_model, t_win, f_normalized_med, p0=[duration_days], bounds=([0], [duration_days*3]))
        ssr_u_med = np.sum((f_normalized_med - trapezoid_model(t_win, *popt_u_med))**2)
        ssr_v_med = np.sum((f_normalized_med - triangle_model(t_win, *popt_v_med))**2)
        fit_ratio_med = ssr_u_med / ssr_v_med if ssr_v_med > 0 else 1.0
    except Exception:
        fit_ratio_med = 1.0

    # Fit under Mean Normalization
    try:
        popt_u_mean, _ = curve_fit(trapezoid_model, t_win, f_normalized_mean, p0=[duration_days, duration_days*0.2], bounds=([0, 0], [duration_days*3, duration_days]))
        popt_v_mean, _ = curve_fit(triangle_model, t_win, f_normalized_mean, p0=[duration_days], bounds=([0], [duration_days*3]))
        ssr_u_mean = np.sum((f_normalized_mean - trapezoid_model(t_win, *popt_u_mean))**2)
        ssr_v_mean = np.sum((f_normalized_mean - triangle_model(t_win, *popt_v_mean))**2)
        fit_ratio_mean = ssr_u_mean / ssr_v_mean if ssr_v_mean > 0 else 1.0
    except Exception:
        fit_ratio_mean = 1.0

    verdict = "PASS (U-Shaped)"
    if flatness_ratio < 0.25 and fit_ratio_med >= 0.85:
        verdict = f"FAIL (V-Shaped EB, SSR U/V ratio (med): {fit_ratio_med:.2f})"
        
    proof_str = f"U/V Ratio (Median-Norm): {fit_ratio_med:.2f} | U/V Ratio (Mean-Norm): {fit_ratio_mean:.2f} (Median OOT: {oot_median:.6f}, Mean OOT: {oot_mean:.6f})"
    return verdict, flatness_ratio, proof_str

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

