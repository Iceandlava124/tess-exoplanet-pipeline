import lightkurve as lk
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
try:
    from advanced_vetting_suite import calculate_odd_even_mismatch, check_uv_shape_vshape
except ImportError:
    def calculate_odd_even_mismatch(*args): return "N/A", 0.0
    def check_uv_shape_vshape(*args): return "N/A", 0.0, "N/A"

# All 10 flagged targets with their known parameters
flagged_targets = {
    287948915: {"period": 2.357955, "epoch": 1491.95, "duration": 4.0, "sectors": [14, 41, 54, 55, 81]},
    258285711: {"period": 7.510148, "epoch": 2800.28, "duration": 4.08, "sectors": [55, 82]},
    305512837: {"period": 3.021719, "epoch": 1491.95, "duration": 4.0, "sectors": [13, 39, 93, 100, 101]},
    319431206: {"period": 7.000774, "epoch": 2800.00, "duration": 4.14, "sectors": [91]},
    1717732429: {"period": 7.861320, "epoch": 1491.95, "duration": 4.0, "sectors": [81, 82, 83]},
    117843067: {"period": 2.376508, "epoch": 1491.95, "duration": 4.0, "sectors": [32, 98]},
    420914536: {"period": 0.736353, "epoch": 1625.50, "duration": 0.67, "sectors": [12]},
    285034141: {"period": 12.264925, "epoch": 1491.95, "duration": 4.0, "sectors": [58, 78, 85]},
    129198098: {"period": 2.979833, "epoch": 1491.95, "duration": 4.0, "sectors": [16, 56, 83]},
    432121978: {"period": 0.712305, "epoch": 1491.95, "duration": 0.58, "sectors": [7, 34]}
}

results = []

print("=" * 90)
print("             RUNNING COMPREHENSIVE ADVANCED VETTING ON ALL FLAGGED TARGETS")
print("=" * 90)

for tic_id, params in flagged_targets.items():
    print(f"\n[TARGET] Processing TIC {tic_id}...", flush=True)
    
    lc = None
    selected_author = "N/A"
    
    # 1. Download & Stitch logic (Tries SPOC, QLP, TGLC, Eleanor)
    authors_to_try = ["SPOC", "QLP", "TGLC", "GSFC-ELEANOR-LITE"]
    
    for author in authors_to_try:
        try:
            # We download up to 3 sectors to stitch for multi-sector targets
            secs_to_download = params["sectors"][:3]
            print(f"  --> Searching for author '{author}' in sectors {secs_to_download}...")
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=secs_to_download, author=author)
            
            if len(search) > 0:
                print(f"  --> Downloading and stitching {len(search)} sectors...")
                lc_col = search.download_all()
                lc = lc_col.stitch().normalize().remove_nans().remove_outliers()
                selected_author = author
                break
        except Exception as e:
            pass
            
    if lc is None:
        print(f"  [ERROR] Could not retrieve any light curve data for TIC {tic_id}.")
        results.append({
            "TIC ID": tic_id,
            "Author": "N/A",
            "Sectors": "N/A",
            "Odd-Even Check": "N/A",
            "U/V Shape Check": "N/A",
            "Final Disposition": "No Data Available"
        })
        continue

    try:
        time = lc.time.value
        flux = lc.flux.value
        
        # 2. Run diagnostics
        oe_verdict, oe_diff = calculate_odd_even_mismatch(time, flux, params["period"], params["epoch"], params["duration"])
        uv_verdict, flatness, uv_proof = check_uv_shape_vshape(time, flux, params["period"], params["epoch"], params["duration"])
        
        # Determine Final Disposition
        if "FAIL" in oe_verdict or "FAIL" in uv_verdict:
            disp = "False Positive (Eclipsing Binary)"
        else:
            disp = "Planet Candidate"
            
        results.append({
            "TIC ID": tic_id,
            "Author": selected_author,
            "Sectors": ", ".join(map(str, params["sectors"][:3])),
            "Odd-Even Check": oe_verdict,
            "U/V Shape Check": uv_verdict,
            "U/V Proof": uv_proof,
            "Final Disposition": disp
        })
        
        print(f"  [RESULT] Odd-Even: {oe_verdict} | U/V Shape: {uv_verdict} | Verdict: {disp}")
        print(f"  [PROOF] {uv_proof}")
    except Exception as e:
        print(f"  [ERROR] Vetting failed: {e}")
        results.append({
            "TIC ID": tic_id,
            "Author": selected_author,
            "Sectors": ", ".join(map(str, params["sectors"][:3])),
            "Odd-Even Check": "Error",
            "U/V Shape Check": "Error",
            "U/V Proof": "Error",
            "Final Disposition": f"Vetting Error: {str(e)}"
        })

print("\n" + "=" * 90)
print("                                MASTER VETTING REPORT")
print("=" * 90)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("=" * 90)
df.to_csv("master_advanced_vetting_results.csv", index=False)
