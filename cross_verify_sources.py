import lightkurve as lk
import numpy as np
import sys
sys.path.insert(0, r"c:\Users\gudae\Desktop\Learn_ml")
from advanced_vetting_suite import calculate_odd_even_mismatch, check_uv_shape_vshape

targets = {
    420914536: {"period": 0.736353, "epoch": 1625.50, "duration": 0.67, "sectors": [12]},
    432121978: {"period": 0.712305, "epoch": 1491.95, "duration": 0.58, "sectors": [7, 34]}
}

authors = ["SPOC", "QLP", "TGLC", "GSFC-ELEANOR-LITE"]

print("=" * 100)
print("             CROSS-SOURCE VETTING VERIFICATION FOR USP TARGETS")
print("=" * 100)

for tic_id, params in targets.items():
    print(f"\n🚀 [TARGET] TIC {tic_id}")
    for author in authors:
        try:
            print(f"  Searching for '{author}' data...")
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=params["sectors"], author=author)
            if len(search) == 0:
                print(f"    --> [NO DATA] No data products found for author {author}")
                continue
                
            print(f"    --> [FOUND] Downloading and stitching {len(search)} sectors...")
            lc_col = search.download_all()
            lc = lc_col.stitch().normalize().remove_nans().remove_outliers()
            
            time = lc.time.value
            flux = lc.flux.value
            
            # Diagnostics
            oe_verdict, oe_diff = calculate_odd_even_mismatch(time, flux, params["period"], params["epoch"], params["duration"])
            uv_verdict, flatness, uv_proof = check_uv_shape_vshape(time, flux, params["period"], params["epoch"], params["duration"])
            
            print(f"    --> [RESULTS] {author}:")
            print(f"        Odd-Even:       {oe_verdict}")
            print(f"        U/V Shape:      {uv_verdict}")
            print(f"        Proof Metrics:  {uv_proof}")
            
        except Exception as e:
            print(f"    --> [ERROR] Failed processing '{author}': {e}")
            
print("=" * 100)
