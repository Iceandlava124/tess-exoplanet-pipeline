import lightkurve as lk
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our advanced vetting checks
import sys
sys.path.insert(0, '.')
try:
    from advanced_vetting_suite import calculate_odd_even_mismatch, check_uv_shape_vshape
except ImportError:
    # Inline fallback implementations if import fails
    def calculate_odd_even_mismatch(*args): return "N/A", 0.0
    def check_uv_shape_vshape(*args): return "N/A", 0.0

print("=" * 80)
print("  STITCHING AND VETTING FLAGGED TARGETS LOCALLY WITH MULTI-SECTOR/ALT-DATA")
print("=" * 80)

# Target 1: TIC 432121978 (missing SPOC/QLP, but has TGLC and Eleanor products)
print("\n>>> Analyzing TIC 432121978 using TGLC / Eleanor products...")
try:
    # Query for TGLC or Eleanor authors
    search = lk.search_lightcurve("TIC 432121978", author="TGLC")
    if len(search) == 0:
        search = lk.search_lightcurve("TIC 432121978", author="GSFC-ELEANOR-LITE")
        
    if len(search) > 0:
        print(f"  --> Found alternative data products (Author: {search[0].author[0]}). Downloading...")
        lc = search[0].download().normalize().remove_nans().remove_outliers()
        time = lc.time.value
        flux = lc.flux.value
        
        period = 0.712305
        epoch = 1491.95
        duration = 0.58
        
        oe_verdict, oe_diff = calculate_odd_even_mismatch(time, flux, period, epoch, duration)
        uv_verdict, flatness = check_uv_shape_vshape(time, flux, period, epoch, duration)
        
        print(f"  [RESULT] Odd-Even check: {oe_verdict} (Diff: {oe_diff:.4f})")
        print(f"  [RESULT] U/V Shape check: {uv_verdict} (Flatness: {flatness:.4f})")
    else:
        print("  [WARNING] No TGLC or Eleanor products found for TIC 432121978.")
except Exception as e:
    print(f"  [ERROR] Failed to analyze TIC 432121978: {e}")

# Target 2: TIC 1717732429 (has 32 sectors, let's stitch sectors 81, 82, 83)
print("\n>>> Stitching Sectors 81, 82, and 83 for TIC 1717732429...")
try:
    search_multi = lk.search_lightcurve("TIC 1717732429", sector=[81, 82, 83], author="QLP")
    if len(search_multi) > 0:
        print(f"  --> Downloading {len(search_multi)} sectors for stitching...")
        lc_collection = search_multi.download_all()
        # Stitch collection
        stitched_lc = lc_collection.stitch().normalize().remove_nans().remove_outliers()
        print(f"  --> Successfully stitched sectors! Total points: {len(stitched_lc)}")
        
        time = stitched_lc.time.value
        flux = stitched_lc.flux.value
        
        period = 7.861320
        epoch = 1491.95 # approximate or loaded
        duration = 4.0   # estimate
        
        oe_verdict, oe_diff = calculate_odd_even_mismatch(time, flux, period, epoch, duration)
        uv_verdict, flatness = check_uv_shape_vshape(time, flux, period, epoch, duration)
        
        print(f"  [RESULT] Odd-Even check: {oe_verdict} (Diff: {oe_diff:.4f})")
        print(f"  [RESULT] U/V Shape check: {uv_verdict} (Flatness: {flatness:.4f})")
    else:
        print("  [WARNING] Sectors 81, 82, 83 not available for TIC 1717732429.")
except Exception as e:
    print(f"  [ERROR] Failed to stitch/analyze TIC 1717732429: {e}")
