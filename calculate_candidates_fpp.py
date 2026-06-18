import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Monkeypatch numpy.int to avoid deprecation errors in pytransit / triceratops
np.int = int

# Disable global SSL verification to prevent stev.oapd.inaf.it connection errors
import ssl
import requests
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()

# Monkeypatch mechanicalsoup and requests to bypass SSL verification completely
from mechanicalsoup import StatefulBrowser
original_open = StatefulBrowser.open
def patched_open(self, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_open(self, url, *args, **kwargs)
StatefulBrowser.open = patched_open

original_submit = StatefulBrowser.submit
def patched_submit(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_submit(self, *args, **kwargs)
StatefulBrowser.open = patched_open
StatefulBrowser.submit = patched_submit

original_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request


try:
    import lightkurve as lk
except ImportError:
    print("Error: lightkurve is not installed. Run 'pip install lightkurve'")
    sys.exit(1)

try:
    import triceratops.triceratops as tr
except ImportError:
    print("Error: triceratops is not installed. Run 'pip install triceratops'")
    sys.exit(1)

# Pipeline parameters for the four candidates
candidates_info = {
    258285711: {"period": 7.510148, "depth": 0.010217, "duration": 0.169841},
    319431206: {"period": 7.000774, "depth": 0.024885, "duration": 0.172391},
    420914536: {"period": 0.736353, "depth": 0.005198, "duration": 0.027784},
    432121978: {"period": 0.712305, "depth": 0.007122, "duration": 0.024168}
}

results = []

print("=" * 70)
print("  TESS Exoplanet Candidates Vetting: TRICERATOPS FPP / NFPP Calculations")
print("=" * 70)

for tic_id, params in candidates_info.items():
    print(f"\nProcessing TIC {tic_id}...")
    try:
        # Find TESS observation sectors using lightkurve search_lightcurve (QLP FFI cutouts)
        print(f"  --> Searching for observation sectors...")
        lc_search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(lc_search) == 0:
            print(f"  [ERROR] No observations found for TIC {tic_id} on MAST. Skipping.")
            results.append({
                "TIC ID": tic_id,
                "Sector": "N/A",
                "Gaia Background Stars": 0,
                "FPP": np.nan,
                "NFPP": np.nan,
                "Final Status": "NO_SECTORS_FOUND"
            })
            continue

        # Extract unique sectors and search for any downloadable light curve, starting from most recent
        sectors = list(set([int(s) for s in lc_search.table['sequence_number']]))
        sectors.sort(reverse=True)
        
        lc = None
        sector = None
        print(f"  --> Found sectors: {sectors}. Searching for downloadable light curve...")
        for sec in sectors:
            lc_matches = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sec, author="SPOC")
            if len(lc_matches) == 0:
                lc_matches = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sec, author="QLP")
            if len(lc_matches) > 0:
                try:
                    lc = lc_matches[0].download().normalize().remove_nans().remove_outliers()
                    sector = sec
                    print(f"  --> Successfully downloaded light curve for Sector {sector}")
                    break
                except Exception as e:
                    print(f"  [Warning] Failed to download Sector {sec}: {e}")
        
        if lc is None:
            print(f"  [ERROR] Could not download any light curve for TIC {tic_id}. Skipping.")
            continue
        time = lc.time.value
        flux = lc.flux.value
        flux_err = lc.flux_err.value if hasattr(lc, "flux_err") else np.ones_like(flux) * np.std(flux)
        
        # Initialize triceratops target directly (which internally downloads TessCut for FFI sectors)
        print(f"  --> Initializing TRICERATOPS target (downloading FFI cutouts)...")
        target = tr.target(ID=tic_id, sectors=[sector])
        
        # Setup transit parameters and compute depths
        depth_ppt = params["depth"] * 1000.0 if params["depth"] < 1.0 else params["depth"] / 1000.0
        print(f"  --> Calculating depths (Period: {params['period']:.4f} d, Depth: {depth_ppt:.4f} ppt)...")
        target.calc_depths(tdepth=depth_ppt)
        
        # Calculate probabilities with the light curve arrays
        print(f"  --> Querying Gaia DR3 for stars in field and calculating scenario probabilities...")
        target.calc_probs(
            time=time,
            flux_0=flux,
            flux_err_0=flux_err,
            P_orb=params["period"]
        )
        
        fpp = target.FPP
        nfpp = target.NFPP
        gaia_stars = len(target.stars) if hasattr(target, "stars") else 0
        
        status = "VALIDATED" if (fpp + nfpp) < 0.01 else "VIABLE" if (fpp + nfpp) < 0.1 else "FALSE_POSITIVE"
        print(f"  [SUCCESS] Complete! FPP: {fpp:.5f}, NFPP: {nfpp:.5f}, Status: {status}")
        
        results.append({
            "TIC ID": tic_id,
            "Sector": sector,
            "Gaia Background Stars": gaia_stars,
            "FPP": fpp,
            "NFPP": nfpp,
            "Final Status": status
        })
        
    except Exception as e:
        print(f"  [ERROR] Error processing TIC {tic_id}: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "TIC ID": tic_id,
            "Sector": "Error",
            "Gaia Background Stars": 0,
            "FPP": np.nan,
            "NFPP": np.nan,
            "Final Status": f"ERROR: {str(e)}"
        })

print("\n" + "=" * 70)
print("                           FINAL VETTING VERDICTS")
print("=" * 70)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("=" * 70)
