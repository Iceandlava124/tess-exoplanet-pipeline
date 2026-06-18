import pandas as pd
import numpy as np
import urllib.request
import json
import os
import sqlite3

# Import download utility to fetch the latest xCTL target list
try:
    import sys
    sys.path.insert(0, '.')
    from src.download import download_xctl
    # Download the latest xCTL catalog and TOI catalog
    download_xctl()
except Exception as e:
    print(f"Warning: could not import/run download_xctl: {e}")

# Combined master list of all 19 flagged stars from both runs
all_flags = [
    # First Run Flags
    {"tic_id": 219698950, "period": 3.422417, "depth": 0.000486, "source": "Run 1"},
    {"tic_id": 233720539, "period": 2.298249, "depth": 0.001273, "source": "Run 1"},
    {"tic_id": 214243287, "period": 5.756117, "depth": 0.000174, "source": "Run 1"},
    {"tic_id": 458857720, "period": 1.440117, "depth": 0.000793, "source": "Run 1"},
    {"tic_id": 75878355,  "period": 8.306342, "depth": 0.000287, "source": "Run 1"},
    {"tic_id": 391903064, "period": 24.069732, "depth": 0.002087, "source": "Run 1"},
    {"tic_id": 442530946, "period": 3.712990, "depth": 0.000241, "source": "Run 1"},
    {"tic_id": 144043410, "period": 0.493043, "depth": 0.004082, "source": "Run 1"},
    {"tic_id": 237808867, "period": 0.567291, "depth": 0.000640, "source": "Run 1"},
    # Second Run Flags
    {"tic_id": 287948915, "period": 2.357955, "depth": 0.010195, "source": "Run 2"},
    {"tic_id": 258285711, "period": 7.510148, "depth": 0.010217, "source": "Run 2"},
    {"tic_id": 305512837, "period": 3.021719, "depth": 0.006495, "source": "Run 2"},
    {"tic_id": 319431206, "period": 7.000774, "depth": 0.024885, "source": "Run 2"},
    {"tic_id": 1717732429,"period": 7.861320, "depth": 0.011304, "source": "Run 2"},
    {"tic_id": 117843067, "period": 2.376508, "depth": 0.008254, "source": "Run 2"},
    {"tic_id": 420914536, "period": 0.736353, "depth": 0.005198, "source": "Run 2"},
    {"tic_id": 285034141, "period": 12.264925, "depth": 0.024942, "source": "Run 2"},
    {"tic_id": 129198098, "period": 2.979833, "depth": 0.008440, "source": "Run 2"},
    {"tic_id": 432121978, "period": 0.712305, "depth": 0.007122, "source": "Run 2"}
]

print(f"Combining and verifying {len(all_flags)} total flags...", flush=True)

toi_path = "data/xctl/toi_catalog.csv"
df_toi = pd.read_csv(toi_path) if os.path.exists(toi_path) else None
if df_toi is not None:
    df_toi['TIC ID'] = pd.to_numeric(df_toi['TIC ID'], errors="coerce")

import time
def query_mast_rest_tic(tic_id):
    # Enforce a huge delay to prevent MAST API rate limits
    time.sleep(5.0)
    url = "https://mast.stsci.edu/api/v0/invoke"
    request_data = {
        "service": "Mast.Catalogs.Filtered.Tic",
        "format": "json",
        "params": {
            "columns": "ID,Teff,rad,mass",
            "filters": [
                {"paramName": "ID", "values": [int(tic_id)]}
            ]
        }
    }
    req_body = f"request={json.dumps(request_data)}".encode("utf-8")
    req = urllib.request.Request(url, data=req_body, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode('utf-8'))
            if "data" in res and len(res["data"]) > 0:
                row = res["data"][0]
                return row.get("rad"), row.get("Teff")
    except Exception as e:
        pass
    return None, None

results = []

for item in all_flags:
    tic = item["tic_id"]
    period = item["period"]
    depth = item["depth"]
    source = item["source"]
    
    # Check TOI catalog match
    toi_num = "New Candidate"
    official_disp = "N/A"
    
    if df_toi is not None:
        matches = df_toi[df_toi['TIC ID'] == tic]
        if len(matches) > 0:
            toi_num = ", ".join(matches['TOI'].astype(str).tolist())
            dispositions = matches['TFOPWG Disposition'].fillna("N/A").tolist()
            tess_dispositions = matches['TESS Disposition'].fillna("N/A").tolist()
            disp_strs = [f"{d}/{t}" for d, t in zip(dispositions, tess_dispositions)]
            official_disp = ", ".join(disp_strs)
            
    # Get Stellar specs from MAST REST
    rad, teff = query_mast_rest_tic(tic)
    
    # Calculate planet radius
    if rad is not None and str(rad) != "None":
        rad = float(rad)
        rp = rad * np.sqrt(depth) * 109.2
        rp_str = f"{rp:.2f}"
    else:
        rad = "N/A"
        rp_str = "N/A"
        
    teff_str = f"{teff:.0f}" if teff is not None and str(teff) != "None" else "N/A"
    rad_str = f"{rad:.2f}" if rad != "N/A" else "N/A"
    
    # Determine Final Verdict Category
    if toi_num == "New Candidate":
        if rad != "N/A" and float(rad) > 0:
            rp_val = float(rp_str)
            if rp_val > 25.0:
                verdict = "False Positive (EB)"
            else:
                verdict = "New Planet Candidate"
        else:
            verdict = "New Planet Candidate"
    else:
        if "CP" in official_disp or "KP" in official_disp:
            verdict = "Confirmed/Known Planet"
        elif "FP" in official_disp or "EB" in official_disp:
            verdict = "False Positive"
        else:
            verdict = "Planet Candidate"
            
    results.append({
        "tic_id": tic,
        "source": source,
        "toi": toi_num,
        "disp": official_disp,
        "period": period,
        "depth_ppm": depth * 1e6,
        "rad_star": rad_str,
        "teff": teff_str,
        "rad_planet": rp_str,
        "verdict": verdict
    })

# Format the results into a markdown file, grouped by Verdict
df_res = pd.DataFrame(results)

report = []
report.append("# Master Verification Report: Flagged Planet Candidates")
report.append("This report lists all 19 flagged planet candidates from both search sessions, cross-referenced with NASA's TESS Object of Interest (TOI) catalog and physical stellar parameters.\n")

categories = ["Confirmed/Known Planet", "Planet Candidate", "New Planet Candidate", "False Positive"]

for cat in categories:
    df_cat = df_res[df_res["verdict"].str.startswith(cat)]
    if len(df_cat) == 0:
        continue
    
    report.append(f"\n## 📂 {cat}s ({len(df_cat)} targets)")
    report.append("| TIC ID | TOI ID | Period (d) | Depth (ppm) | Stellar Rad (R_sun) | Planet Rad (R_earth) | Source | NASA Disposition |")
    report.append("|---|---|---|---|---|---|---|---|")
    
    for idx, row in df_cat.iterrows():
        report.append(f"| {row['tic_id']} | {row['toi']} | {row['period']:.4f} | {row['depth_ppm']:.0f} | {row['rad_star']} | {row['rad_planet']} | {row['source']} | {row['disp']} |")

report_txt = "\n".join(report)
with open("master_verification_report.md", "w", encoding="utf-8") as f:
    f.write(report_txt)
    
print("SUCCESS: master_verification_report.md written.")
