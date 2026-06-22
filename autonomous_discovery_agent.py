"""
autonomous_discovery_agent.py
=============================
A fully autonomous exoplanet discovery AI agent script designed to run end-to-end
photometric search and classification of TESS light curves.

It runs:
1. Environment Setup (folders, packages check).
2. Resumes progress from past runs (loads and copies the cumulative Kaggle dataset).
3. Queries MAST/TIC catalog for Sun-like, high-priority target stars.
4. Executes the Exoplanet Pipeline v2.0 (TLS, caching, detrending, CNN).
5. Generates summary diagnostics and discovery logs.
6. Pushes the updated cumulative results database back to Kaggle.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from datetime import date
from pathlib import Path

# ── USER CONFIGURATION ──────────────────────────────────────
GITHUB_REPO    = "https://github.com/Iceandlava124/tess-exoplanet-pipeline.git"
KAGGLE_DATASET = "bhavishmehta/tess-exoplanet-discovery-results"
SESSION_LABEL  = date.today().strftime("%Y-%m-%d")
TIME_LIMIT_HRS = 5.0
DISK_LIMIT_GB  = 18
STARS_PER_RUN  = 400

# ── ENVIRONMENT PATHS ────────────────────────────────────────
WORKING_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path(".").resolve()
PIPELINE_DIR = WORKING_DIR / "pipeline"
RESULTS_DIR  = WORKING_DIR / "results"
def find_kaggle_input_dir(slug: str) -> Path:
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if slug in Path(root).name:
                return Path(root)
    return Path("/kaggle/input") / slug

INPUT_DIR = find_kaggle_input_dir("exoplanet-pipeline-resources")
INPUT_RESULTS_DIR = find_kaggle_input_dir("tess-exoplanet-discovery-results")

print("=" * 70)
print("STARTING AUTONOMOUS DISCOVERY AGENT")
print("=" * 70)

# ── 1. PACKAGE INSTALLATION ──────────────────────────────────
# Installs packages if not available
try:
    import lightkurve
    import wotan
    import batman
    import transitleastsquares
    import ldtk
    print("SUCCESS: Core packages already installed.")
except ImportError:
    print("Installing required astronomy packages...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "lightkurve", "wotan", "batman-package", "astropy", "astroquery",
             "transitleastsquares", "ldtk", "scipy", "scikit-learn", "imbalanced-learn", "tqdm", "joblib",
             "pandas", "numpy"],
            check=True, capture_output=True
        )
        print("SUCCESS: Packages installed.")
    except Exception as e:
        print(f"WARNING: Package installation error: {e}")

# ── 2. CODEBASE LOAD & EXTRACTION ────────────────────────────
# Prefers extracting src.zip from Kaggle dataset; falls back to GitHub clone.
try:
    if (INPUT_DIR / "src.zip").exists():
        print("Extracting pipeline code from dataset resources...")
        PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(INPUT_DIR / "src.zip", 'r') as zip_ref:
            zip_ref.extractall(PIPELINE_DIR)
        print("SUCCESS: Code extracted.")
    elif (INPUT_DIR / "src").exists():
        print("Copying pipeline code from dataset resources...")
        shutil.copytree(INPUT_DIR / "src", PIPELINE_DIR / "src", dirs_exist_ok=True)
        print("SUCCESS: Code copied.")
    else:
        if (PIPELINE_DIR / ".git").exists():
            print("Updating existing pipeline clone...")
            result = subprocess.run(
                ["git", "-C", str(PIPELINE_DIR), "pull"],
                capture_output=True, text=True, timeout=120
            )
            print("SUCCESS: Pipeline updated.")
        else:
            if PIPELINE_DIR.exists():
                shutil.rmtree(PIPELINE_DIR, ignore_errors=True)
            print("Cloning pipeline repository from GitHub...")
            result = subprocess.run(
                ["git", "clone", "--depth=1", GITHUB_REPO, str(PIPELINE_DIR)],
                capture_output=True, text=True, timeout=180
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip())
            print("SUCCESS: Pipeline cloned.")
except Exception as e:
    print(f"[ERROR] Failed to load source code: {e}")
    sys.exit(1)

# Add pipeline folders to Python path
for p in [str(PIPELINE_DIR), str(PIPELINE_DIR / "src"), str(WORKING_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Create active folders
for folder in ["KEEP", "FLAG", "DISCARD", "plots/flag_analysis", "reports", "figures", "data", "models"]:
    os.makedirs(RESULTS_DIR / folder, exist_ok=True)
    os.makedirs(WORKING_DIR / folder, exist_ok=True)

# Copy dataset metadata so that Kaggle CLI pushes work
src_meta = PIPELINE_DIR / "results" / "dataset-metadata.json"
dest_meta = RESULTS_DIR / "dataset-metadata.json"
if src_meta.exists():
    shutil.copy2(src_meta, dest_meta)
    print("SUCCESS: dataset-metadata.json copied to results directory.")
else:
    import json
    meta = {
        "title": "TESS Exoplanet Discovery Results",
        "id": KAGGLE_DATASET,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(dest_meta, "w") as f:
        json.dump(meta, f, indent=4)
    print("SUCCESS: Generated dataset-metadata.json in results directory.")

# ── 3. RESOURCE LOADING ──────────────────────────────────────
# Copy weights, target files, and caching databases
try:
    if INPUT_DIR.exists():
        print("Loading models, target catalogs, and cached queries...")
        for m in ["random_forest.pkl", "cnn_classifier.h5"]:
            if (INPUT_DIR / m).exists():
                os.makedirs(PIPELINE_DIR / "models", exist_ok=True)
                shutil.copy2(INPUT_DIR / m, WORKING_DIR / "models" / m)
                shutil.copy2(INPUT_DIR / m, PIPELINE_DIR / "models" / m)
        if (INPUT_DIR / "training_targets.csv").exists():
            os.makedirs(PIPELINE_DIR / "data", exist_ok=True)
            shutil.copy2(INPUT_DIR / "training_targets.csv", WORKING_DIR / "data" / "training_targets.csv")
            shutil.copy2(INPUT_DIR / "training_targets.csv", PIPELINE_DIR / "data" / "training_targets.csv")
        if (INPUT_DIR / "pipeline_cache.db").exists():
            os.makedirs(PIPELINE_DIR / "data", exist_ok=True)
            shutil.copy2(INPUT_DIR / "pipeline_cache.db", WORKING_DIR / "data" / "pipeline_cache.db")
            shutil.copy2(INPUT_DIR / "pipeline_cache.db", PIPELINE_DIR / "data" / "pipeline_cache.db")
        print("SUCCESS: Resources loaded.")
except Exception as e:
    print(f"Warning loading resources: {e}")

# ── 4. RESUME FROM PREVIOUS SESSION ──────────────────────────
# Load already processed stars to avoid redundant searches.
OUTPUT_RESULTS_CSV = RESULTS_DIR / "results.csv"
already_done = set()

try:
    input_csv = INPUT_RESULTS_DIR / "results.csv"
    if input_csv.exists():
        shutil.copy2(input_csv, OUTPUT_RESULTS_CSV)
        # Parse using python csv module to be robust against schema/column count variations
        import csv
        rows = []
        COLUMNS_22 = [
            "tic_id", "session_label", "decision", "final_class", "confidence",
            "period", "period_err", "depth", "depth_err", "duration",
            "duration_err", "snr", "flag_reasons", "rp_earth", "is_new_discovery",
            "alias_rejected", "fpp", "combined_fpp", "fpp_status",
            "contamination_ratio", "n_nearby_gaia_stars", "n_sectors_consistent"
        ]
        with open(OUTPUT_RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            
            for row in reader:
                if not row:
                    continue
                new_row = {}
                if len(row) == 12:
                    new_row["tic_id"] = row[0]
                    new_row["session_label"] = ""
                    new_row["decision"] = row[1]
                    new_row["final_class"] = row[2]
                    new_row["confidence"] = row[3]
                    new_row["period"] = row[4]
                    new_row["period_err"] = row[5]
                    new_row["depth"] = row[6]
                    new_row["depth_err"] = row[7]
                    new_row["duration"] = row[8]
                    new_row["duration_err"] = row[9]
                    new_row["snr"] = row[10]
                    new_row["flag_reasons"] = row[11]
                    new_row["rp_earth"] = "0.0"
                    new_row["is_new_discovery"] = "False"
                    new_row["alias_rejected"] = "False"
                    new_row["fpp"] = "None"
                    new_row["combined_fpp"] = "None"
                    new_row["fpp_status"] = "skipped"
                    new_row["contamination_ratio"] = "None"
                    new_row["n_nearby_gaia_stars"] = "0"
                    new_row["n_sectors_consistent"] = "1"
                else:
                    for i, col in enumerate(COLUMNS_22):
                        new_row[col] = row[i] if i < len(row) else ""
                rows.append(new_row)
        
        # Deduplicate
        unique_rows = {}
        for r in rows:
            if r["tic_id"]:
                try:
                    unique_rows[int(r["tic_id"])] = r
                except ValueError:
                    pass
        dedup_rows = list(unique_rows.values())
        
        # Save back
        with open(OUTPUT_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS_22)
            writer.writeheader()
            writer.writerows(dedup_rows)
            
        df_previous = pd.read_csv(OUTPUT_RESULTS_CSV)
        if "tic_id" in df_previous.columns:
            already_done = set(df_previous["tic_id"].astype(int).tolist())
        print(f"[LOAD] Resuming: loaded {len(already_done)} unique processed stars.")
    else:
        print("[INFO] Fresh start: no previous results file found.")
except Exception as e:
    print(f"Warning resuming progress: {e}")

# ── 5. TARGET SELECTION ──────────────────────────────────────
# Query the TESS Input Catalog (TIC) for solar-type candidates.
import pandas as pd
from kaggle_discovery_runner import build_target_list

TARGETS_CSV = RESULTS_DIR / "this_week_targets.csv"
print("\nQuerying TESS Input Catalog for prioritised target stars...")
try:
    targets = build_target_list(
        n_targets            = STARS_PER_RUN,
        already_processed    = already_done,
        mag_range            = (8, 13),
        teff_range           = (3500, 7000),
        radius_range         = (0.5, 2.0),
        exclude_giants       = True,
        exclude_known_contaminated = True,
        prioritise_multi_sector    = True,
        prioritise_not_in_toi      = True,
    )
    targets.to_csv(TARGETS_CSV, index=False)
    print(f"[INFO] Queued {len(targets)} stars for analysis.")
except Exception as e:
    print(f"[ERROR] Failed to build target catalog: {e}")
    targets = pd.DataFrame(columns=["tic_id"])

# ── 6. AUTONOMOUS PIPELINE VETTING ───────────────────────────
# Run the pipeline sequentially.
from kaggle_discovery_runner import run_discovery_session

session_summary = {}
if len(targets) == 0:
    print("[ERROR] No targets to process. Exiting.")
    sys.exit(0)

print("\nStarting exoplanet discovery pipeline loop...")
try:
    session_summary = run_discovery_session(
        targets           = targets,
        output_dir        = RESULTS_DIR,
        models_dir        = WORKING_DIR / "models",
        time_limit_hours  = TIME_LIMIT_HRS,
        disk_limit_gb     = DISK_LIMIT_GB,
        save_every_n      = 50,
        session_label     = SESSION_LABEL,
        run_flag_analyzer      = True,
        run_candidate_export   = True,
        run_toi_crosscheck     = True,
        alias_rejection        = True,
        max_planet_radius_earth = 25.0,
        min_transit_snr         = 5.0,
        min_depth_ppm           = 100,
        log_file = RESULTS_DIR / "discovery_log.txt",
    )
    print("SUCCESS: Pipeline vetting loop finished.")
except Exception as e:
    print(f"[ERROR] Vetting session crashed: {e}")

# ── 7. GENERATE SESSION SUMMARY ──────────────────────────────
from kaggle_discovery_runner import generate_session_summary

print("\nCompiling session statistics and reports...")
try:
    summary = generate_session_summary(
        results_dir   = RESULTS_DIR,
        session_label = SESSION_LABEL,
        session_data  = session_summary,
    )
    
    n_proc  = summary.get("stars_processed", 0)
    n_keep  = summary.get("keep", 0)
    n_flag  = summary.get("flag", 0)
    n_disc  = summary.get("discard", 0)
    n_new   = summary.get("new_discoveries", 0)
    
    print("-" * 50)
    print(f"Processed: {n_proc} | KEEP: {n_keep} | FLAG: {n_flag} | DISCARD: {n_disc}")
    print(f"Potential New Discoveries: {n_new}")
    print("-" * 50)
except Exception as e:
    print(f"Warning generating summary report: {e}")

# ── 8. EXPORT AND AUTO-UPDATE RESULTS ON KAGGLE ─────────────
# Push the updated database back to your Kaggle dataset.
print("\nExporting files and pushing results to Kaggle...")

# Deduplicate results.csv before exporting
if OUTPUT_RESULTS_CSV.exists():
    try:
        import csv
        COLUMNS_22 = [
            "tic_id", "session_label", "decision", "final_class", "confidence",
            "period", "period_err", "depth", "depth_err", "duration",
            "duration_err", "snr", "flag_reasons", "rp_earth", "is_new_discovery",
            "alias_rejected", "fpp", "combined_fpp", "fpp_status",
            "contamination_ratio", "n_nearby_gaia_stars", "n_sectors_consistent"
        ]
        rows = []
        with open(OUTPUT_RESULTS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            for r in reader:
                if not r:
                    continue
                new_r = {}
                if len(r) == 12:
                    new_r["tic_id"] = r[0]
                    new_r["session_label"] = ""
                    new_r["decision"] = r[1]
                    new_r["final_class"] = r[2]
                    new_r["confidence"] = r[3]
                    new_r["period"] = r[4]
                    new_r["period_err"] = r[5]
                    new_r["depth"] = r[6]
                    new_r["depth_err"] = r[7]
                    new_r["duration"] = r[8]
                    new_r["duration_err"] = r[9]
                    new_r["snr"] = r[10]
                    new_r["flag_reasons"] = r[11]
                    new_r["rp_earth"] = "0.0"
                    new_r["is_new_discovery"] = "False"
                    new_r["alias_rejected"] = "False"
                    new_r["fpp"] = "None"
                    new_r["combined_fpp"] = "None"
                    new_r["fpp_status"] = "skipped"
                    new_r["contamination_ratio"] = "None"
                    new_r["n_nearby_gaia_stars"] = "0"
                    new_r["n_sectors_consistent"] = "1"
                else:
                    for i, col in enumerate(COLUMNS_22):
                        new_r[col] = r[i] if i < len(r) else ""
                rows.append(new_r)
        
        unique_rows = {}
        duplicate_count = 0
        for r in rows:
            if r["tic_id"]:
                tic_int = int(r["tic_id"])
                if tic_int in unique_rows:
                    duplicate_count += 1
                unique_rows[tic_int] = r
        
        dedup_rows = list(unique_rows.values())
        if duplicate_count > 0:
            print(f"WARNING: {duplicate_count} duplicate TIC IDs found in results.csv — deduplicating")
            with open(OUTPUT_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS_22)
                writer.writeheader()
                writer.writerows(dedup_rows)
            print(f"   Deduplicated: {len(dedup_rows)} unique stars remain")
        else:
            print("Deduplication check complete: 0 duplicates found.")
    except Exception as e_dedup:
        print(f"Warning during deduplication: {e_dedup}")

export_map = {
    OUTPUT_RESULTS_CSV:                            "results_cumulative.csv",
    RESULTS_DIR / "candidates_submission.csv":     "candidates_all.csv",
    RESULTS_DIR / "manual_review_queue.csv":       "review_queue_latest.csv",
    RESULTS_DIR / "DISCOVERY_LOG.md":              "DISCOVERY_LOG.md",
    RESULTS_DIR / "new_discoveries.txt":           "new_discoveries.txt",
}

for src, dest_name in export_map.items():
    try:
        if src.exists() and src.stat().st_size > 0:
            shutil.copy2(src, WORKING_DIR / dest_name)
            print(f"   Saved: {dest_name}")
    except Exception as e:
        print(f"   Failed to save {dest_name}: {e}")

try:
    commit_msg = f"Auto-update: {SESSION_LABEL} -- {session_summary.get('stars_processed', 0)} stars processed"
    print(f"Pushing results to Kaggle dataset '{KAGGLE_DATASET}'...")
    result = subprocess.run(
        ["kaggle", "datasets", "version",
         "-p", str(RESULTS_DIR),
         "-m", commit_msg,
         "--dir-mode", "zip"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        print("SUCCESS: Kaggle dataset updated. Results are now available.")
    else:
        print(f"Kaggle push warning: {result.stderr.strip()}")
except Exception as e:
    print(f"Kaggle push failed: {e}")

print("\n[SUCCESS] Autonomous discovery execution completed successfully!")
print("=" * 70)
