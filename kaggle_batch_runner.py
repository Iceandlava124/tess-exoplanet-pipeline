"""
kaggle_batch_runner.py
======================
Kaggle entrypoint script.
1. Unzips the source code archive `src.zip` from our Kaggle dataset.
2. Copies target CSVs and model weights to their respective active folders.
3. Invokes the batch processor to vet the exoplanet candidates on Kaggle cloud.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

# Install required astronomy libraries on Kaggle at startup
print("Installing required packages (lightkurve, wotan, batman-package)...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "lightkurve", "wotan", "batman-package"])
    print("SUCCESS: Packages installed.")
except Exception as e:
    print("WARNING: Package installation encountered an error:", e)


# Paths to the attached Kaggle dataset
INPUT_DIR = Path("/kaggle/input/exoplanet-pipeline-resources")
WORKING_DIR = Path("/kaggle/working")

print("=" * 70)
print("KAGGLE ENVIRONMENT SETUP")
print("=" * 70)


# 1. Copy the src folder directly (Kaggle auto-extracts zip archives in datasets, sometimes creating nested folders)
src_candidate1 = INPUT_DIR / "src" / "src"
src_candidate2 = INPUT_DIR / "src"

if src_candidate1.exists() and src_candidate1.is_dir():
    src_path = src_candidate1
elif src_candidate2.exists() and src_candidate2.is_dir():
    src_path = src_candidate2
else:
    raise FileNotFoundError("Could not find src directory in dataset!")

print(f"Copying {src_path} to {WORKING_DIR / 'src'}...")
shutil.copytree(src_path, WORKING_DIR / "src")
print("SUCCESS: Source code loaded.")



# 2. Recreate directory structures
(WORKING_DIR / "data").mkdir(parents=True, exist_ok=True)
(WORKING_DIR / "models").mkdir(parents=True, exist_ok=True)
(WORKING_DIR / "results").mkdir(parents=True, exist_ok=True)

# 3. Copy model weights
print("Copying model weights...")
shutil.copy2(INPUT_DIR / "random_forest.pkl", WORKING_DIR / "models" / "random_forest.pkl")
shutil.copy2(INPUT_DIR / "cnn_classifier.h5", WORKING_DIR / "models" / "cnn_classifier.h5")
print("SUCCESS: Model weights loaded.")

# 4. Copy targets CSV
print("Copying target list...")
shutil.copy2(INPUT_DIR / "training_targets.csv", WORKING_DIR / "data" / "training_targets.csv")
print("SUCCESS: Targets catalog loaded.")

# 4b. Copy batch_processor.py
print("Copying batch processor script...")
shutil.copy2(INPUT_DIR / "batch_processor.py", WORKING_DIR / "batch_processor.py")
print("SUCCESS: Batch processor script loaded.")

# 4c. Copy pipeline cache DB
print("Copying pipeline cache DB...")
cache_src = INPUT_DIR / "pipeline_cache.db"
if cache_src.exists():
    shutil.copy2(cache_src, WORKING_DIR / "data" / "pipeline_cache.db")
    print("SUCCESS: Cache DB loaded.")
else:
    print("WARNING: pipeline_cache.db not found, running without pre-filled cache.")


# 5. Run the batch processor!
print("\n" + "=" * 70)
print("STARTING BATCH PROCESSING RUN ON KAGGLE")
print("=" * 70)

# Import the batch processor now that src is extracted and on PATH
import sys
sys.path.insert(0, str(WORKING_DIR))
from batch_processor import run_batch_processing

# Run batch processing on 200 targets
run_batch_processing(str(WORKING_DIR / "data" / "training_targets.csv"), limit=200)

print("SUCCESS: Batch process complete.")
