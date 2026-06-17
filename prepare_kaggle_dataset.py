"""
prepare_kaggle_dataset.py
=========================
Prepares a local folder with the project's models, targets, caching database, 
and zipped source code, generates the metadata, and uploads it to Kaggle.
"""

import os
import sys
import shutil
import json
import zipfile
import sqlite3
from pathlib import Path

ROOT = Path(r"c:\Users\gudae\Desktop\Learn_ml").resolve()
DATASET_DIR = ROOT / "kaggle_dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PREPARING KAGGLE DATASET UPLOAD")
print("=" * 70)

# 1. Zip the src/ directory and root runner files
src_zip_path = DATASET_DIR / "src.zip"
print(f"Creating zip archive of src/ and runners at {src_zip_path.name}...")
with zipfile.ZipFile(src_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    src_dir = ROOT / "src"
    for file in src_dir.glob("**/*.py"):
        if "__pycache__" in file.parts:
            continue
        # Write file preserving relative structure
        rel_path = file.relative_to(ROOT)
        zipf.write(file, rel_path)
    
    # Write root runner scripts directly to zip root
    root_runners = ["kaggle_discovery_runner.py", "autonomous_discovery_agent.py"]
    for runner in root_runners:
        runner_path = ROOT / runner
        if runner_path.exists():
            zipf.write(runner_path, runner)
print("SUCCESS: Source and runner files zipped successfully.")

# 2. Copy the training targets CSV
targets_src = ROOT / "data" / "training_targets.csv"
targets_dest = DATASET_DIR / "training_targets.csv"
if targets_src.exists():
    print(f"Copying training targets CSV to dataset folder...")
    shutil.copy2(targets_src, targets_dest)
    print("SUCCESS: Targets list copied.")
else:
    print("WARNING: data/training_targets.csv not found!")

# 3. Copy the pipeline cache DB
cache_src = ROOT / "data" / "pipeline_cache.db"
cache_dest = DATASET_DIR / "pipeline_cache.db"
if cache_src.exists():
    print(f"Copying pipeline cache DB to dataset folder...")
    # Checkpoint WAL journals to main db before copying to ensure it's self-contained
    try:
        conn = sqlite3.connect(str(cache_src))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
    except Exception as e:
        print(f"Warning: WAL checkpoint failed: {e}")
    shutil.copy2(cache_src, cache_dest)
    print("SUCCESS: Cache DB copied.")
else:
    print("WARNING: data/pipeline_cache.db not found!")

# 4. Copy batch_processor.py
batch_src = ROOT / "batch_processor.py"
batch_dest = DATASET_DIR / "batch_processor.py"
if batch_src.exists():
    print(f"Copying batch_processor.py to dataset folder...")
    shutil.copy2(batch_src, batch_dest)
    print("SUCCESS: batch_processor.py copied.")
else:
    print("ERROR: batch_processor.py not found!")
    sys.exit(1)

# 5. Copy the model weights
models = ["random_forest.pkl", "cnn_classifier.h5"]
for m in models:
    model_src = ROOT / "models" / m
    model_dest = DATASET_DIR / m
    if model_src.exists():
        print(f"Copying model {m} to dataset folder...")
        shutil.copy2(model_src, model_dest)
        print(f"SUCCESS: Model {m} copied.")
    else:
        print(f"ERROR: models/{m} not found! Run training first.")
        sys.exit(1)

# 6. Create dataset-metadata.json
metadata = {
    "title": "Exoplanet Pipeline Resources",
    "id": "bhavishmehta/exoplanet-pipeline-resources",
    "licenses": [{"name": "CC0-1.0"}]
}
metadata_path = DATASET_DIR / "dataset-metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print("SUCCESS: dataset-metadata.json generated.")

print("\nSUCCESS: Dataset folder ready for upload!")
