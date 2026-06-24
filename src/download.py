"""
src/download.py
===============
Utilities for downloading TESS data from MAST via lightkurve,
and for building target lists from the xCTL catalog.

📚 LEARNING NOTE:
    This module wraps the `lightkurve` library which talks to NASA's
    MAST (Mikulski Archive for Space Telescopes) server to fetch
    FITS files containing stellar light curves.
"""

import os
import time
import logging
import urllib.request
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# lightkurve is the main library for downloading TESS data
import lightkurve as lk

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Directory constants ──────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent          # Learn_ml/
DATA_DIR = ROOT / "data"
FITS_DIR = DATA_DIR / "raw_fits"
XCTL_DIR = DATA_DIR / "xctl"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
for d in [DATA_DIR, FITS_DIR, XCTL_DIR, PROCESSED_DIR,
          ROOT / "models", ROOT / "results" / "figures", ROOT / "results" / "reports"]:
    d.mkdir(parents=True, exist_ok=True)


# ─── xCTL Download ────────────────────────────────────────────────────────────

XCTL_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01.csv"
XCTL_HEADER_URL = "https://archive.stsci.edu/missions/tess/catalogs/xctl/exo_CTL_08.01_header.csv"
TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"


def download_xctl(force: bool = False) -> Path:
    """
    Download the TESS Exoplanet Candidate Target List (xCTL) from MAST,
    and also download the ExoFOP TOI catalog for ground-truth training labels.
    """
    csv_path = XCTL_DIR / "exo_CTL_08.01.csv"
    header_path = XCTL_DIR / "header.csv"
    toi_path = XCTL_DIR / "toi_catalog.csv"

    # 1. Download large xCTL target list if missing
    if not csv_path.exists() or force:
        logger.info("Downloading xCTL CSV (~497 MB). This will take a few minutes...")
        last_b = [0]
        def progress_hook(block_num, block_size, total_size):
            if not hasattr(progress_hook, "pbar"):
                progress_hook.pbar = tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="xCTL download"
                )
            downloaded = block_num * block_size
            progress_hook.pbar.update(downloaded - last_b[0])
            last_b[0] = downloaded
            if downloaded >= total_size:
                progress_hook.pbar.close()

        urllib.request.urlretrieve(XCTL_URL, csv_path, reporthook=progress_hook)
        logger.info(f"xCTL saved to {csv_path}")
    else:
        logger.info(f"xCTL already exists at {csv_path}.")

    # 2. Download xCTL header if missing
    if not header_path.exists() or force:
        logger.info("Downloading xCTL header...")
        urllib.request.urlretrieve(XCTL_HEADER_URL, header_path)
        logger.info(f"Header saved to {header_path}")

    # 3. Download ExoFOP TOI catalog for ground truth labels
    if not toi_path.exists() or force:
        logger.info("Downloading ExoFOP TOI catalog for exoplanet labels...")
        urllib.request.urlretrieve(TOI_URL, toi_path)
        logger.info(f"TOI catalog saved to {toi_path}")
    else:
        logger.info(f"TOI catalog already exists at {toi_path}.")

    return csv_path


# ─── xCTL Processing ─────────────────────────────────────────────────────────

LABEL_NAMES = {
    0: "No Signal",
    1: "Planet Transit",
    2: "Eclipsing Binary",
    3: "False Positive / Blend",
}


def load_xctl(xctl_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the ExoFOP TOI catalog and map its dispositions to integer labels (1, 2, 3).
    """
    import re
    if xctl_path is None:
        # Load the TOI catalog for training labels
        xctl_path = XCTL_DIR / "toi_catalog.csv"

    if not xctl_path.exists():
        # Fallback to downloading
        download_xctl()

    logger.info(f"Loading TOI catalog from {xctl_path}...")
    df = pd.read_csv(xctl_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows from TOI list.")

    # Apply prioritized ground truth mapping logic
    def map_row(row):
        tess_disp = str(row.get("TESS Disposition", "")).strip()
        tfop_disp = str(row.get("TFOPWG Disposition", "")).strip()
        comments = str(row.get("Comments", "")).lower()
        
        # Check if Eclipsing Binary (Label 2)
        has_eb = re.search(r'\b(eb|neb|beb|eclipsing\s+binary|eclipsing\s+binaries)\b', comments) is not None
        if tess_disp == "EB" or tfop_disp == "EB" or has_eb:
            return 2
            
        # Check if False Positive (Label 3)
        if tfop_disp in ["FP", "FA"] or tess_disp in ["O", "IS", "V", "FP"]:
            return 3
            
        # Check if Planet Candidate/Confirmed Planet (Label 1)
        if tfop_disp in ["PC", "KP", "CP", "APC"] or tess_disp in ["PC", "KP", "CP"]:
            return 1
            
        return 0

    df["label"] = df.apply(map_row, axis=1)
    
    logger.info("TOI Label distribution:")
    for k in [1, 2, 3]:
        logger.info(f"  Class {k} ({LABEL_NAMES[k]}): {(df['label'] == k).sum()} stars")

    return df


def build_training_targets(
    df: pd.DataFrame,
    n_per_class: int = 500,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Sample a balanced set of training targets.
    Labels 1, 2, 3 are sampled from the TOI catalog.
    Label 0 (No Signal) is sampled from the large exo_CTL_08.01.csv target list.
    """
    np.random.seed(seed)

    # Rename TIC ID column in TOI catalog if needed
    tic_col = None
    for candidate in ["TIC ID", "TIC", "ticid", "ID"]:
        if candidate in df.columns:
            tic_col = candidate
            break
    if tic_col:
        df = df.rename(columns={tic_col: "tic_id"})
    else:
        df = df.rename(columns={df.columns[0]: "tic_id"})

    sampled_parts = []
    
    # 1. Sample exoplanets, EBs, and FPs from TOI catalog
    for label in [1, 2, 3]:
        subset = df[df["label"] == label]
        if len(subset) >= n_per_class:
            sampled_parts.append(subset.sample(n_per_class, random_state=seed)[["tic_id", "label"]])
        else:
            logger.warning(
                f"Label {label} ({LABEL_NAMES[label]}) only has {len(subset)} samples, using all of them."
            )
            sampled_parts.append(subset[["tic_id", "label"]])

    # 2. Sample Label 0 (No Signal) from the 497 MB CTL list to ensure clean background stars
    ctl_path = XCTL_DIR / "exo_CTL_08.01.csv"
    if not ctl_path.exists():
        logger.info("Downloading exo_CTL_08.01.csv for background stars...")
        download_xctl()

    logger.info("Loading background stars from CTL to sample Class 0 (No Signal)...")
    # Load only the first column of the CTL list to save memory
    ctl_df = pd.read_csv(ctl_path, header=None, usecols=[0], names=["tic_id"], dtype={"tic_id": int})
    
    # Filter out any stars that are in the TOI list (labels 1, 2, 3)
    toi_stars = set(df["tic_id"].unique())
    ctl_df = ctl_df[~ctl_df["tic_id"].isin(toi_stars)]
    
    # Sample and assign label 0
    no_signal_subset = ctl_df.sample(n_per_class, random_state=seed).copy()
    no_signal_subset["label"] = 0
    sampled_parts.append(no_signal_subset[["tic_id", "label"]])

    # 3. Combine and shuffle
    targets = pd.concat(sampled_parts, ignore_index=True)
    targets = targets.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    logger.info(
        f"Final balanced training targets list: {len(targets)} stars\n"
        + "\n".join(
            f"  Class {k} ({LABEL_NAMES[k]}): {(targets['label'] == k).sum()}"
            for k in [0, 1, 2, 3]
        )
    )

    if output_path is None:
        output_path = DATA_DIR / "training_targets.csv"

    targets.to_csv(output_path, index=False)
    logger.info(f"Training targets saved to {output_path}")
    return targets


# ─── Light Curve Download ─────────────────────────────────────────────────────

def download_lightcurve(
    tic_id: int,
    sector: Optional[int] = None,
    cadence: str = "short",
    save_dir: Optional[Path] = None,
    stitch_multisector: bool = False,
) -> Optional[Path]:
    """
    Download a TESS light curve for a given TIC ID from MAST.
    If sector is None and stitch_multisector is True, this downloads and stitches
    all available sectors for the target.

    Args:
        tic_id:             TESS Input Catalog star ID (integer)
        sector:             Which TESS sector to download (None = all/most recent depending on stitch_multisector)
        cadence:            "short" (2-min) or "long" (10-min or 30-min)
        save_dir:           Where to save the FITS file
        stitch_multisector: Whether to download and stitch all available sectors when sector is None

    Returns:
        Path to saved FITS file, or None if not found.
    """
    if save_dir is None:
        save_dir = FITS_DIR

    # Plain English: Check if a stitched multi-sector file already exists
    if sector is None and stitch_multisector:
        stitched_path = save_dir / f"tic_{tic_id}_stitched.fits"
        if stitched_path.exists():
            logger.info(f"TIC {tic_id}: already stitched at {stitched_path}")
            return stitched_path

    # Plain English: Check if a specific single sector file already exists
    if sector is not None:
        # Search for files containing both the TIC ID and the formatted sector number
        fits_sector_pattern = (
            list(save_dir.glob(f"**/*s{sector:04d}*{tic_id}*.fits")) or 
            list(save_dir.glob(f"**/*{tic_id}*s{sector:04d}*.fits")) or 
            list(save_dir.glob(f"**/*{tic_id}*s{sector}*.fits"))
        )
        fits_sector_pattern = [f for f in fits_sector_pattern if "hlsp" not in str(f).lower() and "tasoc" not in str(f).lower()]
        if fits_sector_pattern:
            logger.info(f"TIC {tic_id} Sector {sector}: already downloaded at {fits_sector_pattern[0]}")
            return fits_sector_pattern[0]

    # Plain English: Check if any FITS file exists when not stitching and sector is None
    if sector is None and not stitch_multisector:
        fits_pattern = [f for f in save_dir.glob(f"**/*{tic_id}*.fits") if "hlsp" not in str(f).lower() and "tasoc" not in str(f).lower()]
        if fits_pattern:
            logger.info(f"TIC {tic_id}: already downloaded at {fits_pattern[0]}")
            return fits_pattern[0]

    import time as _time
    import random as _random

    for _attempt in range(3):
        try:
            if sector is None and stitch_multisector:
                # Plain English: Search, download, and stitch all available sectors
                logger.info(f"TIC {tic_id}: Searching for all available TESS sectors...")
                search_result = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    cadence=cadence,
                )

                if len(search_result) == 0:
                    logger.warning(f"TIC {tic_id}: no light curve found on MAST")
                    return None

                if len(search_result) == 1:
                    logger.info(f"TIC {tic_id}: only 1 sector found, downloading standard FITS...")
                    lc = search_result[0].download(download_dir=str(save_dir))
                    fits_files = list(save_dir.glob(f"**/*{tic_id}*.fits"))
                    if fits_files:
                        return fits_files[0]
                    return None

                logger.info(f"TIC {tic_id}: downloading {len(search_result)} sectors...")
                lc_collection = search_result.download_all(download_dir=str(save_dir))

                logger.info(f"TIC {tic_id}: stitching {len(lc_collection)} sectors...")
                stitched_lc = lc_collection.stitch()

                stitched_path = save_dir / f"tic_{tic_id}_stitched.fits"
                stitched_lc.to_fits(str(stitched_path), overwrite=True)
                logger.info(f"TIC {tic_id}: successfully stitched and saved to {stitched_path}")
                return stitched_path

            else:
                # Plain English: Download a single requested sector or default most recent
                search_result = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    mission="TESS",
                    cadence=cadence,
                    sector=sector,
                )

                if len(search_result) == 0:
                    logger.warning(f"TIC {tic_id}: no light curve found on MAST")
                    return None

                lc = search_result[0].download(download_dir=str(save_dir))
                fits_files = list(save_dir.glob(f"**/*{tic_id}*.fits"))
                if fits_files:
                    return fits_files[0]
                return None

        except Exception as e:
            logger.error(f"TIC {tic_id}: download attempt {_attempt+1}/3 failed — {e}")
            if _attempt < 2:
                _time.sleep(2 ** _attempt + _random.random())
            else:
                # ── AWS S3 Public Open Data Backup ──────────────────────────────────────
                # Construct S3 path directly to bypass MAST API gateway.
                # Format: https://tess.s3.amazonaws.com/public/tid/s{sector:04d}/{prefix}/{tic_id}/tess{timestamp}-s{sector:04d}-{tic_id_padded}-0120-s_lc.fits
                # Since constructing the exact filename timestamp is difficult without metadata, 
                # we query the index/directory listing or do a best-effort sector estimation.
                # As a fallback, we attempt to download via public CADC (Canadian Astronomy Data Centre) TESS URL.
                logger.info(f"TIC {tic_id}: MAST API failed. Trying CADC / AWS S3 direct fallback...")
                try:
                    import urllib.request
                    import urllib.parse
                    # CADC search endpoint fallback
                    tic_padded = f"{tic_id:016d}"
                    url = f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/TESS/tic{tic_id}_lc.fits"
                    if sector:
                        url = f"https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/TESS/s{sector:04d}/tic{tic_id}_lc.fits"
                    
                    dest_file = save_dir / f"tic_{tic_id}_cadc_backup.fits"
                    logger.info(f"Downloading from backup URL: {url}")
                    
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=45) as response, open(dest_file, 'wb') as out_file:
                        out_file.write(response.read())
                    
                    if dest_file.exists() and dest_file.stat().st_size > 1000:
                        logger.info(f"TIC {tic_id}: Successfully downloaded from backup mirror to {dest_file}")
                        return dest_file
                except Exception as e_backup:
                    logger.warning(f"Mirror download failed for TIC {tic_id}: {e_backup}")
                
                return None


def batch_download(
    targets: pd.DataFrame,
    cadence: str = "short",
    delay: float = 0.5,
    max_errors: int = 50,
) -> pd.DataFrame:
    """
    Download light curves for all stars in the targets DataFrame.

    Args:
        targets:    DataFrame with 'tic_id' column
        cadence:    "short" (2-min) or "long"
        delay:      Seconds to wait between downloads (be polite to MAST!)
        max_errors: Stop if we hit this many consecutive errors

    Returns:
        Updated DataFrame with 'fits_path' column added.

    📚 LEARNING NOTE:
        When downloading many files from a public server, always add
        a small delay between requests. This is called "rate limiting"
        and is good practice / required by most APIs.
        tqdm gives us a progress bar to see how far along we are.
    """
    paths = []
    error_count = 0

    logger.info(f"Downloading {len(targets)} light curves...")

    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="Downloading LCs"):
        tic_id = int(row["tic_id"])
        path = download_lightcurve(tic_id, cadence=cadence)
        paths.append(str(path) if path else None)

        if path is None:
            error_count += 1
        else:
            error_count = 0

        if error_count >= max_errors:
            logger.error(f"Too many consecutive errors ({max_errors}). Stopping.")
            break

        time.sleep(delay)

    targets = targets.copy()
    targets["fits_path"] = paths + [None] * (len(targets) - len(paths))

    success = targets["fits_path"].notna().sum()
    logger.info(
        f"Download complete: {success}/{len(targets)} light curves fetched "
        f"({100*success/len(targets):.1f}%)"
    )
    return targets


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test: download one well-known exoplanet host
    logger.info("=== Quick Download Test ===")
    logger.info("Trying TIC 261136679 (WASP-121 — famous hot Jupiter host)...")
    path = download_lightcurve(261136679)
    if path:
        logger.info(f"✅ Success! File saved to: {path}")
    else:
        logger.error("❌ Download failed")
