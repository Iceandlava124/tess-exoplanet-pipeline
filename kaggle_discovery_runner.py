"""
kaggle_discovery_runner.py
===========================
Companion module for tess_weekly_runner.ipynb.

Provides three public functions called by the notebook cells:
    build_target_list()        — Cell 3: query TIC, filter, prioritise
    run_discovery_session()    — Cell 4: full 13-step pipeline loop
    generate_session_summary() — Cell 5: statistics and report

All network/file operations have try/except so one bad star never
crashes the whole session. Results are saved incrementally every
save_every_n stars so a Kaggle timeout loses at most ~50 stars.
"""

import csv
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger("kaggle_runner")

# ─── Known TESS systematic alias periods (copied from detect.py) ─────────────
TESS_ALIAS_PERIODS = [0.5, 1.0, 2.0, 13.5]
ALIAS_TOLERANCE    = 0.01

_TOI_TIC_IDS = None


def _load_toi_catalog() -> set:
    global _TOI_TIC_IDS
    if _TOI_TIC_IDS is not None:
        return _TOI_TIC_IDS
    
    _TOI_TIC_IDS = set()
    logger.info("Loading TOI catalog from Caltech TAP...")
    try:
        import urllib.request
        url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+tid+from+toi&format=csv'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            lines = response.read().decode('utf-8').splitlines()
            if lines:
                header = lines[0].split(',')
                try:
                    tid_idx = header.index('tid')
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        parts = line.split(',')
                        if len(parts) > tid_idx:
                            val = parts[tid_idx].strip().replace('"', '')
                            if val:
                                _TOI_TIC_IDS.add(int(float(val)))
                except ValueError:
                    logger.warning("tid column not found in Caltech TOI query header.")
        logger.info(f"Successfully loaded {len(_TOI_TIC_IDS)} TOIs from Caltech TAP.")
    except Exception as e:
        logger.warning(f"Caltech TAP TOI query failed: {e}. Trying ExoFOP fallback.")
        try:
            import urllib.request
            url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                lines = response.read().decode('utf-8').splitlines()
                if lines:
                    header = lines[0].split(',')
                    try:
                        tic_idx = -1
                        for i, col in enumerate(header):
                            if "tic" in col.lower() or "id" in col.lower():
                                tic_idx = i
                                break
                        if tic_idx != -1:
                            for line in lines[1:]:
                                if not line.strip():
                                    continue
                                parts = line.split(',')
                                if len(parts) > tic_idx:
                                    val = parts[tic_idx].strip().replace('"', '')
                                    if val:
                                        _TOI_TIC_IDS.add(int(float(val)))
                    except Exception as ex:
                        logger.warning(f"ExoFOP parsing error: {ex}")
            logger.info(f"Loaded {len(_TOI_TIC_IDS)} TOIs from ExoFOP fallback.")
        except Exception as e2:
            logger.warning(f"ExoFOP fallback failed: {e2}")
            
    if not _TOI_TIC_IDS:
        local_toi_files = [
            Path("data/xctl/toi_catalog.csv"),
            Path("pipeline/data/xctl/toi_catalog.csv"),
            Path(__file__).parent / "data/xctl/toi_catalog.csv"
        ]
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                for file in files:
                    if file == "toi_catalog.csv":
                        local_toi_files.append(Path(root) / file)
        
        for lf in local_toi_files:
            if lf.exists():
                try:
                    df_lf = pd.read_csv(lf)
                    for col in df_lf.columns:
                        if "tic" in col.lower() or "tid" in col.lower():
                            _TOI_TIC_IDS.update(df_lf[col].dropna().astype(int).tolist())
                            logger.info(f"Loaded {len(_TOI_TIC_IDS)} TOIs from local file {lf}")
                            break
                    if _TOI_TIC_IDS:
                        break
                except Exception as lf_err:
                    logger.warning(f"Failed to read local TOI fallback {lf}: {lf_err}")
                    
    return _TOI_TIC_IDS


def _is_alias(period: float) -> bool:
    return any(abs(period - a) < ALIAS_TOLERANCE for a in TESS_ALIAS_PERIODS)



# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — BUILD TARGET LIST
# ═══════════════════════════════════════════════════════════════════════════════

def build_target_list(
    n_targets: int = 800,
    already_processed: set = None,
    mag_range: tuple = (8, 13),
    teff_range: tuple = (3500, 7000),
    radius_range: tuple = (0.5, 2.0),
    exclude_giants: bool = True,
    exclude_known_contaminated: bool = True,
    prioritise_multi_sector: bool = True,
    prioritise_not_in_toi: bool = True,
) -> pd.DataFrame:
    """
    WHAT: Query the TESS Input Catalog (TIC) via astroquery to get a list of
          candidate stars for this week's discovery run.
    WHY:  We apply physical filters to focus on sun-like stars where TESS is
          likely to detect a planet, and skip stars we already processed to
          make the weekly run truly cumulative.

    Args:
        n_targets:              How many stars to return.
        already_processed:      Set of TIC IDs to skip.
        mag_range:              TESS magnitude limits (bright enough for photometry).
        teff_range:             Effective temperature range (planet-hosting range).
        radius_range:           Stellar radius in solar radii (exclude giants).
        exclude_giants:         Remove log g < 3.5 (luminosity class III+).
        exclude_known_contaminated: Remove crowded fields with contamination ratio > 0.1.
        prioritise_multi_sector: Prefer stars observed in 2+ TESS sectors.
        prioritise_not_in_toi:   Prefer stars not yet in the TOI candidate catalog.

    Returns:
        DataFrame with columns: tic_id, ra, dec, tmag, teff, radius, n_sectors, priority_score
    """
    # Force logger to output to stdout for visibility in Kaggle logs/notebooks
    import sys
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(sh)

    if already_processed is None:
        already_processed = set()

    logger.info(f"Querying TESS Input Catalog for up to {n_targets} targets...")

    # ── Query TIC via astroquery ──────────────────────────────────────────────
    # ── Query TIC via direct MAST API with strict 15s timeout ─────────────────
    try:
        import urllib.request
        import json
        
        url = "https://mast.stsci.edu/api/v0/invoke"
        payload = {
            "service": "Mast.Catalogs.Filtered.Tic",
            "format": "json",
            "params": {
                "columns": "ID,ra,dec,Tmag,Teff,rad,logg,contratio,priority,wdflag",
                "filters": [
                    {"paramName": "Tmag", "values": [{"min": float(mag_range[0]), "max": float(mag_range[1])}]},
                    {"paramName": "Teff", "values": [{"min": float(teff_range[0]), "max": float(teff_range[1])}]},
                    {"paramName": "rad", "values": [{"min": float(radius_range[0]), "max": float(radius_range[1])}]},
                    {"paramName": "objType", "values": ["STAR"]}
                ],
                "pagesize": int(max(2000, n_targets * 3))
            }
        }
        
        import threading
        import queue
        
        q = queue.Queue()
        def worker():
            try:
                req_data = f"request={json.dumps(payload)}".encode("utf-8")
                req = urllib.request.Request(url, data=req_data, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    res_data = response.read()
                    q.put((True, res_data))
            except Exception as e_thread:
                q.put((False, e_thread))
                
        t = threading.Thread(target=worker)
        t.daemon = True
        logger.info("Sending request to MAST REST API...")
        t.start()
        
        try:
            success, val = q.get(timeout=15)
            if not success:
                raise val
            res = json.loads(val.decode('utf-8'))
            if "data" not in res:
                raise ValueError("No data field in MAST REST response")
            df = pd.DataFrame(res["data"])
        except queue.Empty:
            raise TimeoutError("TIC query timed out after 15 seconds (wall-clock limit)")
            
        required_cols = ["ID", "ra", "dec", "Tmag", "Teff", "rad", "logg",
                         "contratio", "priority", "wdflag"]
        existing_cols = [c for c in required_cols if c in df.columns]
        df = df[existing_cols].copy()
        df = df.rename(columns={
            "ID": "tic_id", "Tmag": "tmag", "Teff": "teff",
            "rad": "radius", "logg": "logg", "contratio": "contratio",
            "priority": "tic_priority",
        })
        if "tic_priority" in df.columns:
            df = df.sort_values(by="tic_priority", ascending=False)
        df["tic_id"] = df["tic_id"].astype(int)
        logger.info(f"TIC query returned {len(df)} raw candidates.")
    except Exception as e:
        logger.error(f"TIC query failed: {e}. Falling back to cached list.")
        local_files = [
            Path("data/training_targets.csv"),
            Path("data/test_targets.csv"),
            Path("data/validation_targets.csv"),
            Path("pipeline/data/training_targets.csv"),
            Path("pipeline/data/test_targets.csv"),
            Path("pipeline/data/validation_targets.csv")
        ]
        # Check relative to script path
        script_dir = Path(__file__).parent
        local_files.extend([
            script_dir / "data/training_targets.csv",
            script_dir / "data/test_targets.csv",
            script_dir / "data/validation_targets.csv",
            script_dir / "pipeline/data/training_targets.csv",
            script_dir / "pipeline/data/test_targets.csv",
            script_dir / "pipeline/data/validation_targets.csv"
        ])
        # Search recursively in /kaggle/input
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                for file in files:
                    if file.endswith("targets.csv"):
                        local_files.append(Path(root) / file)
                        
        fallback_ids = []
        for lf in local_files:
            for p in [lf, Path(".") / lf, Path("/kaggle/working") / lf]:
                if p.exists():
                    try:
                        df_lf = pd.read_csv(p)
                        if "tic_id" in df_lf.columns:
                            fallback_ids.extend(df_lf["tic_id"].astype(int).tolist())
                    except Exception as e_lf:
                        logger.warning(f"Could not read local targets from {p}: {e_lf}")
        
        fallback_ids = list(set(fallback_ids))
        if fallback_ids:
            logger.info(f"Loaded {len(fallback_ids)} targets from local fallback catalogs.")
            fallback_ids = [t for t in fallback_ids if t not in already_processed]
            logger.info(f"Remaining after excluding already processed: {len(fallback_ids)}")
            
            selected_ids = fallback_ids[:n_targets]
            df_fallback = pd.DataFrame({
                "tic_id": selected_ids,
                "ra": [0.0] * len(selected_ids),
                "dec": [0.0] * len(selected_ids),
                "tmag": [10.0] * len(selected_ids),
                "teff": [5778.0] * len(selected_ids),
                "radius": [1.0] * len(selected_ids),
                "n_sectors": [1] * len(selected_ids),
                "priority_score": [1.0] * len(selected_ids)
            })
            return df_fallback
        
        return pd.DataFrame(columns=["tic_id", "ra", "dec", "tmag", "teff",
                                     "radius", "n_sectors", "priority_score"])

    # ── Remove already-processed stars ───────────────────────────────────────
    before = len(df)
    df = df[~df["tic_id"].isin(already_processed)].copy()
    logger.info(f"Removed {before - len(df)} already-processed stars.")

    # ── Apply physical filters ────────────────────────────────────────────────
    if exclude_giants:
        # log g < 3.5 indicates a giant star (less suitable for planet searches)
        mask = df["logg"].isna() | (df["logg"] >= 3.5)
        df = df[mask].copy()

    if exclude_known_contaminated:
        # contratio > 0.1 means >10% of flux is from nearby contaminating stars
        mask = df["contratio"].isna() | (df["contratio"] <= 0.1)
        df = df[mask].copy()

    # White dwarfs are not good planet hosts
    if "wdflag" in df.columns:
        df = df[df["wdflag"] != 1].copy()

    logger.info(f"After physical filters: {len(df)} candidates remain.")

    # ── Compute sector counts via MAST (optional, improves prioritisation) ────
    df["n_sectors"] = 1   # default; updated below if possible
    if prioritise_multi_sector:
        try:
            from astroquery.mast import Observations
            # Sample dynamically based on n_targets for sector counting (API friendly)
            sample_ids = df["tic_id"].head(min(len(df), max(50, n_targets * 2))).tolist()
            obs = Observations.query_criteria(
                target_name = [f"TIC {t}" for t in sample_ids],
                obs_collection = "TESS",
                dataproduct_type = "timeseries",
            )
            if obs is not None and len(obs) > 0:
                obs_df = obs.to_pandas()
                obs_df["tic_id"] = obs_df["target_name"].str.extract(r"TIC (\d+)").astype(float)
                sector_counts = obs_df.groupby("tic_id").size().reset_index(name="n_sectors")
                sector_counts["tic_id"] = sector_counts["tic_id"].astype(int)
                df = df.merge(sector_counts, on="tic_id", how="left", suffixes=("", "_new"))
                if "n_sectors_new" in df.columns:
                    df["n_sectors"] = df["n_sectors_new"].fillna(1).astype(int)
                    df = df.drop(columns=["n_sectors_new"])
        except Exception as e:
            logger.warning(f"Sector count query failed (will use n_sectors=1): {e}")

    # ── Load TOI catalog for cross-check ─────────────────────────────────────
    toi_tic_ids = set()
    if prioritise_not_in_toi:
        try:
            toi_tic_ids = _load_toi_catalog()
            logger.info(f"TOI catalog loaded: {len(toi_tic_ids)} known targets.")
        except Exception as e:
            logger.warning(f"TOI catalog query failed: {e}")

    df["in_toi"] = df["tic_id"].isin(toi_tic_ids)

    # ── Compute priority score ────────────────────────────────────────────────
    # Higher score = run first. Formula balances data quality, stellar suitability,
    # and discovery potential.
    df["priority_score"] = 0.0

    # Prefer bright stars (more photons = less noise)
    df["priority_score"] += (13.0 - df["tmag"].clip(8, 13)) / 5.0 * 0.3

    # Prefer sun-like temperatures (5000-6000 K is the sweet spot)
    df["priority_score"] += (
        1.0 - np.abs(df["teff"].fillna(5500) - 5500) / 1500.0
    ).clip(0, 1) * 0.3

    # Prefer stars with more TESS sectors (more transits = better statistics)
    df["priority_score"] += (df["n_sectors"].clip(1, 5) - 1) / 4.0 * 0.2

    # Strong bonus for stars NOT already in the TOI catalog
    df["priority_score"] += (~df["in_toi"]).astype(float) * 0.2

    # ── Sort and truncate to n_targets ────────────────────────────────────────
    df = df.sort_values("priority_score", ascending=False).head(n_targets)
    df = df.reset_index(drop=True)

    # Keep only the columns the notebook needs
    keep_cols = ["tic_id", "ra", "dec", "tmag", "teff", "radius",
                 "n_sectors", "in_toi", "priority_score"]
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"Final target list: {len(df)} stars (priority-sorted).")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DISCOVERY SESSION (MAIN LOOP)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_disk_used_gb(path: str) -> float:
    """Return GB used under path (not total disk — used by our output)."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total / 1024**3


def _log_star(log_file: str, tic_id: int, verdict: str, detail: str):
    """Append one line to the plain-text discovery log."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] TIC {tic_id} -> {verdict} ({detail})\n"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _write_csv_row(csv_path: str, row: dict):
    """Append one row to results.csv, writing the header if the file is new."""
    fieldnames = [
        "tic_id", "session_label", "decision", "final_class", "confidence",
        "period", "period_err", "depth", "depth_err", "duration",
        "duration_err", "snr", "flag_reasons", "rp_earth", "is_new_discovery",
        "alias_rejected", "fpp", "combined_fpp", "fpp_status",
        "contamination_ratio", "n_nearby_gaia_stars", "n_sectors_consistent"
    ]
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _copy_to_verdict_folder(output_dir: str, tic_id: int, verdict: str,
                             plot_path: Optional[str], report_path: Optional[str]):
    """Copy plot and report into results/KEEP/, FLAG/, or DISCARD/."""
    dest = os.path.join(output_dir, verdict)
    os.makedirs(dest, exist_ok=True)
    for src in [plot_path, report_path]:
        if src and os.path.isfile(src):
            try:
                shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
            except Exception:
                pass


def _toi_crosscheck(tic_id: int) -> bool:
    """
    WHAT: Check whether this TIC ID is in the TESS Object of Interest (TOI) catalog.
    WHY:  If it's NOT in the catalog, it's potentially a new discovery worth highlighting.
    Returns True if the star is NOT in the TOI catalog (= potentially new).
    """
    try:
        toi_ids = _load_toi_catalog()
        return int(tic_id) not in toi_ids
    except Exception as e:
        logger.warning(f"TOI crosscheck error for TIC {tic_id}: {e}")
        return False   # conservative: assume it is known if query fails



def _run_single_star(
    tic_id: int,
    output_dir: str,
    models_dir: str,
    session_label: str,
    run_flag_analyzer: bool,
    run_toi_crosscheck: bool,
    alias_rejection: bool,
    max_planet_radius_earth: float,
    min_transit_snr: float,
    min_depth_ppm: float,
    log_file: str,
    fast: bool = False,
) -> dict:
    """
    WHAT: Run the complete 13-step discovery pipeline for ONE star.
    WHY:  Encapsulated in a function so try/except in the main loop
    can catch any error and continue to the next star safely.

    Returns a result dict with keys matching results.csv columns.
    """
    result = {
        "tic_id": tic_id, "session_label": session_label,
        "decision": "DISCARD", "final_class": "No Signal",
        "confidence": 0.0, "period": 0.0, "period_err": 0.0,
        "depth": 0.0, "depth_err": 0.0, "duration": 0.0,
        "duration_err": 0.0, "snr": 0.0, "flag_reasons": "",
        "rp_earth": 0.0, "is_new_discovery": False, "alias_rejected": False,
        "fpp": None, "combined_fpp": None, "fpp_status": "skipped",
        "contamination_ratio": None, "n_nearby_gaia_stars": 0, "n_sectors_consistent": 1
    }
    fits_path = None

    try:
        import lightkurve as lk
        from src.preprocess import preprocess_lightcurve, fold_lightcurve
        from src.detect    import run_bls, run_tls, compute_snr
        from src.features  import extract_features
        from src.classify  import classify_target
        from src.fit_transit import fit_batman_transit
        from src.reverse_pipeline import run_reverse_pipeline
        from src.decision_engine  import evaluate_decision, log_to_manual_review_queue

        # ── Step 1: Download light curve ─────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 1: Downloading light curve from MAST...")
        raw_dir = os.path.join(output_dir, "raw_fits")
        os.makedirs(raw_dir, exist_ok=True)

        # Try 2-minute cadence first, fall back to 30-minute
        fits_path = None
        resolved_cadence = None
        for cadence in ["short", "long"]:
            try:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}", mission="TESS", cadence=cadence, limit=1
                )
                if search is not None and len(search) > 0:
                    logger.info(f"TIC {tic_id} -- Step 1: Found {cadence} cadence data. Downloading...")
                    lc_col = search.download_all(download_dir=raw_dir)
                    if lc_col is not None and len(lc_col) > 0:
                        lc_obj = lc_col[0]
                        fits_path = "in_memory"   # we have the lc object
                        resolved_cadence = cadence
                        break
            except Exception as e_dl:
                logger.warning(f"TIC {tic_id} -- Step 1: Cadence {cadence} download attempt failed: {e_dl}")
                continue

        if fits_path is None:
            logger.error(f"TIC {tic_id} -- Step 1: Failed to download light curve from MAST.")
            result["flag_reasons"] = "no_data: could not download from MAST"
            return result
        logger.info(f"TIC {tic_id} -- Step 1: Light curve downloaded successfully ({resolved_cadence} cadence).")

        # ── Step 2: Preprocess ───────────────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 2: Preprocessing light curve...")
        time_arr, flux_arr, flux_err = preprocess_lightcurve(lc_obj)
        quality_flag = "good" if len(time_arr) > 8000 else "poor"
        logger.info(f"TIC {tic_id} -- Step 2: Preprocessing complete. Data points: {len(time_arr)} (Quality: {quality_flag})")

        # ── Step 3: TLS/BLS period search + alias rejection ──────────────────
        if fast:
            logger.info(f"TIC {tic_id} -- Step 3: Fast mode enabled. Running Box Least Squares (BLS)...")
            periods, power, bls_params = run_bls(time_arr, flux_arr, flux_err)
        else:
            logger.info(f"TIC {tic_id} -- Step 3: Running Transit Least Squares (TLS)...")
            periods, power, bls_params = run_tls(time_arr, flux_arr, flux_err)

        best_period = bls_params.get("period", 0.0)
        snr = bls_params.get("snr", 0.0)
        depth_val = bls_params.get("depth", 0.0)
        depth_ppm = abs(depth_val) * 1e6

        logger.info(f"TIC {tic_id} -- Step 3: Search complete. Top Period: {best_period:.4f} d, SNR: {snr:.2f}, Depth: {depth_ppm:.0f} ppm")

        if alias_rejection and bls_params.get("alias_rejected"):
            reason = (
                f"alias_discard: top period {bls_params['period']:.4f} d "
                f"matches TESS systematic alias"
            )
            logger.warning(f"TIC {tic_id} -- Step 3: Target rejected as systematic alias (P={best_period:.4f} d)")
            result["decision"]        = "DISCARD"
            result["flag_reasons"]    = reason
            result["period"]          = bls_params["period"]
            result["snr"]             = bls_params["snr"]
            result["alias_rejected"]  = True
            _log_star(log_file, tic_id, "alias_discard",
                      f"P={bls_params['period']:.3f}d")
            return result

        result["period"] = best_period
        result["snr"]    = snr
        result["depth"]  = depth_val

        # SNR gate
        if snr < min_transit_snr:
            logger.warning(f"TIC {tic_id} -- Step 3: SNR {snr:.2f} < threshold {min_transit_snr}. Discarding.")
            result["flag_reasons"] = f"low_snr: {snr:.2f} < {min_transit_snr}"
            _log_star(log_file, tic_id, "DISCARD", f"SNR={snr:.1f}")
            return result

        # Depth gate (ppm)
        if depth_ppm < min_depth_ppm:
            logger.warning(f"TIC {tic_id} -- Step 3: Depth {depth_ppm:.0f} ppm < threshold {min_depth_ppm} ppm. Discarding.")
            result["flag_reasons"] = f"shallow_transit: {depth_ppm:.0f} ppm < {min_depth_ppm} ppm"
            _log_star(log_file, tic_id, "DISCARD", f"depth={depth_ppm:.0f}ppm")
            return result

        # ── Step 4: Feature extraction ────────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 4: Extracting light curve features...")
        features = extract_features(time_arr, flux_arr, flux_err, bls_params)
        logger.info(f"TIC {tic_id} -- Step 4: Feature extraction complete ({len(features)} features extracted).")

        # ── Step 5: Forward pipeline (RF + CNN) ───────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 5: Running Machine Learning classifiers (RF + CNN)...")
        phase_arr, folded_flux = fold_lightcurve(
            time_arr, flux_arr, bls_params["period"], bls_params["t0"]
        )
        classification = classify_target(features, folded_flux)
        result["final_class"] = classification.get("label_name", "Unknown")
        result["confidence"]  = classification.get("confidence", 0.0)
        logger.info(f"TIC {tic_id} -- Step 5: ML Classification: {result['final_class']} (Confidence: {result['confidence']:.2%})")

        # ── Step 6: Batman fitting + radius plausibility ──────────────────────
        logger.info(f"TIC {tic_id} -- Step 6: Fitting transit model using batman...")
        # Determine sector from light curve metadata
        sector_val = None
        try:
            if hasattr(lc_obj, "sector"):
                sector_val = lc_obj.sector
            elif hasattr(lc_obj, "meta") and "SECTOR" in lc_obj.meta:
                sector_val = lc_obj.meta["SECTOR"]
            
            # Plain English: Safe check to convert list/array sector values to a single integer
            if sector_val is not None:
                if hasattr(sector_val, "__iter__") and not isinstance(sector_val, (str, bytes)):
                    sector_val = int(sector_val[0])
                else:
                    sector_val = int(sector_val)
        except Exception as se_err:
            logger.warning(f"TIC {tic_id} -- Step 6: Failed to resolve sector value: {se_err}")
            sector_val = None

        # Add tic_id to bls_params for stellar parameter lookup inside fit_batman_transit
        bls_params["tic_id"] = tic_id
        transit_params = fit_batman_transit(
            time_arr, flux_arr, flux_err, bls_params, n_bootstrap=25
        )
        rp_earth = transit_params.get("rp_earth", 0.0)
        result["rp_earth"]       = rp_earth
        result["period"]         = transit_params.get("period", result["period"])
        result["depth"]          = transit_params.get("transit_depth", result["depth"])
        result["duration"]       = transit_params.get("transit_duration_hr", 0.0) / 24.0
        logger.info(f"TIC {tic_id} -- Step 6: Batman fit complete. Fitted Radius: {rp_earth:.2f} R_earth")

        # Radius plausibility check
        radius_flag = None
        radius_note = None
        if rp_earth < 0.5:
            radius_flag = "too_small_flag"
            radius_note = f"Fitted radius {rp_earth:.2f} R_earth below TESS sensitivity limit."
            logger.warning(f"TIC {tic_id} -- Step 6: fitted radius too small: {rp_earth:.2f} R_earth")
        elif rp_earth > 100.0:
            radius_flag = "almost_certainly_eb"
            radius_note = (f"Fitted radius {rp_earth:.1f} R_earth in stellar range (>100). "
                           f"Almost certainly an eclipsing binary.")
            logger.warning(f"TIC {tic_id} -- Step 6: fitted radius in stellar range: {rp_earth:.1f} R_earth")
        elif rp_earth > max_planet_radius_earth:
            radius_flag = "giant_radius_eb_suspect"
            radius_note = (f"Fitted radius {rp_earth:.1f} R_earth exceeds {max_planet_radius_earth} R_earth. "
                           f"Likely eclipsing binary or blended EB. Recommend EB catalog cross-check.")
            logger.warning(f"TIC {tic_id} -- Step 6: fitted radius too large: {rp_earth:.1f} R_earth")

        # ── Step 7: Pixel Contamination Check ─────────────────────────────────
        contamination_res = {"contaminated": False, "contamination_ratio": None, "n_nearby_gaia_stars": 0}
        if not fast:
            logger.info(f"TIC {tic_id} -- Step 7: Running pixel-level contamination check...")
            try:
                from flag_analyzer import check_pixel_contamination
                contamination_res = check_pixel_contamination(
                    tic_id=tic_id,
                    sector=sector_val,
                    period=bls_params["period"],
                    epoch=bls_params["t0"],
                    duration=bls_params["duration"]
                )
                logger.info(f"TIC {tic_id} -- Step 7: Contamination check complete. Contaminated: {contamination_res.get('contaminated')} (Ratio: {contamination_res.get('contamination_ratio')})")
            except Exception as e:
                logger.warning(f"TIC {tic_id} -- Step 7: Pixel contamination check failed: {e}")
        else:
            logger.info(f"TIC {tic_id} -- Step 7: Fast mode enabled. Skipping pixel contamination check.")

        # ── Step 8: Reverse pipeline ──────────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 8: Running reverse pipeline vetting...")
        reverse_results = run_reverse_pipeline(
            time_arr, flux_arr, flux_err, bls_params
        )
        logger.info(f"TIC {tic_id} -- Step 8: Reverse pipeline vetting complete.")

        # Determine consistent sector count using lightkurve search
        n_sectors_consistent = 1
        try:
            search_sectors = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            if len(search_sectors) > 0:
                sectors = search_sectors.table["sequence_number"]
                n_sectors_consistent = int(len(np.unique(sectors)))
        except Exception:
            pass

        # ── Step 9: Vetting / FPP Calculation ────────────────────────────────
        fpp_res = {"fpp": None, "combined_fpp": None, "fpp_status": "skipped"}
        # Evaluate initial decision first so we know if we need FPP (FPP only for KEEP)
        decision_res = evaluate_decision(
            classification, reverse_results, bls_params, quality_flag,
            forward_fit=transit_params, n_sectors_consistent=n_sectors_consistent
        )

        if decision_res["decision"] == "KEEP" and not fast:
            logger.info(f"TIC {tic_id} -- Step 9: Running TRICERATOPS False Positive Probability calculation...")
            try:
                from src.fpp_calculator import calculate_fpp
                fpp_res = calculate_fpp(
                    tic_id=tic_id,
                    period=transit_params["period"],
                    epoch=transit_params["t0"],
                    depth=transit_params["transit_depth"],
                    duration=transit_params["transit_duration_hr"] / 24.0,
                    sector=sector_val,
                    time=time_arr,
                    flux=flux_arr
                )
                fpp = fpp_res.get("fpp")
                combined_fpp = fpp_res.get("combined_fpp")
                fpp_status = fpp_res.get("fpp_status")
                logger.info(f"TIC {tic_id} -- Step 9: FPP complete. FPP: {fpp}, Combined FPP: {combined_fpp} ({fpp_status})")
                
                if combined_fpp is not None:
                    if combined_fpp > 0.5:
                        logger.warning(f"TIC {tic_id} -- Step 9: High false positive probability ({combined_fpp:.2f} > 0.5). Downgrading KEEP to FLAG.")
                        decision_res["decision"] = "FLAG"
                        decision_res["flag_reasons"] = f"High false positive probability: {fpp:.2f}"
            except Exception as e:
                logger.warning(f"TIC {tic_id} -- Step 9: FPP calculation failed: {e}")
        else:
            logger.info(f"TIC {tic_id} -- Step 9: Skipping FPP calculation (Verdict: {decision_res['decision']}).")

        # ── Step 10: Vetting / Decision Engine ────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 10: Evaluating final decision using decision engine...")
        # Apply radius overrides AFTER decision (radius evidence always wins)
        if radius_flag in ("giant_radius_eb_suspect", "almost_certainly_eb"):
            original_label = classification.get("label_name", "Unknown")
            classification["label"] = 3
            classification["label_name"] = "Eclipsing Binary"
            classification["confidence"] = 0.0
            decision_res["decision"]            = "DISCARD"
            decision_res["combined_confidence"] = 0.0
            decision_res["flag_reasons"] = f"[{radius_flag}] {radius_note} (original ML label was: {original_label})"
            logger.info(f"TIC {tic_id} -- Step 10: Radius override applied. Forced to DISCARD.")
        elif radius_flag == "too_small_flag":
            existing = decision_res.get("flag_reasons", "")
            decision_res["flag_reasons"] = f"[too_small_flag] {radius_note}" + (f"; {existing}" if existing else "")

        # Apply pixel contamination overrides
        if contamination_res.get("contaminated") is True:
            is_borderline = (decision_res["decision"] == "FLAG") or (decision_res["decision"] == "KEEP" and decision_res["combined_confidence"] < 0.75)
            if rp_earth > 25.0:
                classification["label"] = 3
                classification["label_name"] = "Eclipsing Binary"
                decision_res["decision"] = "DISCARD"
                decision_res["combined_confidence"] = 0.0
                decision_res["flag_reasons"] = "giant_radius_plus_contamination — almost certainly blend"
                logger.warning(f"TIC {tic_id} -- Step 10: Contamination override: giant radius + pixel contamination. Forced to DISCARD.")
            elif is_borderline:
                decision_res["decision"] = "FLAG"
                decision_res["flag_reasons"] = "pixel_contamination_detected"
                logger.warning(f"TIC {tic_id} -- Step 10: Contamination override: borderline signal + pixel contamination. Forced to FLAG.")

        verdict    = decision_res["decision"]
        confidence = decision_res["combined_confidence"]
        flag_reasons = decision_res.get("flag_reasons", "")

        result["decision"]     = verdict
        result["confidence"]   = confidence
        result["final_class"]  = classification.get("label_name", result["final_class"])
        result["flag_reasons"] = flag_reasons
        result["fpp"]          = fpp_res.get("fpp")
        result["combined_fpp"] = fpp_res.get("combined_fpp")
        result["fpp_status"]   = fpp_res.get("fpp_status")
        result["contamination_ratio"] = contamination_res.get("contamination_ratio")
        result["n_nearby_gaia_stars"] = contamination_res.get("n_nearby_gaia_stars")
        result["n_sectors_consistent"] = n_sectors_consistent

        logger.info(f"TIC {tic_id} -- Step 10: Decision engine completed. Final Verdict: {verdict} (Confidence: {confidence:.2%})")

        # Step 10 Flag Deep-Analysis (if flagged)
        if verdict == "FLAG" and run_flag_analyzer:
            logger.info(f"TIC {tic_id} -- Step 10: Running flag deep-analysis...")
            try:
                from flag_analyzer import run_flag_analysis
                fa_result = run_flag_analysis(tic_ids=[tic_id])
                if fa_result.get("upgraded", 0) > 0:
                    verdict = "KEEP"
                elif fa_result.get("downgraded", 0) > 0:
                    verdict = "DISCARD"
                result["decision"] = verdict
                logger.info(f"TIC {tic_id} -- Step 10: Flag deep-analysis complete. Final decision updated to: {verdict}")
            except Exception as e:
                logger.warning(f"TIC {tic_id} -- Step 10: Flag deep-analysis failed ({e}) — keeping FLAG")

            if verdict == "FLAG":
                try:
                    log_to_manual_review_queue(
                        tic_id, decision_res,
                        reverse_results.get("fit_results", {}),
                        output_dir
                    )
                except Exception:
                    pass

        # ── Step 11: TOI Cross-Check ──────────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 11: Running TOI catalog cross-check...")
        is_new = False
        if run_toi_crosscheck and verdict == "KEEP":
            try:
                is_new = _toi_crosscheck(tic_id)
                result["is_new_discovery"] = is_new
                logger.info(f"TIC {tic_id} -- Step 11: TOI cross-check complete. Potential New Discovery: {is_new}")
            except Exception as e:
                logger.warning(f"TIC {tic_id} -- Step 11: TOI cross-check failed: {e}")
        else:
            logger.info(f"TIC {tic_id} -- Step 11: Skipping TOI cross-check (Verdict: {verdict}).")

        if is_new:
            new_file = os.path.join(output_dir, "new_discoveries.txt")
            try:
                with open(new_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"TIC {tic_id} | P={result['period']:.4f}d | "
                        f"depth={result['depth']*100:.4f}% | "
                        f"Rp={rp_earth:.2f}R_earth\n"
                    )
            except Exception:
                pass

        # Generate binned model curves for plotting
        folded_model = None
        try:
            import batman
            params = batman.TransitParams()
            params.t0 = transit_params['t0']
            params.per = transit_params['period']
            params.rp = transit_params['rp_over_rs']
            params.a = transit_params['a_over_rs']
            params.inc = transit_params['inclination']
            params.ecc = 0.0
            params.w = 90.0
            params.u = [0.4, 0.3]
            params.limb_dark = "quadratic"

            phases_grid = np.linspace(-0.5, 0.5, len(folded_flux))
            time_grid = phases_grid * transit_params['period'] + transit_params['t0']
            m_model = batman.TransitModel(params, time_grid)
            folded_model = m_model.light_curve(params)
        except Exception as e:
            logger.warning(f"TIC {tic_id} -- Step 11: Failed to generate folded model curve: {e}")

        # ── Step 12: Diagnostic Plots & Report JSON saving ───────────────────
        logger.info(f"TIC {tic_id} -- Step 12: Generating diagnostic plots and saving report...")
        plot_path = None
        if verdict in ("KEEP", "FLAG"):
            try:
                from src.visualize import plot_diagnostic
                import matplotlib.pyplot as plt
                fig = plot_diagnostic(
                    tic_id=tic_id,
                    time_raw=lc_obj.time.value,
                    flux_raw=lc_obj.flux.value,
                    time_clean=time_arr,
                    flux_clean=flux_arr,
                    bls_periods=periods,
                    bls_power=power,
                    bls_params=bls_params,
                    transit_params=transit_params,
                    classification=classification,
                    phase_folded_flux=folded_flux,
                    model_flux=folded_model,
                    save=True,
                    save_dir=os.path.join(output_dir, "figures"),
                )
                plt.close(fig)
                plot_path = os.path.join(
                    output_dir, "figures", f"TIC_{tic_id}_diagnostic.png"
                )
            except Exception as e:
                logger.warning(f"TIC {tic_id} -- Step 12: Plot generation failed: {e}")

        # Save report JSON
        report_path = None
        try:
            def _clean_for_json(obj):
                if isinstance(obj, dict):
                    return {k: _clean_for_json(v) for k, v in obj.items() if k != "raw_results" and not k.startswith("_")}
                elif isinstance(obj, list):
                    return [_clean_for_json(x) for x in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, "tolist"):
                    return obj.tolist()
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except TypeError:
                        return str(obj)

            report = {
                "tic_id": tic_id, "session_label": session_label,
                "decision": verdict, "classification": classification,
                "transit_parameters": transit_params,
                "bls_parameters": bls_params,
                "reverse_results": reverse_results,
            }
            clean_report = _clean_for_json(report)
            report_dir = os.path.join(output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"TIC_{tic_id}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(clean_report, f, indent=2)
        except Exception as e:
            logger.warning(f"TIC {tic_id} -- Step 12: Report save failed ({e})")

        # Copy to verdict folder
        try:
            _copy_to_verdict_folder(output_dir, tic_id, verdict, plot_path, report_path)
        except Exception:
            pass

        # Log one line to discovery_log.txt
        detail = (f"{result['final_class']}, {confidence*100:.0f}%, "
                  f"P={result['period']:.3f}d")
        _log_star(log_file, tic_id, verdict, detail)
        logger.info(f"TIC {tic_id} -- Step 12: Diagnostic plots and report saved.")

    except Exception as e:
        logger.error(f"TIC {tic_id} -- Pipeline crashed: {e}")
        result["flag_reasons"] = f"pipeline_error: {e}"
        _log_star(log_file, tic_id, "ERROR", str(e)[:60])

    finally:
        # ── Step 13: Cleanup ──────────────────────────────────────────────────
        logger.info(f"TIC {tic_id} -- Step 13: Cleaning up temporary FITS files...")
        # Always delete raw FITS files to conserve disk space
        try:
            if fits_path and fits_path != "in_memory":
                if os.path.isfile(fits_path):
                    os.remove(fits_path)
        except Exception:
            pass

        # Also purge raw_fits directory if it is getting large
        try:
            raw_dir = os.path.join(output_dir, "raw_fits")
            if os.path.isdir(raw_dir):
                for f in os.listdir(raw_dir):
                    fp = os.path.join(raw_dir, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
        except Exception:
            pass
        logger.info(f"TIC {tic_id} -- Step 13: Cleanup complete.")

    return result



def run_discovery_session(
    targets: pd.DataFrame,
    output_dir: str,
    models_dir: str,
    time_limit_hours: float = 8.5,
    disk_limit_gb: float = 18.0,
    save_every_n: int = 50,
    session_label: str = "",
    run_flag_analyzer: bool = True,
    run_candidate_export: bool = True,
    run_toi_crosscheck: bool = True,
    alias_rejection: bool = True,
    max_planet_radius_earth: float = 25.0,
    min_transit_snr: float = 5.0,
    min_depth_ppm: float = 100.0,
    log_file: str = "",
    fast: bool = False,
) -> dict:
    """
    WHAT: Main discovery loop. Runs the 13-step pipeline on every star
          in the target DataFrame until the time or disk limit is hit.
    WHY:  Kaggle sessions time out after ~9 hours. This function respects
          that limit, saves progress incrementally, and never loses results
          from stars already processed.

    Returns a summary dict with counts for the session.
    """
    # Force logger to output to stdout for visibility in Kaggle logs/notebooks
    import sys
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(sh)

    if not session_label:
        session_label = datetime.now().strftime("%Y-%m-%d")
    if not log_file:
        log_file = os.path.join(output_dir, "discovery_log.txt")

    results_csv = os.path.join(output_dir, "results.csv")
    start_time  = time.time()
    time_limit_sec = time_limit_hours * 3600

    os.makedirs(output_dir, exist_ok=True)

    n_attempted  = 0
    n_processed  = 0
    n_skipped    = 0
    n_alias      = 0
    n_keep       = 0
    n_flag       = 0
    n_discard    = 0
    n_new        = 0
    pending_rows = []   # buffer for incremental CSV saves

    target_list = targets["tic_id"].tolist()
    keep_tic_ids = []   # for smart re-prioritisation

    logger.info(f"Starting discovery session: {len(target_list)} stars, "
                f"limit {time_limit_hours} hrs / {disk_limit_gb} GB")

    pbar = tqdm(total=len(target_list), desc="Discovery", unit="star")

    for i, tic_id in enumerate(target_list):
        tic_id = int(tic_id)
        n_attempted += 1

        # ── Time check (every star) ───────────────────────────────────────────
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            logger.info(f"Time limit reached ({elapsed/3600:.1f} hrs). Stopping.")
            pbar.write(f"Time limit reached after {n_processed} stars.")
            break

        # ── Disk check (every 100 stars) ──────────────────────────────────────
        if i % 100 == 0 and i > 0:
            disk_used = _get_disk_used_gb(output_dir)
            if disk_used > disk_limit_gb:
                logger.warning(f"Disk limit reached ({disk_used:.1f} GB). "
                               f"Finishing current queue.")
                pbar.write(f"Disk limit ({disk_limit_gb} GB) reached. Stopping downloads.")
                break

        # ── Smart re-prioritisation (every 500 stars) ─────────────────────────
        if i > 0 and i % 500 == 0 and keep_tic_ids:
            # Move high-SNR stars in the same sky region to the front
            try:
                remaining = target_list[i+1:]
                keep_rows = targets[targets["tic_id"].isin(keep_tic_ids)]
                if len(keep_rows) > 0:
                    mean_ra  = keep_rows["ra"].mean() if "ra"  in keep_rows.columns else None
                    mean_dec = keep_rows["dec"].mean() if "dec" in keep_rows.columns else None
                    if mean_ra is not None and mean_dec is not None:
                        rem_df = targets[targets["tic_id"].isin(remaining)].copy()
                        rem_df["sky_dist"] = np.sqrt(
                            (rem_df.get("ra",  mean_ra)  - mean_ra)**2 +
                            (rem_df.get("dec", mean_dec) - mean_dec)**2
                        )
                        rem_df = rem_df.sort_values("sky_dist")
                        target_list[i+1:] = rem_df["tic_id"].tolist()
                        pbar.write(f"Re-prioritised remaining targets around {mean_ra:.2f}, {mean_dec:.2f}")
            except Exception:
                pass   # re-prioritisation is optional

        # ── Run the pipeline for this star ────────────────────────────────────
        pbar.set_postfix({"tic": tic_id})
        try:
            row = _run_single_star(
                tic_id               = tic_id,
                output_dir           = output_dir,
                models_dir           = models_dir,
                session_label        = session_label,
                run_flag_analyzer    = run_flag_analyzer,
                run_toi_crosscheck   = run_toi_crosscheck,
                alias_rejection      = alias_rejection,
                max_planet_radius_earth = max_planet_radius_earth,
                min_transit_snr      = min_transit_snr,
                min_depth_ppm        = min_depth_ppm,
                log_file             = log_file,
                fast                 = fast,
            )
        except Exception as e:
            logger.error(f"Unexpected error for TIC {tic_id}: {e}")
            n_skipped += 1
            pbar.update(1)
            continue

        n_processed += 1
        verdict = row.get("decision", "DISCARD")

        if row.get("alias_rejected"):
            n_alias += 1
        if "no_data" in row.get("flag_reasons", ""):
            n_skipped += 1
        if verdict == "KEEP":
            n_keep += 1
            keep_tic_ids.append(tic_id)
        elif verdict == "FLAG":
            n_flag += 1
        else:
            n_discard += 1

        if row.get("is_new_discovery"):
            n_new += 1

        pending_rows.append(row)

        # ── Incremental save every save_every_n stars ─────────────────────────
        if len(pending_rows) >= save_every_n:
            try:
                for r in pending_rows:
                    _write_csv_row(results_csv, r)
                pending_rows = []
                logger.info(f"Checkpoint: saved {n_processed} stars so far.")
            except Exception as e:
                logger.warning(f"Incremental save failed: {e}")

        pbar.update(1)

    pbar.close()

    # ── Final save for any remaining rows ─────────────────────────────────────
    try:
        for r in pending_rows:
            _write_csv_row(results_csv, r)
        pending_rows = []
    except Exception as e:
        logger.error(f"Final save failed: {e}")

    # ── Export candidate submission CSV ───────────────────────────────────────
    if run_candidate_export:
        try:
            if os.path.isfile(results_csv):
                df = pd.read_csv(results_csv, on_bad_lines='skip')
                candidates = df[df["decision"] == "KEEP"].copy()
                candidates.to_csv(
                    os.path.join(output_dir, "candidates_submission.csv"),
                    index=False
                )
                logger.info(f"Candidates CSV: {len(candidates)} KEEP targets exported.")
        except Exception as e:
            logger.warning(f"Candidate export failed: {e}")

    elapsed_hrs = (time.time() - start_time) / 3600

    summary = {
        "stars_attempted":  n_attempted,
        "stars_processed":  n_processed,
        "stars_skipped":    n_skipped,
        "alias_discards":   n_alias,
        "elapsed_hours":    round(elapsed_hrs, 2),
        "keep":             n_keep,
        "flag":             n_flag,
        "discard":          n_discard,
        "new_discoveries":  n_new,
    }

    # Plain English: Write metadata.json to save the version of the pipeline.
    try:
        meta_path = os.path.join(output_dir, "metadata.json")
        meta_data = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        meta_data["pipeline_version"] = "2.0"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)
        logger.info("Updated metadata.json with pipeline_version: 2.0")
    except Exception as e:
        logger.warning(f"Failed to update metadata.json: {e}")

    logger.info(f"Session complete: {summary}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SESSION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def generate_session_summary(
    results_dir: str,
    session_label: str,
    session_data: Optional[dict] = None,
) -> dict:
    """
    WHAT: Read results.csv and compute cumulative statistics for this run.
    WHY:  Gives the user an at-a-glance report of what was found, how
          many stars were processed all-time, and whether any new
          discoveries are waiting for follow-up.

    Args:
        results_dir:   Path to the results/ directory.
        session_label: YYYY-MM-DD string for this session.
        session_data:  Dict returned by run_discovery_session() (optional).

    Returns a summary dict.
    """
    if session_data is None:
        session_data = {}

    results_csv = os.path.join(results_dir, "results.csv")

    # ── Load cumulative results ───────────────────────────────────────────────
    df_all = None
    try:
        if os.path.isfile(results_csv):
            df_all = pd.read_csv(results_csv, on_bad_lines='skip')
    except Exception as e:
        logger.warning(f"Could not load results.csv: {e}")

    n_cumulative = len(df_all) if df_all is not None else 0

    # ── Counts from this session ──────────────────────────────────────────────
    n_keep    = session_data.get("keep", 0)
    n_flag    = session_data.get("flag", 0)
    n_discard = session_data.get("discard", 0)
    n_new     = session_data.get("new_discoveries", 0)

    # Refine from CSV if this session's rows are tagged
    if df_all is not None and "session_label" in df_all.columns:
        this_session = df_all[df_all["session_label"] == session_label]
        if len(this_session) > 0:
            from collections import Counter
            counts = Counter(this_session["decision"].fillna("DISCARD"))
            n_keep    = counts.get("KEEP", n_keep)
            n_flag    = counts.get("FLAG", n_flag)
            n_discard = counts.get("DISCARD", n_discard)
            if "is_new_discovery" in this_session.columns:
                n_new = int(this_session["is_new_discovery"].sum())

    summary = {
        "stars_attempted":  session_data.get("stars_attempted", 0),
        "stars_processed":  session_data.get("stars_processed", 0),
        "stars_skipped":    session_data.get("stars_skipped", 0),
        "alias_discards":   session_data.get("alias_discards", 0),
        "elapsed_hours":    session_data.get("elapsed_hours", 0.0),
        "keep":             n_keep,
        "flag":             n_flag,
        "discard":          n_discard,
        "new_discoveries":  n_new,
        "cumulative_total": n_cumulative,
    }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8 — GITHUB INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_github_username(token: str) -> Optional[str]:
    """Fetch the authenticated user's login username from the GitHub API."""
    import urllib.request
    import json
    url = "https://api.github.com/user"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            return res_data.get("login")
    except Exception as e:
        logger.error(f"Error fetching GitHub username: {e}")
        return None

def create_github_repo(token: str, repo_name: str) -> bool:
    """Create a new GitHub repository for the user if it doesn't already exist."""
    import urllib.request
    import json
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "name": repo_name,
        "description": "Autonomous exoplanet discovery results and logs",
        "private": True
    }).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            logger.info(f"GitHub repository '{repo_name}' created successfully.")
            return True
    except Exception as e:
        logger.info(f"GitHub repository check/creation: {e}")
        return False

def push_results_to_github(
    results_dir: str,
    token: str,
    repo_name: str = "tess-discovery-results",
    session_label: str = ""
) -> bool:
    """
    WHAT: Push cumulative exoplanet results and plots to a private GitHub repo.
    WHY:  Allows the agent to act as an independent automation that automatically
          reports back discovery results to the user's GitHub account.
    """
    username = get_github_username(token)
    if not username:
        logger.error("Could not retrieve GitHub username. Skipping GitHub push.")
        return False

    # Check/create the GitHub repo
    create_github_repo(token, repo_name)
    
    # Configure git authentication URL
    remote_url = f"https://{token}@github.com/{username}/{repo_name}.git"
    
    def run_git(args, allow_fail=False):
        res = subprocess.run(args, cwd=str(results_dir), capture_output=True, text=True)
        if res.returncode != 0 and not allow_fail:
            logger.error(f"Git command failed: {' '.join(args)}\nError: {res.stderr.strip()}")
            return False
        return True

    logger.info(f"Pushing results to GitHub repo {username}/{repo_name}...")
    
    # Init git repo inside results_dir
    if not run_git(["git", "init"]): return False
    run_git(["git", "config", "user.name", "TESS Discovery Agent"])
    run_git(["git", "config", "user.email", "agent@tess-pipeline.local"])
    
    # Add remote
    run_git(["git", "remote", "remove", "origin"], allow_fail=True)
    if not run_git(["git", "remote", "add", "origin", remote_url]): return False
    
    # Fetch from remote to sync
    run_git(["git", "fetch", "origin"], allow_fail=True)
    
    # Check out main branch or create it
    if not run_git(["git", "checkout", "main"], allow_fail=True):
        run_git(["git", "checkout", "-b", "main"], allow_fail=True)
        
    # Stage results files
    files_to_add = [
        "results.csv",
        "candidates_submission.csv",
        "manual_review_queue.csv",
        "DISCOVERY_LOG.md",
        "new_discoveries.txt",
        "discovery_log.txt",
        "this_week_targets.csv"
    ]
    for f in files_to_add:
        if os.path.exists(os.path.join(results_dir, f)):
            run_git(["git", "add", f])
            
    # Stage directories if they exist
    for folder in ["plots", "reports"]:
        if os.path.isdir(os.path.join(results_dir, folder)):
            run_git(["git", "add", folder])
            
    # Commit changes
    commit_msg = f"Auto-update exoplanet discovery: {session_label}"
    run_git(["git", "commit", "-m", commit_msg], allow_fail=True)
    
    # Push to GitHub
    if run_git(["git", "push", "-u", "origin", "main"]):
        logger.info(f"SUCCESS: Results successfully pushed to GitHub repo: https://github.com/{username}/{repo_name}")
        return True
    else:
        logger.error("Failed to push to GitHub remote repository.")
        return False
