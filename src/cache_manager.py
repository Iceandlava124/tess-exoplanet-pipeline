"""
src/cache_manager.py
===================
SQLite query caching system for TESS exoplanet pipeline.
Caches expensive external API queries (MAST stellar parameters, Gaia cone searches/TPFs)
locally to avoid rate limits and minimize run time.
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Define cache database path
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "pipeline_cache.db"

def get_db_connection():
    """Create a connection to the SQLite database and ensure tables exist."""
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    # Use WAL mode for concurrent read/write support
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # Create tables if they do not exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stellar_params (
            tic_id INTEGER PRIMARY KEY,
            teff REAL,
            logg REAL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pixel_contamination (
            tic_id INTEGER,
            sector INTEGER,
            contamination_ratio REAL,
            n_nearby_gaia_stars INTEGER,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (tic_id, sector)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_results (
            tic_id INTEGER PRIMARY KEY,
            decision TEXT,
            final_class TEXT,
            confidence REAL,
            period REAL,
            period_err REAL,
            depth REAL,
            depth_err REAL,
            duration REAL,
            duration_err REAL,
            snr REAL,
            flag_reasons TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def get_stellar_params(tic_id):
    """
    Retrieve cached stellar parameters (Teff, logg) for a TIC ID.
    Returns (teff, logg) if found, otherwise None.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT teff, logg FROM stellar_params WHERE tic_id = ?",
            (int(tic_id),)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            # Plain English: found cached stellar parameters
            return row[0], row[1]
    except Exception as e:
        logger.warning(f"Cache read error for stellar params (TIC {tic_id}): {e}")
    return None

def save_stellar_params(tic_id, teff, logg):
    """Save stellar parameters to the cache."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO stellar_params (tic_id, teff, logg, timestamp) VALUES (?, ?, ?, ?)",
            (int(tic_id), teff, logg, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache write error for stellar params (TIC {tic_id}): {e}")

def get_pixel_contamination(tic_id, sector):
    """
    Retrieve cached pixel contamination results.
    Use sector = -1 for unspecified sector (None).
    Returns (contamination_ratio, n_nearby_gaia_stars) if found, otherwise None.
    """
    sec_key = -1 if sector is None else int(sector)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT contamination_ratio, n_nearby_gaia_stars FROM pixel_contamination WHERE tic_id = ? AND sector = ?",
            (int(tic_id), sec_key)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            # Plain English: found cached pixel contamination values
            return row[0], row[1]
    except Exception as e:
        logger.warning(f"Cache read error for pixel contamination (TIC {tic_id}, Sector {sector}): {e}")
    return None

def save_pixel_contamination(tic_id, sector, contamination_ratio, n_nearby_gaia_stars):
    """Save pixel contamination values to the cache."""
    sec_key = -1 if sector is None else int(sector)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO pixel_contamination (tic_id, sector, contamination_ratio, n_nearby_gaia_stars, timestamp) VALUES (?, ?, ?, ?, ?)",
            (int(tic_id), sec_key, contamination_ratio, n_nearby_gaia_stars, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache write error for pixel contamination (TIC {tic_id}, Sector {sector}): {e}")


def save_pipeline_result(tic_id, decision, final_class, confidence, period, period_err, depth, depth_err, duration, duration_err, snr, flag_reasons):
    """Save pipeline result to SQLite database to avoid flat CSV corruption."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO pipeline_results (
                tic_id, decision, final_class, confidence, period, period_err, 
                depth, depth_err, duration, duration_err, snr, flag_reasons, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(tic_id), str(decision), str(final_class), float(confidence), 
            float(period), float(period_err), float(depth), float(depth_err), 
            float(duration), float(duration_err), float(snr), str(flag_reasons), 
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()
        logger.info(f"Successfully cached result for TIC {tic_id} in SQLite pipeline_results database.")
    except Exception as e:
        logger.error(f"SQL database write error for pipeline result (TIC {tic_id}): {e}")


def get_all_results():
    """Retrieve all cached pipeline results as a pandas DataFrame."""
    try:
        import pandas as pd
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM pipeline_results", conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"SQL database read error for all results: {e}")
        import pandas as pd
        return pd.DataFrame()
