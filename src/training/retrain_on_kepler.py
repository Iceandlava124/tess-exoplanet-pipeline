"""
src/training/retrain_on_kepler.py
=================================
Kepler DR25 is the gold standard labelled dataset for exoplanet ML.
It has ~35,000 threshold crossing events with confirmed labels:
PC (planet candidate), FP (false positive), EB (eclipsing binary).
Training on this gives our models far more edge cases to learn from.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def download_kepler_dr25_labels():
    """
    Download the Kepler DR25 TCE table from NASA Exoplanet Archive.
    """
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        logger.info("Querying NASA Exoplanet Archive for cumulative KOI table...")
        # Query cumulative table
        koi_table = NasaExoplanetArchive.query_criteria(
            table="cumulative",
            select="kepid,koi_period,koi_depth,koi_duration,koi_time0bk,koi_disposition,koi_model_snr",
            cache=True
        )
        return koi_table.to_pandas()
    except Exception as e:
        logger.error(f"Failed to download Kepler DR25 labels: {e}")
        # Return empty df with expected columns
        return pd.DataFrame(columns=["kepid", "koi_period", "koi_depth", "koi_duration", "koi_time0bk", "koi_disposition", "koi_model_snr"])

def download_kepler_lightcurve(kepid):
    """Download a Kepler light curve by KeplerID using lightkurve"""
    try:
        import lightkurve as lk
        result = lk.search_lightcurve(
            f"KIC {kepid}", mission="Kepler", cadence="long"
        )
        if len(result) == 0:
            return None
        return result[0].download()
    except Exception as e:
        logger.warning(f"Could not download lightcurve for KIC {kepid}: {e}")
        return None

def extract_features_from_lc(lc, row):
    """Preprocess and extract features from a light curve using pipeline methods."""
    try:
        from src.preprocess import preprocess_lightcurve, fold_lightcurve
        from src.features import extract_features
        
        time_arr, flux_arr, flux_err = preprocess_lightcurve(lc)
        
        # Build bls_params structure matching pipeline expect
        bls_params = {
            "period": float(row["koi_period"]),
            "t0": float(row["koi_time0bk"]),
            "depth": float(row["koi_depth"]) / 1e6, # convert ppm to fractional
            "duration": float(row["koi_duration"]) / 24.0, # convert hours to days
            "snr": float(row.get("koi_model_snr", 10.0)) if not pd.isna(row.get("koi_model_snr")) else 10.0
        }
        
        features = extract_features(time_arr, flux_arr, flux_err, bls_params)
        
        # Also extract phase-folded light curve for CNN
        phase, folded_flux = fold_lightcurve(time_arr, flux_arr, bls_params["period"], bls_params["t0"])
        
        # Resample folded light curve to 200 bins
        if len(folded_flux) != 200:
            x_old = np.linspace(0, 1, len(folded_flux))
            x_new = np.linspace(0, 1, 200)
            f = interp1d(x_old, folded_flux, kind='linear', fill_value='extrapolate')
            folded_flux_resampled = f(x_new)
        else:
            folded_flux_resampled = folded_flux
            
        return features, folded_flux_resampled
    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return None, None

def save_partial_features(features_list, labels_list, folded_curves_list, output_dir):
    """Save training progress to temporary files."""
    try:
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"features": features_list, "labels": labels_list, "curves": folded_curves_list}, temp_dir / "progress.pkl")
    except Exception as e:
        logger.warning(f"Failed to save partial progress: {e}")

def generate_synthetic_data(n_samples_per_class):
    """Generate synthetic features and folded curves for Class 0 (Noise) and Class 2 (EB)."""
    n_bins = 200
    
    # Class 0: No Signal (Noise)
    features_0 = []
    curves_0 = np.zeros((n_samples_per_class, n_bins))
    for i in range(n_samples_per_class):
        snr = np.random.exponential(1.5) + 1.0
        depth = np.abs(np.random.normal(0.0004, 0.0003))
        odd_even = np.abs(np.random.normal(0.0, 0.0001))
        sec_depth = np.abs(np.random.normal(0.0, 0.0001))
        period = np.random.uniform(0.5, 20.0)
        duration = np.random.uniform(1.0, 8.0)
        oot_rms = np.random.uniform(0.0001, 0.0025)
        
        features_0.append({
            'bls_snr': snr, 'transit_depth': depth, 'odd_even_diff': odd_even,
            'secondary_depth': sec_depth, 'bls_period': period,
            'transit_duration_hrs': duration, 'oot_rms': oot_rms
        })
        
        # folded curve (pure noise + occasional sine)
        flux = np.ones(n_bins) + np.random.normal(0, oot_rms, n_bins)
        if np.random.rand() < 0.25:
            flux += np.random.uniform(0.0001, 0.0008) * np.sin(np.linspace(0, 4 * np.pi, n_bins))
        curves_0[i] = flux
        
    # Class 2: Eclipsing Binary (EB)
    features_2 = []
    curves_2 = np.zeros((n_samples_per_class, n_bins))
    for i in range(n_samples_per_class):
        snr = np.random.uniform(15.0, 150.0)
        depth = np.random.uniform(0.04, 0.40)
        odd_even = np.random.uniform(0.005, 0.12)
        sec_depth = np.random.uniform(0.01, 0.10)
        period = np.random.uniform(0.5, 20.0)
        duration = np.random.uniform(1.0, 8.0)
        oot_rms = np.random.uniform(0.0001, 0.0025)
        
        features_2.append({
            'bls_snr': snr, 'transit_depth': depth, 'odd_even_diff': odd_even,
            'secondary_depth': sec_depth, 'bls_period': period,
            'transit_duration_hrs': duration, 'oot_rms': oot_rms
        })
        
        # folded curve (primary + secondary)
        phase = np.linspace(-0.5, 0.5, n_bins)
        flux = np.ones(n_bins)
        dur_frac = (duration / 24.0) / period
        in_primary = np.abs(phase) < (dur_frac / 2)
        if np.any(in_primary):
            norm_phase = np.abs(phase[in_primary]) / (dur_frac / 2)
            flux[in_primary] -= depth * (1.0 - norm_phase)
        sec_mask = np.abs(phase) > (0.5 - dur_frac / 2)
        if np.any(sec_mask):
            dist = 0.5 - np.abs(phase[sec_mask])
            norm_sec = dist / (dur_frac / 2)
            flux[sec_mask] -= sec_depth * norm_sec
        flux += np.random.normal(0, oot_rms, n_bins)
        curves_2[i] = flux
        
    return features_0, curves_0, features_2, curves_2

def retrain_on_kepler_dr25(n_targets=5000, output_dir="models/", batch_size=100):
    """
    Downloads Kepler light curves with known labels, extracts features,
    generates synthetic noise/EB samples to balance classes, and retrains both models.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Get labels
    labels_df = download_kepler_dr25_labels()
    if len(labels_df) == 0:
        logger.error("No labels available. Aborting retraining.")
        return
        
    # Map Kepler dispositions to our label system
    label_map = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 3}
    labels_df["label"] = labels_df["koi_disposition"].map(label_map)
    labels_df = labels_df.dropna(subset=["label"])
    
    # Balance classes 1 and 3
    min_class = min(labels_df["label"].value_counts().min(), n_targets // 2)
    logger.info(f"Targeting {min_class} Kepler light curves per class (Planets vs FPs)...")
    
    balanced = labels_df.groupby("label").apply(
        lambda x: x.sample(min(len(x), min_class), random_state=42)
    ).reset_index(drop=True)
    
    features_list = []
    labels_list = []
    curves_list = []
    
    # Step 2: Download Kepler light curves and extract features
    logger.info("Downloading Kepler light curves and extracting features...")
    for _, row in tqdm(balanced.iterrows(), total=len(balanced)):
        try:
            lc = download_kepler_lightcurve(int(row["kepid"]))
            if lc is None:
                continue
            
            features, curve = extract_features_from_lc(lc, row)
            if features is not None and curve is not None:
                features_list.append(features)
                curves_list.append(curve)
                labels_list.append(int(row["label"]))
                
        except Exception as e:
            continue
            
        if len(features_list) % batch_size == 0 and len(features_list) > 0:
            save_partial_features(features_list, labels_list, curves_list, output_dir)
            
    n_kepler_samples = len(labels_list)
    logger.info(f"Extracted features for {n_kepler_samples} Kepler light curves.")
    
    if n_kepler_samples == 0:
        logger.error("No Kepler features could be extracted. Retraining cannot proceed.")
        return
        
    # Step 3: Integrate synthetic noise and EB classes to maintain 4-class classifiers
    n_samples_per_class = max(50, n_kepler_samples // 2) # balance the dataset
    logger.info(f"Adding {n_samples_per_class} synthetic Noise (Class 0) and EB (Class 2) samples...")
    features_0, curves_0, features_2, curves_2 = generate_synthetic_data(n_samples_per_class)
    
    # Combine Random Forest feature inputs
    feature_keys = [
        'bls_snr', 'transit_depth', 'odd_even_diff', 'secondary_depth',
        'bls_period', 'transit_duration_hrs', 'oot_rms'
    ]
    
    rf_features = []
    for f in features_list:
        rf_features.append({k: f.get(k, 0.0) for k in feature_keys})
        
    # Append class 0 and class 2
    rf_features.extend(features_0)
    rf_features.extend(features_2)
    
    rf_labels = labels_list + [0] * n_samples_per_class + [2] * n_samples_per_class
    
    # Retrain Random Forest
    logger.info("Training Random Forest Classifier on Kepler + Synthetic ensemble...")
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    X_rf = pd.DataFrame(rf_features)
    y_rf = np.array(rf_labels)
    rf_clf.fit(X_rf, y_rf)
    
    rf_path = output_path / "random_forest.pkl"
    joblib.dump(rf_clf, rf_path)
    logger.info(f"✅ Random Forest model saved to: {rf_path}")
    
    # Combine CNN inputs
    all_curves = []
    all_curves.extend(curves_list)
    all_curves.extend(curves_0)
    all_curves.extend(curves_2)
    
    X_cnn = np.array(all_curves).reshape(-1, 200, 1).astype(np.float32)
    y_cnn = tf.keras.utils.to_categorical(np.array(rf_labels), num_classes=4)
    
    # Retrain CNN
    logger.info("Training CNN Classifier on Kepler + Synthetic ensemble...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(200, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=9, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    
    model.fit(
        X_cnn, y_cnn,
        epochs=40,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    
    cnn_path = output_path / "cnn_classifier.h5"
    model.save(str(cnn_path))
    logger.info(f"✅ CNN model saved to: {cnn_path}")
    logger.info("🎉 Model retraining successfully completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ML models on Kepler DR25 database.")
    parser.add_argument("--n-targets", type=int, default=5000, help="Number of targets to pull (default: 5000)")
    parser.add_argument("--output-dir", type=str, default="models/", help="Output directory for trained models")
    args = parser.parse_args()
    
    retrain_on_kepler_dr25(n_targets=args.n_targets, output_dir=args.output_dir)
