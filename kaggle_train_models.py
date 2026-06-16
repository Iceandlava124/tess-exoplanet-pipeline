"""
kaggle_train_models.py
======================
Self-contained script to generate synthetic data and train both the Random Forest
and the 1D-CNN models for exoplanet detection. Designed to run directly on Kaggle.

Saves:
  - models/random_forest.pkl
  - models/cnn_classifier.h5
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. Setup paths
ROOT = Path(".").resolve()
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# STEP 1: Generate Synthetic Feature Data for Random Forest (10,000 pts)
# =====================================================================
print("=" * 70)
print("STEP 1: Generating high-fidelity synthetic features for Random Forest")
print("=" * 70)

n_rf_samples = 10000
n_rf_per_class = n_rf_samples // 4

# Class labels: 0=No Signal, 1=Planet Transit, 2=Eclipsing Binary, 3=False Positive
rf_labels = np.concatenate([
    np.zeros(n_rf_per_class, dtype=int),
    np.ones(n_rf_per_class, dtype=int),
    np.full(n_rf_per_class, 2, dtype=int),
    np.full(n_rf_per_class, 3, dtype=int),
])

# Feature: SNR
rf_snr = np.zeros(n_rf_samples)
rf_snr[:n_rf_per_class] = np.random.exponential(1.5, n_rf_per_class) + 1.0
rf_snr[n_rf_per_class:2*n_rf_per_class] = np.random.uniform(7.1, 40.0, n_rf_per_class)
rf_snr[2*n_rf_per_class:3*n_rf_per_class] = np.random.uniform(15.0, 150.0, n_rf_per_class)
rf_snr[3*n_rf_per_class:] = np.random.uniform(3.0, 16.0, n_rf_per_class)

# Feature: Depth
rf_depth = np.zeros(n_rf_samples)
rf_depth[:n_rf_per_class] = np.abs(np.random.normal(0.0004, 0.0003, n_rf_per_class))
rf_depth[n_rf_per_class:2*n_rf_per_class] = np.random.uniform(0.001, 0.025, n_rf_per_class)
rf_depth[2*n_rf_per_class:3*n_rf_per_class] = np.random.uniform(0.04, 0.40, n_rf_per_class)
rf_depth[3*n_rf_per_class:] = np.random.uniform(0.0005, 0.012, n_rf_per_class)

# Feature: Odd-Even difference
rf_odd_even = np.zeros(n_rf_samples)
rf_odd_even[:n_rf_per_class] = np.abs(np.random.normal(0.0, 0.0001, n_rf_per_class))
rf_odd_even[n_rf_per_class:2*n_rf_per_class] = np.abs(np.random.normal(0.0, 0.00015, n_rf_per_class))
rf_odd_even[2*n_rf_per_class:3*n_rf_per_class] = np.random.uniform(0.005, 0.12, n_rf_per_class)
rf_odd_even[3*n_rf_per_class:] = np.abs(np.random.normal(0.0, 0.0002, n_rf_per_class))

# Feature: Secondary eclipse depth
rf_sec_depth = np.zeros(n_rf_samples)
rf_sec_depth[:n_rf_per_class] = np.abs(np.random.normal(0.0, 0.0001, n_rf_per_class))
rf_sec_depth[n_rf_per_class:2*n_rf_per_class] = np.abs(np.random.normal(0.0, 0.00012, n_rf_per_class))
rf_sec_depth[2*n_rf_per_class:3*n_rf_per_class] = np.random.uniform(0.01, 0.10, n_rf_per_class)
rf_sec_depth[3*n_rf_per_class:] = np.abs(np.random.normal(0.0, 0.0003, n_rf_per_class))

rf_period = np.random.uniform(0.5, 20.0, n_rf_samples)
rf_duration = np.random.uniform(1.0, 8.0, n_rf_samples)
rf_oot_rms = np.random.uniform(0.0001, 0.0025, n_rf_samples)

df_rf = pd.DataFrame({
    'bls_snr': np.clip(rf_snr, 0.1, None),
    'transit_depth': np.clip(rf_depth, 0.0, None),
    'odd_even_diff': np.clip(rf_odd_even, 0.0, None),
    'secondary_depth': np.clip(rf_sec_depth, 0.0, None),
    'bls_period': rf_period,
    'transit_duration_hrs': rf_duration,
    'oot_rms': rf_oot_rms,
})

# Training Random Forest Classifier
print("Training Random Forest Classifier (n_estimators=300)...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(df_rf, rf_labels)

# Evaluate via CV
scores = cross_val_score(rf_clf, df_rf, rf_labels, cv=5, scoring='accuracy')
print(f"✅ Random Forest 5-Fold CV Accuracy: {np.mean(scores)*100:.2f}%")

rf_model_path = MODELS_DIR / "random_forest.pkl"
joblib.dump(rf_clf, rf_model_path)
print(f"✅ Random Forest model saved to: {rf_model_path}")


# =====================================================================
# STEP 2: Generate Synthetic Folded Curves for CNN (12,000 pts)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Generating synthetic folded light curves for CNN")
print("=" * 70)

n_bins = 200
n_cnn_samples = 12000
n_cnn_per_class = n_cnn_samples // 4

cnn_labels = np.concatenate([
    np.zeros(n_cnn_per_class, dtype=int),
    np.ones(n_cnn_per_class, dtype=int),
    np.full(n_cnn_per_class, 2, dtype=int),
    np.full(n_cnn_per_class, 3, dtype=int),
])

def gen_noise(n_bins, noise_level):
    flux = np.ones(n_bins)
    flux += np.random.normal(0, noise_level, n_bins)
    if np.random.rand() < 0.25:
        amp = np.random.uniform(0.0001, 0.0008)
        flux += amp * np.sin(np.linspace(0, 4 * np.pi, n_bins))
    return flux

def gen_transit(n_bins, depth, duration_frac, noise_level):
    phase = np.linspace(-0.5, 0.5, n_bins)
    flux = np.ones(n_bins)
    in_transit = np.abs(phase) < (duration_frac / 2)
    if np.any(in_transit):
        norm_phase = phase[in_transit] / (duration_frac / 2)
        shape = 1.0 - 0.3 * norm_phase**2 - 0.2 * norm_phase**4
        shape /= np.max(shape)
        flux[in_transit] -= depth * shape
    flux += np.random.normal(0, noise_level, n_bins)
    return flux

def gen_eb(n_bins, primary_depth, secondary_depth, duration_frac, noise_level):
    phase = np.linspace(-0.5, 0.5, n_bins)
    flux = np.ones(n_bins)
    in_primary = np.abs(phase) < (duration_frac / 2)
    if np.any(in_primary):
        norm_phase = np.abs(phase[in_primary]) / (duration_frac / 2)
        flux[in_primary] -= primary_depth * (1.0 - norm_phase)
    sec_mask = np.abs(phase) > (0.5 - duration_frac / 2)
    if np.any(sec_mask):
        dist = 0.5 - np.abs(phase[sec_mask])
        norm_sec = dist / (duration_frac / 2)
        flux[sec_mask] -= secondary_depth * norm_sec
    flux += np.random.normal(0, noise_level, n_bins)
    return flux

def gen_fp(n_bins, depth, noise_level):
    phase = np.linspace(-0.5, 0.5, n_bins)
    flux = np.ones(n_bins)
    fp_type = np.random.choice(['skewed', 'step', 'single_v'])
    if fp_type == 'skewed':
        center = np.random.uniform(-0.08, 0.08)
        width = np.random.uniform(0.05, 0.12)
        diff = phase - center
        dip = depth * np.exp(-0.5 * (diff / width)**2) * (1 + tf.math.erf(2.0 * diff / (width * np.sqrt(2))).numpy())
        if np.max(dip) > 0:
            dip = (dip / np.max(dip)) * depth
        flux -= dip
    elif fp_type == 'step':
        jump = np.random.randint(50, 150)
        flux[jump:] -= depth
    elif fp_type == 'single_v':
        width = np.random.uniform(0.02, 0.08)
        in_transit = np.abs(phase) < (width / 2)
        flux[in_transit] -= depth * (1.0 - np.abs(phase[in_transit]) / (width / 2))
    flux += np.random.normal(0, noise_level, n_bins)
    return flux

X_cnn = np.zeros((n_cnn_samples, n_bins))

for i in range(n_cnn_per_class):
    X_cnn[i] = gen_noise(n_bins, np.random.uniform(0.0001, 0.0018))
    X_cnn[n_cnn_per_class + i] = gen_transit(n_bins, np.random.uniform(0.001, 0.025), np.random.uniform(0.02, 0.08), np.random.uniform(0.0001, 0.0012))
    X_cnn[2*n_cnn_per_class + i] = gen_eb(n_bins, np.random.uniform(0.04, 0.35), np.random.uniform(0.005, 0.15), np.random.uniform(0.03, 0.12), np.random.uniform(0.0002, 0.0018))
    X_cnn[3*n_cnn_per_class + i] = gen_fp(n_bins, np.random.uniform(0.002, 0.04), np.random.uniform(0.0002, 0.0018))

# Shuffle and reshape
shuffle_idx = np.random.permutation(n_cnn_samples)
X_cnn = X_cnn[shuffle_idx].reshape(-1, n_bins, 1).astype(np.float32)
y_cnn = tf.keras.utils.to_categorical(cnn_labels[shuffle_idx], num_classes=4)

# Build deeper CNN with dropout regularization
print("Building 1D CNN classifier architecture...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(n_bins, 1)),
    
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

# Training with callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

print("Training CNN model...")
model.fit(
    X_cnn, y_cnn,
    epochs=45,
    batch_size=64,
    validation_split=0.15,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

cnn_model_path = MODELS_DIR / "cnn_classifier.h5"
model.save(str(cnn_model_path))
print(f"✅ CNN model saved to: {cnn_model_path}")
print("\n🎉 ALL MODEL WEIGHTS TRAINED AND SAVED SUCCESSFULLY!")
