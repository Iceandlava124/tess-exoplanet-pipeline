"""
src/classify.py
===============
Load trained ML models and run inference (classification) on new light curves.

Supports both:
  - Random Forest (Phase 5) — feature-based
  - CNN (Phase 6) — raw phase-folded light curve input

Also computes a combined confidence score.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

LABEL_NAMES = {
    0: "No Signal",
    1: "Planet Transit",
    2: "Eclipsing Binary",
    3: "False Positive / Blend",
}


def load_random_forest(model_path: Optional[Path] = None):
    """Load the trained Random Forest classifier."""
    import joblib
    if model_path is None:
        model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Random Forest model not found at {model_path}. Train it first in notebook 05.")
    return joblib.load(model_path)


def load_cnn(model_path: Optional[Path] = None):
    """Load the trained CNN classifier."""
    import tensorflow as tf
    
    # Patch BatchNormalization for Keras 2 -> Keras 3 deserialization compatibility
    try:
        from tensorflow.keras.layers import BatchNormalization
        original_init = BatchNormalization.__init__
        def patched_init(self, *args, **kwargs):
            for k in ['renorm', 'renorm_clipping', 'renorm_momentum']:
                kwargs.pop(k, None)
            original_init(self, *args, **kwargs)
        BatchNormalization.__init__ = patched_init
    except Exception as e:
        pass

    if model_path is None:
        model_path = MODELS_DIR / "cnn_classifier.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"CNN model not found at {model_path}. Train it first in notebook 06.")
    return tf.keras.models.load_model(str(model_path))


def classify_with_rf(
    features,
    model=None,
) -> Tuple[int, np.ndarray]:
    """
    Classify a feature vector using the Random Forest.

    Args:
        features: Dict of features or array of feature values
        model:    Pre-loaded RF model (loads from disk if None)

    Returns:
        (predicted_label, class_probabilities)

    📚 LEARNING NOTE:
        Random Forest gives us PROBABILITIES for each class, not just
        a single prediction. This is called "soft classification" and
        is much more useful than just getting "Planet" or "Not Planet."

        Example output: [0.05, 0.82, 0.08, 0.05]
        → 82% probability of being a planet transit
        → 8% probability of being an eclipsing binary
        → 5% each for no signal and false positive

        We report the max probability as the "confidence score."
    """
    import pandas as pd
    if model is None:
        model = load_random_forest()

    feature_keys = [
        'bls_snr', 'transit_depth', 'odd_even_diff', 'secondary_depth',
        'bls_period', 'transit_duration_hrs', 'oot_rms'
    ]

    if isinstance(features, dict):
        features_df = pd.DataFrame([{k: features.get(k, 0.0) for k in feature_keys}])
    else:
        # If passed as an array, map back to keys to find the 7 features
        features = np.array(features).flatten()
        if len(features) == len(feature_keys):
            features_df = pd.DataFrame([features], columns=feature_keys)
        else:
            from src.features import FEATURE_NAMES
            sorted_keys = sorted(FEATURE_NAMES)
            features_dict = {k: val for k, val in zip(sorted_keys, features)}
            features_df = pd.DataFrame([{k: features_dict.get(k, 0.0) for k in feature_keys}])

    proba = model.predict_proba(features_df)[0]
    predicted = int(np.argmax(proba))
    return predicted, proba


def classify_with_cnn(
    phase_folded_lc: np.ndarray,
    model=None,
    n_bins: int = 200,
) -> Tuple[int, np.ndarray]:
    """
    Classify a phase-folded light curve using the 1D CNN.

    Args:
        phase_folded_lc: Phase-folded flux array (will be resampled to n_bins)
        model:           Pre-loaded CNN model (loads from disk if None)
        n_bins:          Expected input size for CNN

    Returns:
        (predicted_label, class_probabilities)

    📚 LEARNING NOTE:
        Unlike the Random Forest which needs handcrafted features,
        the CNN learns its OWN features directly from the raw waveform.
        We feed in the phase-folded light curve as a 1D signal,
        and the convolutional filters learn what "transit shape" looks like.

        Input shape: (1, n_bins, 1) — batch=1, time_steps=200, channels=1
    """
    if model is None:
        model = load_cnn()

    # Resample to n_bins if needed
    if len(phase_folded_lc) != n_bins:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(phase_folded_lc))
        x_new = np.linspace(0, 1, n_bins)
        f = interp1d(x_old, phase_folded_lc, kind='linear', fill_value='extrapolate')
        phase_folded_lc = f(x_new)

    # Add batch and channel dimensions
    input_tensor = phase_folded_lc.reshape(1, n_bins, 1).astype(np.float32)
    proba = model.predict(input_tensor, verbose=0)[0]
    predicted = int(np.argmax(proba))
    return predicted, proba


def ensemble_classify(
    rf_proba: np.ndarray,
    cnn_proba: np.ndarray,
    rf_weight: float = 0.4,
    cnn_weight: float = 0.6,
) -> Tuple[int, np.ndarray, float]:
    """
    Combine RF and CNN predictions for a final ensemble decision.

    Args:
        rf_proba:   RF class probabilities [4]
        cnn_proba:  CNN class probabilities [4]
        rf_weight:  Weight for RF (default 0.4)
        cnn_weight: Weight for CNN (default 0.6)

    Returns:
        (final_label, combined_proba, confidence)

    📚 LEARNING NOTE:
        Ensemble methods combine multiple models to get better results
        than any single model alone. This is the same idea as:
        - "Ask multiple experts and take a weighted vote"
        - The CNN is better at shape patterns, RF is better at
          global features like odd-even depth difference.

        We weight the CNN higher (0.6) because in practice it
        outperforms the RF on this task.
    """
    combined = rf_weight * np.array(rf_proba) + cnn_weight * np.array(cnn_proba)
    combined /= combined.sum()  # renormalise

    final_label = int(np.argmax(combined))
    confidence = float(combined[final_label])

    return final_label, combined, confidence


def classify_target(
    features: np.ndarray,
    phase_folded_lc: np.ndarray,
    rf_model=None,
    cnn_model=None,
) -> Dict:
    """
    Full classification pipeline for a single target.

    Returns a dict with all classification results and confidence scores.
    """
    try:
        rf_label, rf_proba = classify_with_rf(features, rf_model)
    except Exception as e:
        logger.warning(f"RF classification failed: {e}")
        rf_label, rf_proba = 0, np.array([0.25, 0.25, 0.25, 0.25])

    try:
        cnn_label, cnn_proba = classify_with_cnn(phase_folded_lc, cnn_model)
    except Exception as e:
        logger.warning(f"CNN classification failed: {e}")
        cnn_label, cnn_proba = rf_label, rf_proba

    final_label, combined_proba, confidence = ensemble_classify(rf_proba, cnn_proba)

    return {
        "label": final_label,
        "label_name": LABEL_NAMES[final_label],
        "confidence": confidence,
        "rf_label": rf_label,
        "rf_proba": rf_proba.tolist(),
        "cnn_label": cnn_label,
        "cnn_proba": cnn_proba.tolist(),
        "combined_proba": combined_proba.tolist(),
    }
