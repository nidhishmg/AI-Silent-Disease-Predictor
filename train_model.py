"""
train_model.py — Model training pipeline for AI Silent Disease Predictor.

Generates synthetic medically-informed training data for the 9-feature
biomarker schema, trains a RandomForestClassifier, and saves:
    • models/health_model.pkl
    • models/scaler.pkl

Run once before launching the application::

    python train_model.py

Dataset References (for future real-data integration)
-----------------------------------------------------
- UCI Heart Disease:  https://archive.ics.uci.edu/ml/datasets/heart+disease
- PIMA Diabetes:      https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

The synthetic data uses medically-informed correlation rules inspired by
these datasets to produce a realistic mapping from facial / vocal
biomarkers to health risk labels.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Add project root to path so config is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    FEATURE_NAMES,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    SCALER_PATH,
    TEST_SPLIT,
    TRAINING_SAMPLES,
    TRAINING_SEED,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Synthetic data generation
# ==============================================================================

def generate_synthetic_data(
    n_samples: int = TRAINING_SAMPLES,
    seed: int = TRAINING_SEED,
) -> pd.DataFrame:
    """Generate medically-informed synthetic biomarker data.

    Correlation rules (medical logic):
    - High fatigue + high voice stress → likely risk=1
    - High symmetry + stable pitch → likely risk=0
    - Brightness variance (stress proxy) contributes additively
    - Breathing irregularity correlates with respiratory/cardiac risk
    - Composite risk scores are correlated with individual markers
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic training samples (seed=%d)", n_samples, seed)

    # ------- Base feature distributions ------- #
    face_fatigue = rng.beta(2, 5, n_samples)          # skewed low (most people healthy)
    symmetry_score = rng.beta(8, 2, n_samples)         # skewed high (most faces symmetric)
    blink_instability = rng.beta(2, 6, n_samples)
    brightness_variance = rng.beta(2, 5, n_samples)

    voice_stress = rng.beta(2, 5, n_samples)
    breathing_score = rng.beta(2, 5, n_samples)
    pitch_instability = rng.beta(2, 6, n_samples)

    # Composite scores (must correlate with components)
    face_risk_score = (
        0.3 * face_fatigue
        + 0.3 * (1.0 - symmetry_score)
        + 0.2 * blink_instability
        + 0.2 * brightness_variance
    ) * 100 + rng.normal(0, 3, n_samples)
    face_risk_score = np.clip(face_risk_score, 0, 100)

    voice_risk_score = (
        0.4 * voice_stress
        + 0.3 * breathing_score
        + 0.3 * pitch_instability
    ) * 100 + rng.normal(0, 3, n_samples)
    voice_risk_score = np.clip(voice_risk_score, 0, 100)

    # ------- Health risk label ------- #
    # Probability of risk=1 is a sigmoid of combined risk factors
    risk_linear = (
        1.5 * face_fatigue
        + 1.2 * voice_stress
        + 1.0 * breathing_score
        + 0.8 * pitch_instability
        + 0.8 * blink_instability
        + 0.6 * brightness_variance
        - 1.0 * symmetry_score
        + 0.01 * face_risk_score
        + 0.01 * voice_risk_score
        + rng.normal(0, 0.3, n_samples)   # noise
    )
    risk_prob = 1.0 / (1.0 + np.exp(-2.0 * (risk_linear - 1.5)))
    health_risk = (rng.random(n_samples) < risk_prob).astype(int)

    logger.info(
        "Label distribution — risk=0: %d,  risk=1: %d",
        int(np.sum(health_risk == 0)),
        int(np.sum(health_risk == 1)),
    )

    df = pd.DataFrame(
        {
            "face_fatigue": face_fatigue,
            "symmetry_score": symmetry_score,
            "blink_instability": blink_instability,
            "brightness_variance": brightness_variance,
            "voice_stress": voice_stress,
            "breathing_score": breathing_score,
            "pitch_instability": pitch_instability,
            "face_risk_score": face_risk_score,
            "voice_risk_score": voice_risk_score,
            "health_risk": health_risk,
        }
    )
    return df


# ==============================================================================
# Training pipeline
# ==============================================================================

def train_and_save() -> None:
    """Complete training pipeline: generate data → train → evaluate → save."""
    logger.info("=" * 60)
    logger.info("AI Silent Disease Predictor — Model Training v%s", MODEL_VERSION)
    logger.info("=" * 60)

    # 1. Generate data
    df = generate_synthetic_data()

    X = df[FEATURE_NAMES].values
    y = df["health_risk"].values

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=TRAINING_SEED, stratify=y  # type: ignore
    )
    logger.info(
        "Split — train: %d,  test: %d", len(X_train), len(X_test)
    )

    # 3. Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train classifier
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=TRAINING_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    report: str = classification_report(  # type: ignore
        y_test, y_pred, target_names=["Low Risk", "High Risk"]
    )
    logger.info("\nClassification Report:\n%s", report)
    print("\n" + report)

    # 6. Feature importances
    importances = model.feature_importances_
    print("\nFeature Importances:")
    print("-" * 40)
    for fname, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"  {fname:<22s} {imp:.4f}  {bar}")
        logger.info("Feature importance: %s = %.4f", fname, imp)

    # 7. Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    print(f"✓ Scaler saved to {SCALER_PATH}")

    # 8. Validate saved artefacts
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    test_input = X_test[:1]
    test_scaled = loaded_scaler.transform(test_input)
    proba = loaded_model.predict_proba(test_scaled)
    print(f"✓ Validation — sample predict_proba: {proba[0]}")
    logger.info("Validation passed. Model is ready.")


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    train_and_save()
