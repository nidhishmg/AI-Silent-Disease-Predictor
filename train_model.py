"""
train_model.py — Model training pipeline for AI Silent Disease Predictor.

Trains a RandomForestClassifier on **real clinical datasets** with optional
synthetic augmentation, and saves:
    • models/health_model.pkl
    • models/scaler.pkl

Usage::

    # 1. Download real datasets (one-time)
    python data/download_datasets.py

    # 2. Train
    python train_model.py

Integrated Datasets
-------------------
- UCI Heart Disease  (303 records)  — cardiovascular clinical attributes
- PIMA Diabetes      (768 records)  — metabolic / demographic attributes

If dataset files are not found, the pipeline falls back to synthetic data
and prints a warning.
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
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
    SYNTHETIC_AUGMENT,
    TEST_SPLIT,
    TRAINING_SAMPLES,
    TRAINING_SEED,
)
from data.data_loader import load_real_datasets
from utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Synthetic data generation (fallback + augmentation)
# ==============================================================================

def generate_synthetic_data(
    n_samples: int = TRAINING_SAMPLES,
    seed: int = TRAINING_SEED,
) -> pd.DataFrame:
    """Generate medically-informed synthetic biomarker data.

    Used as:
        - **Fallback** when no real datasets are available
        - **Augmentation** alongside real data for larger training set
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic samples (seed=%d)", n_samples, seed)

    # ------- Base feature distributions ------- #
    face_fatigue = rng.beta(2, 5, n_samples)
    symmetry_score = rng.beta(8, 2, n_samples)
    blink_instability = rng.beta(2, 6, n_samples)
    brightness_variance = rng.beta(2, 5, n_samples)

    voice_stress = rng.beta(2, 5, n_samples)
    breathing_score = rng.beta(2, 5, n_samples)
    pitch_instability = rng.beta(2, 6, n_samples)

    # Composite scores
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

    # Health risk label via sigmoid
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
        + rng.normal(0, 0.3, n_samples)
    )
    risk_prob = 1.0 / (1.0 + np.exp(-2.0 * (risk_linear - 1.5)))
    health_risk = (rng.random(n_samples) < risk_prob).astype(int)

    return pd.DataFrame({
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
    })


# ==============================================================================
# Data assembly: real + synthetic
# ==============================================================================

def assemble_training_data(seed: int = TRAINING_SEED) -> pd.DataFrame:
    """Load real datasets and optionally augment with synthetic samples.

    Priority:
        1. Real clinical data (UCI Heart Disease + PIMA Diabetes)
        2. Synthetic augmentation to reach target sample count
        3. Full synthetic fallback if no real data available
    """
    print("\n" + "=" * 60)
    print("  DATA ASSEMBLY")
    print("=" * 60)

    real_df = load_real_datasets(seed)

    if real_df is not None:
        # Drop internal _source column
        if "_source" in real_df.columns:
            real_df = real_df.drop(columns=["_source"])

        n_real = len(real_df)
        print(f"\n  ✓ Loaded {n_real} real clinical records")

        # Augment with synthetic data
        n_augment = max(SYNTHETIC_AUGMENT, 0)
        if n_augment > 0:
            print(f"  + Generating {n_augment} synthetic augmentation samples")
            synth_df = generate_synthetic_data(n_samples=n_augment, seed=seed + 100)
            df = pd.concat([real_df, synth_df], ignore_index=True)
        else:
            df = real_df

        # Shuffle
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n_total = len(df)
        real_pct = n_real / n_total * 100
        print(f"\n  Total training samples: {n_total}")
        print(f"  Real data:     {n_real} ({real_pct:.1f}%)")
        print(f"  Synthetic:     {n_total - n_real} ({100 - real_pct:.1f}%)")

    else:
        print("\n  ⚠  No real datasets found — using synthetic data only")
        print("     Run:  python data/download_datasets.py")
        df = generate_synthetic_data(n_samples=TRAINING_SAMPLES, seed=seed)

    n_risk0 = int((df["health_risk"] == 0).sum())
    n_risk1 = int((df["health_risk"] == 1).sum())
    logger.info("Final dataset — total: %d, risk=0: %d, risk=1: %d", len(df), n_risk0, n_risk1)
    print(f"  Label dist:    risk=0: {n_risk0},  risk=1: {n_risk1}")

    return df


# ==============================================================================
# Training pipeline
# ==============================================================================

def train_and_save() -> None:
    """Complete training pipeline: load data → train → evaluate → save."""
    logger.info("=" * 60)
    logger.info("AI Silent Disease Predictor — Model Training v%s", MODEL_VERSION)
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print(f"  AI Silent Disease Predictor — Training Pipeline v{MODEL_VERSION}")
    print("=" * 60)

    # 1. Assemble data
    df = assemble_training_data()

    X = df[FEATURE_NAMES].values
    y = df["health_risk"].values

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=TRAINING_SEED, stratify=y  # type: ignore
    )
    logger.info("Split — train: %d,  test: %d", len(X_train), len(X_test))
    print(f"\n  Train: {len(X_train)} samples,  Test: {len(X_test)} samples")

    # 3. Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train classifier
    print("\n  Training RandomForest ...")
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=TRAINING_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # 5. Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    logger.info("CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # 6. Test evaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report: str = classification_report(  # type: ignore
        y_test, y_pred, target_names=["Low Risk", "High Risk"]
    )
    logger.info("\nClassification Report:\n%s", report)
    print(f"\n  Test Accuracy: {acc:.4f}  ({acc * 100:.1f}%)")
    print("\n" + report)

    # 7. Feature importances
    importances = model.feature_importances_
    print("Feature Importances:")
    print("-" * 50)
    for fname, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"  {fname:<22s} {imp:.4f}  {bar}")
        logger.info("Feature importance: %s = %.4f", fname, imp)

    # 8. Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved → %s", MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    print(f"✓ Scaler saved to {SCALER_PATH}")

    # 9. Validate saved artefacts
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    test_input = X_test[:1]
    test_scaled = loaded_scaler.transform(test_input)
    proba = loaded_model.predict_proba(test_scaled)
    print(f"✓ Validation — sample predict_proba: {proba[0]}")
    logger.info("Validation passed. Model is ready.")

    # 10. Summary
    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Accuracy:  {acc * 100:.1f}%")
    print(f"  CV Mean:   {cv_scores.mean() * 100:.1f}%")
    print(f"  Samples:   {len(df)}")
    print("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    train_and_save()
