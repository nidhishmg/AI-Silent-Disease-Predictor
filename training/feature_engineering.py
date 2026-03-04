"""
feature_engineering.py — Generate AI biomarker features from clinical data.

Reads the cleaned clinical dataset and produces the 9 biomarker features
used by the prediction engine, plus 4 interaction features that boost
model signal strength.

Biomarker Features (9)
----------------------
face_fatigue         — Composite of BP, age, inverse heart-rate
symmetry_score       — ECG/cholesterol-derived structural indicator
blink_instability    — Glucose & insulin-resistance proxy
brightness_variance  — Skin/metabolic surface indicator
voice_stress         — Glucose + BMI stress proxy
breathing_score      — Age + BMI respiratory load
pitch_instability    — Glucose volatility proxy
face_risk_score      — Composite face risk (0–100)
voice_risk_score     — Composite voice risk (0–100)

Interaction Features (4)
------------------------
cardio_stress        — face_fatigue × breathing_score
metabolic_score      — brightness_variance × voice_stress
fatigue_stress       — face_fatigue × voice_stress
respiratory_variation— breathing_score × pitch_instability

Usage::

    python training/feature_engineering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_CSV = PROCESSED_DIR / "cleaned_dataset.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"


# ======================================================================
# Helpers
# ======================================================================

def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def _sigmoid(arr: np.ndarray, center: float = 0.0, k: float = 5.0) -> np.ndarray:
    """Sigmoid stretch to amplify class separation."""
    return 1.0 / (1.0 + np.exp(-k * (arr - center)))


def _rescale(series: pd.Series) -> np.ndarray:
    """Rescale any column to [0, 1] using min-max on the actual values.

    The cleaned dataset has been *StandardScaler*-normalised; this maps
    the z-scores back to a bounded [0, 1] range per-column.
    """
    vals = series.values.astype(np.float64)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return np.full_like(vals, 0.5)
    return (vals - lo) / (hi - lo)


# ======================================================================
# Biomarker mapping
# ======================================================================

def generate_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """Map cleaned clinical features → 9 biomarkers + 4 interactions + 9 raw.

    Returns 24 features total:
      - 9 base biomarkers (softer sigmoid, k=3.0 for better signal)
      - 4 biomarker interaction features
      - 3 clinical cross-interaction features
      - 8 rescaled raw clinical features (no raw_heart_rate — zero variance)
    """

    # Rescale each clinical feature to [0, 1]
    age = _rescale(df["age"])
    sex = df["sex"].values.astype(np.float64)
    bp  = _rescale(df["blood_pressure"])
    chol = _rescale(df["cholesterol"])
    gluc = _rescale(df["glucose"])
    bmi  = _rescale(df["bmi"])
    hr   = _rescale(df["heart_rate"])
    smoke = df["smoking"].values.astype(np.float64)
    exercise = df["exercise"].values.astype(np.float64)

    # ── Clinical interaction features (Step 5) ─────────────────────
    cardio_stress_clin = bp * chol       # blood_pressure × cholesterol
    metabolic_clin     = gluc * bmi      # glucose × BMI
    fatigue_clin       = age * bp        # age × BP proxy

    # ── Biomarker generation (softer sigmoid k=3.0) ────────────────
    # Each biomarker now uses k=3.0 (was 8.0) and center=0.50 (was 0.35)
    # to preserve gradient information instead of binarising features.

    face_fatigue = _sigmoid(_clip01(
        0.30 * bp
        + 0.25 * age
        + 0.20 * (1 - hr)
        + 0.15 * cardio_stress_clin
        + 0.10 * smoke
    ), center=0.50, k=3.0)

    symmetry_score = _sigmoid(_clip01(
        1.0 - (
            0.30 * chol
            + 0.25 * age
            + 0.20 * bp
            + 0.15 * cardio_stress_clin
            + 0.10 * (1 - exercise)
        )
    ), center=0.50, k=-3.0)

    blink_instability = _sigmoid(_clip01(
        0.30 * gluc
        + 0.25 * metabolic_clin
        + 0.20 * age
        + 0.15 * bp
        + 0.10 * (1 - exercise)
    ), center=0.50, k=3.0)

    brightness_variance = _sigmoid(_clip01(
        0.30 * bmi
        + 0.25 * gluc
        + 0.20 * age
        + 0.15 * chol
        + 0.10 * smoke
    ), center=0.50, k=3.0)

    voice_stress = _sigmoid(_clip01(
        0.30 * gluc
        + 0.25 * bmi
        + 0.20 * metabolic_clin
        + 0.15 * bp
        + 0.10 * smoke
    ), center=0.50, k=3.0)

    breathing_score = _sigmoid(_clip01(
        0.30 * age
        + 0.25 * bmi
        + 0.20 * (1 - hr)
        + 0.15 * bp
        + 0.10 * smoke
    ), center=0.50, k=3.0)

    pitch_instability = _sigmoid(_clip01(
        0.30 * gluc
        + 0.25 * age
        + 0.20 * chol
        + 0.15 * metabolic_clin
        + 0.10 * (1 - exercise)
    ), center=0.50, k=3.0)

    # ── Composite risk scores ──────────────────────────────────────
    face_risk_score = np.clip(
        (0.3 * face_fatigue + 0.3 * (1 - symmetry_score)
         + 0.2 * blink_instability + 0.2 * brightness_variance) * 100,
        0, 100,
    )
    voice_risk_score = np.clip(
        (0.4 * voice_stress + 0.3 * breathing_score
         + 0.3 * pitch_instability) * 100,
        0, 100,
    )

    # ── Biomarker interaction features (Step 5) ────────────────────
    cardio_stress_bio = face_fatigue * breathing_score
    metabolic_score_bio = brightness_variance * voice_stress
    fatigue_stress = face_fatigue * voice_stress
    respiratory_variation = breathing_score * pitch_instability

    # ── Clinical cross-interaction features ─────────────────────
    bp_chol_risk   = bp * chol          # blood pressure × cholesterol
    age_metabolic  = age * gluc         # age-modified metabolic risk
    clinical_cardio = bp * face_fatigue  # clinical × AI biomarker cross

    result = pd.DataFrame({
        # 9 base biomarkers
        "face_fatigue": face_fatigue,
        "symmetry_score": symmetry_score,
        "blink_instability": blink_instability,
        "brightness_variance": brightness_variance,
        "voice_stress": voice_stress,
        "breathing_score": breathing_score,
        "pitch_instability": pitch_instability,
        "face_risk_score": face_risk_score,
        "voice_risk_score": voice_risk_score,
        # 4 biomarker interaction features
        "cardio_stress": cardio_stress_bio,
        "metabolic_score": metabolic_score_bio,
        "fatigue_stress": fatigue_stress,
        "respiratory_variation": respiratory_variation,
        # 3 clinical cross-interaction features
        "bp_chol_risk": bp_chol_risk,
        "age_metabolic": age_metabolic,
        "clinical_cardio": clinical_cardio,
        # 8 rescaled raw clinical features (no raw_heart_rate — zero variance)
        "raw_age": age,
        "raw_sex": sex,
        "raw_bp": bp,
        "raw_cholesterol": chol,
        "raw_glucose": gluc,
        "raw_bmi": bmi,
        "raw_smoking": smoke,
        "raw_exercise": exercise,
        # Target
        "target": df["target"].values,
    })
    return result


# ======================================================================
# CLI entry point
# ======================================================================

def run_feature_engineering() -> pd.DataFrame:
    """Load cleaned dataset, generate features, save."""
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING")
    print("=" * 60)

    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            f"{CLEANED_CSV} not found.  Run preprocessing.py first."
        )

    df = pd.read_csv(CLEANED_CSV)
    print(f"\n  Loaded {len(df):,} cleaned records.")

    features_df = generate_biomarkers(df)

    n0 = int((features_df["target"] == 0).sum())
    n1 = int((features_df["target"] == 1).sum())

    n_feats = len(features_df.columns) - 1
    print(f"\n  Generated features:")
    print(f"    Rows:       {len(features_df):>7,}")
    print(f"    Columns:    {n_feats}  (9 biomarkers + 4 interactions + 3 cross + 8 raw clinical)")
    print(f"    target = 0: {n0:>7,}")
    print(f"    target = 1: {n1:>7,}")

    # Feature statistics
    feature_cols = [c for c in features_df.columns if c != "target"]
    print(f"\n  Feature ranges:")
    for col in feature_cols:
        vals = features_df[col]
        print(f"    {col:25s}  mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}  "
              f"[{vals.min():.3f}, {vals.max():.3f}]")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(FEATURES_CSV, index=False)
    print(f"\n  ✓ Features saved → {FEATURES_CSV}")

    # Step 7: Cache as parquet for fast training reload
    parquet_path = FEATURES_CSV.with_suffix(".parquet")
    features_df.to_parquet(parquet_path, index=False)
    print(f"  ✓ Parquet cache  → {parquet_path}")
    print("=" * 60)

    return features_df


if __name__ == "__main__":
    run_feature_engineering()
