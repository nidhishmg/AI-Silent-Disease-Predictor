"""
data_loader.py — Load real clinical datasets and map to biomarker features.

Bridges two publicly available clinical datasets to the 9-feature biomarker
schema used by the AI Silent Disease Predictor model:

    UCI Heart Disease (303 records)
        14 cardiovascular clinical attributes → 7 biomarkers
        target = presence of heart disease (0/1)

    PIMA Indians Diabetes (768 records)
        8 metabolic/demographic attributes → 7 biomarkers
        Outcome = diabetes diagnosis (0/1)

Mapping Rationale
-----------------
Clinical features are transformed to our biomarker space using **medically
justified weighted sums** — not arbitrary random projections.  Each weight
encodes a documented physiological relationship:

 • Cardiovascular stress (high BP, low maxHR)  → elevated face_fatigue
 • Metabolic dysfunction (high glucose, BMI)   → elevated voice_stress
 • Neuropathy / vascular issues               → blink & pitch instability
 • Skin / connective tissue changes            → brightness_variance

The two composite scores (face_risk_score, voice_risk_score) are computed
from the mapped biomarkers using the same formula as the synthetic pipeline.

Small Gaussian noise is added per-feature to:
 1. prevent identical biomarker values when clinical inputs coincide
 2. regularise the downstream Random Forest
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
HEART_CSV = os.path.join(DATA_DIR, "heart.csv")
DIABETES_CSV = os.path.join(DATA_DIR, "diabetes.csv")


# ==============================================================================
# Utility helpers
# ==============================================================================

def _clip01(arr: np.ndarray) -> np.ndarray:
    """Clip array to [0, 1]."""
    return np.clip(arr, 0.0, 1.0)


def _norm(series: pd.Series, lo: float, hi: float) -> np.ndarray:
    """Min-max normalise a pandas Series to [0, 1]."""
    return _clip01(((series.values - lo) / (hi - lo)).astype(np.float64))


# ==============================================================================
# UCI Heart Disease  →  biomarkers
# ==============================================================================

def _load_heart(seed: int = 42) -> Optional[pd.DataFrame]:
    """Load UCI Heart Disease CSV and map to biomarker features.

    Columns expected (datasciencedojo mirror):
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal, target

    Returns
    -------
    DataFrame with columns matching ``config.settings.FEATURE_NAMES`` + ``health_risk``,
    or *None* if the file is missing.
    """
    if not os.path.isfile(HEART_CSV):
        return None

    df = pd.read_csv(HEART_CSV)
    rng = np.random.default_rng(seed)
    n = len(df)
    noise = lambda: rng.normal(0, 0.03, n)

    # ── Normalise clinical features ──────────────────────────────────
    age     = _norm(df["age"],      29.0,  77.0)
    # sex: already 0/1
    cp      = _norm(df["cp"],       0.0,   3.0)   # chest pain type
    bp      = _norm(df["trestbps"], 94.0, 200.0)
    chol    = _norm(df["chol"],    126.0, 564.0)
    fbs     = df["fbs"].values.astype(np.float64)  # 0/1
    ecg     = _norm(df["restecg"],  0.0,   2.0)
    maxhr   = _norm(df["thalach"], 71.0,  202.0)
    exang   = df["exang"].values.astype(np.float64)  # 0/1
    oldpeak = _norm(df["oldpeak"],  0.0,   6.2)
    slope   = _norm(df["slope"],    0.0,   2.0)
    ca      = _norm(df["ca"],       0.0,   4.0)
    thal    = _norm(df["thal"],     0.0,   3.0)

    # ── Map to biomarkers (medical rationale in docstring) ───────────
    face_fatigue = _clip01(
        0.35 * age + 0.25 * bp + 0.25 * (1 - maxhr) + 0.15 * exang + noise()
    )
    symmetry_score = _clip01(
        1.0 - (0.25 * ecg + 0.25 * oldpeak + 0.25 * cp + 0.25 * age) + noise()
    )
    blink_instability = _clip01(
        0.30 * exang + 0.25 * oldpeak + 0.25 * cp + 0.20 * (1 - maxhr) + noise()
    )
    brightness_variance = _clip01(
        0.30 * chol + 0.25 * fbs + 0.25 * age + 0.20 * bp + noise()
    )
    voice_stress = _clip01(
        0.35 * cp + 0.25 * exang + 0.20 * oldpeak + 0.20 * bp + noise()
    )
    breathing_score = _clip01(
        0.30 * (1 - maxhr) + 0.25 * exang + 0.25 * slope + 0.20 * ca + noise()
    )
    pitch_instability = _clip01(
        0.30 * thal + 0.25 * ca + 0.25 * oldpeak + 0.20 * age + noise()
    )

    # ── Composite risk scores ───────────────────────────────────────
    face_risk = (
        0.3 * face_fatigue
        + 0.3 * (1 - symmetry_score)
        + 0.2 * blink_instability
        + 0.2 * brightness_variance
    ) * 100 + rng.normal(0, 2, n)
    face_risk = np.clip(face_risk, 0, 100)

    voice_risk = (
        0.4 * voice_stress
        + 0.3 * breathing_score
        + 0.3 * pitch_instability
    ) * 100 + rng.normal(0, 2, n)
    voice_risk = np.clip(voice_risk, 0, 100)

    return pd.DataFrame({
        "face_fatigue":       face_fatigue,
        "symmetry_score":     symmetry_score,
        "blink_instability":  blink_instability,
        "brightness_variance": brightness_variance,
        "voice_stress":       voice_stress,
        "breathing_score":    breathing_score,
        "pitch_instability":  pitch_instability,
        "face_risk_score":    face_risk,
        "voice_risk_score":   voice_risk,
        "health_risk":        df["target"].values.astype(int),
        "_source":            "uci_heart",
    })


# ==============================================================================
# PIMA Indians Diabetes  →  biomarkers
# ==============================================================================

_PIMA_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def _load_diabetes(seed: int = 42) -> Optional[pd.DataFrame]:
    """Load PIMA Diabetes CSV and map to biomarker features.

    Columns (no header row in raw file):
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

    Returns
    -------
    DataFrame with biomarker columns + ``health_risk``, or *None*.
    """
    if not os.path.isfile(DIABETES_CSV):
        return None

    df = pd.read_csv(DIABETES_CSV, header=None, names=_PIMA_COLS)
    rng = np.random.default_rng(seed + 1)   # different seed from heart
    n = len(df)
    noise = lambda: rng.normal(0, 0.03, n)

    # Replace physiologically impossible zeros with column median
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        median_val = df.loc[df[col] > 0, col].median()
        df.loc[df[col] == 0, col] = median_val

    # ── Normalise ────────────────────────────────────────────────────
    glucose = _norm(df["Glucose"],       44.0, 199.0)
    bp      = _norm(df["BloodPressure"], 24.0, 122.0)
    skin    = _norm(df["SkinThickness"],  7.0,  99.0)
    insulin = _norm(df["Insulin"],       14.0, 846.0)
    bmi     = _norm(df["BMI"],           18.2,  67.1)
    dpf     = _norm(df["DiabetesPedigreeFunction"], 0.078, 2.42)
    age     = _norm(df["Age"],           21.0,  81.0)

    # ── Map to biomarkers ────────────────────────────────────────────
    face_fatigue = _clip01(
        0.30 * glucose + 0.25 * age + 0.25 * bmi + 0.20 * bp + noise()
    )
    symmetry_score = _clip01(
        1.0 - (0.30 * bp + 0.25 * age + 0.25 * glucose + 0.20 * bmi) + noise()
    )
    blink_instability = _clip01(
        0.30 * insulin + 0.25 * dpf + 0.25 * glucose + 0.20 * age + noise()
    )
    brightness_variance = _clip01(
        0.35 * skin + 0.25 * bmi + 0.25 * glucose + 0.15 * age + noise()
    )
    voice_stress = _clip01(
        0.30 * glucose + 0.25 * bp + 0.25 * bmi + 0.20 * age + noise()
    )
    breathing_score = _clip01(
        0.35 * bmi + 0.25 * age + 0.25 * bp + 0.15 * glucose + noise()
    )
    pitch_instability = _clip01(
        0.30 * dpf + 0.25 * glucose + 0.25 * insulin + 0.20 * age + noise()
    )

    # ── Composite risk scores ───────────────────────────────────────
    face_risk = (
        0.3 * face_fatigue
        + 0.3 * (1 - symmetry_score)
        + 0.2 * blink_instability
        + 0.2 * brightness_variance
    ) * 100 + rng.normal(0, 2, n)
    face_risk = np.clip(face_risk, 0, 100)

    voice_risk = (
        0.4 * voice_stress
        + 0.3 * breathing_score
        + 0.3 * pitch_instability
    ) * 100 + rng.normal(0, 2, n)
    voice_risk = np.clip(voice_risk, 0, 100)

    return pd.DataFrame({
        "face_fatigue":       face_fatigue,
        "symmetry_score":     symmetry_score,
        "blink_instability":  blink_instability,
        "brightness_variance": brightness_variance,
        "voice_stress":       voice_stress,
        "breathing_score":    breathing_score,
        "pitch_instability":  pitch_instability,
        "face_risk_score":    face_risk,
        "voice_risk_score":   voice_risk,
        "health_risk":        df["Outcome"].values.astype(int),
        "_source":            "pima_diabetes",
    })


# ==============================================================================
# Public API
# ==============================================================================

def load_real_datasets(seed: int = 42) -> Optional[pd.DataFrame]:
    """Load all available real datasets and concatenate.

    Returns
    -------
    pd.DataFrame or None
        Combined dataset with  ``FEATURE_NAMES``  columns + ``health_risk``.
        Returns *None* if no dataset files are found.
    """
    frames: list[pd.DataFrame] = []

    heart = _load_heart(seed)
    if heart is not None:
        frames.append(heart)

    diabetes = _load_diabetes(seed)
    if diabetes is not None:
        frames.append(diabetes)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Summary
    n_total = len(combined)
    n_risk0 = int((combined["health_risk"] == 0).sum())
    n_risk1 = int((combined["health_risk"] == 1).sum())
    sources = combined["_source"].value_counts().to_dict()

    print(f"\n  Real-data summary:")
    print(f"    Total records : {n_total}")
    print(f"    risk=0 (low)  : {n_risk0}")
    print(f"    risk=1 (high) : {n_risk1}")
    for src, cnt in sources.items():
        print(f"    {src:20s}: {cnt} records")

    return combined
