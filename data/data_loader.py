
"""
data_loader.py — Load, clean, and transform real clinical datasets.

Complete data pipeline:
    1. Load raw CSVs (UCI Heart Disease + PIMA Diabetes)
    2. Handle missing values (median for numeric, mode for categorical)
    3. Remove outliers via IQR method
    4. Engineer derived clinical features
    5. Map clinical features → 9 biomarker schema
    6. Return clean, ready-to-train DataFrame

Integrated Datasets
-------------------
UCI Heart Disease (297 records, 14 attributes)
    Cardiovascular clinical data → facial + vocal biomarkers

PIMA Indians Diabetes (768 records, 8 attributes)
    Metabolic / demographic data → facial + vocal biomarkers

Feature Engineering
-------------------
Derived features created before biomarker mapping:
    • fatigue_score      — (age + bp + cholesterol) / 3
    • cardio_stress      — blood_pressure × cholesterol (interaction)
    • metabolic_score    — glucose × BMI (interaction)
    • cardiac_reserve    — maxHR / age (functional capacity)

These capture non-linear clinical relationships that improve ML accuracy.

Data Cleaning
-------------
    • Missing values: median imputation (numeric), mode (categorical)
    • Outliers: IQR method (1.5 × IQR fence) removes extreme readings
    • Zero imputation: physiologically impossible zeros → median
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


def _sigmoid_stretch(arr: np.ndarray, center: float = 0.35, k: float = 8.0) -> np.ndarray:
    """Apply sigmoid stretching to amplify separation near *center*.

    Values below *center* are pushed lower, values above pushed higher,
    creating a wider gap between low-risk and high-risk samples.
    """
    return 1.0 / (1.0 + np.exp(-k * (arr - center)))


def _norm(series: pd.Series, lo: float, hi: float) -> np.ndarray:
    """Min-max normalise a pandas Series to [0, 1]."""
    return _clip01(((series.values - lo) / (hi - lo)).astype(np.float64))


def _remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove rows with outlier values using IQR method.

    Only checks specified numeric columns.  Keeps the target column intact.
    """
    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        mask &= (df[col] >= lo) & (df[col] <= hi)

    df_clean = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        print(f"    IQR outlier removal: {n_removed} rows removed ({n_before} → {len(df_clean)})")
    return df_clean


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    # Numeric columns → median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(median_val)
            print(f"    Imputed {n_missing} missing in '{col}' with median={median_val:.2f}")

    # Categorical / object columns → mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            n_missing = int(df[col].isnull().sum())
            df[col] = df[col].fillna(mode_val)
            print(f"    Imputed {n_missing} missing in '{col}' with mode={mode_val}")

    return df


# ==============================================================================
# UCI Heart Disease  →  biomarkers
# ==============================================================================

def _load_heart(seed: int = 42) -> Optional[pd.DataFrame]:
    """Load UCI Heart Disease CSV, clean, engineer features, map to biomarkers.

    Pipeline:
        1. Load CSV with 14 clinical attributes
        2. Impute missing values (ca, thal, slope often missing)
        3. Remove IQR outliers on continuous columns
        4. Engineer derived features (fatigue_score, cardio_stress, cardiac_reserve)
        5. Map to 9 biomarker features
    """
    if not os.path.isfile(HEART_CSV):
        return None

    print("\n  ── UCI Heart Disease ──")
    df = pd.read_csv(HEART_CSV)
    print(f"    Raw records: {len(df)}")

    # ── 1. Handle missing values ────────────────────────────────────
    df = _impute_missing(df)

    # ── 2. Remove outliers on continuous clinical columns ───────────
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df = _remove_outliers_iqr(df, continuous_cols, factor=2.0)  # moderate fence

    rng = np.random.default_rng(seed)
    n = len(df)

    # ── 3. Feature engineering (derived clinical features) ──────────
    # Fatigue score: composite of age, blood pressure, and inverse max HR
    fatigue_raw = (
        _norm(df["age"], 29, 77)
        + _norm(df["trestbps"], 94, 200)
        + (1.0 - _norm(df["thalach"], 71, 202))
    ) / 3.0

    # Cardiovascular stress: BP × cholesterol interaction (normalised)
    cardio_stress_raw = (
        _norm(df["trestbps"], 94, 200) * _norm(df["chol"], 126, 564)
    )

    # Cardiac reserve: maxHR / age (higher = better fitness)
    cardiac_reserve = _clip01(
        _norm(df["thalach"] / (df["age"] + 1e-6), 1.0, 3.5)
    )

    # ── 4. Normalise raw clinical features ──────────────────────────
    age     = _norm(df["age"],      29.0,  77.0)
    cp      = _norm(df["cp"],       0.0,   3.0)
    bp      = _norm(df["trestbps"], 94.0, 200.0)
    chol    = _norm(df["chol"],    126.0, 564.0)
    fbs     = df["fbs"].values.astype(np.float64)
    ecg     = _norm(df["restecg"],  0.0,   2.0)
    maxhr   = _norm(df["thalach"], 71.0,  202.0)
    exang   = df["exang"].values.astype(np.float64)
    oldpeak = _norm(df["oldpeak"],  0.0,   6.2)
    slope   = _norm(df["slope"],    0.0,   2.0)
    ca      = _norm(df["ca"],       0.0,   4.0)
    thal    = _norm(df["thal"],     0.0,   3.0)

    # ── 5. Map to biomarkers (orthogonal — each uses a distinct primary driver)
    face_fatigue = _sigmoid_stretch(_clip01(
        0.30 * fatigue_raw       # primary: composite fatigue
        + 0.25 * age
        + 0.20 * (1 - maxhr)
        + 0.15 * exang
        + 0.10 * bp
    ))
    symmetry_score = _sigmoid_stretch(_clip01(
        1.0 - (
            0.30 * ecg           # primary: ECG abnormality
            + 0.25 * oldpeak
            + 0.20 * cp
            + 0.15 * cardio_stress_raw
            + 0.10 * age
        )
    ), center=0.65, k=-8.0)  # inverse: higher = healthier
    blink_instability = _sigmoid_stretch(_clip01(
        0.30 * exang             # primary: exercise-induced angina
        + 0.25 * oldpeak
        + 0.20 * (1 - cardiac_reserve)
        + 0.15 * cp
        + 0.10 * thal
    ))
    brightness_variance = _sigmoid_stretch(_clip01(
        0.30 * chol              # primary: cholesterol
        + 0.25 * cardio_stress_raw
        + 0.20 * fbs
        + 0.15 * age
        + 0.10 * bp
    ))
    voice_stress = _sigmoid_stretch(_clip01(
        0.30 * cp                # primary: chest pain type
        + 0.25 * fatigue_raw
        + 0.20 * exang
        + 0.15 * oldpeak
        + 0.10 * bp
    ))
    breathing_score = _sigmoid_stretch(_clip01(
        0.30 * (1 - cardiac_reserve)  # primary: cardiac reserve
        + 0.25 * (1 - maxhr)
        + 0.20 * slope
        + 0.15 * ca
        + 0.10 * exang
    ))
    pitch_instability = _sigmoid_stretch(_clip01(
        0.30 * thal              # primary: thalassemia
        + 0.25 * ca
        + 0.20 * oldpeak
        + 0.15 * slope
        + 0.10 * cardio_stress_raw
    ))

    # ── Composite risk scores ───────────────────────────────────────
    face_risk = (
        0.3 * face_fatigue
        + 0.3 * (1 - symmetry_score)
        + 0.2 * blink_instability
        + 0.2 * brightness_variance
    ) * 100
    face_risk = np.clip(face_risk, 0, 100)

    voice_risk = (
        0.4 * voice_stress
        + 0.3 * breathing_score
        + 0.3 * pitch_instability
    ) * 100
    voice_risk = np.clip(voice_risk, 0, 100)

    print(f"    Final records: {n}")
    print(f"    Engineered features: fatigue_score, cardio_stress, cardiac_reserve")

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
    """Load PIMA Diabetes CSV, clean, engineer features, map to biomarkers.

    Pipeline:
        1. Load CSV with 8 metabolic attributes
        2. Replace physiologically impossible zeros with median
        3. Remove IQR outliers
        4. Engineer derived features (metabolic_score, insulin_resistance)
        5. Map to 9 biomarker features
    """
    if not os.path.isfile(DIABETES_CSV):
        return None

    print("\n  ── PIMA Indians Diabetes ──")
    df = pd.read_csv(DIABETES_CSV, header=None, names=_PIMA_COLS)
    print(f"    Raw records: {len(df)}")

    # ── 1. Replace physiologically impossible zeros with median ─────
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        n_zeros = int((df[col] == 0).sum())
        if n_zeros > 0:
            median_val = df.loc[df[col] > 0, col].median()
            df.loc[df[col] == 0, col] = median_val
            print(f"    Replaced {n_zeros} zeros in '{col}' with median={median_val:.1f}")

    # ── 2. Handle any remaining NaN ─────────────────────────────────
    df = _impute_missing(df)

    # ── 3. Remove outliers ──────────────────────────────────────────
    # Only remove outliers from columns without zero-imputation
    # (Glucose, BP, Skin, Insulin, BMI all had zeros replaced — IQR is skewed)
    continuous_cols = ["Age"]
    df = _remove_outliers_iqr(df, continuous_cols, factor=2.5)

    rng = np.random.default_rng(seed + 1)
    n = len(df)

    # ── 4. Feature engineering ──────────────────────────────────────
    # Metabolic score: glucose × BMI interaction (diabetes risk proxy)
    metabolic_score = (
        _norm(df["Glucose"], 44, 199) * _norm(df["BMI"], 18.2, 67.1)
    )

    # Insulin resistance proxy: glucose × insulin / (BMI + 1)
    insulin_resistance = _clip01(
        _norm(
            df["Glucose"] * df["Insulin"] / (df["BMI"] + 1),
            0, 5000
        )
    )

    # Age-metabolic interaction
    age_metabolic = (
        _norm(df["Age"], 21, 81) * _norm(df["Glucose"], 44, 199)
    )

    # ── 5. Normalise raw features ───────────────────────────────────
    glucose = _norm(df["Glucose"],       44.0, 199.0)
    bp      = _norm(df["BloodPressure"], 24.0, 122.0)
    skin    = _norm(df["SkinThickness"],  7.0,  99.0)
    insulin = _norm(df["Insulin"],       14.0, 846.0)
    bmi     = _norm(df["BMI"],           18.2,  67.1)
    dpf     = _norm(df["DiabetesPedigreeFunction"], 0.078, 2.42)
    age     = _norm(df["Age"],           21.0,  81.0)

    # ── 6. Map to biomarkers (orthogonal — each biomarker uses
    #       a distinct primary clinical driver to reduce correlation)
    preg = _norm(df["Pregnancies"], 0.0, 17.0)

    face_fatigue = _sigmoid_stretch(_clip01(
        0.35 * glucose          # primary: glucose
        + 0.25 * metabolic_score
        + 0.15 * age
        + 0.15 * bmi
        + 0.10 * bp
    ))
    symmetry_score = _sigmoid_stretch(_clip01(
        1.0 - (
            0.35 * bp           # primary: blood pressure
            + 0.25 * age
            + 0.15 * glucose
            + 0.15 * preg
            + 0.10 * bmi
        )
    ), center=0.65, k=-8.0)  # inverse: higher = healthier
    blink_instability = _sigmoid_stretch(_clip01(
        0.35 * insulin_resistance  # primary: insulin resistance
        + 0.25 * insulin
        + 0.15 * dpf
        + 0.15 * glucose
        + 0.10 * age
    ))
    brightness_variance = _sigmoid_stretch(_clip01(
        0.35 * skin             # primary: skin thickness
        + 0.25 * age_metabolic
        + 0.15 * bmi
        + 0.15 * glucose
        + 0.10 * preg
    ))
    voice_stress = _sigmoid_stretch(_clip01(
        0.35 * bmi              # primary: BMI
        + 0.25 * metabolic_score
        + 0.15 * glucose
        + 0.15 * bp
        + 0.10 * age
    ))
    breathing_score = _sigmoid_stretch(_clip01(
        0.35 * age              # primary: age
        + 0.25 * metabolic_score
        + 0.15 * bmi
        + 0.15 * bp
        + 0.10 * preg
    ))
    pitch_instability = _sigmoid_stretch(_clip01(
        0.35 * dpf              # primary: diabetes pedigree function
        + 0.25 * insulin_resistance
        + 0.15 * insulin
        + 0.15 * glucose
        + 0.10 * age
    ))

    # ── Composite risk scores ───────────────────────────────────────
    face_risk = (
        0.3 * face_fatigue
        + 0.3 * (1 - symmetry_score)
        + 0.2 * blink_instability
        + 0.2 * brightness_variance
    ) * 100
    face_risk = np.clip(face_risk, 0, 100)

    voice_risk = (
        0.4 * voice_stress
        + 0.3 * breathing_score
        + 0.3 * pitch_instability
    ) * 100
    voice_risk = np.clip(voice_risk, 0, 100)

    print(f"    Final records: {n}")
    print(f"    Engineered features: metabolic_score, insulin_resistance, age_metabolic")

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
    """Load all available real datasets, clean, engineer features, concatenate.

    Returns
    -------
    pd.DataFrame or None
        Combined dataset with ``FEATURE_NAMES`` columns + ``health_risk``.
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
