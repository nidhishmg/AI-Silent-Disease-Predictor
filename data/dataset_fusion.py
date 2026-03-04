"""
dataset_fusion.py — Fuse heterogeneous clinical datasets into a unified schema.

Reads all raw datasets from ``data/raw/``, normalises column names, maps
clinical attributes into a common schema, and produces a single fused
DataFrame saved to ``data/processed/fused_dataset.csv``.

Unified Schema
--------------
age             float   Years
sex             int     0 = Female, 1 = Male
blood_pressure  float   Systolic BP (mmHg)
cholesterol     float   Total cholesterol (mg/dL)
glucose         float   Blood glucose (mg/dL)
bmi             float   Body-mass index (kg/m²)
heart_rate      float   Heart rate (bpm)
smoking         int     0 = no, 1 = yes
exercise        int     0 = inactive, 1 = active
target          int     0 = low risk, 1 = high risk

Usage::

    python data/dataset_fusion.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FUSED_CSV = PROCESSED_DIR / "fused_dataset.csv"

# Population medians for imputing missing schema columns (CDC / WHO refs)
_POP = {
    "blood_pressure": 120.0,
    "cholesterol": 200.0,
    "glucose": 100.0,
    "bmi": 26.6,
    "heart_rate": 72.0,
}


# ======================================================================
# Per-dataset loaders (raw → unified schema)
# ======================================================================

def _load_uci_heart() -> pd.DataFrame | None:
    path = RAW_DIR / "heart.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["age"].astype(float)
    out["sex"] = df["sex"].astype(int)
    out["blood_pressure"] = df["trestbps"].astype(float)
    out["cholesterol"] = df["chol"].astype(float)
    out["glucose"] = np.where(df["fbs"] == 1, 140.0, 90.0)  # fbs>120 → binary
    out["bmi"] = _POP["bmi"]  # not in dataset
    out["heart_rate"] = df["thalach"].astype(float)
    out["smoking"] = 0  # not in dataset
    out["exercise"] = (1 - df["exang"]).astype(int)  # inverse of exercise angina
    out["target"] = df["target"].astype(int)
    out["_source"] = "uci_heart"
    return out


def _load_pima() -> pd.DataFrame | None:
    path = RAW_DIR / "diabetes.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["Age"].astype(float)
    out["sex"] = 0  # all female
    out["blood_pressure"] = df["BloodPressure"].astype(float).replace(0, np.nan)
    out["cholesterol"] = _POP["cholesterol"]  # not in dataset
    out["glucose"] = df["Glucose"].astype(float).replace(0, np.nan)
    out["bmi"] = df["BMI"].astype(float).replace(0, np.nan)
    out["heart_rate"] = _POP["heart_rate"]  # not in dataset
    out["smoking"] = 0  # not in dataset
    out["exercise"] = 1  # not in dataset
    out["target"] = df["Outcome"].astype(int)
    out["_source"] = "pima_diabetes"
    return out


def _load_framingham() -> pd.DataFrame | None:
    path = RAW_DIR / "framingham.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    out = pd.DataFrame()
    out["age"] = df["age"].astype(float)
    out["sex"] = df["male"].astype(int)
    out["blood_pressure"] = df["sysBP"].astype(float)
    out["cholesterol"] = df["totChol"].astype(float)
    out["glucose"] = df["glucose"].astype(float)
    out["bmi"] = df["BMI"].astype(float)
    out["heart_rate"] = df["heartRate"].astype(float)
    out["smoking"] = df["currentSmoker"].astype(int)
    out["exercise"] = 1  # not explicitly in dataset
    out["target"] = df["TenYearCHD"].astype(int)
    out["_source"] = "framingham"
    return out


def _load_stroke() -> pd.DataFrame | None:
    path = RAW_DIR / "stroke.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["age"] = df["age"].astype(float)
    out["sex"] = (df["gender"].str.strip().str.lower() == "male").astype(int)
    # No direct BP column — derive from hypertension flag
    out["blood_pressure"] = np.where(df["hypertension"] == 1, 150.0, 120.0)
    out["cholesterol"] = _POP["cholesterol"]  # not in dataset
    out["glucose"] = df["avg_glucose_level"].astype(float)
    # BMI may have "N/A"
    out["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    out["heart_rate"] = _POP["heart_rate"]  # not in dataset
    out["smoking"] = df["smoking_status"].isin(["smokes", "formerly smoked"]).astype(int)
    out["exercise"] = 1  # not in dataset
    out["target"] = df["stroke"].astype(int)
    out["_source"] = "stroke"
    return out


def _load_cardiovascular() -> pd.DataFrame | None:
    path = RAW_DIR / "cardiovascular.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";")

    out = pd.DataFrame()
    out["age"] = (df["age"] / 365.25).round(1)  # days → years
    out["sex"] = (df["gender"] == 2).astype(int)  # 1=F, 2=M
    out["blood_pressure"] = df["ap_hi"].astype(float)
    # Ordinal cholesterol → continuous estimate
    chol_map = {1: 180.0, 2: 240.0, 3: 300.0}
    out["cholesterol"] = df["cholesterol"].map(chol_map).astype(float)
    # Ordinal glucose → continuous estimate
    gluc_map = {1: 85.0, 2: 140.0, 3: 200.0}
    out["glucose"] = df["gluc"].map(gluc_map).astype(float)
    # BMI from height and weight
    h_m = df["height"] / 100.0
    out["bmi"] = (df["weight"] / (h_m ** 2)).round(1)
    out["heart_rate"] = _POP["heart_rate"]  # not in dataset
    out["smoking"] = df["smoke"].astype(int)
    out["exercise"] = df["active"].astype(int)
    out["target"] = df["cardio"].astype(int)
    out["_source"] = "cardiovascular"
    return out


# ======================================================================
# Fusion orchestrator
# ======================================================================

_LOADERS = [
    ("UCI Heart Disease", _load_uci_heart),
    ("PIMA Diabetes", _load_pima),
    ("Framingham Heart Study", _load_framingham),
    ("Stroke Prediction", _load_stroke),
    ("Cardiovascular Disease", _load_cardiovascular),
]


def fuse_datasets() -> pd.DataFrame:
    """Load all available raw datasets and fuse into unified schema."""
    print("\n" + "=" * 60)
    print("  DATASET FUSION")
    print("=" * 60)

    frames: list[pd.DataFrame] = []
    for name, loader in _LOADERS:
        df = loader()
        if df is not None:
            print(f"  ✓ {name:30s} → {len(df):>6,} rows")
            frames.append(df)
        else:
            print(f"  ✗ {name:30s} — file not found, skipping")

    if not frames:
        raise FileNotFoundError(
            "No datasets found in data/raw/.  Run download_datasets.py first."
        )

    fused = pd.concat(frames, ignore_index=True)

    # --- Summary ---
    n_total = len(fused)
    n_risk0 = int((fused["target"] == 0).sum())
    n_risk1 = int((fused["target"] == 1).sum())
    sources = fused["_source"].value_counts().to_dict()

    print(f"\n  Fused dataset:")
    print(f"    Total rows:   {n_total:>7,}")
    print(f"    target = 0:   {n_risk0:>7,}  ({n_risk0 / n_total:.1%})")
    print(f"    target = 1:   {n_risk1:>7,}  ({n_risk1 / n_total:.1%})")
    print(f"    Sources:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"      {src:25s} {cnt:>7,}")

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fused.to_csv(FUSED_CSV, index=False)
    print(f"\n  ✓ Saved → {FUSED_CSV}")
    print("=" * 60)

    return fused


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    fuse_datasets()
