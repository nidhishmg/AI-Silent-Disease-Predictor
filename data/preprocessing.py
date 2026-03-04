"""
preprocessing.py — Data cleaning and normalisation pipeline.

Pipeline Steps
--------------
1. Load fused dataset from ``data/processed/fused_dataset.csv``.
2. Drop metadata columns (``_source``).
3. Handle missing values:
   • numeric → median imputation
   • categorical → mode imputation
4. Remove outliers via IQR method (1.5 × IQR fence).
5. Normalise numeric features with ``StandardScaler``.
6. Save cleaned dataset and scaler artifact.

Outputs
-------
• ``data/processed/cleaned_dataset.csv``
• ``models/scaler.pkl``

Usage::

    python data/preprocessing.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FUSED_CSV = PROCESSED_DIR / "fused_dataset.csv"
CLEANED_CSV = PROCESSED_DIR / "cleaned_dataset.csv"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

NUMERIC_COLS = [
    "age", "blood_pressure", "cholesterol", "glucose",
    "bmi", "heart_rate",
]
CATEGORICAL_COLS = ["sex", "smoking", "exercise"]
TARGET_COL = "target"


# ======================================================================
# 1. Missing values
# ======================================================================

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    print("\n  ── Missing Value Imputation ──")
    total_missing = 0
    for col in NUMERIC_COLS:
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            print(f"    {col:20s}  {n_miss:>6,} NaN → median {median:.1f}")
            total_missing += n_miss

    for col in CATEGORICAL_COLS:
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode)
            print(f"    {col:20s}  {n_miss:>6,} NaN → mode {mode}")
            total_missing += n_miss

    if total_missing == 0:
        print("    No missing values found.")
    else:
        print(f"    Total imputed: {total_missing:,}")

    return df


# ======================================================================
# 2. Outlier removal — IQR method
# ======================================================================

def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove rows with values outside [Q1 − k·IQR, Q3 + k·IQR]."""
    if columns is None:
        columns = NUMERIC_COLS
    print(f"\n  ── IQR Outlier Removal (factor={factor}) ──")
    n_before = len(df)
    mask = pd.Series(True, index=df.index)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        col_mask = (df[col] >= lo) & (df[col] <= hi)
        n_out = int(~col_mask.sum())
        if n_out > 0:
            print(f"    {col:20s}  {n_out:>6,} outliers  "
                  f"[{lo:.1f}, {hi:.1f}]")
        mask &= col_mask

    df_clean = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    print(f"    Total removed:  {n_removed:,} / {n_before:,}  "
          f"({n_removed / n_before:.1%})")
    return df_clean


# ======================================================================
# 3. Normalisation
# ======================================================================

def normalise(
    df: pd.DataFrame,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalise numeric columns with StandardScaler."""
    print("\n  ── StandardScaler Normalisation ──")
    scaler = StandardScaler()
    if fit:
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    print(f"    Scaled {len(NUMERIC_COLS)} numeric columns.")
    return df, scaler


# ======================================================================
# Full pipeline
# ======================================================================

def preprocess(iqr_factor: float = 1.5) -> tuple[pd.DataFrame, StandardScaler]:
    """Run the complete preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING")
    print("=" * 60)

    if not FUSED_CSV.exists():
        raise FileNotFoundError(
            f"{FUSED_CSV} not found.  Run dataset_fusion.py first."
        )

    df = pd.read_csv(FUSED_CSV)
    print(f"\n  Loaded {len(df):,} rows from fused dataset.")

    # Drop metadata
    if "_source" in df.columns:
        df = df.drop(columns=["_source"])

    # 1. Impute missing values
    df = impute_missing(df)

    # 2. Remove outliers
    df = remove_outliers_iqr(df, factor=iqr_factor)

    # 3. Normalise
    df_scaled, scaler = normalise(df)

    # Save artifacts
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_scaled.to_csv(CLEANED_CSV, index=False)
    joblib.dump(scaler, SCALER_PATH)

    n0 = int((df_scaled[TARGET_COL] == 0).sum())
    n1 = int((df_scaled[TARGET_COL] == 1).sum())
    print(f"\n  ✓ Cleaned dataset saved → {CLEANED_CSV}")
    print(f"    Rows: {len(df_scaled):>7,}")
    print(f"    target=0: {n0:>7,}  target=1: {n1:>7,}")
    print(f"  ✓ Scaler saved → {SCALER_PATH}")
    print("=" * 60)

    return df_scaled, scaler


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    preprocess()
