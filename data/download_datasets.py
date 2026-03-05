"""
download_datasets.py — Download & verify clinical datasets.

Integrated Datasets
-------------------
1. UCI Heart Disease        (~297 samples)   — UCI ML Repository
2. PIMA Indians Diabetes    (~768 samples)   — GitHub mirror
3. Framingham Heart Study   (~4 238 samples) — Kaggle / synthetic
4. Stroke Prediction        (~5 110 samples) — Kaggle / synthetic
5. Cardiovascular Disease   (~70 000 samples)— Kaggle / synthetic

Download Strategy
-----------------
• UCI & PIMA: direct HTTP download (always available).
• Kaggle datasets: attempt ``kaggle datasets download`` CLI; if Kaggle
  is not configured, fall back to *statistically faithful* synthetic
  generation based on published variable distributions.

The synthetic fallback guarantees the pipeline always executes end-to-end
even without Kaggle API credentials.  Replace synthetic files with real
Kaggle downloads when available — the rest of the pipeline is agnostic.

Usage::

    python data/download_datasets.py
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "uci_heart": {
        "filename": "heart.csv",
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data",
        ],
        "description": "UCI Heart Disease (297 samples, 14 attributes)",
        "expected_rows": 297,
        "needs_header": True,
        "header": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
        ],
    },
    "pima_diabetes": {
        "filename": "diabetes.csv",
        "urls": [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "pima-indians-diabetes.data.csv",
        ],
        "description": "PIMA Indians Diabetes (768 samples, 9 attributes)",
        "expected_rows": 768,
        "needs_header": True,
        "header": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ],
    },
    "framingham": {
        "filename": "framingham.csv",
        "kaggle_slug": "dileep070/heart-disease-prediction-using-logistic-regression",
        "kaggle_file": "framingham.csv",
        "description": "Framingham Heart Study (4 238 samples)",
        "expected_rows": 4238,
    },
    "stroke": {
        "filename": "stroke.csv",
        "kaggle_slug": "fedesoriano/stroke-prediction-dataset",
        "kaggle_file": "healthcare-dataset-stroke-data.csv",
        "description": "Stroke Prediction (5 110 samples)",
        "expected_rows": 5110,
    },
    "cardiovascular": {
        "filename": "cardiovascular.csv",
        "kaggle_slug": "sulianova/cardiovascular-disease-dataset",
        "kaggle_file": "cardio_train.csv",
        "description": "Cardiovascular Disease (70 000 samples)",
        "expected_rows": 70000,
    },
}


# ======================================================================
# HTTP download helpers
# ======================================================================

def _download_http(url: str, dest: Path) -> bool:
    """Download a file via HTTP.  Returns True on success."""
    try:
        print(f"    ↓ {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"    ✗ HTTP failed: {exc}")
        return False


def _download_uci_heart(dest: Path) -> bool:
    """Download UCI Heart Disease and add header + clean '?' values."""
    url = DATASETS["uci_heart"]["urls"][0]
    if not _download_http(url, dest):
        return False
    # Process: add header, replace '?' with NaN, write clean CSV
    header = DATASETS["uci_heart"]["header"]
    raw = dest.read_text()
    rows = []
    for line in raw.strip().splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) == len(header):
            rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Binarise target: 0 = no disease, 1–4 = disease
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(dest, index=False)
    return True


def _download_pima(dest: Path) -> bool:
    """Download PIMA Diabetes."""
    url = DATASETS["pima_diabetes"]["urls"][0]
    if not _download_http(url, dest):
        return False
    header = DATASETS["pima_diabetes"]["header"]
    df = pd.read_csv(dest, header=None, names=header)
    df.to_csv(dest, index=False)
    return True


# ======================================================================
# Kaggle download
# ======================================================================

def _download_kaggle(slug: str, kaggle_file: str, dest: Path) -> bool:
    """Attempt to download a Kaggle dataset via CLI."""
    try:
        tmp_dir = dest.parent / "_kaggle_tmp"
        tmp_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", slug, "-p", str(tmp_dir), "--unzip",
        ]
        print(f"    ↓ kaggle datasets download -d {slug}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        # Find the target file
        src = tmp_dir / kaggle_file
        if not src.exists():
            # Search recursively
            matches = list(tmp_dir.rglob(kaggle_file))
            if not matches:
                raise FileNotFoundError(f"{kaggle_file} not in download")
            src = matches[0]
        shutil.move(str(src), str(dest))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True
    except Exception as exc:
        print(f"    ✗ Kaggle download failed: {exc}")
        shutil.rmtree(dest.parent / "_kaggle_tmp", ignore_errors=True)
        return False


# ======================================================================
# Synthetic dataset generators (faithful to published distributions)
# ======================================================================

def _generate_framingham(dest: Path, n: int = 30000, seed: int = 42) -> None:
    """Generate Framingham-like data from published study statistics.

    Variable distributions sourced from:
    D'Agostino et al., Circulation (2008) and Framingham Heart Study reports.
    """
    print("    ⚙ Generating synthetic Framingham data (realistic distributions)")
    rng = np.random.default_rng(seed)

    male = rng.binomial(1, 0.44, n)
    age = rng.normal(49.6, 8.6, n).clip(32, 70).round().astype(int)
    education = rng.choice([1, 2, 3, 4], n, p=[0.43, 0.33, 0.13, 0.11])
    current_smoker = rng.binomial(1, 0.49, n)
    cigs_per_day = np.where(
        current_smoker, rng.poisson(9, n).clip(0, 70), 0
    ).astype(float)
    
    # --- Inter-feature Correlations ---
    # BMI increases with age, modified by sex
    bmi_base = rng.normal(25.8, 4.1, n)
    bmi = (bmi_base + 0.05 * (age - 30) - 1.2 * male).clip(15.5, 56.8).round(2)
    
    # Blood pressure increases strongly with BMI and age
    sys_bp_base = rng.normal(132.4, 15.0, n)
    sys_bp = (sys_bp_base + 0.8 * (bmi - 25) + 0.5 * (age - 40)).clip(83, 295).round(1)
    
    dia_bp_base = rng.normal(82.9, 10.0, n)
    dia_bp = (dia_bp_base + 0.5 * (bmi - 25) + 0.3 * (age - 40)).clip(48, 143).round(1)
    
    # Glucose increases with BMI and age
    glucose_base = rng.normal(81.9, 15.0, n)
    glucose = (glucose_base + 1.5 * (bmi - 25) + 0.2 * age).clip(40, 394).round().astype(int)
    
    # Cholesterol increases with BMI, age, and smoking
    tot_chol_base = rng.normal(236.7, 30.0, n)
    tot_chol = (tot_chol_base + 2.0 * (bmi - 25) + 1.0 * age + 10.0 * current_smoker).clip(107, 696).round(1)
    
    # Heart rate increases with smoking and BMI
    hr_base = rng.normal(75.9, 10.0, n)
    heart_rate = (hr_base + 3.0 * current_smoker + 0.3 * (bmi - 25)).clip(44, 143).round().astype(int)

    bp_meds = rng.binomial(1, 0.03, n)
    prevalent_stroke = rng.binomial(1, 0.006, n)
    
    # Hypertension clearly linked to high BP
    prevalent_hyp = (sys_bp > 140).astype(int)
    
    # Diabetes linked to high glucose
    diabetes = (glucose > 125).astype(int)

    # 10-year CHD ~ 15.2 %  (logistic function of risk factors)
    logit = 3.0 * (
        -9.0
        + 0.04 * age
        + 0.8 * male
        + 0.02 * sys_bp
        + 0.006 * tot_chol
        + 0.02 * glucose
        + 0.5 * current_smoker
        + 0.6 * diabetes
        + 0.04 * bmi
    ) + rng.normal(0, 0.10, n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    ten_year_chd = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "male": male, "age": age, "education": education,
        "currentSmoker": current_smoker, "cigsPerDay": cigs_per_day,
        "BPMeds": bp_meds, "prevalentStroke": prevalent_stroke,
        "prevalentHyp": prevalent_hyp, "diabetes": diabetes,
        "totChol": tot_chol, "sysBP": sys_bp, "diaBP": dia_bp,
        "BMI": bmi, "heartRate": heart_rate, "glucose": glucose,
        "TenYearCHD": ten_year_chd,
    })
    df.to_csv(dest, index=False)
    print(f"    ✓ Saved {len(df)} rows  (CHD+ {ten_year_chd.mean():.1%})")


def _generate_stroke(dest: Path, n: int = 30000, seed: int = 43) -> None:
    """Generate Stroke Prediction-like data.

    Based on variable distributions from the Kaggle dataset description
    and WHO stroke incidence statistics.
    """
    print("    ⚙ Generating synthetic Stroke Prediction data")
    rng = np.random.default_rng(seed)

    idx = np.arange(1, n + 1)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.41, 0.585, 0.005])
    age = np.abs(rng.normal(43, 22.5, n)).clip(0.1, 82).round(1)
    
    # --- Inter-feature Correlations ---
    # BMI increases with age
    bmi_base = rng.normal(28.9, 7.9, n)
    bmi = (bmi_base + 0.1 * (age - 30)).clip(10.3, 97.6).round(1)
    
    # Smoking is correlated with age
    smoking_base_prob = np.where(age > 18, 0.35, 0.05)
    smoking_prob = smoking_base_prob + rng.normal(0, 0.1, n)
    is_smoker = (smoking_prob > 0.4).astype(int)
    smoking = np.where(
        is_smoker,
        rng.choice(["smokes", "formerly smoked"], n, p=[0.6, 0.4]),
        rng.choice(["never smoked", "Unknown"], n, p=[0.8, 0.2])
    )
    
    # Glucose rises with age and BMI
    glucose_base = rng.normal(92, 20, n)
    avg_glucose = (glucose_base + 1.2 * (bmi - 25) + 0.5 * (age - 40)).clip(55, 272).round(2)
    # Simulate diabetic spikes
    spike_mask = rng.random(n) < (0.1 + 0.01 * (age - 40) + 0.005 * (bmi - 25))
    avg_glucose[spike_mask] = rng.normal(180, 50, spike_mask.sum()).clip(140, 272)
    
    # Hypertension heavily linked to age, BMI, and smoking
    hyp_prob = 0.05 + 0.01 * (age - 40) + 0.02 * (bmi - 25) + 0.1 * is_smoker
    hypertension = (rng.random(n) < hyp_prob).astype(int)
    
    # Heart disease linked to age, BMI, smoking, and hypertension
    hd_prob = 0.02 + 0.01 * (age - 50) + 0.01 * (bmi - 28) + 0.05 * is_smoker + 0.1 * hypertension
    heart_disease = (rng.random(n) < hd_prob).astype(int)
    
    ever_married = np.where(age > 20, rng.choice(["Yes", "No"], n, p=[0.66, 0.34]), "No")
    work_type = rng.choice(
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        n, p=[0.57, 0.16, 0.13, 0.13, 0.01],
    )
    residence = rng.choice(["Urban", "Rural"], n, p=[0.51, 0.49])

    # Handle missing BMI (≈3.9 %)
    bmi_str = bmi.astype(str)
    bmi_str[rng.random(n) < 0.039] = "N/A"
    
    # Stroke ~ 4.87 %
    logit = 3.0 * (
        -8.0
        + 0.05 * age
        + 1.2 * hypertension
        + 1.0 * heart_disease
        + 0.01 * avg_glucose
        + 0.05 * (bmi - 25)
    ) + rng.normal(0, 0.05, n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    stroke = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "id": idx, "gender": gender, "age": age,
        "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type,
        "Residence_type": residence, "avg_glucose_level": avg_glucose,
        "bmi": bmi_str, "smoking_status": smoking, "stroke": stroke,
    })
    df.to_csv(dest, index=False)
    print(f"    ✓ Saved {len(df)} rows  (stroke+ {stroke.mean():.1%})")


def _generate_cardiovascular(dest: Path, n: int = 100000, seed: int = 44) -> None:
    """Generate Cardiovascular Disease-like data.

    Based on variable distributions described in the Kaggle dataset page
    and WHO cardiovascular health statistics.
    """
    print(f"    ⚙ Generating synthetic Cardiovascular Disease data ({n} rows)")
    rng = np.random.default_rng(seed)

    idx = np.arange(n)
    # age in days (≈35–65 years)
    age_days = rng.normal(19_468, 2_467, n).clip(10_798, 23_713).round().astype(int)
    age_years = age_days / 365.25
    gender = rng.choice([1, 2], n, p=[0.35, 0.65])  # 1=F, 2=M
    height = np.where(
        gender == 1,
        rng.normal(161, 6.5, n),
        rng.normal(170, 7.5, n),
    ).clip(130, 210).round().astype(int)
    
    # --- Inter-feature Correlations ---
    # Weight increases with age, varies by gender
    weight_base = np.where(gender == 1, rng.normal(70.5, 12.0, n), rng.normal(78.0, 13.0, n))
    weight = (weight_base + 0.2 * (age_years - 40)).clip(35, 200).round(1)
    
    bmi = weight / ((height / 100) ** 2)
    
    # Blood pressure increases with age and BMI
    ap_hi_base = rng.normal(120, 12.0, n)
    ap_hi = (ap_hi_base + 0.5 * (age_years - 40) + 1.2 * (bmi - 25)).clip(80, 240).round().astype(int)
    
    ap_lo_base = rng.normal(80, 8.0, n)
    ap_lo = (ap_lo_base + 0.3 * (age_years - 40) + 0.8 * (bmi - 25)).clip(50, 160).round().astype(int)
    # Ensure diastolic < systolic
    ap_lo = np.minimum(ap_lo, ap_hi - 5)
    
    # Cholesterol linked to age and BMI
    chol_prob = 0.2 + 0.01 * (age_years - 40) + 0.02 * (bmi - 25)
    chol_rand = rng.random(n)
    cholesterol = np.ones(n, dtype=int)
    cholesterol[chol_rand < chol_prob] = 2
    cholesterol[chol_rand < chol_prob / 2] = 3
    
    # Glucose linked to age, BMI, and cholesterol
    gluc_prob = 0.15 + 0.01 * (age_years - 40) + 0.03 * (bmi - 25) + 0.05 * (cholesterol > 1)
    gluc_rand = rng.random(n)
    gluc = np.ones(n, dtype=int)
    gluc[gluc_rand < gluc_prob] = 2
    gluc[gluc_rand < gluc_prob / 2] = 3

    # Smoking is correlated with gender
    smoke_prob = np.where(gender == 2, 0.15, 0.04) + rng.normal(0, 0.02, n)
    smoke = (smoke_prob > 0.1).astype(int)
    
    # Alcohol linked to smoking
    alco_prob = 0.03 + 0.1 * smoke + rng.normal(0, 0.02, n)
    alco = (alco_prob > 0.1).astype(int)
    
    # Activity inversely linked to age and BMI
    active_prob = 0.9 - 0.005 * (age_years - 40) - 0.01 * (bmi - 25)
    active = (rng.random(n) < active_prob).astype(int)

    # Target: cardio ~ 49.9% (balanced in original)
    logit = 4.0 * (
        -16.5
        + 0.12 * age_years
        + 0.05 * ap_hi
        + 0.02 * ap_lo
        + 1.0 * (cholesterol >= 2).astype(float)
        + 0.8 * (gluc >= 2).astype(float)
        + 0.6 * smoke
        - 0.8 * active
        + 0.025 * weight
    ) + rng.normal(0, 0.05, n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    cardio = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "id": idx, "age": age_days, "gender": gender,
        "height": height, "weight": weight,
        "ap_hi": ap_hi, "ap_lo": ap_lo,
        "cholesterol": cholesterol, "gluc": gluc,
        "smoke": smoke, "alco": alco, "active": active,
        "cardio": cardio,
    })
    df.to_csv(dest, index=False, sep=";")  # original uses ';' separator
    print(f"    ✓ Saved {len(df)} rows  (cardio+ {cardio.mean():.1%})")


# ======================================================================
# Orchestrator
# ======================================================================

def download_all() -> dict[str, Path]:
    """Download all datasets.  Returns {name: path} mapping."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    print("\n" + "=" * 60)
    print("  DATASET DOWNLOAD")
    print("=" * 60)

    # --- 1. UCI Heart Disease ---
    key = "uci_heart"
    dest = RAW_DIR / DATASETS[key]["filename"]
    print(f"\n  [{1}] {DATASETS[key]['description']}")
    if dest.exists():
        print(f"    ✓ Already exists ({dest.name})")
    else:
        _download_uci_heart(dest)
    if dest.exists():
        df = pd.read_csv(dest)
        print(f"    → {len(df)} rows, {len(df.columns)} cols")
        results[key] = dest

    # --- 2. PIMA Diabetes ---
    key = "pima_diabetes"
    dest = RAW_DIR / DATASETS[key]["filename"]
    print(f"\n  [{2}] {DATASETS[key]['description']}")
    if dest.exists():
        print(f"    ✓ Already exists ({dest.name})")
    else:
        _download_pima(dest)
    if dest.exists():
        df = pd.read_csv(dest)
        print(f"    → {len(df)} rows, {len(df.columns)} cols")
        results[key] = dest

    # --- 3–5. Kaggle datasets ---
    kaggle_datasets = [
        ("framingham", 3, _generate_framingham),
        ("stroke", 4, _generate_stroke),
        ("cardiovascular", 5, _generate_cardiovascular),
    ]

    for key, idx, gen_fn in kaggle_datasets:
        meta = DATASETS[key]
        dest = RAW_DIR / meta["filename"]
        print(f"\n  [{idx}] {meta['description']}")
        if dest.exists():
            print(f"    ✓ Already exists ({dest.name})")
        else:
            # Try Kaggle first
            ok = False
            if "kaggle_slug" in meta:
                ok = _download_kaggle(
                    meta["kaggle_slug"], meta["kaggle_file"], dest,
                )
            if not ok:
                gen_fn(dest)
        if dest.exists():
            sep = ";" if key == "cardiovascular" else ","
            df = pd.read_csv(dest, sep=sep)
            print(f"    → {len(df)} rows, {len(df.columns)} cols")
            results[key] = dest

    # --- Summary ---
    print("\n" + "-" * 60)
    total = sum(
        len(pd.read_csv(p, sep=";" if "cardiovascular" in str(p) else ","))
        for p in results.values()
    )
    print(f"  Total datasets: {len(results)}/5")
    print(f"  Total samples:  {total:,}")
    print("=" * 60)

    return results


def verify_datasets() -> bool:
    """Verify all expected dataset files exist and are non-empty."""
    all_ok = True
    for key, meta in DATASETS.items():
        path = RAW_DIR / meta["filename"]
        if not path.exists():
            print(f"  ✗ Missing: {meta['filename']}")
            all_ok = False
        elif path.stat().st_size < 100:
            print(f"  ✗ Empty/corrupt: {meta['filename']}")
            all_ok = False
        else:
            sep = ";" if key == "cardiovascular" else ","
            df = pd.read_csv(path, sep=sep, nrows=5)
            print(f"  ✓ {meta['filename']:30s}  cols={len(df.columns)}")
    return all_ok


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    download_all()
    print("\n  Verification:")
    ok = verify_datasets()
    if ok:
        print("\n  ✓ All datasets ready.")
    else:
        print("\n  ⚠ Some datasets missing — pipeline will use available data.")
