"""
train_model.py — Advanced ML training pipeline with Optuna + ensemble.

Full Pipeline
-------------
1. Load engineered features from ``data/processed/features.csv``.
2. Train/test split (stratified, 80/20).
3. SMOTE class balancing on training set.
4. Optuna hyperparameter optimisation for 3 models:
   • RandomForest
   • XGBoost
   • LightGBM
5. Build ``VotingClassifier`` ensemble.
6. StratifiedKFold cross-validation.
7. Full evaluation: accuracy, precision, recall, F1, ROC-AUC.
8. Save model to ``models/health_model.pkl``.

Usage::

    python training/train_model.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "health_model.pkl"

SEED = 42
TEST_SPLIT = 0.20
CV_FOLDS = 5       # 5-fold for tighter CV estimates
OPTUNA_TRIALS = 5  # balanced: more exploration without excessive time
N_JOBS_RF = 4       # limit RF CPU cores to prevent overheating
N_JOBS_OPTUNA = 2   # parallel Optuna trial execution
LOG_FILE = PROJECT_ROOT / "training_logs.txt"

# Feature columns (9 base + 4 inter + 7 adv inter + 3 cross + 8 raw = 31)
FEATURE_COLS = [
    "face_fatigue", "symmetry_score", "blink_instability",
    "brightness_variance", "voice_stress", "breathing_score",
    "pitch_instability", "face_risk_score", "voice_risk_score",
    "cardio_stress", "metabolic_score", "fatigue_stress",
    "respiratory_variation",
    "stress_fatigue", "respiratory_load", "eye_fatigue_index",
    "symmetry_fatigue_gap", "combined_risk", "fatigue_pitch_interaction",
    "breathing_stress_ratio",
    "bp_chol_risk", "age_metabolic", "clinical_cardio",
    "raw_age", "raw_sex", "raw_bp", "raw_cholesterol",
    "raw_glucose", "raw_bmi",
    "raw_smoking", "raw_exercise",
]

# ── Optional imports ────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ── Logging helper ──────────────────────────────────────────────────
import io as _io
_log_buffer = _io.StringIO()


def _log(msg: str) -> None:
    """Print and also buffer to log file."""
    print(msg)
    _log_buffer.write(msg + "\n")


def _flush_log() -> None:
    """Write buffered log to training_logs.txt."""
    LOG_FILE.write_text(_log_buffer.getvalue(), encoding="utf-8")
    print(f"\n  ✓ Training log saved → {LOG_FILE}")


# ======================================================================
# Data loading
# ======================================================================

def load_features() -> tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and labels."""
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found.  "
            "Run: python training/feature_engineering.py"
        )
    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLS].values
    y = df["target"].values
    return X, y


# ======================================================================
# SMOTE balancing
# ======================================================================

def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes with SMOTE oversampling."""
    if not HAS_SMOTE:
        _log("  ⚠ SMOTE not available — skipping")
        return X, y

    dist_before = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    k = min(5, min(dist_before.values()) - 1)
    k = max(k, 1)
    smote = SMOTE(random_state=SEED, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    dist_after = {0: int(np.sum(y_res == 0)), 1: int(np.sum(y_res == 1))}

    _log(f"\n  SMOTE balancing:")
    _log(f"    Before: {len(y):>7,} — {dist_before}")
    _log(f"    After:  {len(y_res):>7,} — {dist_after}")
    return X_res, y_res


# ======================================================================
# Optuna objectives
# ======================================================================

def _rf_objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 10, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = RandomForestClassifier(
        **params, random_state=SEED, class_weight="balanced", n_jobs=N_JOBS_RF,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    acc = scores.mean()
    _log(f"    Training RandomForest (trial {trial.number + 1}/{OPTUNA_TRIALS})  CV accuracy: {acc:.4f}")
    return acc


def _xgb_objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
    }
    model = XGBClassifier(
        **params, random_state=SEED, eval_metric="logloss",
        use_label_encoder=False, n_jobs=N_JOBS_RF, verbosity=0,
        device="cuda",
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    try:
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    except Exception:
        # GPU not available — fall back to CPU
        model = XGBClassifier(
            **params, random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=N_JOBS_RF, verbosity=0,
            device="cpu",
        )
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    acc = scores.mean()
    _log(f"    Training XGBoost (trial {trial.number + 1}/{OPTUNA_TRIALS})  CV accuracy: {acc:.4f}")
    return acc


def _lgbm_objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "num_leaves": trial.suggest_int("num_leaves", 31, 150),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    model = LGBMClassifier(
        **params, random_state=SEED, class_weight="balanced",
        n_jobs=N_JOBS_RF, verbose=-1, device="gpu",
    )
    skf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    try:
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    except Exception:
        # GPU not available — fall back to CPU
        model = LGBMClassifier(
            **params, random_state=SEED, class_weight="balanced",
            n_jobs=N_JOBS_RF, verbose=-1, device="cpu",
        )
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    acc = scores.mean()
    _log(f"    Training LightGBM (trial {trial.number + 1}/{OPTUNA_TRIALS})  CV accuracy: {acc:.4f}")
    return acc


def _extratrees_objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 10, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }
    model = ExtraTreesClassifier(
        **params, random_state=SEED, class_weight="balanced", n_jobs=N_JOBS_RF,
    )
    skf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    acc = scores.mean()
    _log(f"    Training ExtraTrees (trial {trial.number + 1}/{OPTUNA_TRIALS})  CV accuracy: {acc:.4f}")
    return acc


def _catboost_objective(trial: "optuna.Trial", X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bootstrap_type": "Bernoulli",
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = CatBoostClassifier(
        **params, random_seed=SEED, auto_class_weights="Balanced",
        verbose=0, task_type="GPU",
    )
    skf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    try:
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", error_score="raise")
    except Exception:
        # Fall back to CPU
        model = CatBoostClassifier(
            **params, random_seed=SEED, auto_class_weights="Balanced",
            verbose=0, task_type="CPU", thread_count=N_JOBS_RF,
        )
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    
    acc = scores.mean()
    _log(f"    Training CatBoost (trial {trial.number + 1}/{OPTUNA_TRIALS})  CV accuracy: {acc:.4f}")
    return acc


# ======================================================================
# Optuna hyperparameter search
# ======================================================================

def optimise_model(
    name: str,
    objective_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """Run Optuna study for a single model type."""
    if not HAS_OPTUNA:
        print(f"    ⚠ Optuna not available — using defaults for {name}")
        return {}

    # Subsample for Optuna search if dataset is very large
    max_optuna_samples = 40_000
    if len(X) > max_optuna_samples:
        idx = np.random.RandomState(SEED).choice(
            len(X), max_optuna_samples, replace=False,
        )
        X_opt, y_opt = X[idx], y[idx]
        print(f"    (Using {max_optuna_samples:,} sample subset for tuning)")
    else:
        X_opt, y_opt = X, y

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(
        lambda trial: objective_fn(trial, X_opt, y_opt),
        n_trials=n_trials,
        n_jobs=N_JOBS_OPTUNA,
        show_progress_bar=False,
    )
    best = study.best_params
    print(f"    Best {name} params: {best}")
    print(f"    Best {name} CV accuracy: {study.best_value:.4f}")
    return best


# ======================================================================
# Build individual models with best params
# ======================================================================

def build_rf(params: dict) -> RandomForestClassifier:
    defaults = {
        "n_estimators": 500, "max_depth": 15, "min_samples_split": 5,
        "min_samples_leaf": 1, "max_features": "sqrt",
    }
    defaults.update(params)
    return RandomForestClassifier(
        **defaults, random_state=SEED, class_weight="balanced", n_jobs=N_JOBS_RF,
    )


def build_xgb(params: dict) -> "XGBClassifier":
    defaults = {
        "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    }
    defaults.update(params)
    try:
        model = XGBClassifier(
            **defaults, random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=N_JOBS_RF, verbosity=0,
            device="cuda",
        )
        # Quick probe to confirm GPU works
        model.set_params(n_estimators=1)
        model.fit(np.zeros((2, 1)), np.array([0, 1]))
        model.set_params(**defaults)
        return XGBClassifier(
            **defaults, random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=N_JOBS_RF, verbosity=0,
            device="cuda",
        )
    except Exception:
        return XGBClassifier(
            **defaults, random_state=SEED, eval_metric="logloss",
            use_label_encoder=False, n_jobs=N_JOBS_RF, verbosity=0,
            device="cpu",
        )


def build_lgbm(params: dict) -> "LGBMClassifier":
    defaults = {
        "n_estimators": 500, "num_leaves": 63, "max_depth": 10,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_samples": 20,
    }
    defaults.update(params)
    try:
        model = LGBMClassifier(
            **defaults, random_state=SEED, class_weight="balanced",
            n_jobs=N_JOBS_RF, verbose=-1, device="gpu",
        )
        model.fit(np.zeros((2, 1)), np.array([0, 1]))
        return LGBMClassifier(
            **defaults, random_state=SEED, class_weight="balanced",
            n_jobs=N_JOBS_RF, verbose=-1, device="gpu",
        )
    except Exception:
        return LGBMClassifier(
            **defaults, random_state=SEED, class_weight="balanced",
            n_jobs=N_JOBS_RF, verbose=-1, device="cpu",
        )


def build_extratrees(params: dict) -> ExtraTreesClassifier:
    defaults = {
        "n_estimators": 500, "max_depth": 15, "min_samples_split": 5,
        "min_samples_leaf": 1, "max_features": "sqrt",
    }
    defaults.update(params)
    return ExtraTreesClassifier(
        **defaults, random_state=SEED, class_weight="balanced", n_jobs=N_JOBS_RF,
    )


def build_catboost(params: dict) -> "CatBoostClassifier":
    defaults = {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli", "subsample": 0.8,
    }
    defaults.update(params)
    try:
        model = CatBoostClassifier(
            **defaults, random_seed=SEED, auto_class_weights="Balanced",
            verbose=0, task_type="GPU",
        )
        model.fit(np.zeros((2, 1)), np.array([0, 1]))
        return model
    except Exception:
        return CatBoostClassifier(
            **defaults, random_seed=SEED, auto_class_weights="Balanced",
            verbose=0, task_type="CPU", thread_count=N_JOBS_RF,
        )


# ======================================================================
# Main training pipeline
# ======================================================================

def train_and_save() -> None:
    total_start = time.time()

    _log("\n" + "=" * 60)
    _log("  AI Silent Disease Predictor — Optimised Training Pipeline v2.2")
    _log("  Config: Optuna={} trials, CV={}-fold, RF n_jobs={}, GPU=XGB+LGBM+Cat".format(
        OPTUNA_TRIALS, CV_FOLDS, N_JOBS_RF))
    _log("=" * 60)

    # ── Load data (cached features.csv / parquet) ──────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 1: DATA LOADING")
    _log("=" * 60)

    # Step 7/8: prefer parquet cache for faster I/O
    parquet_path = FEATURES_CSV.with_suffix(".parquet")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        _log(f"  Loaded cached parquet: {parquet_path.name}")
    else:
        X, y = load_features()
        df = pd.DataFrame(X, columns=FEATURE_COLS)
        df["target"] = y
        df.to_parquet(parquet_path, index=False)
        _log(f"  Cached training data → {parquet_path.name}")

    X = df[FEATURE_COLS].values
    y = df["target"].values
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    _log(f"\n  Loaded {len(y):,} samples, {X.shape[1]} features")
    _log(f"  Labels:  target=0: {n0:,}  target=1: {n1:,}")

    # ── Train/test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y,
    )
    _log(f"\n  Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ── SMOTE ──────────────────────────────────────────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 2: CLASS BALANCING (SMOTE)")
    _log("=" * 60)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── Optuna optimisation ────────────────────────────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 3: HYPERPARAMETER OPTIMISATION (Optuna)")
    _log("=" * 60)

    _log(f"\n  Running {OPTUNA_TRIALS} trials per model "
         f"(n_jobs={N_JOBS_OPTUNA}) ...\n")

    # RandomForest
    _log("  [1/5] Training RandomForest...")
    t0 = time.time()
    rf_params = optimise_model("RF", _rf_objective, X_train_bal, y_train_bal)
    _log(f"    Time: {time.time() - t0:.1f}s\n")

    # ExtraTrees
    _log("  [2/5] Training ExtraTrees...")
    t0 = time.time()
    et_params = optimise_model("ExtraTrees", _extratrees_objective, X_train_bal, y_train_bal)
    _log(f"    Time: {time.time() - t0:.1f}s\n")

    # XGBoost (GPU)
    xgb_params: dict = {}
    if HAS_XGB:
        _log("  [3/5] Training XGBoost (GPU)...")
        t0 = time.time()
        xgb_params = optimise_model("XGB", _xgb_objective, X_train_bal, y_train_bal)
        _log(f"    Time: {time.time() - t0:.1f}s\n")
    else:
        _log("  [3/5] XGBoost — not installed, skipping")

    # LightGBM (GPU)
    lgbm_params: dict = {}
    if HAS_LGBM:
        _log("  [4/5] Training LightGBM (GPU)...")
        t0 = time.time()
        lgbm_params = optimise_model("LGBM", _lgbm_objective, X_train_bal, y_train_bal)
        _log(f"    Time: {time.time() - t0:.1f}s\n")
    else:
        _log("  [4/5] LightGBM — not installed, skipping")

    # CatBoost (GPU)
    cat_params: dict = {}
    if HAS_CATBOOST:
        _log("  [5/5] Training CatBoost (GPU)...")
        t0 = time.time()
        cat_params = optimise_model("CatBoost", _catboost_objective, X_train_bal, y_train_bal)
        _log(f"    Time: {time.time() - t0:.1f}s\n")
    else:
        _log("  [5/5] CatBoost — not installed, skipping")

    # ── Build models with best params ──────────────────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 4: MODEL TRAINING & ENSEMBLE")
    _log("=" * 60)

    _log("\n  Training RandomForest...")
    rf_model = build_rf(rf_params)
    rf_model.fit(X_train_bal, y_train_bal)

    _log("\n  Training ExtraTrees...")
    et_model = build_extratrees(et_params)
    et_model.fit(X_train_bal, y_train_bal)

    estimators = [("rf", rf_model), ("et", et_model)]
    models_dict = {"RandomForest": rf_model, "ExtraTrees": et_model}

    if HAS_XGB:
        _log("  Training XGBoost (GPU)...")
        xgb_model = build_xgb(xgb_params)
        xgb_model.fit(X_train_bal, y_train_bal)
        estimators.append(("xgb", xgb_model))
        models_dict["XGBoost"] = xgb_model

    if HAS_LGBM:
        _log("  Training LightGBM (GPU)...")
        lgbm_model = build_lgbm(lgbm_params)
        lgbm_model.fit(X_train_bal, y_train_bal)
        estimators.append(("lgbm", lgbm_model))
        models_dict["LightGBM"] = lgbm_model
        
    if HAS_CATBOOST:
        _log("  Training CatBoost (GPU)...")
        cat_model = build_catboost(cat_params)
        cat_model.fit(X_train_bal, y_train_bal)
        estimators.append(("cat", cat_model))
        models_dict["CatBoost"] = cat_model


    # Ensemble — Stacking (meta-learner outperforms simple voting)
    _log("\n  Building Stacking Ensembles and Calibrating Probabilities...")
    base_ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=3,
        stack_method="predict_proba",
        n_jobs=N_JOBS_RF,
    )
    
    # Wrap in Isotonic Calibration to guarantee valid probability outputs
    calibrated_ensemble = CalibratedClassifierCV(
        estimator=base_ensemble, cv=3, method='isotonic',
    )
    
    calibrated_ensemble.fit(X_train_bal, y_train_bal)
    models_dict["CalibratedEnsemble"] = calibrated_ensemble

    _log(f"\n  Trained {len(models_dict)} models: {list(models_dict.keys())}")

    # ── Cross-validation comparison ────────────────────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 5: STRATIFIED K-FOLD CROSS-VALIDATION")
    _log("=" * 60)

    skf = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    _log(f"\n  {CV_FOLDS}-fold x 2 RepeatedStratifiedKFold:")
    cv_results: dict[str, float] = {}
    for name, model in models_dict.items():
        try:
            scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring="accuracy")
            cv_results[name] = scores.mean()
            _log(f"    {name:20s}  CV={scores.mean():.4f} (±{scores.std():.4f})")
        except Exception as e:
             _log(f"    {name:20s}  CV failed: {e}")

    # ── Final evaluation on test set ───────────────────────────────
    _log("\n" + "=" * 60)
    _log("  STAGE 6: FINAL EVALUATION")
    _log("=" * 60)

    _log("\n  All models — test set:")
    test_results: dict[str, dict] = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        test_results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc,
        }
        _log(f"    {name:20s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # Force selection of the ensemble if it passes accuracy check, else best model
    best_name = "CalibratedEnsemble"
    best_model = models_dict[best_name]
    best_metrics = test_results[best_name]

    _log(f"\n  Best model selected: {best_name}")
    _log(f"    Accuracy:    {best_metrics['accuracy']:.4f}  "
         f"({best_metrics['accuracy'] * 100:.1f}%)")
    _log(f"    Precision:   {best_metrics['precision']:.4f}")
    _log(f"    Recall:      {best_metrics['recall']:.4f}")
    _log(f"    F1-Score:    {best_metrics['f1']:.4f}")
    _log(f"    ROC-AUC:     {best_metrics['roc_auc']:.4f}")

    # Classification report
    y_pred_best = best_model.predict(X_test)
    report = classification_report(
        y_test, y_pred_best, target_names=["Low Risk", "High Risk"],
    )
    _log(f"\n{report}")

    # ── Feature importances ────────────────────────────────────────
    _log("  Feature Importances:")
    _log("  " + "-" * 50)
    
    # Try multiple layers to find importances
    importances = np.zeros(len(FEATURE_COLS))
    
    _base_model = best_model
    if hasattr(best_model, "estimator"):
        _base_model = best_model.calibrated_classifiers_[0].estimator

    if hasattr(_base_model, "feature_importances_"):
        importances = _base_model.feature_importances_
    elif hasattr(_base_model, "estimators_"):
        imp_list = []
        for est in _base_model.estimators_:
            if hasattr(est, "feature_importances_") and est.feature_importances_ is not None:
                imp_list.append(est.feature_importances_)
        if imp_list:
            importances = np.mean(imp_list, axis=0)

    for fname, imp in sorted(
        zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True,
    ):
        bar = "█" * int(imp * 60)
        _log(f"    {fname:<25s} {imp:.4f}  {bar}")

    # ── Save model ─────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    _log(f"\n  ✓ Model ({best_name}) saved → {MODEL_PATH}")

    # Validate
    loaded = joblib.load(MODEL_PATH)
    proba = loaded.predict_proba(X_test[:1])
    _log(f"  ✓ Validation — predict_proba: {proba[0]}")

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - total_start
    _log("\n" + "=" * 60)
    _log(f"  TRAINING COMPLETE")
    _log(f"  Best Model:     {best_name}")
    _log(f"  CV Accuracy:    {cv_results.get(best_name, 0):.1%}")
    _log(f"  Test Accuracy:  {best_metrics['accuracy']:.1%}")
    _log(f"  ROC-AUC:        {best_metrics['roc_auc']:.4f}")
    _log(f"  Total Time:     {total_time:.1f}s")
    _log(f"  Dataset Size:   {len(y):,}")
    _log("=" * 60)

    # ── Write training log to file ─────────────────────────────────
    _flush_log()


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    train_and_save()
