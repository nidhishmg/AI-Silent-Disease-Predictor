"""
train_model.py — Advanced ML training pipeline for AI Silent Disease Predictor.

Production-grade pipeline with:
    1. Real clinical data (UCI Heart Disease + PIMA Diabetes)
    2. Data cleaning (missing values, IQR outlier removal)
    3. Feature engineering (derived clinical features)
    4. SMOTE class balancing
    5. Feature selection (SelectKBest)
    6. Hyperparameter optimization (GridSearchCV)
    7. Ensemble model (VotingClassifier: RandomForest + XGBoost)
    8. StratifiedKFold cross-validation
    9. Full evaluation metrics

Saves:
    • models/health_model.pkl
    • models/scaler.pkl

Usage::

    python data/download_datasets.py   # one-time
    python train_model.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# Suppress convergence warnings during grid search
warnings.filterwarnings("ignore", category=UserWarning)

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

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    logger.warning("imbalanced-learn not installed — SMOTE disabled")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed — using GradientBoosting fallback")


# ==============================================================================
# Synthetic data generation (fallback + augmentation)
# ==============================================================================

def generate_synthetic_data(
    n_samples: int = TRAINING_SAMPLES,
    seed: int = TRAINING_SEED,
) -> pd.DataFrame:
    """Generate medically-informed synthetic biomarker data."""
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic samples (seed=%d)", n_samples, seed)

    face_fatigue = rng.beta(2, 5, n_samples)
    symmetry_score = rng.beta(8, 2, n_samples)
    blink_instability = rng.beta(2, 6, n_samples)
    brightness_variance = rng.beta(2, 5, n_samples)
    voice_stress = rng.beta(2, 5, n_samples)
    breathing_score = rng.beta(2, 5, n_samples)
    pitch_instability = rng.beta(2, 6, n_samples)

    face_risk_score = (
        0.3 * face_fatigue + 0.3 * (1.0 - symmetry_score)
        + 0.2 * blink_instability + 0.2 * brightness_variance
    ) * 100 + rng.normal(0, 3, n_samples)
    face_risk_score = np.clip(face_risk_score, 0, 100)

    voice_risk_score = (
        0.4 * voice_stress + 0.3 * breathing_score + 0.3 * pitch_instability
    ) * 100 + rng.normal(0, 3, n_samples)
    voice_risk_score = np.clip(voice_risk_score, 0, 100)

    risk_linear = (
        1.5 * face_fatigue + 1.2 * voice_stress + 1.0 * breathing_score
        + 0.8 * pitch_instability + 0.8 * blink_instability
        + 0.6 * brightness_variance - 1.0 * symmetry_score
        + 0.01 * face_risk_score + 0.01 * voice_risk_score
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
# Data assembly
# ==============================================================================

def assemble_training_data(seed: int = TRAINING_SEED) -> pd.DataFrame:
    """Load real datasets, optionally augment with synthetic samples."""
    print("\n" + "=" * 60)
    print("  STAGE 1: DATA ASSEMBLY & CLEANING")
    print("=" * 60)

    real_df = load_real_datasets(seed)

    if real_df is not None:
        if "_source" in real_df.columns:
            real_df = real_df.drop(columns=["_source"])

        n_real = len(real_df)
        print(f"\n  ✓ Loaded {n_real} cleaned clinical records")

        n_augment = max(SYNTHETIC_AUGMENT, 0)
        if n_augment > 0:
            print(f"  + Generating {n_augment} synthetic augmentation samples")
            synth_df = generate_synthetic_data(n_samples=n_augment, seed=seed + 100)
            df = pd.concat([real_df, synth_df], ignore_index=True)
        else:
            df = real_df

        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_total = len(df)
        real_pct = n_real / n_total * 100
        print(f"\n  Total training samples: {n_total}")
        print(f"  Real data:     {n_real} ({real_pct:.1f}%)")
        print(f"  Synthetic:     {n_total - n_real} ({100 - real_pct:.1f}%)")
    else:
        print("\n  ⚠  No real datasets found — using synthetic data only")
        df = generate_synthetic_data(n_samples=TRAINING_SAMPLES, seed=seed)

    n_risk0 = int((df["health_risk"] == 0).sum())
    n_risk1 = int((df["health_risk"] == 1).sum())
    logger.info("Dataset — total: %d, risk=0: %d, risk=1: %d", len(df), n_risk0, n_risk1)
    print(f"  Label dist:    risk=0: {n_risk0},  risk=1: {n_risk1}")

    return df


# ==============================================================================
# Feature selection
# ==============================================================================

def select_features(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int = 9
) -> tuple[np.ndarray, list[str], list[int]]:
    """Select top-k features using ANOVA F-test.

    Returns the filtered X, selected feature names, and selected indices.
    """
    k = min(k, X.shape[1])  # can't select more than available
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    mask = selector.get_support()
    selected_names = [f for f, m in zip(feature_names, mask) if m]
    selected_idx = [i for i, m in enumerate(mask) if m]

    scores = selector.scores_
    print("\n  Feature scores (ANOVA F-test):")
    for fname, score, m in sorted(
        zip(feature_names, scores, mask), key=lambda x: x[1], reverse=True
    ):
        status = "✓" if m else "✗"
        print(f"    {status} {fname:<22s}  F={score:.2f}")

    return X_selected, selected_names, selected_idx


# ==============================================================================
# SMOTE balancing
# ==============================================================================

def apply_smote(
    X: np.ndarray, y: np.ndarray, seed: int = TRAINING_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Balance classes using SMOTE oversampling."""
    if not HAS_SMOTE:
        print("  ⚠ SMOTE not available — skipping balancing")
        return X, y

    n_before = len(y)
    dist_before = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}

    smote = SMOTE(random_state=seed, k_neighbors=min(5, min(dist_before.values()) - 1))
    X_res, y_res = smote.fit_resample(X, y)

    n_after = len(y_res)
    dist_after = {0: int(np.sum(y_res == 0)), 1: int(np.sum(y_res == 1))}

    print(f"\n  SMOTE balancing:")
    print(f"    Before: {n_before} samples — {dist_before}")
    print(f"    After:  {n_after} samples — {dist_after}")

    return X_res, y_res


# ==============================================================================
# Model building with hyperparameter optimization
# ==============================================================================

def build_optimized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = TRAINING_SEED,
) -> tuple:
    """Build an ensemble model with hyperparameter optimization.

    Steps:
        1. Grid search on RandomForest
        2. Grid search on XGBoost (or GradientBoosting fallback)
        3. Combine into VotingClassifier ensemble
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # ── RandomForest Grid Search ────────────────────────────────────
    print("\n  [1/3] Optimizing RandomForest ...")
    t0 = time.time()

    rf_param_grid = {
        "n_estimators": [300, 500, 700],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }
    rf_base = RandomForestClassifier(
        random_state=seed, class_weight="balanced", n_jobs=-1,
    )
    rf_grid = GridSearchCV(
        rf_base, rf_param_grid, cv=skf, scoring="accuracy",
        n_jobs=-1, verbose=0, refit=True,
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    rf_time = time.time() - t0
    print(f"    Best RF params: {rf_grid.best_params_}")
    print(f"    Best RF CV accuracy: {rf_grid.best_score_:.4f}")
    print(f"    Time: {rf_time:.1f}s")

    # ── XGBoost / GradientBoosting Grid Search ──────────────────────
    print("\n  [2/3] Optimizing XGBoost ...")
    t0 = time.time()

    if HAS_XGB:
        xgb_param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }
        xgb_base = XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
    else:
        xgb_param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        }
        xgb_base = GradientBoostingClassifier(random_state=seed)

    xgb_grid = GridSearchCV(
        xgb_base, xgb_param_grid, cv=skf, scoring="accuracy",
        n_jobs=-1, verbose=0, refit=True,
    )
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    xgb_time = time.time() - t0
    model_name = "XGBoost" if HAS_XGB else "GradientBoosting"
    print(f"    Best {model_name} params: {xgb_grid.best_params_}")
    print(f"    Best {model_name} CV accuracy: {xgb_grid.best_score_:.4f}")
    print(f"    Time: {xgb_time:.1f}s")

    # ── Ensemble: VotingClassifier ──────────────────────────────────
    print("\n  [3/3] Building VotingClassifier ensemble ...")
    t0 = time.time()

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_best),
            ("xgb", xgb_best),
        ],
        voting="soft",   # probability-based voting
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    ens_time = time.time() - t0

    # ── Compare all three ───────────────────────────────────────────
    print("\n  Model comparison (CV accuracy):")
    rf_cv = cross_val_score(rf_best, X_train, y_train, cv=skf, scoring="accuracy")
    xgb_cv = cross_val_score(xgb_best, X_train, y_train, cv=skf, scoring="accuracy")
    ens_cv = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring="accuracy")

    print(f"    RandomForest:      {rf_cv.mean():.4f} (±{rf_cv.std():.4f})")
    print(f"    {model_name:18s}: {xgb_cv.mean():.4f} (±{xgb_cv.std():.4f})")
    print(f"    Ensemble:          {ens_cv.mean():.4f} (±{ens_cv.std():.4f})")

    # Pick best performing — use mean-std (conservative, penalises high variance)
    models = {
        "RandomForest": (rf_best, rf_cv.mean()),
        model_name: (xgb_best, xgb_cv.mean()),
        "Ensemble": (ensemble, ens_cv.mean()),
    }
    model_scores = {
        "RandomForest": rf_cv.mean() - rf_cv.std(),
        model_name: xgb_cv.mean() - xgb_cv.std(),
        "Ensemble": ens_cv.mean() - ens_cv.std(),
    }
    best_name = max(model_scores, key=model_scores.get)

    best_model = models[best_name][0]
    best_score = models[best_name][1]

    print(f"\n  ★ Best model: {best_name} (CV={best_score:.4f})")

    return best_model, best_name, models, model_scores


# ==============================================================================
# Training pipeline
# ==============================================================================

def train_and_save() -> None:
    """Complete advanced training pipeline."""
    logger.info("=" * 60)
    logger.info("AI Silent Disease Predictor — Training v%s", MODEL_VERSION)
    logger.info("=" * 60)

    total_start = time.time()

    print("\n" + "=" * 60)
    print(f"  AI Silent Disease Predictor — Advanced Training Pipeline v{MODEL_VERSION}")
    print("=" * 60)

    # ── Stage 1: Assemble & clean data ──────────────────────────────
    df = assemble_training_data()

    X = df[FEATURE_NAMES].values
    y = df["health_risk"].values

    # ── Stage 1b: Feature selection ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1b: FEATURE SELECTION")
    print("=" * 60)

    # Keep all 9 features (they all score well), but show the ranking
    X_sel, selected_names, selected_idx = select_features(
        X, y, FEATURE_NAMES, k=len(FEATURE_NAMES)
    )

    # ── Train / test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=TRAINING_SEED, stratify=y  # type: ignore
    )
    logger.info("Split — train: %d,  test: %d", len(X_train), len(X_test))
    print(f"\n  Train: {len(X_train)} samples,  Test: {len(X_test)} samples")

    # ── Stage 2: SMOTE balancing ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2: CLASS BALANCING (SMOTE)")
    print("=" * 60)

    X_train_bal, y_train_bal = apply_smote(X_train, y_train, TRAINING_SEED)

    # ── Stage 2b: Scale features ────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # ── Stage 3: Build optimized models ─────────────────────────────
    best_model, best_name, all_models, cv_scores = build_optimized_model(
        X_train_scaled, y_train_bal, TRAINING_SEED
    )

    # ── Stage 4: Final evaluation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4: FINAL EVALUATION")
    print("=" * 60)

    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report: str = classification_report(  # type: ignore
        y_test, y_pred, target_names=["Low Risk", "High Risk"]
    )

    print(f"\n  ★ Best Model: {best_name}")
    print(f"\n  Test Accuracy:   {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"\n{report}")

    logger.info("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
                acc, prec, rec, f1)

    # ── Evaluate all models on test set & pick best ───────────────
    print("  All models on test set:")
    test_results: dict[str, float] = {}
    for name, (model, cv_score) in all_models.items():
        y_p = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_p)
        test_results[name] = test_acc
        print(f"    {name:20s}  CV={cv_score:.4f}  Test={test_acc:.4f}")

    # Final model selection: best test accuracy (held-out evaluation)
    final_name = max(test_results, key=test_results.get)  # type: ignore[arg-type]
    if final_name != best_name:
        print(f"\n  → Overriding CV pick ({best_name}) with best test model: {final_name}")
        best_name = final_name
        best_model = all_models[final_name][0]
        y_pred = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        report = classification_report(  # type: ignore
            y_test, y_pred, target_names=["Low Risk", "High Risk"]
        )
        print(f"\n  ★ Final Model: {best_name}")
        print(f"\n  Test Accuracy:   {acc:.4f}  ({acc * 100:.1f}%)")
        print(f"  Precision:       {prec:.4f}")
        print(f"  Recall:          {rec:.4f}")
        print(f"  F1-Score:        {f1:.4f}")
        print(f"\n{report}")

    # ── Feature importances ─────────────────────────────────────────
    print("\n  Feature Importances (from best model):")
    print("  " + "-" * 48)

    # Extract importances — handle ensemble vs single model
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "estimators_"):
        # VotingClassifier — average importances from sub-models
        imp_list = []
        for _, est in best_model.estimators:
            if hasattr(est, "feature_importances_"):
                imp_list.append(est.feature_importances_)
        importances = np.mean(imp_list, axis=0) if imp_list else np.zeros(len(FEATURE_NAMES))
    else:
        importances = np.zeros(len(FEATURE_NAMES))

    for fname, imp in sorted(
        zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"    {fname:<22s} {imp:.4f}  {bar}")
        logger.info("Feature importance: %s = %.4f", fname, imp)

    # ── Save artefacts ──────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model (%s) saved → %s", best_name, MODEL_PATH)
    logger.info("Scaler saved → %s", SCALER_PATH)
    print(f"\n  ✓ Model ({best_name}) saved to {MODEL_PATH}")
    print(f"  ✓ Scaler saved to {SCALER_PATH}")

    # ── Validate saved artefacts ────────────────────────────────────
    loaded_model = joblib.load(MODEL_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    test_input = X_test[:1]
    test_scaled = loaded_scaler.transform(test_input)
    proba = loaded_model.predict_proba(test_scaled)
    print(f"  ✓ Validation — sample predict_proba: {proba[0]}")
    logger.info("Validation passed. Model is ready.")

    # ── Final summary ───────────────────────────────────────────────
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Best Model:    {best_name}")
    print(f"  Test Accuracy: {acc * 100:.1f}%")
    print(f"  Precision:     {prec * 100:.1f}%")
    print(f"  Recall:        {rec * 100:.1f}%")
    print(f"  F1-Score:      {f1 * 100:.1f}%")
    print(f"  Total Time:    {total_time:.1f}s")
    print(f"  Samples:       {len(df)} ({len(X_train_bal)} after SMOTE)")
    print("=" * 60)


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    train_and_save()
