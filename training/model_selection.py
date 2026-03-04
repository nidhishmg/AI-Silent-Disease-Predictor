"""
model_selection.py — Model comparison and selection utilities.

Provides functions for comparing candidate models via cross-validation
and generating comparison reports.  Used by ``train_model.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Run StratifiedKFold CV and return accuracy statistics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "cv_min": float(scores.min()),
        "cv_max": float(scores.max()),
    }


def evaluate_on_test(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a trained model on a test set."""
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "report": classification_report(
            y_test, y_pred, target_names=["Low Risk", "High Risk"]
        ),
    }


def compare_models(
    models: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare multiple models on CV and test metrics.

    Returns a DataFrame with one row per model and columns for each metric.
    """
    rows = []
    for name, model in models.items():
        cv_stats = cross_validate_model(model, X_train, y_train, cv, seed)
        test_stats = evaluate_on_test(model, X_test, y_test)
        rows.append({
            "model": name,
            "cv_accuracy": cv_stats["cv_mean"],
            "cv_std": cv_stats["cv_std"],
            "test_accuracy": test_stats["accuracy"],
            "precision": test_stats["precision"],
            "recall": test_stats["recall"],
            "f1": test_stats["f1"],
            "roc_auc": test_stats["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)


def select_best(
    comparison_df: pd.DataFrame,
    models: dict[str, Any],
) -> tuple[str, Any]:
    """Select the best model from comparison results.

    Prefers the model with highest test accuracy.
    """
    best_row = comparison_df.iloc[0]
    best_name = best_row["model"]
    return best_name, models[best_name]
