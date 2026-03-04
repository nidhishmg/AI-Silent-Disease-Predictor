"""
feature_importance.py — SHAP-based feature importance analysis.

Generates:
- SHAP summary plot (beeswarm)
- SHAP feature importance bar chart
- Saves images to ``visualization/`` directory.

Usage::

    python visualization/feature_importance.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "health_model.pkl"
VIZ_DIR = PROJECT_ROOT / "visualization"

FEATURE_COLS = [
    "face_fatigue", "symmetry_score", "blink_instability",
    "brightness_variance", "voice_stress", "breathing_score",
    "pitch_instability", "face_risk_score", "voice_risk_score",
    "cardio_stress", "metabolic_score", "fatigue_stress",
    "respiratory_variation",
]


def _load_data_and_model():
    """Load feature matrix and trained model."""
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found.  Run feature_engineering first."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found.  Run training first."
        )

    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLS]
    model = joblib.load(MODEL_PATH)
    return X, model


def generate_shap_analysis(max_samples: int = 500) -> None:
    """Generate SHAP plots and save to visualization directory."""
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  ⚠ Cannot generate SHAP plots: {e}")
        print("    Install: pip install shap matplotlib")
        return

    X, model = _load_data_and_model()

    # Subsample for speed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    print(f"\n  Computing SHAP values for {len(X_sample)} samples ...")

    # Use TreeExplainer for tree-based models
    from sklearn.ensemble import VotingClassifier
    if isinstance(model, VotingClassifier):
        # Use a single estimator for SHAP (first one)
        base_model = model.estimators_[0]
        explainer = shap.TreeExplainer(base_model)
    elif hasattr(model, "predict_proba"):
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(
                model.predict_proba, X_sample.iloc[:50],
            )
    else:
        print("  ⚠ Model type not supported for SHAP analysis")
        return

    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1]  # positive class
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]  # positive class
    else:
        shap_vals = shap_values

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # ── Summary (beeswarm) plot ──────────────────────────────────
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=FEATURE_COLS,
        show=False,
        max_display=13,
    )
    plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path1 = VIZ_DIR / "shap_summary.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP summary plot saved → {path1}")

    # ── Bar plot (mean |SHAP|) ───────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals, X_sample,
        feature_names=FEATURE_COLS,
        plot_type="bar",
        show=False,
        max_display=13,
    )
    plt.title("Mean |SHAP| per Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path2 = VIZ_DIR / "shap_bar.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ SHAP bar chart saved → {path2}")

    # ── Custom importance bar chart ──────────────────────────────
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feature_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": mean_abs,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feature_imp["feature"], feature_imp["importance"],
                   color="#2196F3", edgecolor="#1565C0", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, feature_imp["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path3 = VIZ_DIR / "feature_importance.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Feature importance chart saved → {path3}")

    print("\n  SHAP analysis complete ✓")


if __name__ == "__main__":
    generate_shap_analysis()
