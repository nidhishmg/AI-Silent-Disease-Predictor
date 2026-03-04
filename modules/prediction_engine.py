"""
prediction_engine.py — ML inference and risk scoring module.

Loads the trained RandomForest model and scaler, validates input,
computes probability-based health risk, and returns structured results.

Elite features
--------------
- Feature drift detection (out-of-distribution warning)
- SHAP-ready explainability placeholder
- Lazy model loading (singleton pattern, no global state leak)
- Confidence scoring via entropy

Architecture rules
------------------
- Does NOT import Streamlit.
- Does NOT import face_analysis or voice_analysis.
- Pure dict-in → dict-out interface (API-ready).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from config.settings import (
    DRIFT_Z_THRESHOLD,
    FEATURE_NAMES,
    LOW_THRESHOLD,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    MODERATE_THRESHOLD,
    NUM_FEATURES,
    SCALER_PATH,
)
from utils.feature_utils import compute_entropy
from utils.logger import get_logger
from utils.preprocessing import clip_score, validate_feature_dict

logger = get_logger(__name__)


# ==============================================================================
# Lazy model cache (module-private, not true global state)
# ==============================================================================

class _ModelCache:
    """Holds loaded model artefacts.  Prevents repeated disk I/O."""

    def __init__(self) -> None:
        self.model: Any = None
        self.scaler: Any = None
        self.poly: Any = None  # PolynomialFeatures transformer (optional)
        self.loaded: bool = False

    def load(self) -> bool:
        """Attempt to load model and scaler from disk.  Returns success flag."""
        if self.loaded:
            return True
        if not os.path.isfile(MODEL_PATH):
            logger.error("Model file not found: %s", MODEL_PATH)
            return False
        if not os.path.isfile(SCALER_PATH):
            logger.error("Scaler file not found: %s", SCALER_PATH)
            return False
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            # Optional poly transformer (for polynomial feature models)
            poly_path = os.path.join(MODEL_DIR, "poly.pkl")
            if os.path.isfile(poly_path):
                self.poly = joblib.load(poly_path)
                logger.info("PolynomialFeatures loaded from %s", poly_path)
            self.loaded = True
            logger.info(
                "Model v%s loaded successfully (%s, %s)",
                MODEL_VERSION,
                MODEL_PATH,
                SCALER_PATH,
            )
            return True
        except Exception:
            logger.exception("Failed to load model artefacts.")
            return False


_cache = _ModelCache()


# ==============================================================================
# PUBLIC API
# ==============================================================================

def predict_health_risk(features: Dict[str, float]) -> Dict[str, Any]:
    """Predict overall health risk from fused biomarker features.

    Parameters
    ----------
    features : dict
        Must contain all keys listed in ``config.settings.FEATURE_NAMES``,
        each mapping to a numeric value.

    Returns
    -------
    dict
        ``overall_risk``  (float 0–100),
        ``risk_level``    ("Low" | "Moderate" | "High" | "Unknown"),
        ``confidence_score`` (float 0–100),
        ``feature_contribution`` (dict[str, float]),
        ``model_version`` (str),
        ``drift_warning`` (bool).
    """
    # --- Validate input ---
    if not validate_feature_dict(features, FEATURE_NAMES):
        logger.error("Invalid feature dict: %s", list(features.keys()) if isinstance(features, dict) else type(features))
        return _fallback_result("Invalid input features")

    # --- Load model ---
    if not _cache.load():
        return _fallback_result("Model not loaded — run train_model.py first")

    try:
        # Build ordered feature vector
        feature_vector = np.array(
            [features[name] for name in FEATURE_NAMES], dtype=np.float64
        ).reshape(1, -1)

        # --- Feature drift detection ---
        scaled = _cache.scaler.transform(feature_vector)
        drift_warning = bool(np.any(np.abs(scaled) > DRIFT_Z_THRESHOLD))
        if drift_warning:
            flagged = [
                FEATURE_NAMES[i]
                for i in range(NUM_FEATURES)
                if abs(scaled[0, i]) > DRIFT_Z_THRESHOLD
            ]
            logger.warning(
                "Feature drift detected — out-of-distribution features: %s",
                flagged,
            )

        # --- Apply polynomial feature expansion if used during training ---
        model_input = scaled
        if _cache.poly is not None:
            model_input = _cache.poly.transform(scaled)

        # --- Prediction ---
        probabilities = _cache.model.predict_proba(model_input)[0]  # [p_low, p_high]
        risk_probability = float(probabilities[1])
        overall_risk = clip_score(risk_probability * 100, 0.0, 100.0)

        # --- Risk level ---
        risk_level = _map_risk_level(overall_risk)

        # --- Confidence ---
        max_entropy = compute_entropy(np.array([0.5, 0.5]))  # binary max entropy
        current_entropy = compute_entropy(probabilities)
        confidence_score = clip_score(
            (1.0 - current_entropy / max_entropy) * 100 if max_entropy > 0 else 50.0,
            0.0,
            100.0,
        )

        # --- Feature contributions ---
        # Handle both single models (feature_importances_) and ensembles
        if hasattr(_cache.model, "feature_importances_"):
            importances_full = _cache.model.feature_importances_
        elif hasattr(_cache.model, "estimators_"):
            # VotingClassifier — average importances from sub-models
            imp_list = []
            for _, est in _cache.model.estimators:
                if hasattr(est, "feature_importances_"):
                    imp_list.append(est.feature_importances_)
            importances_full = np.mean(imp_list, axis=0) if imp_list else None
        else:
            importances_full = None

        # Aggregate poly importances back to original features
        if importances_full is not None and _cache.poly is not None:
            # Map each poly feature back to its constituent original features
            n_orig = NUM_FEATURES
            importances = np.zeros(n_orig)
            poly_powers = _cache.poly.powers_
            for poly_idx, power_vec in enumerate(poly_powers):
                # Distribute this poly feature's importance to its base features
                contributing = [i for i in range(n_orig) if power_vec[i] > 0]
                if contributing:
                    share = importances_full[poly_idx] / len(contributing)
                    for i in contributing:
                        importances[i] += share
            # Normalise so they sum to 1
            total = importances.sum()
            if total > 0:
                importances = importances / total
        elif importances_full is not None:
            importances = importances_full
        else:
            importances = np.ones(NUM_FEATURES) / NUM_FEATURES

        feature_contribution = {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

        result: Dict[str, Any] = {
            "overall_risk": round(overall_risk, 2),
            "risk_level": risk_level,
            "confidence_score": round(confidence_score, 2),
            "feature_contribution": feature_contribution,
            "model_version": MODEL_VERSION,
            "drift_warning": drift_warning,
        }
        logger.info("Prediction complete: risk=%.2f%% (%s)", overall_risk, risk_level)
        return result

    except Exception:
        logger.exception("Prediction failed.")
        return _fallback_result("Prediction error")


# ==============================================================================
# SHAP-Ready Placeholder
# ==============================================================================

def explain_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    """Placeholder for SHAP-based model explainability.

    When fully implemented this will return per-feature SHAP values
    explaining the model's decision for a specific input.

    Parameters
    ----------
    features : dict
        Same schema as ``predict_health_risk``.

    Returns
    -------
    dict
        ``shap_values``: dict of feature → SHAP value.
        ``base_value``: expected model output.
        ``explanation_ready``: bool.
    """
    logger.info("SHAP explainability called — returning placeholder.")

    # Future: import shap, build TreeExplainer, compute SHAP values
    # explainer = shap.TreeExplainer(_cache.model)
    # shap_values = explainer.shap_values(scaled_features)

    return {
        "shap_values": {name: 0.0 for name in FEATURE_NAMES},
        "base_value": 0.5,
        "explanation_ready": False,
        "note": "SHAP integration module ready — awaiting shap package installation.",
    }


# ==============================================================================
# Warm-up
# ==============================================================================

def warm_up() -> bool:
    """Pre-load model and run a dummy prediction to eliminate first-call lag.

    Call this once at application startup.

    Returns
    -------
    bool
        True if warm-up succeeded.
    """
    logger.info("Model warm-up initiated.")
    dummy = {name: 0.5 for name in FEATURE_NAMES}
    result = predict_health_risk(dummy)
    success = result["risk_level"] != "Unknown"
    if success:
        logger.info("Model warm-up complete.")
    else:
        logger.warning("Model warm-up failed — model may not be trained yet.")
    return success


# ==============================================================================
# INTERNAL helpers
# ==============================================================================

def _map_risk_level(risk_score: float) -> str:
    """Map numeric risk to categorical level."""
    if risk_score < LOW_THRESHOLD:
        return "Low"
    elif risk_score < MODERATE_THRESHOLD:
        return "Moderate"
    return "High"


def _fallback_result(reason: str) -> Dict[str, Any]:
    """Return a safe fallback result when prediction cannot be performed."""
    logger.warning("Returning fallback result — reason: %s", reason)
    return {
        "overall_risk": 0.0,
        "risk_level": "Unknown",
        "confidence_score": 0.0,
        "feature_contribution": {name: 0.0 for name in FEATURE_NAMES},
        "model_version": MODEL_VERSION,
        "drift_warning": False,
        "error": reason,
    }
