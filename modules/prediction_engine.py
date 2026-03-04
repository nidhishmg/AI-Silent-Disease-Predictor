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
    ALL_FEATURE_NAMES,
    CLINICAL_CROSS_FEATURES,
    DRIFT_Z_THRESHOLD,
    FEATURE_NAMES,
    INTERACTION_FEATURES,
    LOW_THRESHOLD,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_VERSION,
    MODERATE_THRESHOLD,
    NUM_ALL_FEATURES,
    NUM_FEATURES,
    RAW_CLINICAL_FEATURES,
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
        self.loaded: bool = False
        self.uses_interactions: bool = False  # True if model expects 13 features

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
            # Detect whether model expects 13 features (with interactions)
            n_expected = _detect_n_features(self.model)
            self.uses_interactions = (n_expected == NUM_ALL_FEATURES)
            logger.info(
                "Model expects %d features (interactions=%s)",
                n_expected, self.uses_interactions,
            )
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
# Interaction feature computation
# ==============================================================================

def _compute_interactions(features: Dict[str, float]) -> Dict[str, float]:
    """Compute 4 interaction features from 9 base biomarkers.

    Must match the formulas used in ``training/feature_engineering.py``.
    """
    return {
        "cardio_stress": features["face_fatigue"] * features["breathing_score"],
        "metabolic_score": features["brightness_variance"] * features["voice_stress"],
        "fatigue_stress": features["face_fatigue"] * features["voice_stress"],
        "respiratory_variation": features["breathing_score"] * features["pitch_instability"],
    }


def _estimate_raw_clinical(features: Dict[str, float]) -> Dict[str, float]:
    """Estimate raw clinical features from biomarker values for inference.

    During training the model sees rescaled raw clinical features alongside
    biomarkers.  At inference time (face/voice analysis) these raw values
    are not available, so we estimate them from the biomarker values using
    the dominant contribution weights defined in feature_engineering.py.

    The estimates are approximate but provide the model with a reasonable
    signal in the same feature space it was trained on.
    """
    ff = features.get("face_fatigue", 0.5)
    ss = features.get("symmetry_score", 0.5)
    bi = features.get("blink_instability", 0.5)
    bv = features.get("brightness_variance", 0.5)
    vs = features.get("voice_stress", 0.5)
    bs = features.get("breathing_score", 0.5)
    pi = features.get("pitch_instability", 0.5)
    frs = features.get("face_risk_score", 50.0) / 100.0
    vrs = features.get("voice_risk_score", 50.0) / 100.0

    # Approximate inverse: each raw feature is estimated from the
    # biomarker(s) where it has the strongest weight contribution.
    raw_bp       = 0.40 * ff + 0.30 * (1 - ss) + 0.15 * bs + 0.15 * bi
    raw_age      = 0.30 * bs + 0.25 * ff + 0.25 * pi + 0.20 * (1 - ss)
    raw_chol     = 0.40 * (1 - ss) + 0.30 * pi + 0.30 * bv
    raw_gluc     = 0.30 * bi + 0.25 * vs + 0.25 * pi + 0.20 * bv
    raw_bmi      = 0.35 * bv + 0.30 * vs + 0.20 * bs + 0.15 * bi
    raw_smoke    = frs * 0.5 + vrs * 0.5      # rough proxy
    raw_exercise = 1.0 - (0.5 * bi + 0.3 * pi + 0.2 * (1 - ss))
    raw_sex      = 0.5  # no reliable inference from biomarkers

    return {
        "raw_age": float(np.clip(raw_age, 0, 1)),
        "raw_sex": float(np.clip(raw_sex, 0, 1)),
        "raw_bp": float(np.clip(raw_bp, 0, 1)),
        "raw_cholesterol": float(np.clip(raw_chol, 0, 1)),
        "raw_glucose": float(np.clip(raw_gluc, 0, 1)),
        "raw_bmi": float(np.clip(raw_bmi, 0, 1)),
        "raw_smoking": float(np.clip(raw_smoke, 0, 1)),
        "raw_exercise": float(np.clip(raw_exercise, 0, 1)),
    }


def _compute_clinical_cross(features: Dict[str, float],
                            raw: Dict[str, float]) -> Dict[str, float]:
    """Compute 3 clinical cross-interaction features.

    Must match the formulas used in ``training/feature_engineering.py``.
    """
    return {
        "bp_chol_risk": raw["raw_bp"] * raw["raw_cholesterol"],
        "age_metabolic": raw["raw_age"] * raw["raw_glucose"],
        "clinical_cardio": raw["raw_bp"] * features["face_fatigue"],
    }


def _detect_n_features(model: Any) -> int:
    """Detect the number of features the model was trained on."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "feature_importances_"):
        return len(model.feature_importances_)
    if hasattr(model, "estimators_"):
        for _, est in model.estimators:
            if hasattr(est, "n_features_in_"):
                return int(est.n_features_in_)
    return NUM_FEATURES  # fallback


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
        # Build ordered feature vector (9 base biomarkers)
        feature_vector = np.array(
            [features[name] for name in FEATURE_NAMES], dtype=np.float64
        ).reshape(1, -1)

        # --- Compute interaction features if model requires them ---
        if _cache.uses_interactions:
            interactions = _compute_interactions(features)
            interaction_vec = np.array(
                [interactions[name] for name in INTERACTION_FEATURES],
                dtype=np.float64,
            ).reshape(1, -1)

            # Estimate raw clinical features from biomarkers
            raw_clinical = _estimate_raw_clinical(features)

            # Compute clinical cross-interaction features
            clinical_cross = _compute_clinical_cross(features, raw_clinical)
            cross_vec = np.array(
                [clinical_cross[name] for name in CLINICAL_CROSS_FEATURES],
                dtype=np.float64,
            ).reshape(1, -1)

            raw_vec = np.array(
                [raw_clinical[name] for name in RAW_CLINICAL_FEATURES],
                dtype=np.float64,
            ).reshape(1, -1)

            full_vector = np.hstack([feature_vector, interaction_vec, cross_vec, raw_vec])
            all_names = ALL_FEATURE_NAMES
            n_feats = NUM_ALL_FEATURES
        else:
            full_vector = feature_vector
            all_names = FEATURE_NAMES
            n_feats = NUM_FEATURES

        # --- Feature drift detection (on base features only) ---
        scaled_base = _cache.scaler.transform(feature_vector)
        drift_warning = bool(np.any(np.abs(scaled_base) > DRIFT_Z_THRESHOLD))
        if drift_warning:
            flagged = [
                FEATURE_NAMES[i]
                for i in range(NUM_FEATURES)
                if abs(scaled_base[0, i]) > DRIFT_Z_THRESHOLD
            ]
            logger.warning(
                "Feature drift detected — out-of-distribution features: %s",
                flagged,
            )

        # --- Prediction ---
        probabilities = _cache.model.predict_proba(full_vector)[0]  # [p_low, p_high]
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
            importances = _cache.model.feature_importances_
        elif hasattr(_cache.model, "estimators_"):
            # VotingClassifier — average importances from sub-models
            imp_list = []
            for _, est in _cache.model.estimators:
                if hasattr(est, "feature_importances_"):
                    imp_list.append(est.feature_importances_)
            importances = np.mean(imp_list, axis=0) if imp_list else None
        else:
            importances = None

        # Map back to base features (aggregate interaction importances)
        if importances is not None and len(importances) > NUM_FEATURES:
            base_imp = importances[:NUM_FEATURES].copy()
            interaction_imp = importances[NUM_FEATURES:]
            total_interaction = float(interaction_imp.sum())
            if total_interaction > 0:
                base_imp += total_interaction / NUM_FEATURES
            total = base_imp.sum()
            if total > 0:
                base_imp = base_imp / total
            importances = base_imp
        elif importances is not None:
            total = importances.sum()
            if total > 0:
                importances = importances / total
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
    """SHAP-based model explainability.

    Uses TreeExplainer if ``shap`` is installed, otherwise returns zeros.

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
    if not _cache.load():
        return {
            "shap_values": {name: 0.0 for name in FEATURE_NAMES},
            "base_value": 0.5,
            "explanation_ready": False,
            "note": "Model not loaded.",
        }

    try:
        import shap

        feature_vector = np.array(
            [features[name] for name in FEATURE_NAMES], dtype=np.float64,
        ).reshape(1, -1)

        if _cache.uses_interactions:
            interactions = _compute_interactions(features)
            int_vec = np.array(
                [interactions[n] for n in INTERACTION_FEATURES], dtype=np.float64,
            ).reshape(1, -1)
            full_vec = np.hstack([feature_vector, int_vec])
        else:
            full_vec = feature_vector

        # For VotingClassifier, use first estimator
        from sklearn.ensemble import VotingClassifier
        model_for_shap = _cache.model
        if isinstance(_cache.model, VotingClassifier):
            model_for_shap = _cache.model.estimators_[0]

        explainer = shap.TreeExplainer(model_for_shap)
        sv = explainer.shap_values(full_vec)
        if isinstance(sv, list):
            sv = sv[1]  # positive class

        shap_dict = {}
        names = ALL_FEATURE_NAMES if _cache.uses_interactions else FEATURE_NAMES
        for i, name in enumerate(names):
            shap_dict[name] = float(sv[0][i])

        return {
            "shap_values": shap_dict,
            "base_value": float(explainer.expected_value[1])
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value),
            "explanation_ready": True,
        }
    except ImportError:
        logger.info("shap package not installed — returning placeholder.")
        return {
            "shap_values": {name: 0.0 for name in FEATURE_NAMES},
            "base_value": 0.5,
            "explanation_ready": False,
            "note": "Install shap package: pip install shap",
        }
    except Exception:
        logger.exception("SHAP explanation failed.")
        return {
            "shap_values": {name: 0.0 for name in FEATURE_NAMES},
            "base_value": 0.5,
            "explanation_ready": False,
            "note": "SHAP computation failed — see logs.",
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
