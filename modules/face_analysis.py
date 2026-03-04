"""
face_analysis.py — Facial biomarker extraction module.

Uses OpenCV + MediaPipe FaceMesh to compute:
    • Eye Aspect Ratio (EAR)
    • Blink instability
    • Facial symmetry index
    • Skin brightness variance

Returns a structured dictionary.  Falls back to simulated data when
the webcam is unavailable or no face is detected.

Architecture rules
------------------
- Does NOT import Streamlit.
- Does NOT import other modules (voice_analysis, prediction_engine).
- Communicates only via dict return values.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    FACE_WEIGHTS,
    LEFT_EYE_LANDMARKS,
    MIDLINE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    SYMMETRIC_PAIRS,
)
from utils.feature_utils import (
    compute_brightness_variance,
    compute_ear,
    compute_symmetry_index,
    weighted_composite,
)
from utils.logger import get_logger
from utils.preprocessing import clip_score, normalize_to_range

logger = get_logger(__name__)

# Healthy EAR reference (open eye)
_HEALTHY_EAR = 0.27


# ==============================================================================
# PUBLIC API
# ==============================================================================

def analyze_face(
    image: Optional[np.ndarray] = None,
    deterministic_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Extract facial biomarkers from a BGR image.

    Parameters
    ----------
    image : np.ndarray or None
        BGR image captured from the webcam.  If ``None``, simulated
        data is returned.
    deterministic_seed : int or None
        If provided, simulation uses this seed for reproducible output.

    Returns
    -------
    dict
        Keys: ``face_fatigue``, ``symmetry_score``, ``blink_instability``,
        ``brightness_variance``, ``face_risk_score``.
    """
    if image is None or image.size == 0:
        logger.warning("No image provided — returning simulated face data.")
        return _simulate_face_data(deterministic_seed)

    try:
        return _extract_face_features(image)
    except Exception:
        logger.exception("Face analysis failed — falling back to simulation.")
        return _simulate_face_data(deterministic_seed)


# ==============================================================================
# INTERNAL — Real feature extraction
# ==============================================================================

def _extract_face_features(image: np.ndarray) -> Dict[str, float]:
    """Run MediaPipe FaceMesh and compute biomarkers."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        logger.warning("No face detected in frame — falling back to simulation.")
        return _simulate_face_data()

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    # Convert to numpy — (468, 3) normalised coordinates
    coords = np.array(
        [(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks],
        dtype=np.float64,
    )

    # ----- EAR -----
    right_eye_pts = [coords[i] for i in RIGHT_EYE_LANDMARKS]
    left_eye_pts = [coords[i] for i in LEFT_EYE_LANDMARKS]
    ear_right = compute_ear(right_eye_pts)
    ear_left = compute_ear(left_eye_pts)
    avg_ear = (ear_right + ear_left) / 2.0

    # Fatigue: how far EAR deviates below the healthy reference
    fatigue_raw = max(0.0, _HEALTHY_EAR - avg_ear) / _HEALTHY_EAR
    face_fatigue = normalize_to_range(fatigue_raw, 0.0, 1.0)

    # Blink instability: deviation magnitude
    ear_diff = abs(ear_right - ear_left)
    blink_instability = normalize_to_range(ear_diff, 0.0, 0.15)

    # ----- Symmetry -----
    left_sym = np.array([coords[lp] for lp, _ in SYMMETRIC_PAIRS])
    right_sym = np.array([coords[rp] for _, rp in SYMMETRIC_PAIRS])
    midline = np.array([coords[i] for i in MIDLINE_LANDMARKS])
    symmetry_score = compute_symmetry_index(left_sym, right_sym, midline)

    # ----- Brightness variance -----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute face bounding box from landmarks
    xs = coords[:, 0]
    ys = coords[:, 1]
    x_min, x_max = int(max(xs.min(), 0)), int(min(xs.max(), w))
    y_min, y_max = int(max(ys.min(), 0)), int(min(ys.max(), h))
    roi = gray[y_min:y_max, x_min:x_max]
    brightness_variance = compute_brightness_variance(roi)

    # ----- Composite face risk score -----
    # asymmetry_score: invert symmetry (1 = symmetric → low risk)
    asymmetry_score = 1.0 - symmetry_score
    composite_values = {
        "fatigue": face_fatigue,
        "asymmetry": asymmetry_score,
        "blink": blink_instability,
        "brightness": brightness_variance,
    }
    face_risk_score = clip_score(
        weighted_composite(composite_values, FACE_WEIGHTS) * 100, 0.0, 100.0
    )

    result = {
        "face_fatigue": round(face_fatigue, 4),
        "symmetry_score": round(symmetry_score, 4),
        "blink_instability": round(blink_instability, 4),
        "brightness_variance": round(brightness_variance, 4),
        "face_risk_score": round(face_risk_score, 2),
    }
    logger.info("Face analysis complete: %s", result)
    return result


# ==============================================================================
# INTERNAL — Simulated data (fallback / demo)
# ==============================================================================

def _simulate_face_data(
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Generate physiologically plausible simulated face metrics.

    Uses normal distributions centred on healthy ranges.
    """
    rng = np.random.default_rng(seed)

    face_fatigue = float(np.clip(rng.normal(0.30, 0.15), 0.0, 1.0))
    symmetry_score = float(np.clip(rng.normal(0.85, 0.08), 0.0, 1.0))
    blink_instability = float(np.clip(rng.normal(0.20, 0.10), 0.0, 1.0))
    brightness_variance = float(np.clip(rng.normal(0.25, 0.12), 0.0, 1.0))

    asymmetry = 1.0 - symmetry_score
    composite_values = {
        "fatigue": face_fatigue,
        "asymmetry": asymmetry,
        "blink": blink_instability,
        "brightness": brightness_variance,
    }
    face_risk_score = clip_score(
        weighted_composite(composite_values, FACE_WEIGHTS) * 100, 0.0, 100.0
    )

    result = {
        "face_fatigue": round(face_fatigue, 4),
        "symmetry_score": round(symmetry_score, 4),
        "blink_instability": round(blink_instability, 4),
        "brightness_variance": round(brightness_variance, 4),
        "face_risk_score": round(face_risk_score, 2),
    }
    logger.info("Simulated face data generated: %s", result)
    return result
