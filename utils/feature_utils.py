"""
feature_utils.py — Reusable, pure-math feature computation helpers.

No OpenCV, Librosa, MediaPipe, or Streamlit imports.
Only numpy for vectorised math.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ==============================================================================
# Eye Aspect Ratio (EAR)
# ==============================================================================

def compute_ear(eye_landmarks: List[Tuple[float, float, float]]) -> float:
    """Compute the Eye Aspect Ratio from six 3-D landmark coordinates.

    Landmark ordering (per MediaPipe convention)::

        p1 = outer corner    p4 = inner corner
        p2, p3 = upper lid   p5, p6 = lower lid

    Formula::

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    eye_landmarks : list of 6 (x, y, z) tuples
        Landmarks in the order [p1, p2, p3, p4, p5, p6].

    Returns
    -------
    float
        EAR value; healthy open-eye is ~0.25–0.30.
    """
    if len(eye_landmarks) < 6:
        return 0.0

    pts = np.array(eye_landmarks, dtype=np.float64)
    # Vertical distances
    vert_a = np.linalg.norm(pts[1] - pts[5])
    vert_b = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    horiz = np.linalg.norm(pts[0] - pts[3])

    if horiz < 1e-6:
        return 0.0

    return float((vert_a + vert_b) / (2.0 * horiz))


# ==============================================================================
# Facial Symmetry Index
# ==============================================================================

def compute_symmetry_index(
    left_points: np.ndarray,
    right_points: np.ndarray,
    midline_points: np.ndarray,
) -> float:
    """Compute a 0–1 symmetry score from symmetric landmark pairs.

    A score of **1.0** means perfectly symmetric; lower values indicate
    greater asymmetry.

    Algorithm
    ---------
    1.  Define the midline as the line through the first and last midline
        landmark.
    2.  For each pair, compute the perpendicular distance of the left and
        right points from the midline.
    3.  Compute the mean absolute difference of the paired distances,
        normalised by the mean face width.

    Parameters
    ----------
    left_points : np.ndarray, shape (N, 3)
    right_points : np.ndarray, shape (N, 3)
    midline_points : np.ndarray, shape (M, 3)  (M >= 2)

    Returns
    -------
    float  (0–1)
    """
    if (
        len(left_points) == 0
        or len(right_points) == 0
        or len(midline_points) < 2
    ):
        return 1.0  # default to perfect symmetry (no data)

    # Use 2-D projection (x, y)
    mid_top = midline_points[0, :2]
    mid_bot = midline_points[-1, :2]
    midline_vec = mid_bot - mid_top
    midline_len = np.linalg.norm(midline_vec)
    if midline_len < 1e-6:
        return 1.0

    midline_unit = midline_vec / midline_len

    def _signed_dist(pt: np.ndarray) -> float:
        v = pt[:2] - mid_top
        # Perpendicular signed distance
        return float(np.cross(midline_unit, v))

    diffs: list[float] = []
    for lp, rp in zip(left_points, right_points):
        d_left = abs(_signed_dist(lp))
        d_right = abs(_signed_dist(rp))
        mean_d = (d_left + d_right) / 2.0
        if mean_d < 1e-6:
            continue
        diffs.append(abs(d_left - d_right) / mean_d)

    if not diffs:
        return 1.0

    asymmetry = float(np.mean(diffs))
    # Convert asymmetry (0 = symmetric) to score (1 = symmetric)
    return float(np.clip(1.0 - asymmetry, 0.0, 1.0))


# ==============================================================================
# Skin / ROI Brightness Variance
# ==============================================================================

def compute_brightness_variance(roi_pixels: np.ndarray) -> float:
    """Compute normalised variance of pixel brightness in a face ROI.

    Parameters
    ----------
    roi_pixels : np.ndarray
        Grayscale pixel values (uint8 or float).

    Returns
    -------
    float
        Variance normalised to 0–1 (divided by 255² for uint8 input).
    """
    if roi_pixels is None or roi_pixels.size == 0:
        return 0.0

    pixels = roi_pixels.astype(np.float64).ravel()
    variance = float(np.var(pixels))

    # Normalise: for uint8 images max variance is 255^2 / 4 ≈ 16256
    max_possible = 16256.0
    return float(np.clip(variance / max_possible, 0.0, 1.0))


# ==============================================================================
# Generic Weighted Composite
# ==============================================================================

def weighted_composite(values: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute a weighted sum using matching keys.

    Parameters
    ----------
    values : dict[str, float]
    weights : dict[str, float]

    Returns
    -------
    float
    """
    total = 0.0
    weight_sum = 0.0
    for key, w in weights.items():
        if key in values:
            total += w * values[key]
            weight_sum += w
    if weight_sum < 1e-9:
        return 0.0
    return total / weight_sum * weight_sum  # keep unnormalised — caller scales


# ==============================================================================
# Shannon Entropy (for confidence scoring)
# ==============================================================================

def compute_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Parameters
    ----------
    probabilities : np.ndarray
        1-D array of probabilities (must sum to ~1).

    Returns
    -------
    float
        Entropy in nats.  Lower = more confident.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))
