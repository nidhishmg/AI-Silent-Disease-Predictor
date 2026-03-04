"""
preprocessing.py — Data validation, normalisation, and safety utilities.

Pure-function helpers used across modules.  No ML / CV / audio imports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


# ==============================================================================
# Normalisation
# ==============================================================================

def normalize_to_range(
    value: float,
    min_val: float = 0.0,
    max_val: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """Clamp *value* to [min_val, max_val] then linearly scale to
    [target_min, target_max].

    Parameters
    ----------
    value : float
        Raw value to normalise.
    min_val, max_val : float
        Expected input range.
    target_min, target_max : float
        Desired output range.

    Returns
    -------
    float
    """
    clamped = max(min_val, min(value, max_val))
    if max_val == min_val:
        return target_min
    ratio = (clamped - min_val) / (max_val - min_val)
    return target_min + ratio * (target_max - target_min)


# ==============================================================================
# Validation
# ==============================================================================

def validate_feature_dict(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """Return ``True`` if *data* contains all *required_keys* with numeric
    values.

    Parameters
    ----------
    data : dict
        Feature dictionary to validate.
    required_keys : list[str]
        Keys that must be present with numeric (int / float) values.

    Returns
    -------
    bool
    """
    if not isinstance(data, dict):
        return False
    for key in required_keys:
        if key not in data:
            return False
        val = data[key]
        if not isinstance(val, (int, float)):
            return False
        if math.isnan(val) or math.isinf(val):
            return False
    return True


# ==============================================================================
# Type-safe conversion
# ==============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to ``float``; return *default* on failure.

    Parameters
    ----------
    value : Any
        Value to cast.
    default : float
        Fallback value.

    Returns
    -------
    float
    """
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


# ==============================================================================
# Score clamping
# ==============================================================================

def clip_score(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to [lo, hi].

    Parameters
    ----------
    value : float
        Raw score.
    lo, hi : float
        Bounds.

    Returns
    -------
    float
    """
    return max(lo, min(value, hi))
