"""
voice_analysis.py — Vocal biomarker extraction module.

Uses Librosa and NumPy to extract:
    • MFCC mean + variance
    • Pitch mean and standard deviation
    • RMS energy
    • Speech-rate estimate

Returns a structured dictionary.  Falls back to simulated data when
the microphone is unavailable or audio is invalid.

Architecture rules
------------------
- Does NOT import Streamlit.
- Does NOT import other modules (face_analysis, prediction_engine).
- Communicates only via dict return values.
"""

from __future__ import annotations

from typing import Dict, Optional

import librosa
import numpy as np

from config.settings import SAMPLE_RATE, VOICE_WEIGHTS
from utils.feature_utils import weighted_composite
from utils.logger import get_logger
from utils.preprocessing import clip_score, normalize_to_range

logger = get_logger(__name__)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def analyze_voice(
    audio_data: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    deterministic_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Extract vocal biomarkers from raw audio samples.

    Parameters
    ----------
    audio_data : np.ndarray or None
        1-D float32 waveform.  If ``None``, simulated data is returned.
    sample_rate : int
        Sampling rate in Hz (default from settings).
    deterministic_seed : int or None
        If provided, simulation uses this seed for reproducible output.

    Returns
    -------
    dict
        Keys: ``voice_stress``, ``breathing_score``, ``pitch_instability``,
        ``voice_risk_score``.
    """
    if audio_data is None or (isinstance(audio_data, np.ndarray) and audio_data.size < 1024):
        logger.warning("No valid audio provided — returning simulated voice data.")
        return _simulate_voice_data(deterministic_seed)

    try:
        return _extract_voice_features(audio_data, sample_rate)
    except Exception:
        logger.exception("Voice analysis failed — falling back to simulation.")
        return _simulate_voice_data(deterministic_seed)


# ==============================================================================
# INTERNAL — Real feature extraction
# ==============================================================================

def _extract_voice_features(
    audio: np.ndarray, sr: int
) -> Dict[str, float]:
    """Run Librosa feature extraction pipeline."""

    # Ensure mono float32
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    # ---- MFCCs ----
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)        # shape (13,)
    mfcc_var = np.var(mfccs, axis=1)           # shape (13,)
    mfcc_instability = float(np.mean(mfcc_var))

    # ---- Pitch (pYIN) ----
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    # Filter NaN (unvoiced frames)
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    if len(f0_voiced) > 1:
        pitch_mean = float(np.mean(f0_voiced))
        pitch_std = float(np.std(f0_voiced))
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 1e-6 else 0.0
    else:
        pitch_mean = 150.0   # fallback neutral
        pitch_std = 20.0
        pitch_cv = 0.13

    # ---- RMS Energy ----
    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))

    # ---- Speech-rate estimate ----
    # Count voiced segments (transitions from unvoiced → voiced)
    if voiced_flag is not None and len(voiced_flag) > 1:
        transitions = np.diff(voiced_flag.astype(int))
        onset_count = int(np.sum(transitions == 1))
        duration_sec = len(audio) / sr
        speech_rate = onset_count / max(duration_sec, 0.1)
    else:
        speech_rate = 2.0  # neutral

    # ---- Derived biomarkers ----

    # Voice stress: composite of MFCC instability + pitch variability
    mfcc_stress = normalize_to_range(mfcc_instability, 0.0, 50.0)
    pitch_stress = normalize_to_range(pitch_cv, 0.0, 0.5)
    voice_stress = float(np.clip(0.5 * mfcc_stress + 0.5 * pitch_stress, 0.0, 1.0))

    # Breathing score: low RMS energy + high RMS variance → irregular breathing
    low_energy = normalize_to_range(rms_mean, 0.0, 0.1, target_min=1.0, target_max=0.0)
    energy_irregularity = normalize_to_range(rms_var, 0.0, 0.01)
    breathing_score = float(np.clip(0.6 * low_energy + 0.4 * energy_irregularity, 0.0, 1.0))

    # Pitch instability: coefficient of variation
    pitch_instability = normalize_to_range(pitch_cv, 0.0, 0.5)

    # ---- Voice risk score (composite) ----
    composite_values = {
        "stress": voice_stress,
        "breathing": breathing_score,
        "pitch": pitch_instability,
    }
    voice_risk_score = clip_score(
        weighted_composite(composite_values, VOICE_WEIGHTS) * 100, 0.0, 100.0
    )

    result = {
        "voice_stress": round(voice_stress, 4),
        "breathing_score": round(breathing_score, 4),
        "pitch_instability": round(pitch_instability, 4),
        "voice_risk_score": round(voice_risk_score, 2),
    }
    logger.info("Voice analysis complete: %s", result)
    return result


# ==============================================================================
# INTERNAL — Simulated data (fallback / demo)
# ==============================================================================

def _simulate_voice_data(
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Generate physiologically plausible simulated voice metrics."""
    rng = np.random.default_rng(seed)

    voice_stress = float(np.clip(rng.normal(0.35, 0.15), 0.0, 1.0))
    breathing_score = float(np.clip(rng.normal(0.30, 0.12), 0.0, 1.0))
    pitch_instability = float(np.clip(rng.normal(0.25, 0.10), 0.0, 1.0))

    composite_values = {
        "stress": voice_stress,
        "breathing": breathing_score,
        "pitch": pitch_instability,
    }
    voice_risk_score = clip_score(
        weighted_composite(composite_values, VOICE_WEIGHTS) * 100, 0.0, 100.0
    )

    result = {
        "voice_stress": round(voice_stress, 4),
        "breathing_score": round(breathing_score, 4),
        "pitch_instability": round(pitch_instability, 4),
        "voice_risk_score": round(voice_risk_score, 2),
    }
    logger.info("Simulated voice data generated: %s", result)
    return result
