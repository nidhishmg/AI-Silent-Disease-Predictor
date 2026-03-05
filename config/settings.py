"""
settings.py — Central configuration for AI Silent Disease Predictor.

All tuneable parameters, paths, thresholds, and constants live here.
Supports environment variable overrides for 12-factor app compliance.
"""

import os

# ==============================================================================
# APPLICATION METADATA
# ==============================================================================
APP_TITLE = "AI Silent Disease Predictor"
APP_VERSION = os.environ.get("APP_VERSION", "1.0.0")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
APP_DESCRIPTION = (
    "Preventive healthcare intelligence system using multimodal AI — "
    "facial biomarkers, vocal stress analysis, and ML risk fusion."
)

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(MODEL_DIR, "health_model.pkl")
)
SCALER_PATH = os.environ.get(
    "SCALER_PATH", os.path.join(MODEL_DIR, "scaler.pkl")
)
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
DATA_DIR = os.path.join(BASE_DIR, "data", "datasets")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Dataset file paths
HEART_CSV = os.path.join(DATA_DIR, "heart.csv")
DIABETES_CSV = os.path.join(DATA_DIR, "diabetes.csv")
FUSED_CSV = os.path.join(PROCESSED_DIR, "fused_dataset.csv")
CLEANED_CSV = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
FEATURES_CSV = os.path.join(PROCESSED_DIR, "features.csv")

# ==============================================================================
# FEATURE SCHEMA (strict ordering — must match train_model.py)
# ==============================================================================
FEATURE_NAMES = [
    "face_fatigue",
    "symmetry_score",
    "blink_instability",
    "brightness_variance",
    "voice_stress",
    "breathing_score",
    "pitch_instability",
    "face_risk_score",
    "voice_risk_score",
]

# Interaction features (computed from base biomarkers)
INTERACTION_FEATURES = [
    "cardio_stress",           # face_fatigue × breathing_score
    "metabolic_score",         # brightness_variance × voice_stress
    "fatigue_stress",          # face_fatigue × voice_stress
    "respiratory_variation",   # breathing_score × pitch_instability
]

# Advanced Interaction features (Phase 12)
ADVANCED_INTERACTION_FEATURES = [
    "stress_fatigue",
    "respiratory_load",
    "eye_fatigue_index",
    "symmetry_fatigue_gap",
    "combined_risk",
    "fatigue_pitch_interaction",
    "breathing_stress_ratio",
]

# Clinical cross-interaction features
CLINICAL_CROSS_FEATURES = [
    "bp_chol_risk",            # raw_bp × raw_cholesterol
    "age_metabolic",           # raw_age × raw_glucose
    "clinical_cardio",         # raw_bp × face_fatigue
]

# Raw rescaled clinical features (preserve original signal for training)
RAW_CLINICAL_FEATURES = [
    "raw_age", "raw_sex", "raw_bp", "raw_cholesterol",
    "raw_glucose", "raw_bmi",
    "raw_smoking", "raw_exercise",
]

# All features used by the trained model (9 base + 4 interactions + 7 advanced + 3 cross + 8 raw = 31)
ALL_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_FEATURES + ADVANCED_INTERACTION_FEATURES + CLINICAL_CROSS_FEATURES + RAW_CLINICAL_FEATURES
NUM_FEATURES = len(FEATURE_NAMES)
NUM_ALL_FEATURES = len(ALL_FEATURE_NAMES)

# ==============================================================================
# RISK THRESHOLDS
# ==============================================================================
LOW_THRESHOLD = float(os.environ.get("LOW_THRESHOLD", 40))
MODERATE_THRESHOLD = float(os.environ.get("MODERATE_THRESHOLD", 70))

# ==============================================================================
# FACE ANALYSIS WEIGHTS
# ==============================================================================
FACE_WEIGHTS = {
    "fatigue": 0.3,
    "asymmetry": 0.3,
    "blink": 0.2,
    "brightness": 0.2,
}

# ==============================================================================
# VOICE ANALYSIS WEIGHTS
# ==============================================================================
VOICE_WEIGHTS = {
    "stress": 0.4,
    "breathing": 0.3,
    "pitch": 0.3,
}

# ==============================================================================
# AUDIO SETTINGS
# ==============================================================================
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 22050))
RECORD_DURATION = int(os.environ.get("RECORD_DURATION", 5))

# ==============================================================================
# FACEMESH LANDMARK INDICES
# ==============================================================================
# Right eye — (outer, top1, top2, inner, bottom2, bottom1)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# Left eye — (outer, top1, top2, inner, bottom2, bottom1)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Midline landmarks (forehead, nose tip, chin)
MIDLINE_LANDMARKS = [10, 1, 152]
# Symmetric landmark pairs (left_idx, right_idx) for symmetry analysis
SYMMETRIC_PAIRS = [
    (33, 263),    # outer eye corners
    (133, 362),   # inner eye corners
    (70, 300),    # eyebrow inner
    (105, 334),   # eyebrow outer
    (54, 284),    # cheek upper
    (67, 297),    # cheek mid
    (109, 338),   # nose side
    (132, 361),   # jaw upper
    (172, 397),   # jaw mid
    (136, 365),   # jaw lower
]

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# ==============================================================================
# UI COLOUR PALETTE (Medical theme)
# ==============================================================================
COLOR_PRIMARY = "#1a73e8"
COLOR_SUCCESS = "#0f9d58"
COLOR_WARNING = "#f4b400"
COLOR_DANGER = "#db4437"
COLOR_BACKGROUND = "#ffffff"
COLOR_TEXT = "#202124"
COLOR_LIGHT_GRAY = "#f8f9fa"

# ==============================================================================
# SECURITY & PRIVACY
# ==============================================================================
PERSIST_DATA = False  # Never persist patient data to disk

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
TRAINING_SAMPLES = int(os.environ.get("TRAINING_SAMPLES", 5000))
SYNTHETIC_AUGMENT = int(os.environ.get("SYNTHETIC_AUGMENT", 0))  # 0 = real data only (SMOTE handles balance)
TRAINING_SEED = int(os.environ.get("TRAINING_SEED", 42))
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", 500))
RF_MAX_DEPTH = int(os.environ.get("RF_MAX_DEPTH", 15))
TEST_SPLIT = float(os.environ.get("TEST_SPLIT", 0.2))

# ==============================================================================
# FEATURE DRIFT DETECTION
# ==============================================================================
DRIFT_Z_THRESHOLD = float(os.environ.get("DRIFT_Z_THRESHOLD", 3.0))

# ==============================================================================
# DIGITAL TWIN SIMULATION MODIFIERS
# ==============================================================================
TWIN_MODIFIERS = {
    "sleep_deficit": 10.0,     # +10% risk if sleep < 6 hours
    "smoking_active": 15.0,    # +15% risk if currently smoking
    "exercise_bonus": -10.0,   # -10% risk if exercise >=3x/week
    "no_exercise_penalty": 5.0, # +5% risk if no exercise
}

# ==============================================================================
# DEMO MODE
# ==============================================================================
DEMO_SEED = int(os.environ.get("DEMO_SEED", 12345))
