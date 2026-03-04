<p align="center">
  <img src="https://img.icons8.com/fluency/96/heart-health.png" alt="Logo" width="80"/>
</p>

<h1 align="center">AI Silent Disease Predictor</h1>

<p align="center">
  <strong>Preventive Healthcare Intelligence using Multimodal AI</strong><br>
  Face Biomarkers &bull; Vocal Stress Analysis &bull; ML Risk Fusion Engine
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh-4285F4?logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Model%20Version-1.0.0-purple" alt="Model Version"/>
</p>

---

## Overview

**AI Silent Disease Predictor** is a production-grade preventive healthcare system that uses **multimodal AI** to assess health risk from non-invasive biomarkers. It captures **facial biomarkers** (via OpenCV + MediaPipe FaceMesh), extracts **vocal stress markers** (via Librosa), fuses both through a trained **RandomForest ML model**, and outputs a probabilistic health risk score displayed on a professional medical dashboard.

### Key Capabilities

| Modality | Technology | Biomarkers Extracted |
|----------|-----------|---------------------|
| **Face** | OpenCV + MediaPipe FaceMesh | Eye Aspect Ratio, Blink Instability, Facial Symmetry, Skin Brightness Variance |
| **Voice** | Librosa + NumPy | MFCC, Pitch Instability, RMS Energy, Breathing Score |
| **ML Engine** | scikit-learn RandomForest | Probabilistic Risk Score, Confidence, Feature Contributions |

---

## Architecture

```
AI-Silent-Disease-Predictor/
│
├── app.py                        # Streamlit UI controller (ONLY UI file)
├── train_model.py                # Model training pipeline
├── requirements.txt              # Python dependencies
│
├── config/
│   └── settings.py               # Central config (paths, thresholds, weights)
│
├── models/
│   ├── health_model.pkl          # Trained RandomForest model
│   └── scaler.pkl                # Fitted StandardScaler
│
├── modules/
│   ├── face_analysis.py          # Facial biomarker extraction
│   ├── voice_analysis.py         # Vocal biomarker extraction
│   └── prediction_engine.py      # ML inference + risk scoring
│
├── utils/
│   ├── preprocessing.py          # Normalization & validation
│   ├── feature_utils.py          # Pure-math feature helpers (EAR, symmetry)
│   └── logger.py                 # Centralized rotating file logger
│
├── data/datasets/                # Training data directory
├── assets/                       # Static assets
└── logs/                         # Application logs
```

### Design Principles

- **Clean Separation of Concerns** — UI only in `app.py`; ML modules never import Streamlit
- **No Circular Imports** — modules communicate only via structured dictionaries
- **API-Ready** — `predict_health_risk(dict) → dict` is a pure function, directly wrappable by FastAPI
- **Resilient** — camera/mic/model failures auto-fallback to simulation; app never crashes
- **Secure** — no images, audio, or personal data persisted; in-memory processing only

---

## Quick Start

### Prerequisites

- **Python 3.10–3.12** (3.11 recommended; MediaPipe does not support 3.13)
- **Git**
- **Webcam** (optional — simulated data available)
- **Microphone** (optional — file upload or simulated data available)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/AI-Silent-Disease-Predictor.git
cd AI-Silent-Disease-Predictor

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the ML model (run once)
python train_model.py

# 5. Launch the application
python -m streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## How It Works

```
┌──────────────┐     ┌──────────────┐
│  Face Scan   │     │ Voice Record │
│  (Webcam /   │     │ (Upload /    │
│   Simulate)  │     │  Simulate)   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│face_analysis │     │voice_analysis│
│   .py        │     │   .py        │
│              │     │              │
│ EAR, Symm,   │     │ MFCC, Pitch, │
│ Blink, Bright│     │ RMS, Breath  │
└──────┬───────┘     └──────┬───────┘
       │    Dict            │   Dict
       └────────┬───────────┘
                ▼
       ┌────────────────┐
       │prediction_engine│
       │     .py         │
       │                 │
       │ Scale → Predict │
       │ → Risk Level   │
       │ → Confidence   │
       └────────┬───────┘
                │  Dict
                ▼
       ┌────────────────┐
       │    app.py       │
       │  (Dashboard)    │
       │                 │
       │ Gauge, Charts,  │
       │ PDF Report,     │
       │ Digital Twin    │
       └────────────────┘
```

### Pipeline

1. **Face Scan** → MediaPipe FaceMesh extracts 468 facial landmarks → computes EAR, symmetry, blink instability, brightness variance → composite `face_risk_score`
2. **Voice Scan** → Librosa extracts MFCCs, pitch (pYIN), RMS energy → computes voice stress, breathing score, pitch instability → composite `voice_risk_score`
3. **ML Fusion** → 9-feature vector scaled via StandardScaler → RandomForest `predict_proba` → probabilistic risk percentage
4. **Dashboard** → Plotly gauge, feature contribution chart, AI recommendations, downloadable PDF report
5. **Digital Twin** → Lifestyle inputs (sleep, exercise, smoking) simulate 6-month projected risk

---

## Features

### Core

- **Facial Biomarker Extraction** — Eye Aspect Ratio using 12 MediaPipe landmarks, facial symmetry via 10 symmetric pairs, skin brightness variance
- **Vocal Biomarker Extraction** — 13 MFCCs, pYIN pitch tracking, RMS energy patterns, speech-rate estimation
- **ML Risk Engine** — RandomForest (200 trees, max_depth=10) trained on 3000 synthetic medically-informed samples
- **Risk Scoring** — Probabilistic 0–100% scale with Low/Moderate/High classification

### Elite Features

| Feature | Description |
|---------|-------------|
| **Model Versioning** | `MODEL_VERSION` tracked in config, displayed in sidebar, dashboard, and PDF reports |
| **Feature Drift Detection** | Z-score check before inference — warns when inputs deviate from training distribution |
| **SHAP-Ready Explainability** | `explain_prediction()` placeholder ready for SHAP integration |
| **Model Warm-Up** | Pre-loads model on app startup via `@st.cache_resource` to eliminate first-call lag |
| **Deterministic Demo Mode** | Sidebar toggle uses fixed seed for reproducible outputs during live presentations |

### Dashboard

- Circular risk gauge (Plotly) with color-coded zones
- Color-coded risk banner (Low/Moderate/High)
- Confidence percentage metric
- Feature contribution horizontal bar chart
- AI-generated recommendations based on risk level and dominant features
- Medical disclaimer
- Downloadable PDF health report

### Digital Twin Simulation

- Adjustable lifestyle inputs: sleep hours, exercise frequency, smoking status
- Projects 6-month risk based on configurable modifiers
- Side-by-side current vs. projected risk gauges with delta comparison

---

## Resilience

| Failure Scenario | Behavior |
|-----------------|----------|
| Camera unavailable | "Use Simulated Data" button → physiologically plausible synthetic face metrics |
| Microphone unavailable | File upload fallback + "Use Simulated Data" button |
| No face detected | Auto-fallback to simulated data with warning |
| Audio processing fails | Auto-fallback to simulated voice data |
| Model files missing | UI warning: "Run train_model.py first" — app does not crash |
| Feature drift | Warning banner in dashboard; results flagged as less reliable |

---

## API-Ready Design

The prediction engine is a pure function — no Streamlit dependency:

```python
from modules.prediction_engine import predict_health_risk

result = predict_health_risk({
    "face_fatigue": 0.35,
    "symmetry_score": 0.88,
    "blink_instability": 0.15,
    "brightness_variance": 0.22,
    "voice_stress": 0.40,
    "breathing_score": 0.28,
    "pitch_instability": 0.20,
    "face_risk_score": 25.5,
    "voice_risk_score": 30.2,
})

# Returns:
# {
#     "overall_risk": 42.5,
#     "risk_level": "Moderate",
#     "confidence_score": 78.3,
#     "feature_contribution": { ... },
#     "model_version": "1.0.0",
#     "drift_warning": False
# }
```

FastAPI integration (future):

```python
from fastapi import FastAPI
from modules.prediction_engine import predict_health_risk

app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    return predict_health_risk(payload)
```

---

## Training

The model is trained on **synthetic medically-informed data** (3000 samples) using:

- **Beta distributions** for realistic biomarker ranges
- **Sigmoid-based labeling** with medically-informed correlation rules
- **Stratified train/test split** (80/20)
- **Balanced class weights** for fair risk classification

```bash
python train_model.py
```

Output:
```
Classification Report:
              precision    recall  f1-score   support
    Low Risk       0.69      0.75      0.72       328
   High Risk       0.66      0.59      0.62       272
    accuracy                           0.68       600

Feature Importances:
  voice_risk_score       0.1571  ███████
  face_risk_score        0.1421  ███████
  face_fatigue           0.1222  ██████
  breathing_score        0.1199  █████
  voice_stress           0.1034  █████
  blink_instability      0.0941  ████
  symmetry_score         0.0890  ████
  pitch_instability      0.0875  ████
  brightness_variance    0.0847  ████
```

### Dataset References (for future real-data integration)

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Configuration

All settings are centralized in `config/settings.py` with environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VERSION` | `1.0.0` | Model version tag |
| `LOW_THRESHOLD` | `40` | Risk score boundary: Low → Moderate |
| `MODERATE_THRESHOLD` | `70` | Risk score boundary: Moderate → High |
| `TRAINING_SAMPLES` | `3000` | Number of synthetic training samples |
| `RF_N_ESTIMATORS` | `200` | RandomForest number of trees |
| `DRIFT_Z_THRESHOLD` | `3.0` | Z-score threshold for drift detection |
| `DEMO_SEED` | `12345` | Fixed seed for deterministic demo mode |

Override via environment:
```bash
set MODEL_VERSION=2.0.0
set TRAINING_SAMPLES=5000
python train_model.py
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit 1.30+ |
| Face Analysis | OpenCV, MediaPipe FaceMesh |
| Voice Analysis | Librosa, NumPy |
| ML Model | scikit-learn RandomForestClassifier |
| Data Processing | Pandas, NumPy, StandardScaler |
| Visualization | Plotly |
| PDF Reports | fpdf2 |
| Logging | Python logging (rotating file handler) |

---

## Security & Privacy

- **No image storage** — webcam frames processed in-memory, immediately discarded
- **No audio storage** — audio buffers deleted after feature extraction
- **No persistent data** — no database, no user accounts, no PII
- **In-memory PDF** — generated as bytes, streamed to browser download

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

> This AI system is for **research and educational purposes only**. It does NOT constitute medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition.

---

<p align="center">
  Built with precision engineering for preventive healthcare intelligence.
</p>
