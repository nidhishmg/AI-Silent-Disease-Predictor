"""
app.py — Streamlit UI controller for AI Silent Disease Predictor.

This is the ONLY file that imports Streamlit.
All ML / CV / audio logic lives in the modules/ package.


Run::

    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

from config.settings import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DEMO_SEED,
    FEATURE_NAMES,
    LOW_THRESHOLD,
    MODEL_VERSION,
    MODERATE_THRESHOLD,
    SAMPLE_RATE,
    TWIN_MODIFIERS,
)
from modules.face_analysis import analyze_face
from modules.prediction_engine import predict_health_risk, warm_up
from modules.voice_analysis import analyze_voice
from utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# PAGE CONFIG & CUSTOM CSS
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    /* Medical colour palette */
    :root {
        --primary: #1a73e8;
        --success: #0f9d58;
        --warning: #f4b400;
        --danger: #db4437;
    }
    .stApp { background-color: #ffffff; }
    h1, h2, h3 { color: #202124; font-family: 'Segoe UI', sans-serif; }
    .risk-low { background-color: #e6f4ea; border-left: 5px solid #0f9d58;
                padding: 1rem; border-radius: 4px; }
    .risk-moderate { background-color: #fef7e0; border-left: 5px solid #f4b400;
                     padding: 1rem; border-radius: 4px; }
    .risk-high { background-color: #fce8e6; border-left: 5px solid #db4437;
                 padding: 1rem; border-radius: 4px; }
    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem;
                   text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .disclaimer { background: #f1f3f4; padding: 1rem; border-radius: 8px;
                   font-size: 0.85rem; color: #5f6368; margin-top: 2rem; }
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALISATION
# ==============================================================================

def _init_state() -> None:
    """Initialise session state variables."""
    defaults = {
        "page": "home",
        "face_data": None,
        "voice_data": None,
        "prediction": None,
        "demo_mode": False,
        "model_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ==============================================================================
# MODEL WARM-UP (runs once on first load)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def _warm_up_model() -> bool:
    """Pre-load model to eliminate first-call latency."""
    return warm_up()


st.session_state.model_ready = _warm_up_model()


# ==============================================================================
# SIDEBAR
# ==============================================================================

def _render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/heart-health.png",
            width=64,
        )
        st.title("Navigation")
        pages = {
            "🏠 Home": "home",
            "ℹ️ About": "about",
            "📷 Face Scan": "face",
            "🎙️ Voice Scan": "voice",
            "🤖 AI Analysis": "analyze",
            "📊 Dashboard": "dashboard",
            "🔮 Digital Twin": "twin",
        }
        for label, key in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.page = key

        st.divider()

        # Deterministic demo mode toggle
        st.session_state.demo_mode = st.toggle(
            "Deterministic Demo Mode",
            value=st.session_state.demo_mode,
            help="Fixed seed for reproducible results in presentations.",
        )

        st.divider()
        st.caption(f"v{APP_VERSION} · Model v{MODEL_VERSION}")
        if not st.session_state.model_ready:
            st.warning("⚠️ Model not trained. Run `train_model.py` first.")
        st.caption("© 2026 AI Silent Disease Predictor")


_render_sidebar()


# ==============================================================================
# PAGE: HOME
# ==============================================================================

def _page_home() -> None:
    st.markdown(
        f"<h1 style='text-align:center; color:{COLOR_PRIMARY};'>"
        f"🏥 {APP_TITLE}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center; font-size:1.2rem; color:#5f6368;'>"
        f"{APP_DESCRIPTION}</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "<div class='metric-card'><h3>📷</h3>"
            "<p><strong>Face Analysis</strong></p>"
            "<p>EAR, symmetry, skin brightness, fatigue detection</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            "<div class='metric-card'><h3>🎙️</h3>"
            "<p><strong>Voice Analysis</strong></p>"
            "<p>MFCC, pitch, RMS energy, respiratory markers</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            "<div class='metric-card'><h3>🤖</h3>"
            "<p><strong>ML Risk Engine</strong></p>"
            "<p>RandomForest prediction with confidence scoring</p></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(
            "🚀 Start Health Scan",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "face"
            st.rerun()


# ==============================================================================
# PAGE: ABOUT
# ==============================================================================

def _page_about() -> None:
    st.header("About This System")
    st.markdown(
        """
        **AI Silent Disease Predictor** is a preventive healthcare intelligence
        system that uses multimodal AI to assess health risk from non-invasive
        biomarkers.

        ### Architecture
        | Layer | Technology |
        |-------|-----------|
        | Face Analysis | OpenCV + MediaPipe FaceMesh |
        | Voice Analysis | Librosa + NumPy |
        | ML Engine | scikit-learn RandomForestClassifier |
        | UI | Streamlit |
        | Report | fpdf2 |

        ### How It Works
        1. **Capture** — facial image & voice recording
        2. **Extract** — biomarker features from both modalities
        3. **Fuse** — combined feature vector fed to trained ML model
        4. **Predict** — probabilistic health risk with confidence score
        5. **Report** — downloadable PDF with recommendations

        ### Privacy
        - No images or audio are stored
        - All processing is in-memory only
        - No personal data persisted
        """
    )
    st.markdown(
        "<div class='disclaimer'>"
        "⚕️ <strong>Medical Disclaimer:</strong> This is an AI research tool and "
        "does NOT provide medical diagnoses.  Always consult a qualified healthcare "
        "professional for medical advice.</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE: FACE SCAN
# ==============================================================================

def _page_face() -> None:
    st.header("📷 Face Scan")
    st.info(
        "Capture a facial image for biomarker extraction. "
        "If your camera is unavailable, use simulated data."
    )

    col_cam, col_sim = st.columns([3, 1])

    with col_cam:
        camera_image = st.camera_input("Take a photo", key="face_camera")

    with col_sim:
        st.write("")
        st.write("")
        use_sim = st.button(
            "🔄 Use Simulated Data",
            use_container_width=True,
            help="Generate demo face biomarkers without a camera.",
        )

    # Process camera image
    if camera_image is not None:
        with st.spinner("Analyzing facial biomarkers..."):
            img_bytes = camera_image.getvalue()
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            seed = DEMO_SEED if st.session_state.demo_mode else None
            face_data = analyze_face(frame, deterministic_seed=seed)
            st.session_state.face_data = face_data
    elif use_sim:
        with st.spinner("Generating simulated face data..."):
            seed = DEMO_SEED if st.session_state.demo_mode else None
            face_data = analyze_face(None, deterministic_seed=seed)
            st.session_state.face_data = face_data

    # Display results
    if st.session_state.face_data:
        fd = st.session_state.face_data
        st.success("Face analysis complete!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fatigue", f"{fd['face_fatigue']:.3f}")
        c2.metric("Symmetry", f"{fd['symmetry_score']:.3f}")
        c3.metric("Blink Instability", f"{fd['blink_instability']:.3f}")
        c4.metric("Brightness Var.", f"{fd['brightness_variance']:.3f}")
        st.metric("Face Risk Score", f"{fd['face_risk_score']:.1f} / 100")

        if st.button("➡️ Proceed to Voice Scan", type="primary"):
            st.session_state.page = "voice"
            st.rerun()


# ==============================================================================
# PAGE: VOICE SCAN
# ==============================================================================

def _page_voice() -> None:
    st.header("🎙️ Voice Analysis")
    st.info(
        "Upload a voice recording (.wav) for vocal biomarker extraction, "
        "or use simulated data."
    )

    col_upload, col_sim = st.columns([3, 1])

    with col_upload:
        uploaded_audio = st.file_uploader(
            "Upload a .wav file (5–10 seconds of speech)",
            type=["wav"],
            key="voice_upload",
        )

    with col_sim:
        st.write("")
        st.write("")
        use_sim = st.button(
            "🔄 Use Simulated Data",
            use_container_width=True,
            help="Generate demo voice biomarkers without audio.",
        )

    # Process uploaded audio
    if uploaded_audio is not None:
        with st.spinner("Analyzing vocal biomarkers..."):
            try:
                import librosa
                audio_bytes = uploaded_audio.getvalue()
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)
                seed = DEMO_SEED if st.session_state.demo_mode else None
                voice_data = analyze_voice(audio_data, sr, deterministic_seed=seed)
                st.session_state.voice_data = voice_data
            except Exception as e:
                logger.exception("Audio upload processing failed.")
                st.error(f"Audio processing failed: {e}. Using simulated data.")
                seed = DEMO_SEED if st.session_state.demo_mode else None
                st.session_state.voice_data = analyze_voice(
                    None, deterministic_seed=seed
                )
    elif use_sim:
        with st.spinner("Generating simulated voice data..."):
            seed = DEMO_SEED if st.session_state.demo_mode else None
            voice_data = analyze_voice(None, deterministic_seed=seed)
            st.session_state.voice_data = voice_data

    # Display results
    if st.session_state.voice_data:
        vd = st.session_state.voice_data
        st.success("Voice analysis complete!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Voice Stress", f"{vd['voice_stress']:.3f}")
        c2.metric("Breathing Score", f"{vd['breathing_score']:.3f}")
        c3.metric("Pitch Instability", f"{vd['pitch_instability']:.3f}")
        st.metric("Voice Risk Score", f"{vd['voice_risk_score']:.1f} / 100")

        if st.button("➡️ Run AI Analysis", type="primary"):
            st.session_state.page = "analyze"
            st.rerun()


# ==============================================================================
# PAGE: AI ANALYSIS (processing)
# ==============================================================================

def _page_analyze() -> None:
    st.header("🤖 AI Health Risk Analysis")

    face_data = st.session_state.face_data
    voice_data = st.session_state.voice_data

    if face_data is None or voice_data is None:
        st.warning("Please complete both Face Scan and Voice Scan first.")
        c1, c2 = st.columns(2)
        if face_data is None:
            with c1:
                if st.button("Go to Face Scan"):
                    st.session_state.page = "face"
                    st.rerun()
        if voice_data is None:
            with c2:
                if st.button("Go to Voice Scan"):
                    st.session_state.page = "voice"
                    st.rerun()
        return

    if not st.session_state.model_ready:
        st.error(
            "⚠️ ML model is not trained. Please run `python train_model.py` "
            "from the project directory before using AI analysis."
        )
        return

    # Merge features
    combined_features: Dict[str, float] = {**face_data, **voice_data}

    # Progress animation
    progress = st.progress(0, text="Initialising AI engine...")
    stages = [
        (20, "Loading biomarker features..."),
        (40, "Scaling feature vectors..."),
        (60, "Running RandomForest inference..."),
        (80, "Computing confidence scores..."),
        (100, "Generating risk assessment..."),
    ]
    for pct, msg in stages:
        time.sleep(0.3)
        progress.progress(pct, text=msg)

    prediction = predict_health_risk(combined_features)
    st.session_state.prediction = prediction

    # Drift warning
    if prediction.get("drift_warning"):
        st.warning(
            "⚠️ Feature drift detected — some biomarker values are outside "
            "the training distribution. Results may be less reliable."
        )

    st.success("Analysis complete!")
    time.sleep(0.5)
    st.session_state.page = "dashboard"
    st.rerun()


# ==============================================================================
# PAGE: DASHBOARD
# ==============================================================================

def _page_dashboard() -> None:
    prediction = st.session_state.prediction
    face_data = st.session_state.face_data
    voice_data = st.session_state.voice_data

    if prediction is None:
        st.warning("No analysis results yet. Please run AI Analysis first.")
        if st.button("Go to AI Analysis"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    risk = prediction["overall_risk"]
    level = prediction["risk_level"]
    confidence = prediction["confidence_score"]
    contributions = prediction["feature_contribution"]

    st.markdown(
        f"<h1 style='text-align:center; color:{COLOR_PRIMARY};'>"
        f"📊 Health Risk Dashboard</h1>",
        unsafe_allow_html=True,
    )

    # ----- Risk banner -----
    if level == "Low":
        st.markdown(
            "<div class='risk-low'><h3>✅ Low Risk</h3>"
            "<p>Your biomarkers indicate a low health risk profile.</p></div>",
            unsafe_allow_html=True,
        )
    elif level == "Moderate":
        st.markdown(
            "<div class='risk-moderate'><h3>⚠️ Moderate Risk</h3>"
            "<p>Some biomarkers warrant attention. Consider a professional check-up.</p></div>",
            unsafe_allow_html=True,
        )
    elif level == "High":
        st.markdown(
            "<div class='risk-high'><h3>🚨 High Risk</h3>"
            "<p>Elevated biomarkers detected. Please consult a healthcare professional.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Risk level could not be determined.")

    st.write("")

    # ----- Gauge + confidence -----
    col_gauge, col_conf = st.columns([2, 1])

    with col_gauge:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": "%", "font": {"size": 48}},
                title={"text": "Overall Health Risk", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 2},
                    "bar": {"color": _risk_color(level)},
                    "steps": [
                        {"range": [0, LOW_THRESHOLD], "color": "#e6f4ea"},
                        {"range": [LOW_THRESHOLD, MODERATE_THRESHOLD], "color": "#fef7e0"},
                        {"range": [MODERATE_THRESHOLD, 100], "color": "#fce8e6"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": risk,
                    },
                },
            )
        )
        fig.update_layout(height=320, margin=dict(t=40, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_conf:
        st.write("")
        st.write("")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Risk Level", level)
        st.metric("Model Version", prediction.get("model_version", MODEL_VERSION))
        if prediction.get("drift_warning"):
            st.warning("Drift detected")

    # ----- Feature contribution chart -----
    st.subheader("Feature Contributions")
    sorted_contrib = dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    )
    fig_bar = go.Figure(
        go.Bar(
            x=list(sorted_contrib.values()),
            y=[name.replace("_", " ").title() for name in sorted_contrib.keys()],
            orientation="h",
            marker_color=COLOR_PRIMARY,
        )
    )
    fig_bar.update_layout(
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        height=350,
        margin=dict(t=10, b=30, l=10, r=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----- Detailed metrics -----
    st.subheader("Biomarker Details")
    col_f, col_v = st.columns(2)
    with col_f:
        st.markdown("**Face Biomarkers**")
        if face_data:
            for k, v in face_data.items():
                st.text(f"  {k.replace('_', ' ').title():.<30s} {v}")
    with col_v:
        st.markdown("**Voice Biomarkers**")
        if voice_data:
            for k, v in voice_data.items():
                st.text(f"  {k.replace('_', ' ').title():.<30s} {v}")

    # ----- AI Recommendations -----
    st.subheader("🩺 AI Recommendations")
    _render_recommendations(level, contributions, face_data, voice_data)

    # ----- PDF report download -----
    st.subheader("📄 Download Report")
    pdf_bytes = _generate_pdf_report(
        risk, level, confidence, contributions, face_data, voice_data
    )
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name=f"health_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
        mime="application/pdf",
        type="primary",
    )

    # ----- Disclaimer -----
    st.markdown(
        "<div class='disclaimer'>"
        "⚕️ <strong>Medical Disclaimer:</strong> This AI system is for research and "
        "educational purposes only.  It does NOT constitute medical advice, diagnosis, "
        "or treatment.  Always seek the advice of a qualified healthcare provider with "
        "any questions regarding a medical condition.</div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE: DIGITAL TWIN
# ==============================================================================

def _page_twin() -> None:
    st.header("🔮 Digital Twin Simulation")
    st.info(
        "Simulate how lifestyle changes could affect your projected health risk "
        "over the next 6 months."
    )

    prediction = st.session_state.prediction
    if prediction is None:
        st.warning("Run AI Analysis first to use the Digital Twin.")
        if st.button("Go to AI Analysis"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    current_risk = prediction["overall_risk"]
    current_level = prediction["risk_level"]

    st.subheader("Lifestyle Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        sleep_hours = st.slider("Average Sleep (hours/night)", 0, 12, 7)
    with col2:
        exercise_freq = st.selectbox(
            "Exercise Frequency",
            ["None", "1–2x / week", "3–4x / week", "5+ / week"],
        )
    with col3:
        smoking = st.radio("Smoking Status", ["Non-smoker", "Former", "Active"])

    # ----- Compute projected risk -----
    modifier = 0.0
    if sleep_hours < 6:
        modifier += TWIN_MODIFIERS["sleep_deficit"]
    if smoking == "Active":
        modifier += TWIN_MODIFIERS["smoking_active"]
    elif smoking == "Former":
        modifier += TWIN_MODIFIERS["smoking_active"] * 0.3
    if exercise_freq in ("3–4x / week", "5+ / week"):
        modifier += TWIN_MODIFIERS["exercise_bonus"]
    elif exercise_freq == "None":
        modifier += TWIN_MODIFIERS["no_exercise_penalty"]

    projected_risk = max(0.0, min(100.0, current_risk + modifier))
    projected_level = (
        "Low" if projected_risk < LOW_THRESHOLD
        else "Moderate" if projected_risk < MODERATE_THRESHOLD
        else "High"
    )
    delta = projected_risk - current_risk

    # ----- Dual gauge display -----
    st.subheader("Current vs. Projected Risk (6 months)")
    col_now, col_proj = st.columns(2)

    with col_now:
        fig_now = _make_gauge(current_risk, "Current Risk", current_level)
        st.plotly_chart(fig_now, use_container_width=True)

    with col_proj:
        fig_proj = _make_gauge(projected_risk, "Projected Risk", projected_level)
        st.plotly_chart(fig_proj, use_container_width=True)

    # Delta
    delta_color = COLOR_SUCCESS if delta <= 0 else COLOR_DANGER
    delta_sign = "+" if delta > 0 else ""
    st.markdown(
        f"<h3 style='text-align:center; color:{delta_color};'>"
        f"Projected change: {delta_sign}{delta:.1f}%</h3>",
        unsafe_allow_html=True,
    )

    # Recommendations based on twin
    if delta > 0:
        st.warning(
            "Your current lifestyle trajectory may increase health risk. "
            "Consider improving sleep, adding exercise, or reducing smoking."
        )
    elif delta < 0:
        st.success(
            "Your lifestyle choices are projected to reduce health risk. "
            "Keep it up!"
        )
    else:
        st.info("No significant change projected based on current inputs.")


# ==============================================================================
# HELPER — Risk colour mapping
# ==============================================================================

def _risk_color(level: str) -> str:
    return {
        "Low": COLOR_SUCCESS,
        "Moderate": COLOR_WARNING,
        "High": COLOR_DANGER,
    }.get(level, COLOR_PRIMARY)


# ==============================================================================
# HELPER — Gauge factory
# ==============================================================================

def _make_gauge(value: float, title: str, level: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _risk_color(level)},
                "steps": [
                    {"range": [0, LOW_THRESHOLD], "color": "#e6f4ea"},
                    {"range": [LOW_THRESHOLD, MODERATE_THRESHOLD], "color": "#fef7e0"},
                    {"range": [MODERATE_THRESHOLD, 100], "color": "#fce8e6"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


# ==============================================================================
# HELPER — AI Recommendations
# ==============================================================================

def _render_recommendations(
    level: str,
    contributions: Dict[str, float],
    face_data: Optional[Dict[str, float]],
    voice_data: Optional[Dict[str, float]],
) -> None:
    """Generate rule-based AI recommendations."""

    recs: list[str] = []

    # General level-based
    if level == "High":
        recs.append(
            "🔴 **Immediate Action:** Your overall risk is elevated. "
            "Schedule a comprehensive health check-up with your physician."
        )
    elif level == "Moderate":
        recs.append(
            "🟡 **Monitor:** Some risk markers are elevated. Consider a "
            "routine check-up within the next 1–2 months."
        )
    else:
        recs.append(
            "🟢 **Maintain:** Your risk profile is within healthy range. "
            "Continue your current healthy habits."
        )

    # Feature-specific recommendations
    top_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    for fname, imp in top_features:
        if fname == "face_fatigue" and face_data and face_data.get("face_fatigue", 0) > 0.5:
            recs.append(
                "😴 **Fatigue detected:** Ensure adequate sleep (7–9 hours). "
                "Consider a sleep quality assessment."
            )
        if fname == "voice_stress" and voice_data and voice_data.get("voice_stress", 0) > 0.5:
            recs.append(
                "🗣️ **Vocal stress elevated:** Practice stress management techniques "
                "such as deep breathing or mindfulness meditation."
            )
        if fname == "breathing_score" and voice_data and voice_data.get("breathing_score", 0) > 0.5:
            recs.append(
                "🫁 **Breathing irregularity:** Consider a pulmonary function test. "
                "Regular aerobic exercise can improve respiratory capacity."
            )
        if fname == "symmetry_score" and face_data and face_data.get("symmetry_score", 0) < 0.7:
            recs.append(
                "🧑 **Facial asymmetry noted:** While often benign, significant "
                "asymmetry changes should be evaluated by a neurologist."
            )
        if fname == "pitch_instability" and voice_data and voice_data.get("pitch_instability", 0) > 0.5:
            recs.append(
                "🎵 **Pitch instability:** May indicate stress or thyroid changes. "
                "Consider a thyroid panel if persistent."
            )

    # Always add lifestyle
    recs.append(
        "💪 **General wellness:** Maintain regular exercise (150 min/week), "
        "balanced diet, adequate hydration, and annual health screenings."
    )

    for r in recs:
        st.markdown(r)


# ==============================================================================
# HELPER — PDF Report Generation
# ==============================================================================

def _generate_pdf_report(
    risk: float,
    level: str,
    confidence: float,
    contributions: Dict[str, float],
    face_data: Optional[Dict[str, float]],
    voice_data: Optional[Dict[str, float]],
) -> bytes:
    """Generate a downloadable PDF health report."""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 15, APP_TITLE, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 8,
        f"Report generated: {datetime.now():%Y-%m-%d %H:%M:%S}  |  "
        f"Model v{MODEL_VERSION}  |  App v{APP_VERSION}",
        new_x="LMARGIN", new_y="NEXT", align="C",
    )
    pdf.ln(10)

    # Risk summary
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Risk Assessment Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Overall Risk Score: {risk:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Risk Level: {level}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Confidence: {confidence:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Face biomarkers
    if face_data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Face Biomarkers", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for k, v in face_data.items():
            label = k.replace("_", " ").title()
            pdf.cell(0, 7, f"  {label}: {v}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Voice biomarkers
    if voice_data:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Voice Biomarkers", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        for k, v in voice_data.items():
            label = k.replace("_", " ").title()
            pdf.cell(0, 7, f"  {label}: {v}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Feature importances
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Feature Contributions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    sorted_c = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_c:
        label = name.replace("_", " ").title()
        bar = "█" * int(imp * 50)
        pdf.cell(
            0, 7, f"  {label}: {imp:.4f}  {bar}",
            new_x="LMARGIN", new_y="NEXT",
        )
    pdf.ln(8)

    # Recommendations
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    if level == "High":
        pdf.multi_cell(
            0, 7,
            "Your overall risk is elevated. Schedule a comprehensive "
            "health check-up with your physician promptly.",
        )
    elif level == "Moderate":
        pdf.multi_cell(
            0, 7,
            "Some risk markers are elevated. Consider a routine check-up "
            "within the next 1-2 months.",
        )
    else:
        pdf.multi_cell(
            0, 7,
            "Your risk profile is within healthy range. Continue your "
            "current healthy habits and maintain annual screenings.",
        )
    pdf.ln(5)
    pdf.multi_cell(
        0, 7,
        "General: maintain regular exercise (150 min/week), balanced diet, "
        "adequate hydration, 7-9 hours sleep, and annual health screenings.",
    )

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "DISCLAIMER: This report is generated by an AI research tool and does "
        "not constitute medical advice, diagnosis, or treatment. Always seek "
        "the advice of a qualified healthcare provider.",
    )

    return bytes(pdf.output())


# ==============================================================================
# PAGE ROUTER
# ==============================================================================

_PAGES = {
    "home": _page_home,
    "about": _page_about,
    "face": _page_face,
    "voice": _page_voice,
    "analyze": _page_analyze,
    "dashboard": _page_dashboard,
    "twin": _page_twin,
}

page_fn = _PAGES.get(st.session_state.page, _page_home)
page_fn()
