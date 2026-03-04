"""
app.py — Modern Healthcare Dashboard UI for AI Silent Disease Predictor.

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
import uuid
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
from streamlit_option_menu import option_menu

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
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="AI Silent Disease Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS — Modern medical SaaS theme
# ==============================================================================

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #1a73e8;
    --primary-light: #e8f0fe;
    --primary-dark: #1557b0;
    --success: #0f9d58;
    --success-light: #e6f4ea;
    --warning: #f4b400;
    --warning-light: #fef7e0;
    --danger: #db4437;
    --danger-light: #fce8e6;
    --bg: #f0f4f8;
    --card-bg: #ffffff;
    --text-primary: #1a202c;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border: #e2e8f0;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -1px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.03);
    --radius: 12px;
    --radius-lg: 16px;
}

/* Global */
.stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}

/* Card component */
.med-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    margin-bottom: 1rem;
}
.med-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}

/* Metric card */
.metric-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
}
.metric-card .metric-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}
.metric-card .metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.metric-card .metric-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card.blue::after { background: var(--primary); }
.metric-card.green::after { background: var(--success); }
.metric-card.amber::after { background: var(--warning); }
.metric-card.red::after { background: var(--danger); }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 50%, #1a237e 100%);
    border-radius: var(--radius-lg);
    padding: 2.5rem 3rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-header h1 {
    color: white !important;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
    position: relative;
    z-index: 1;
}
.hero-header p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem;
    margin-bottom: 0;
    position: relative;
    z-index: 1;
}

/* Badge */
.info-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    padding: 0.35rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    color: white;
    margin-right: 0.5rem;
    margin-top: 0.75rem;
    position: relative;
    z-index: 1;
}
.badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}
.badge-dot.green { background: #4ade80; }
.badge-dot.blue { background: #60a5fa; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Risk banners */
.risk-banner {
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.risk-banner .risk-icon { font-size: 2rem; }
.risk-banner .risk-text h3 {
    margin: 0 0 0.25rem 0;
    font-size: 1.1rem;
}
.risk-banner .risk-text p {
    margin: 0;
    font-size: 0.9rem;
}
.risk-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 5px solid var(--success);
    color: #065f46;
}
.risk-low h3 { color: #065f46 !important; }
.risk-moderate {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 5px solid var(--warning);
    color: #92400e;
}
.risk-moderate h3 { color: #92400e !important; }
.risk-high {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-left: 5px solid var(--danger);
    color: #991b1b;
}
.risk-high h3 { color: #991b1b !important; }

/* Recommendation cards */
.rec-card {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    border-left: 4px solid;
    font-size: 0.9rem;
    line-height: 1.5;
}
.rec-green {
    background: #f0fdf4;
    border-color: #22c55e;
    color: #166534;
}
.rec-yellow {
    background: #fffbeb;
    border-color: #eab308;
    color: #854d0e;
}
.rec-red {
    background: #fef2f2;
    border-color: #ef4444;
    color: #991b1b;
}
.rec-blue {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1e40af;
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}
.section-header .section-icon {
    font-size: 1.4rem;
}
.section-header h2 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* Footer */
.app-footer {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.25rem 2rem;
    margin-top: 2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}
.app-footer p {
    color: var(--text-muted);
    font-size: 0.8rem;
    margin: 0.25rem 0;
}
.app-footer .footer-brand {
    font-weight: 600;
    color: var(--primary);
}

/* Feature card for home page */
.feature-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
    transition: all 0.3s ease;
    height: 100%;
}
.feature-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-4px);
    border-color: var(--primary);
}
.feature-card .feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}
.feature-card h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}
.feature-card p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Scan card (face / voice pages) */
.scan-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
}

/* Biomarker tag */
.bio-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--primary-light);
    color: var(--primary);
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.25rem;
}

/* Table styling */
.bio-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}
.bio-table th {
    background: #f8fafc;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: left;
    border-bottom: 2px solid var(--border);
}
.bio-table td {
    padding: 0.65rem 1rem;
    font-size: 0.9rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}
.bio-table tr:last-child td { border-bottom: none; }
.bio-table tr:hover td { background: #f8fafc; }

/* Privacy shield */
.privacy-shield {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border-radius: var(--radius);
    padding: 1rem 1.5rem;
    border: 1px solid #bbf7d0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1rem;
}
.privacy-shield .shield-icon { font-size: 1.5rem; }
.privacy-shield p {
    margin: 0;
    font-size: 0.8rem;
    color: #166534;
    line-height: 1.4;
}

/* Streamlit button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a73e8, #1557b0) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(26,115,232,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(26,115,232,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Download button override */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a73e8, #1557b0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(26,115,232,0.35) !important;
}

/* Spinner */
.stSpinner > div { color: var(--primary) !important; }

/* Plotly chart background fix */
.js-plotly-plot .plotly .main-svg { border-radius: var(--radius); }

</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE
# ==============================================================================

def _init_state() -> None:
    defaults = {
        "page": "home",
        "face_data": None,
        "voice_data": None,
        "prediction": None,
        "demo_mode": False,
        "model_ready": False,
        "scan_id": str(uuid.uuid4())[:8].upper(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ==============================================================================
# MODEL WARM-UP
# ==============================================================================

@st.cache_resource(show_spinner=False)
def _warm_up_model() -> bool:
    return warm_up()


st.session_state.model_ready = _warm_up_model()


# ==============================================================================
# SIDEBAR — Dark modern navigation
# ==============================================================================

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding: 1.5rem 0 1rem 0;'>"
            "<span style='font-size:2.5rem;'>🧬</span><br/>"
            "<span style='font-size:1.1rem; font-weight:700; color:#f1f5f9;'>"
            "AI Disease Predictor</span><br/>"
            "<span style='font-size:0.7rem; color:#64748b; letter-spacing:1px;'>"
            "PREVENTIVE HEALTHCARE AI</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        selected = option_menu(
            menu_title=None,
            options=[
                "Home", "About", "Face Scan", "Voice Scan",
                "AI Analysis", "Dashboard", "Digital Twin",
            ],
            icons=[
                "house-heart", "info-circle", "camera",
                "mic", "cpu", "speedometer2", "person-bounding-box",
            ],
            default_index=_page_index(),
            styles={
                "container": {
                    "padding": "0 !important",
                    "background-color": "transparent",
                },
                "icon": {"color": "#94a3b8", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.9rem",
                    "font-weight": "500",
                    "text-align": "left",
                    "margin": "2px 0",
                    "padding": "0.6rem 1rem",
                    "border-radius": "8px",
                    "color": "#cbd5e1",
                    "--hover-color": "rgba(255,255,255,0.08)",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, #1a73e8, #1557b0)",
                    "color": "white",
                    "font-weight": "600",
                },
            },
            key="nav_menu",
        )

        _PAGE_MAP = {
            "Home": "home",
            "About": "about",
            "Face Scan": "face",
            "Voice Scan": "voice",
            "AI Analysis": "analyze",
            "Dashboard": "dashboard",
            "Digital Twin": "twin",
        }
        st.session_state.page = _PAGE_MAP.get(selected, "home")

        st.markdown("<br/>", unsafe_allow_html=True)

        # Demo mode toggle
        st.session_state.demo_mode = st.toggle(
            "🧪 Demo Mode",
            value=st.session_state.demo_mode,
            help="Fixed seed for reproducible results in presentations.",
        )

        st.divider()

        # System status
        status_color = "#4ade80" if st.session_state.model_ready else "#f87171"
        status_text = "Online" if st.session_state.model_ready else "Model Missing"
        st.markdown(
            f"<div style='padding: 0.5rem 0;'>"
            f"<span style='font-size:0.7rem; color:#64748b; text-transform:uppercase; "
            f"letter-spacing:1px;'>System Status</span><br/>"
            f"<span style='color:{status_color}; font-size:0.85rem; font-weight:600;'>"
            f"● {status_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='padding: 0.5rem 0; font-size:0.75rem; color:#64748b;'>"
            f"App v{APP_VERSION} · Model v{MODEL_VERSION}<br/>"
            f"© 2026 AI Silent Disease Predictor"
            f"</div>",
            unsafe_allow_html=True,
        )


def _page_index() -> int:
    return {
        "home": 0, "about": 1, "face": 2, "voice": 3,
        "analyze": 4, "dashboard": 5, "twin": 6,
    }.get(st.session_state.page, 0)


_render_sidebar()


# ==============================================================================
# HELPER — Hero header
# ==============================================================================

def _render_hero(title: str, subtitle: str, show_badges: bool = True) -> None:
    badges_html = ""
    if show_badges:
        badges_html = (
            f'<div>'
            f'<span class="info-badge">'
            f'<span class="badge-dot blue"></span> Scan #{st.session_state.scan_id}</span>'
            f'<span class="info-badge">'
            f'<span class="badge-dot blue"></span> Model v{MODEL_VERSION}</span>'
            f'<span class="info-badge">'
            f'<span class="badge-dot green"></span> '
            f'{"System Online" if st.session_state.model_ready else "Model Offline"}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div class="hero-header">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'{badges_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# HELPER — Section header
# ==============================================================================

def _section(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">'
        f'<span class="section-icon">{icon}</span>'
        f'<h2>{title}</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# HELPER — Metric card
# ==============================================================================

def _metric_card(
    icon: str, value: str, label: str, sub: str = "", color: str = "blue"
) -> str:
    return (
        f'<div class="metric-card {color}">'
        f'<div class="metric-icon">{icon}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>'
    )


# ==============================================================================
# HELPER — Risk colour
# ==============================================================================

def _risk_color(level: str) -> str:
    return {
        "Low": COLOR_SUCCESS,
        "Moderate": COLOR_WARNING,
        "High": COLOR_DANGER,
    }.get(level, COLOR_PRIMARY)


# ==============================================================================
# HELPER — Gauge chart factory
# ==============================================================================

def _make_gauge(
    value: float,
    title: str,
    level: str,
    height: int = 300,
) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 52, "family": "Inter", "color": "#1a202c"},
            },
            title={
                "text": title,
                "font": {"size": 16, "family": "Inter", "color": "#64748b"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#e2e8f0",
                    "tickfont": {"size": 11, "color": "#94a3b8"},
                },
                "bar": {
                    "color": _risk_color(level),
                    "thickness": 0.75,
                },
                "bgcolor": "#f8fafc",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, LOW_THRESHOLD], "color": "#dcfce7"},
                    {"range": [LOW_THRESHOLD, MODERATE_THRESHOLD], "color": "#fef9c3"},
                    {"range": [MODERATE_THRESHOLD, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1a202c", "width": 3},
                    "thickness": 0.85,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ==============================================================================
# HELPER — Footer
# ==============================================================================

def _render_footer() -> None:
    st.markdown(
        f'<div class="app-footer">'
        f'<p><span class="footer-brand">🧬 AI Silent Disease Predictor</span> · '
        f'Model v{MODEL_VERSION} · App v{APP_VERSION}</p>'
        f'<p>🔒 No biometric data is stored. All processing occurs locally in-memory.</p>'
        f'<p>⚕️ This is an AI research tool — not medical advice. '
        f'Always consult a qualified healthcare professional.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE: HOME
# ==============================================================================

def _page_home() -> None:
    _render_hero(
        "🧬 AI Silent Disease Predictor",
        "Multimodal Preventive Healthcare AI — "
        "Detect early health risks through facial & vocal biomarker analysis",
    )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            _metric_card("🔬", "9", "Biomarkers", "Face + Voice features", "blue"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card("🌳", "200", "Decision Trees", "RandomForest ensemble", "green"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card("⚡", "<1s", "Inference", "Real-time prediction", "amber"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card("🔒", "100%", "Private", "Zero data retention", "blue"),
            unsafe_allow_html=True,
        )

    st.write("")

    # Feature cards
    _section("🧬", "How It Works")
    col_a, col_b, col_c, col_d = st.columns(4)
    features = [
        ("📷", "Face Analysis", "OpenCV + MediaPipe FaceMesh extracts fatigue, symmetry, blink & brightness biomarkers."),
        ("🎙️", "Voice Analysis", "Librosa extracts MFCC, pYIN pitch, RMS energy & respiratory stress markers."),
        ("🤖", "ML Risk Engine", "RandomForest classifier fuses 9 features into a calibrated risk probability."),
        ("📄", "Health Report", "Downloadable PDF with risk scores, biomarker details & AI recommendations."),
    ]
    for col, (icon, title, desc) in zip([col_a, col_b, col_c, col_d], features):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<span class="feature-icon">{icon}</span>'
                f'<h3>{title}</h3>'
                f'<p>{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    # CTA
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button(
            "🚀  Start Health Scan",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "face"
            st.rerun()

    st.write("")

    # Privacy
    st.markdown(
        '<div class="privacy-shield">'
        '<span class="shield-icon">🛡️</span>'
        '<p><strong>Privacy-First Architecture:</strong> No images, audio, or personal data '
        'are stored. All analysis runs locally in-memory and is discarded after your session.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_footer()


# ==============================================================================
# PAGE: ABOUT
# ==============================================================================

def _page_about() -> None:
    _render_hero(
        "ℹ️ About This System",
        "Architecture, technology stack, and privacy information",
        show_badges=False,
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(
            '<div class="med-card">'
            '<h3>🏗️ System Architecture</h3>'
            '<p style="color:var(--text-secondary); line-height:1.8;">'
            '<strong>AI Silent Disease Predictor</strong> is a preventive healthcare '
            'intelligence system that uses multimodal AI to assess health risk from '
            'non-invasive biomarkers.<br/><br/>'
            '<strong>Pipeline:</strong><br/>'
            '1️⃣ <strong>Capture</strong> — facial image & voice recording<br/>'
            '2️⃣ <strong>Extract</strong> — biomarker features from both modalities<br/>'
            '3️⃣ <strong>Fuse</strong> — combined feature vector fed to trained ML model<br/>'
            '4️⃣ <strong>Predict</strong> — probabilistic health risk with confidence score<br/>'
            '5️⃣ <strong>Report</strong> — downloadable PDF with recommendations</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            '<div class="med-card">'
            '<h3>⚙️ Tech Stack</h3>'
            '<table class="bio-table">'
            '<tr><th>Layer</th><th>Technology</th></tr>'
            '<tr><td>Face Analysis</td><td>OpenCV + MediaPipe</td></tr>'
            '<tr><td>Voice Analysis</td><td>Librosa + NumPy</td></tr>'
            '<tr><td>ML Engine</td><td>scikit-learn RF</td></tr>'
            '<tr><td>Dashboard</td><td>Streamlit + Plotly</td></tr>'
            '<tr><td>Reports</td><td>fpdf2</td></tr>'
            '<tr><td>Drift Detection</td><td>Z-score monitor</td></tr>'
            '</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="privacy-shield">'
        '<span class="shield-icon">🛡️</span>'
        '<p><strong>Privacy Guarantee:</strong> No images or audio are stored. '
        'All processing is in-memory only. No personal data is persisted. '
        'This system is for research and educational purposes only.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_footer()


# ==============================================================================
# PAGE: FACE SCAN
# ==============================================================================

def _page_face() -> None:
    _render_hero(
        "📷 Face Scan",
        "Capture a facial image for biomarker extraction using MediaPipe FaceMesh",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        camera_image = st.camera_input(
            "Take a photo for analysis", key="face_camera"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown(
            '<div class="med-card" style="text-align:center;">'
            '<span style="font-size:2.5rem;">🔬</span>'
            '<h4 style="font-size:0.95rem !important;">Biomarkers Extracted</h4>'
            '<div class="bio-tag">EAR (Blink)</div>'
            '<div class="bio-tag">Symmetry</div>'
            '<div class="bio-tag">Fatigue</div>'
            '<div class="bio-tag">Brightness</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        use_sim = st.button(
            "🔄 Use Simulated Data",
            use_container_width=True,
            help="Generate demo face biomarkers without a camera.",
        )

    # Process
    if camera_image is not None:
        with st.spinner("🔬 Analyzing facial biomarkers..."):
            img_bytes = camera_image.getvalue()
            img_array = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            seed = DEMO_SEED if st.session_state.demo_mode else None
            st.session_state.face_data = analyze_face(frame, deterministic_seed=seed)
    elif use_sim:
        with st.spinner("🔬 Generating simulated face data..."):
            seed = DEMO_SEED if st.session_state.demo_mode else None
            st.session_state.face_data = analyze_face(None, deterministic_seed=seed)

    # Results
    if st.session_state.face_data:
        fd = st.session_state.face_data
        st.write("")
        _section("✅", "Face Analysis Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                _metric_card("😴", f"{fd['face_fatigue']:.3f}", "Fatigue", "", "blue"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                _metric_card("🔄", f"{fd['symmetry_score']:.3f}", "Symmetry", "", "green"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _metric_card("👁️", f"{fd['blink_instability']:.3f}", "Blink Instability", "", "amber"),
                unsafe_allow_html=True,
            )
        with c4:
            color = "green" if fd['face_risk_score'] < LOW_THRESHOLD else (
                "amber" if fd['face_risk_score'] < MODERATE_THRESHOLD else "red"
            )
            st.markdown(
                _metric_card(
                    "📊",
                    f"{fd['face_risk_score']:.1f}%",
                    "Face Risk",
                    f"Brightness Var: {fd['brightness_variance']:.3f}",
                    color,
                ),
                unsafe_allow_html=True,
            )

        st.write("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button(
                "➡️  Proceed to Voice Scan",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.page = "voice"
                st.rerun()


# ==============================================================================
# PAGE: VOICE SCAN
# ==============================================================================

def _page_voice() -> None:
    _render_hero(
        "🎙️ Voice Analysis",
        "Upload a voice recording for vocal biomarker extraction via Librosa",
    )

    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<div class="scan-card">', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader(
            "Upload a .wav file (5–10 seconds of speech)",
            type=["wav"],
            key="voice_upload",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown(
            '<div class="med-card" style="text-align:center;">'
            '<span style="font-size:2.5rem;">🎵</span>'
            '<h4 style="font-size:0.95rem !important;">Biomarkers Extracted</h4>'
            '<div class="bio-tag">MFCC</div>'
            '<div class="bio-tag">pYIN Pitch</div>'
            '<div class="bio-tag">RMS Energy</div>'
            '<div class="bio-tag">Stress</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        use_sim = st.button(
            "🔄 Use Simulated Data",
            use_container_width=True,
            help="Generate demo voice biomarkers without audio.",
        )

    # Process
    if uploaded_audio is not None:
        with st.spinner("🎵 Analyzing vocal biomarkers..."):
            try:
                import librosa
                audio_bytes = uploaded_audio.getvalue()
                audio_buffer = io.BytesIO(audio_bytes)
                audio_data, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE)
                seed = DEMO_SEED if st.session_state.demo_mode else None
                st.session_state.voice_data = analyze_voice(
                    audio_data, int(sr), deterministic_seed=seed  # type: ignore
                )
            except Exception as e:
                logger.exception("Audio upload processing failed.")
                st.error(f"Audio processing failed: {e}. Using simulated data.")
                seed = DEMO_SEED if st.session_state.demo_mode else None
                st.session_state.voice_data = analyze_voice(
                    None, deterministic_seed=seed
                )
    elif use_sim:
        with st.spinner("🎵 Generating simulated voice data..."):
            seed = DEMO_SEED if st.session_state.demo_mode else None
            st.session_state.voice_data = analyze_voice(
                None, deterministic_seed=seed
            )

    # Results
    if st.session_state.voice_data:
        vd = st.session_state.voice_data
        st.write("")
        _section("✅", "Voice Analysis Results")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                _metric_card("🗣️", f"{vd['voice_stress']:.3f}", "Voice Stress", "", "blue"),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                _metric_card("🫁", f"{vd['breathing_score']:.3f}", "Breathing", "", "green"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                _metric_card("🎵", f"{vd['pitch_instability']:.3f}", "Pitch Instability", "", "amber"),
                unsafe_allow_html=True,
            )
        with c4:
            color = "green" if vd['voice_risk_score'] < LOW_THRESHOLD else (
                "amber" if vd['voice_risk_score'] < MODERATE_THRESHOLD else "red"
            )
            st.markdown(
                _metric_card(
                    "📊",
                    f"{vd['voice_risk_score']:.1f}%",
                    "Voice Risk",
                    "",
                    color,
                ),
                unsafe_allow_html=True,
            )

        st.write("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button(
                "🤖  Run AI Analysis",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.page = "analyze"
                st.rerun()


# ==============================================================================
# PAGE: AI ANALYSIS
# ==============================================================================

def _page_analyze() -> None:
    _render_hero(
        "🤖 AI Health Risk Analysis",
        "Fusing facial & vocal biomarkers through ML risk engine",
    )

    face_data = st.session_state.face_data
    voice_data = st.session_state.voice_data

    if face_data is None or voice_data is None:
        st.markdown(
            '<div class="med-card" style="text-align:center; padding:3rem;">'
            '<span style="font-size:3rem;">⚠️</span>'
            '<h3>Scans Required</h3>'
            '<p style="color:var(--text-secondary);">'
            'Please complete both Face Scan and Voice Scan before running AI analysis.'
            '</p></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        if face_data is None:
            with c1:
                if st.button("📷  Go to Face Scan", use_container_width=True):
                    st.session_state.page = "face"
                    st.rerun()
        if voice_data is None:
            with c2:
                if st.button("🎙️  Go to Voice Scan", use_container_width=True):
                    st.session_state.page = "voice"
                    st.rerun()
        return

    if not st.session_state.model_ready:
        st.error(
            "⚠️ ML model not trained. Run `python train_model.py` first."
        )
        return

    # Merge features
    combined_features: Dict[str, float] = {**face_data, **voice_data}

    # Progress
    st.markdown(
        '<div class="med-card" style="text-align:center;">'
        '<h3>🧬 AI Engine Processing</h3>'
        '</div>',
        unsafe_allow_html=True,
    )
    progress = st.progress(0, text="Initialising AI engine...")
    stages = [
        (20, "🔬 Loading biomarker features..."),
        (40, "📐 Scaling feature vectors..."),
        (60, "🌳 Running RandomForest inference..."),
        (80, "📊 Computing confidence scores..."),
        (100, "✅ Generating risk assessment..."),
    ]
    for pct, msg in stages:
        time.sleep(0.35)
        progress.progress(pct, text=msg)

    prediction = predict_health_risk(combined_features)
    st.session_state.prediction = prediction

    if prediction.get("drift_warning"):
        st.warning(
            "⚠️ Feature drift detected — some biomarker values are outside "
            "the training distribution. Results may be less reliable."
        )

    time.sleep(0.5)
    st.session_state.page = "dashboard"
    st.rerun()


# ==============================================================================
# PAGE: DASHBOARD — Main results view
# ==============================================================================

def _page_dashboard() -> None:
    prediction = st.session_state.prediction
    face_data = st.session_state.face_data
    voice_data = st.session_state.voice_data

    if prediction is None:
        st.markdown(
            '<div class="med-card" style="text-align:center; padding:3rem;">'
            '<span style="font-size:3rem;">📊</span>'
            '<h3>No Results Yet</h3>'
            '<p style="color:var(--text-secondary);">Run AI Analysis to see your dashboard.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("🤖  Go to AI Analysis", type="primary"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    risk = prediction["overall_risk"]
    level = prediction["risk_level"]
    confidence = prediction["confidence_score"]
    contributions = prediction["feature_contribution"]
    health_index = max(0.0, 100.0 - risk)

    # Header
    _render_hero(
        "📊 Health Risk Dashboard",
        f"Scan #{st.session_state.scan_id} · "
        f"Analysis completed {datetime.now():%B %d, %Y at %H:%M}",
    )

    # Risk banner
    if level == "Low":
        st.markdown(
            '<div class="risk-banner risk-low">'
            '<span class="risk-icon">✅</span>'
            '<div class="risk-text">'
            '<h3>Low Risk — Looking Great!</h3>'
            '<p>Your biomarkers indicate a healthy risk profile. '
            'Continue your current healthy lifestyle habits.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif level == "Moderate":
        st.markdown(
            '<div class="risk-banner risk-moderate">'
            '<span class="risk-icon">⚠️</span>'
            '<div class="risk-text">'
            '<h3>Moderate Risk — Monitor Closely</h3>'
            '<p>Some biomarkers warrant attention. '
            'Consider scheduling a professional check-up within 1–2 months.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="risk-banner risk-high">'
            '<span class="risk-icon">🚨</span>'
            '<div class="risk-text">'
            '<h3>High Risk — Action Recommended</h3>'
            '<p>Elevated biomarkers detected. '
            'Please consult a healthcare professional promptly.</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Top metric cards ----
    col1, col2, col3 = st.columns(3)
    risk_color = "green" if level == "Low" else ("amber" if level == "Moderate" else "red")
    with col1:
        st.markdown(
            _metric_card(
                "🎯", f"{risk:.1f}%", "Risk Score",
                f"{level} Risk", risk_color,
            ),
            unsafe_allow_html=True,
        )
    with col2:
        conf_color = "green" if confidence >= 70 else ("amber" if confidence >= 50 else "red")
        st.markdown(
            _metric_card(
                "📊", f"{confidence:.1f}%", "Confidence",
                "Model certainty", conf_color,
            ),
            unsafe_allow_html=True,
        )
    with col3:
        hi_color = "green" if health_index >= 60 else ("amber" if health_index >= 30 else "red")
        st.markdown(
            _metric_card(
                "🫀", f"{health_index:.1f}%", "Health Index",
                "Overall wellness", hi_color,
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    # ---- Gauge + details ----
    col_gauge, col_detail = st.columns([3, 2])

    with col_gauge:
        _section("📈", "Risk Gauge")
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        fig = _make_gauge(risk, "Overall Health Risk", level, height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_detail:
        _section("🔍", "Quick Summary")
        st.markdown(
            '<div class="med-card">'
            f'<table class="bio-table">'
            f'<tr><th>Metric</th><th>Value</th></tr>'
            f'<tr><td>Overall Risk</td><td><strong>{risk:.1f}%</strong></td></tr>'
            f'<tr><td>Risk Level</td><td>{level}</td></tr>'
            f'<tr><td>Confidence</td><td>{confidence:.1f}%</td></tr>'
            f'<tr><td>Health Index</td><td>{health_index:.1f}%</td></tr>'
            f'<tr><td>Model Version</td><td>v{prediction.get("model_version", MODEL_VERSION)}</td></tr>'
            f'<tr><td>Drift Warning</td><td>'
            f'{"⚠️ Yes" if prediction.get("drift_warning") else "✅ No"}</td></tr>'
            f'</table>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # ---- Feature contributions ----
    _section("📊", "Feature Contributions")
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    sorted_contrib = dict(
        sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    )
    names = [n.replace("_", " ").title() for n in sorted_contrib.keys()]
    values = list(sorted_contrib.values())

    colors = []
    for v in values:
        if v > 0.15:
            colors.append("#ef4444")
        elif v > 0.10:
            colors.append("#f59e0b")
        else:
            colors.append("#1a73e8")

    fig_bar = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            textfont={"family": "Inter", "size": 11, "color": "#64748b"},
        )
    )
    fig_bar.update_layout(
        xaxis_title="Importance",
        xaxis=dict(
            showgrid=True,
            gridcolor="#f1f5f9",
            tickfont={"size": 11, "color": "#94a3b8"},
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont={"size": 12, "color": "#374151", "family": "Inter"},
        ),
        height=380,
        margin=dict(t=10, b=40, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # ---- Biomarker details ----
    _section("🧬", "Biomarker Details")
    col_f, col_v = st.columns(2)

    with col_f:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown("##### 📷 Face Biomarkers")
        if face_data:
            rows = ""
            for k, v in face_data.items():
                label = k.replace("_", " ").title()
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                rows += f"<tr><td>{label}</td><td><strong>{val}</strong></td></tr>"
            st.markdown(
                f'<table class="bio-table">'
                f'<tr><th>Biomarker</th><th>Value</th></tr>'
                f'{rows}</table>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_v:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        st.markdown("##### 🎙️ Voice Biomarkers")
        if voice_data:
            rows = ""
            for k, v in voice_data.items():
                label = k.replace("_", " ").title()
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                rows += f"<tr><td>{label}</td><td><strong>{val}</strong></td></tr>"
            st.markdown(
                f'<table class="bio-table">'
                f'<tr><th>Biomarker</th><th>Value</th></tr>'
                f'{rows}</table>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # ---- Recommendations ----
    _section("🩺", "AI Health Recommendations")
    _render_recommendations(level, contributions, face_data, voice_data)

    st.write("")

    # ---- PDF Download ----
    _section("📄", "Medical Report")
    st.markdown('<div class="med-card" style="text-align:center; padding:2rem;">', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:2.5rem;">📋</span><br/>'
        '<p style="color:var(--text-secondary); margin:0.5rem 0 1rem 0;">'
        'Download a comprehensive PDF report with all biomarker data, '
        'risk scores, and AI recommendations.</p>',
        unsafe_allow_html=True,
    )
    pdf_bytes = _generate_pdf_report(
        risk, level, confidence, contributions, face_data, voice_data
    )
    st.download_button(
        label="📥  Download Health Report",
        data=pdf_bytes,
        file_name=f"health_report_{st.session_state.scan_id}_{datetime.now():%Y%m%d_%H%M%S}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # Disclaimer
    st.markdown(
        '<div class="med-card" style="border-left: 4px solid #94a3b8;">'
        '<p style="font-size:0.8rem; color:var(--text-muted); margin:0;">'
        '⚕️ <strong>Medical Disclaimer:</strong> This AI system is for research and '
        'educational purposes only. It does NOT constitute medical advice, diagnosis, '
        'or treatment. Always seek the advice of a qualified healthcare provider.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _render_footer()


# ==============================================================================
# PAGE: DIGITAL TWIN
# ==============================================================================

def _page_twin() -> None:
    _render_hero(
        "🔮 Digital Twin Simulation",
        "Simulate how lifestyle changes could affect your projected health risk over 6 months",
    )

    prediction = st.session_state.prediction
    if prediction is None:
        st.markdown(
            '<div class="med-card" style="text-align:center; padding:3rem;">'
            '<span style="font-size:3rem;">🔮</span>'
            '<h3>Analysis Required</h3>'
            '<p style="color:var(--text-secondary);">'
            'Run AI Analysis first to use the Digital Twin.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("🤖  Go to AI Analysis", type="primary"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    current_risk = prediction["overall_risk"]
    current_level = prediction["risk_level"]

    # Lifestyle inputs
    _section("⚙️", "Lifestyle Parameters")
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 😴 Sleep Pattern")
        sleep_hours = st.slider(
            "Average sleep (hours/night)",
            0, 12, 7,
            help="Hours of sleep per night on average",
        )
    with col2:
        st.markdown("##### 🏃 Exercise Routine")
        exercise_freq = st.selectbox(
            "Exercise frequency",
            ["None", "1–2x / week", "3–4x / week", "5+ / week"],
        )
    with col3:
        st.markdown("##### 🚭 Smoking Status")
        smoking = st.radio(
            "Current status",
            ["Non-smoker", "Former", "Active"],
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Compute projected risk
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

    st.write("")

    # Dual gauge
    _section("📈", "Current vs. Projected Risk (6 months)")
    col_now, col_arrow, col_proj = st.columns([5, 1, 5])

    with col_now:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        fig_now = _make_gauge(current_risk, "Current Risk", current_level, 300)
        st.plotly_chart(fig_now, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_arrow:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        arrow_color = COLOR_SUCCESS if delta <= 0 else COLOR_DANGER
        arrow = "↘️" if delta < 0 else ("↗️" if delta > 0 else "➡️")
        st.markdown(
            f'<div style="text-align:center; padding-top:4rem;">'
            f'<span style="font-size:2.5rem;">{arrow}</span><br/>'
            f'<span style="font-size:1.1rem; font-weight:700; color:{arrow_color};">'
            f'{"+" if delta > 0 else ""}{delta:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_proj:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)
        fig_proj = _make_gauge(projected_risk, "Projected Risk (6mo)", projected_level, 300)
        st.plotly_chart(fig_proj, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    # Insight
    if delta > 0:
        st.markdown(
            '<div class="rec-card rec-red">'
            '🔴 <strong>Risk Increasing:</strong> Your current lifestyle trajectory '
            'may increase health risk. Consider improving sleep quality, adding regular '
            'exercise, or reducing smoking to reverse this trend.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif delta < 0:
        st.markdown(
            '<div class="rec-card rec-green">'
            '🟢 <strong>Risk Decreasing:</strong> Your lifestyle choices are projected '
            'to reduce health risk over the next 6 months. Keep up the great work!'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rec-card rec-blue">'
            '🔵 <strong>Stable:</strong> No significant change projected based on '
            'current inputs. Consider optimizing sleep and exercise for improvement.'
            '</div>',
            unsafe_allow_html=True,
        )

    _render_footer()


# ==============================================================================
# HELPER — Recommendations
# ==============================================================================

def _render_recommendations(
    level: str,
    contributions: Dict[str, float],
    face_data: Optional[Dict[str, float]],
    voice_data: Optional[Dict[str, float]],
) -> None:
    """Render styled AI health recommendations."""

    # Level-based
    if level == "High":
        st.markdown(
            '<div class="rec-card rec-red">'
            '🚨 <strong>Immediate Action:</strong> Your overall risk is elevated. '
            'Schedule a comprehensive health check-up with your physician as soon as possible.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif level == "Moderate":
        st.markdown(
            '<div class="rec-card rec-yellow">'
            '⚠️ <strong>Monitor Closely:</strong> Some risk markers are elevated. '
            'Consider a routine check-up within the next 1–2 months.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rec-card rec-green">'
            '✅ <strong>Maintain:</strong> Your risk profile is within healthy range. '
            'Continue your current healthy habits and annual screenings.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Feature-specific
    top_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    for fname, imp in top_features:
        if fname == "face_fatigue" and face_data and face_data.get("face_fatigue", 0) > 0.5:
            st.markdown(
                '<div class="rec-card rec-yellow">'
                '😴 <strong>Fatigue Detected:</strong> Ensure adequate sleep (7–9 hours). '
                'Consider a sleep quality assessment and establish a consistent bedtime routine.'
                '</div>',
                unsafe_allow_html=True,
            )
        if fname == "voice_stress" and voice_data and voice_data.get("voice_stress", 0) > 0.5:
            st.markdown(
                '<div class="rec-card rec-yellow">'
                '🗣️ <strong>Vocal Stress Elevated:</strong> Practice stress management '
                'techniques such as deep breathing, mindfulness meditation, or yoga.'
                '</div>',
                unsafe_allow_html=True,
            )
        if fname == "breathing_score" and voice_data and voice_data.get("breathing_score", 0) > 0.5:
            st.markdown(
                '<div class="rec-card rec-yellow">'
                '🫁 <strong>Breathing Irregularity:</strong> Consider a pulmonary function test. '
                'Regular aerobic exercise can improve respiratory capacity.'
                '</div>',
                unsafe_allow_html=True,
            )
        if fname == "symmetry_score" and face_data and face_data.get("symmetry_score", 0) < 0.7:
            st.markdown(
                '<div class="rec-card rec-yellow">'
                '🧑 <strong>Facial Asymmetry Noted:</strong> While often benign, '
                'significant asymmetry changes should be evaluated by a neurologist.'
                '</div>',
                unsafe_allow_html=True,
            )
        if fname == "pitch_instability" and voice_data and voice_data.get("pitch_instability", 0) > 0.5:
            st.markdown(
                '<div class="rec-card rec-yellow">'
                '🎵 <strong>Pitch Instability:</strong> May indicate stress or thyroid changes. '
                'Consider a thyroid panel if symptoms persist.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Always — wellness
    st.markdown(
        '<div class="rec-card rec-blue">'
        '💪 <strong>General Wellness:</strong> Maintain regular exercise (150 min/week), '
        'balanced diet, adequate hydration, 7–9 hours sleep, and annual health screenings.'
        '</div>',
        unsafe_allow_html=True,
    )


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
        f"Model v{MODEL_VERSION}  |  App v{APP_VERSION}  |  "
        f"Scan #{st.session_state.scan_id}",
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
    pdf.cell(0, 8, f"Health Confidence Index: {max(0, 100 - risk):.1f}%", new_x="LMARGIN", new_y="NEXT")
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
            "URGENT: Your overall risk is elevated. Schedule a comprehensive "
            "health check-up with your physician promptly.",
        )
    elif level == "Moderate":
        pdf.multi_cell(
            0, 7,
            "MONITOR: Some risk markers are elevated. Consider a routine check-up "
            "within the next 1-2 months.",
        )
    else:
        pdf.multi_cell(
            0, 7,
            "MAINTAIN: Your risk profile is within healthy range. Continue your "
            "current healthy habits and maintain annual screenings.",
        )
    pdf.ln(5)
    pdf.multi_cell(
        0, 7,
        "General: Maintain regular exercise (150 min/week), balanced diet, "
        "adequate hydration, 7-9 hours sleep, and annual health screenings.",
    )

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "DISCLAIMER: This report is generated by an AI research tool and does "
        "not constitute medical advice, diagnosis, or treatment. Always seek "
        "the advice of a qualified healthcare provider. "
        "No biometric data is stored. All processing occurs locally.",
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
