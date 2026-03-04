"""
architecture_diagram.py — Generate pipeline architecture diagram with Graphviz.

Produces a full system architecture overview of the ML pipeline,
saved as PNG and PDF in the ``visualization/`` directory.

Usage::

    python visualization/architecture_diagram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIZ_DIR = PROJECT_ROOT / "visualization"


def generate_diagram() -> None:
    """Create pipeline architecture diagram using graphviz."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("  ⚠ graphviz not installed — pip install graphviz")
        print("    Also install system Graphviz: https://graphviz.org/download/")
        return

    dot = Digraph(
        "AI_Silent_Disease_Predictor",
        format="png",
        engine="dot",
    )
    dot.attr(rankdir="TB", fontsize="14", fontname="Helvetica")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
             fontsize="11", margin="0.2,0.1")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # ── Color palette ────────────────────────────────────────────
    C_DATA = "#E3F2FD"     # blue light
    C_PROC = "#FFF3E0"     # orange light
    C_TRAIN = "#E8F5E9"    # green light
    C_EVAL = "#F3E5F5"     # purple light
    C_DEPLOY = "#FFEBEE"   # red light
    C_EDGE = "#455A64"     # dark gray

    # ── Data Acquisition ─────────────────────────────────────────
    with dot.subgraph(name="cluster_data") as c:
        c.attr(label="1. Data Acquisition", style="dashed", color="#1565C0",
               fontcolor="#1565C0", fontsize="13")
        c.node("ds1", "UCI Heart Disease\n(297)", fillcolor=C_DATA)
        c.node("ds2", "PIMA Diabetes\n(768)", fillcolor=C_DATA)
        c.node("ds3", "Framingham\n(4,238)", fillcolor=C_DATA)
        c.node("ds4", "Stroke Prediction\n(5,110)", fillcolor=C_DATA)
        c.node("ds5", "Cardiovascular\n(70,000)", fillcolor=C_DATA)

    # ── Data Processing ──────────────────────────────────────────
    with dot.subgraph(name="cluster_proc") as c:
        c.attr(label="2. Data Processing", style="dashed", color="#E65100",
               fontcolor="#E65100", fontsize="13")
        c.node("fusion", "Dataset Fusion\n(Unified Schema)", fillcolor=C_PROC)
        c.node("clean", "Preprocessing\n(Impute / IQR / Scale)", fillcolor=C_PROC)
        c.node("feat", "Feature Engineering\n(9 Biomarkers + 4 Interactions)", fillcolor=C_PROC)

    # ── Training ─────────────────────────────────────────────────
    with dot.subgraph(name="cluster_train") as c:
        c.attr(label="3. Model Training", style="dashed", color="#2E7D32",
               fontcolor="#2E7D32", fontsize="13")
        c.node("smote", "SMOTE Balancing", fillcolor=C_TRAIN)
        c.node("optuna", "Optuna\nHyperparameter Tuning\n(30 trials/model)", fillcolor=C_TRAIN)
        c.node("rf", "RandomForest", fillcolor=C_TRAIN)
        c.node("xgb", "XGBoost", fillcolor=C_TRAIN)
        c.node("lgbm", "LightGBM", fillcolor=C_TRAIN)
        c.node("ensemble", "VotingClassifier\n(Soft Voting)", fillcolor=C_TRAIN,
               shape="doubleoctagon", style="filled")

    # ── Evaluation ───────────────────────────────────────────────
    with dot.subgraph(name="cluster_eval") as c:
        c.attr(label="4. Evaluation", style="dashed", color="#6A1B9A",
               fontcolor="#6A1B9A", fontsize="13")
        c.node("cv", "StratifiedKFold CV\n(5-fold)", fillcolor=C_EVAL)
        c.node("metrics", "Metrics\nAcc / F1 / AUC / SHAP", fillcolor=C_EVAL)
        c.node("shap", "SHAP\nExplainability", fillcolor=C_EVAL)

    # ── Deployment ───────────────────────────────────────────────
    with dot.subgraph(name="cluster_deploy") as c:
        c.attr(label="5. Deployment", style="dashed", color="#C62828",
               fontcolor="#C62828", fontsize="13")
        c.node("model_save", "health_model.pkl\n(Best model)", fillcolor=C_DEPLOY)
        c.node("api", "Flask REST API\n/predict", fillcolor=C_DEPLOY)
        c.node("ui", "React Frontend\n(Face/Voice Scanner)", fillcolor=C_DEPLOY)

    # ── Edges ────────────────────────────────────────────────────
    for ds in ["ds1", "ds2", "ds3", "ds4", "ds5"]:
        dot.edge(ds, "fusion", color=C_EDGE)

    dot.edge("fusion", "clean", label="fused_dataset.csv", color=C_EDGE)
    dot.edge("clean", "feat", label="cleaned_dataset.csv", color=C_EDGE)
    dot.edge("feat", "smote", label="features.csv\n(13 features)", color=C_EDGE)
    dot.edge("smote", "optuna", color=C_EDGE)
    dot.edge("optuna", "rf", color=C_EDGE)
    dot.edge("optuna", "xgb", color=C_EDGE)
    dot.edge("optuna", "lgbm", color=C_EDGE)
    dot.edge("rf", "ensemble", color=C_EDGE)
    dot.edge("xgb", "ensemble", color=C_EDGE)
    dot.edge("lgbm", "ensemble", color=C_EDGE)
    dot.edge("ensemble", "cv", color=C_EDGE)
    dot.edge("ensemble", "metrics", color=C_EDGE)
    dot.edge("metrics", "shap", color=C_EDGE)
    dot.edge("ensemble", "model_save", color=C_EDGE)
    dot.edge("model_save", "api", color=C_EDGE)
    dot.edge("api", "ui", color=C_EDGE)

    # ── Render ───────────────────────────────────────────────────
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VIZ_DIR / "architecture_diagram"
    try:
        dot.render(str(output_path), cleanup=True)
        print(f"  ✓ Architecture diagram saved → {output_path}.png")
    except Exception as e:
        # Graphviz system binary not installed — save source only
        source_path = output_path.with_suffix(".gv")
        with open(source_path, "w") as f:
            f.write(dot.source)
        print(f"  ⚠ Graphviz rendering failed ({e})")
        print(f"    Source saved → {source_path}")
        print("    Install system Graphviz to render: https://graphviz.org/download/")


if __name__ == "__main__":
    generate_diagram()
