from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_intermediate_values,
        plot_edf,
        plot_timeline,
        plot_slice,
        plot_contour,
        plot_rank,
    )
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False



st.set_page_config(
    page_title="MH Multimodal Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',system-ui,sans-serif}
.block-container{padding-top:1rem;max-width:1340px}
.hero{background:linear-gradient(135deg,#0f172a,#1e3a5f 45%,#0d9488);border-radius:16px;padding:1.25rem 1.6rem;color:#f8fafc;margin-bottom:1rem;box-shadow:0 8px 24px rgba(15,23,42,.25)}
.hero h1{margin:0 0 .2rem;font-size:1.55rem;font-weight:700}
.hero p{margin:0;opacity:.88;font-size:.92rem}
.kpi-row{display:flex;gap:.7rem;flex-wrap:wrap;margin:.4rem 0 1rem}
.kpi{flex:1 1 120px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:.75rem .9rem}
.kpi .label{font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:#64748b;font-weight:600}
.kpi .value{font-size:1.3rem;font-weight:700;color:#0f172a;font-family:'JetBrains Mono',monospace}
.kpi .sub{font-size:.72rem;color:#94a3b8}
.section-title{font-size:1.02rem;font-weight:700;color:#0f172a;margin:.7rem 0 .35rem;border-left:3px solid #0d9488;padding-left:.5rem}
.pill{display:inline-block;padding:.15rem .55rem;border-radius:999px;font-size:.72rem;font-weight:600;margin-right:.3rem}
.pill-ok{background:#ccfbf1;color:#0f766e}.pill-off{background:#f1f5f9;color:#64748b}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0f172a,#1e293b);
}
/* Labels & captions: muted light text on dark sidebar */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
  color:#cbd5e1!important;
  font-size:.78rem!important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown strong{
  color:#f1f5f9!important;
}
/* Widget values: dark text on light field so selection is always readable */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="input"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="select"] div{
  color:#0f172a!important;
  background-color:#f8fafc!important;
}
/* text/number inputs only (not radio/checkbox) */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] [data-testid="stTextInput"] input{
  color:#0f172a!important;
  background-color:#ffffff!important;
  -webkit-text-fill-color:#0f172a!important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  border-color:#94a3b8!important;
}
/* Number input steppers / selected chips */
section[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  color:#0f172a!important;
  -webkit-text-fill-color:#0f172a!important;
  background-color:#ffffff!important;
  caret-color:#0f172a!important;
}
/* Multiselect tags */
section[data-testid="stSidebar"] [data-baseweb="tag"]{
  background-color:#e2e8f0!important;
  color:#0f172a!important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] span{
  color:#0f172a!important;
}
/* Radio / checkbox — keep option text visible on dark sidebar */
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] p,
section[data-testid="stSidebar"] [role="radiogroup"] label,
section[data-testid="stSidebar"] [role="radiogroup"] label p,
section[data-testid="stSidebar"] [role="radiogroup"] label span,
section[data-testid="stSidebar"] [role="radio"],
section[data-testid="stSidebar"] [role="radio"] p,
section[data-testid="stSidebar"] [role="radio"] span,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] span{
  color:#f1f5f9!important;
  -webkit-text-fill-color:#f1f5f9!important;
  opacity:1!important;
}
/* Do not force light backgrounds onto radio circles / checkbox controls */
section[data-testid="stSidebar"] [data-testid="stRadio"] input,
section[data-testid="stSidebar"] [role="radiogroup"] input,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] input{
  background-color:transparent!important;
  accent-color:#2dd4bf!important;
}
/* Horizontal radio often renders as BaseWeb buttons */
section[data-testid="stSidebar"] [data-testid="stRadio"] button,
section[data-testid="stSidebar"] [data-baseweb="button"][role="radio"],
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div{
  color:#f1f5f9!important;
  -webkit-text-fill-color:#f1f5f9!important;
  border-color:#64748b!important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] button[aria-checked="true"],
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"]{
  color:#0f172a!important;
  -webkit-text-fill-color:#0f172a!important;
  background-color:#5eead4!important;
  border-color:#2dd4bf!important;
}
/* Slider value */
section[data-testid="stSidebar"] [data-testid="stThumbValue"],
section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stTickBarMax"]{
  color:#e2e8f0!important;
}
div[data-baseweb="notification"]{border-radius:10px}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

# ----- Sidebar -----
with st.sidebar:
    st.markdown("### 🧠 MH Studio")
    st.caption("Unified multimodal analysis")
    st.markdown("---")
    data_mode = st.radio("Data source", ["demo", "upload"], horizontal=True)
    st.session_state.data_mode = data_mode

    model_choices = ["rf"] + (["xgb"] if HAS_XGB else [])
    model_type = st.radio("Model", model_choices, format_func=lambda x: "Random Forest" if x == "rf" else "XGBoost", horizontal=True)
    imbalance_mode = st.selectbox(
        "Imbalance handling",
        ["balanced", "focal", "none"],
        index=0,
        help="balanced = class/sample weights; focal = emphasize minority; none = uniform",
    )
    task_choice = st.selectbox("Task", ["auto", "classification", "regression", "ordinal_phq"])
    window_size = st.slider("Window size", 2, 21, 7)
    sliding = st.checkbox("Sliding windows", True)
    predict_next = st.checkbox("Next-day target", True)
    n_splits = st.slider("CV n_splits (GroupKFold)", 2, 8, 5)
    eval_mode = st.selectbox(
        "Eval protocol",
        ["groupkfold", "repeated_groupkfold", "nested_groupkfold", "nested_optuna", "time_aware", "hybrid_subject_time"],
        index=0,
        help="nested_optuna=outer GroupKFold + inner Optuna; hybrid=subject holdout + temporal span; time_aware=last days within subject",
    )
    n_repeats = st.slider("Repeats (if repeated)", 1, 10, 3)

    st.markdown("**Feature engineering**")
    fe_preset = st.radio(
        "Preset",
        ["Minimal", "Default", "Rich", "Trend"],
        horizontal=True,
        key="fe_preset",
        help="Chooses default aggregation stats for window features",
    )
    preset_map = {
        "Minimal": ["mean", "std", "last"],
        "Default": ["mean", "std", "min", "max", "last"],
        "Rich": ["mean", "std", "min", "max", "last", "median", "range", "cv", "slope"],
        "Trend": ["mean", "last", "slope", "delta"],
    }
    st.markdown("**Stats & ratios**")
    # Reset multiselect defaults when preset changes
    _preset_key = f"agg_stats_{fe_preset}"
    if st.session_state.get("_last_fe_preset") != fe_preset:
        st.session_state["_last_fe_preset"] = fe_preset
        st.session_state["agg_stats_ms"] = list(preset_map[fe_preset])
    if "agg_stats_ms" not in st.session_state:
        st.session_state["agg_stats_ms"] = list(preset_map[fe_preset])
    agg_stats = st.multiselect(
        "Agg stats",
        ["mean", "std", "min", "max", "last", "median", "sum", "range", "cv", "slope", "delta"],
        default=st.session_state.get("agg_stats_ms", preset_map[fe_preset]),
        key="agg_stats_ms",
    )
    if not agg_stats:
        agg_stats = list(preset_map[fe_preset])
        st.caption(f"Empty selection → using preset **{fe_preset}**: {agg_stats}")
    use_ratios = st.checkbox("Use ratio features", True)
    use_composites = st.checkbox("Use composites", True)
    include_eeg = st.checkbox("EEG trials branch", True)
    include_audio = st.checkbox("Audio clips branch", True)
    n_subjects = st.number_input("Demo subjects", 4, 50, 10)
    n_days = st.number_input("Demo days", 14, 120, 35)
    seed = st.number_input("Seed", 0, 9999, 42)
    out_dir = st.text_input("Export directory", value=str(ROOT / "data" / "exports"))
    st.markdown("---")
    st.caption(f"XGB {'✓' if HAS_XGB else '✗'} · Optuna {'✓' if HAS_OPTUNA else '✗'} · SHAP {'✓' if HAS_SHAP else '✗'}")

st.markdown("""
<div class="hero">
  <h1>Mental Health Multimodal Studio</h1>
  <p>Demo or uploads · RF / XGBoost · GroupKFold · Optuna · Calibration · SHAP · Ablation · EEG/Audio</p>
</div>
""", unsafe_allow_html=True)


tabs = st.tabs([
    "Overview", "Data panel", "Guide",
])

tabs = st.tabs([
    "Overview", "Data panel", "Workflows", "Run pipeline", "Results", "Ablation", "EEG / Audio",
    "Optuna", "Calibration", "SHAP", "Interactions", "Playground",
    "Dataset cards", "Tokenizers", "Stimuli / trials", "FM Probe",
    "Foundation Model", "Export", "Guide", "Settings",
])

# ===== Overview =====
with tabs[0]:
    st.markdown('<div class="section-title">Studio overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Stages**\n1. Load demo/uploads\n2. Align + engineer\n3. Window tokenize\n4. Export / GroupKFold\n5. RF or XGBoost\n6. Optuna · SHAP · calibration")
    with c2:
        st.markdown("**Modalities**\nActivity · vitals · sleep · behavior · weather\nLabs · weight · meds (as-of)\nEEG trials · audio clips")
    with c3:
        st.markdown(f"**Settings**\n- `{label_col}` · **{model_type.upper()}**\n- Window **{window_size}** · splits **{n_splits}**\n- EEG `{include_eeg}` · Audio `{include_audio}`\n- Data: **{data_mode}**")
    st.dataframe(pd.DataFrame({"outcome": OUTCOME_CANDIDATES, "selected": [c == label_col for c in OUTCOME_CANDIDATES]}), use_container_width=True, hide_index=True)

# ===== Data panel =====
with tabs[1]:
    st.markdown('<div class="section-title">Data contract · multi-file loaders · QC</div>', unsafe_allow_html=True)
    st.caption(
        "Core clinical/behavioral: surveys (PHQ/GAD/energy/pain/substance), activity, sleep, vitals (HR/HRV/BP), labs, notes.\n\n"
        "Neuroscience: EEG/MEG/fNIRS continuous + events → daily neuro features mergeable into canonical panel + FM fusion.\n\n"
        "Surveys + activity + labs + notes (and optional vitals/sleep/…) → validated column maps → "
        "**canonical panel** (`subject_id` × `date`). All train / FM / merge tabs read this single panel."
    )

# ===== Guide =====
with tabs[2]:
    st.markdown(
        '<div class="section-title">Studio guide · methods · diagrams</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Full application manual (token contract, subject-level metrics, SHAP, FM, artifacts) "
        "plus diagrams, procedures, and glossaries — same depth as the subject-level metrics notes."
    )

    gtab1, gtab2, gtab3, gtab4, gtab5 = st.tabs([
        "Diagrams", "Full manual", "Procedures", "Glossaries", "Decisions",
    ])

    with gtab1:
        st.markdown("**A. End-to-end workflow**")
        if HAS_PLOTLY:
            fig = go.Figure(go.Scatter(
                x=list(range(len(FLOW_STEPS))),
                y=[1] * len(FLOW_STEPS),
                mode="markers+text+lines",
                text=FLOW_STEPS,
                textposition="top center",
                marker=dict(size=26, color="#0d9488"),
                line=dict(color="#94a3b8", width=3),
            ))
            fig.update_layout(
                height=200, margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                title="Suggested analysis flow",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(" → ".join(FLOW_STEPS))

        st.markdown("**B. Data-flow Sankey (logical volumes)**")
        if HAS_PLOTLY:
            sk = architecture_sankey_data()
            fig = go.Figure(go.Sankey(
                node=dict(label=sk["labels"], pad=12, thickness=16,
                          color=["#64748b", "#0d9488", "#0891b2", "#4f46e5", "#7c3aed",
                                 "#ea580c", "#ca8a04", "#e11d48", "#334155"]),
                link=dict(source=sk["source"], target=sk["target"], value=sk["value"]),
            ))
            fig.update_layout(height=420, title="Raw → align → tokens → probes / FM")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**C. Layered architecture**")
        for layer in ARCH_LAYERS:
            st.markdown(f"**{layer['layer']}**")
            st.write(" · ".join(layer["items"]))

        st.markdown("**D. Subject-wise GroupKFold schematic**")
        st.dataframe(pd.DataFrame(split_diagram_levels()), hide_index=True, use_container_width=True)
        st.caption("No subject appears in both train and test within a fold — required for person-level generalization claims.")

        st.markdown("**E. Token schema (prefix map)**")
        if HAS_PLOTLY:
            pref = pd.DataFrame({
                "prefix": ["f__", "y__", "e__", "m__", "mask__"],
                "role": ["features", "labels", "embeddings", "meta", "observed?"],
                "weight": [5, 3, 2, 2, 4],
            })
            fig = px.bar(pref, x="prefix", y="weight", color="role", text="role",
                         title="TokenBatch column families")
            fig.update_layout(height=320, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        st.code(
            "subject_id | session_id | t0 | t1 | modality | tokenizer_id | f__* | y__* | e__* | m__* | mask__*",
            language="text",
        )

        st.markdown("**F. ASCII system map**")
        st.code(
            """
Raw modalities ──► Align (daily / asof / trial) ──► TOKENIZER_REGISTRY
                                                      │
                         ┌────────────────────────────┼────────────────────────────┐
                         ▼                            ▼                            ▼
                   Flat panel                   Sequence JSONL              Contrastive pairs
                         │                            │                            │
              RF/XGB · linear probe              Transformer FM (ext)            SSL pretrain
                         │
                    SHAP · Optuna · calibration · export
            """,
            language="text",
        )

    with gtab2:
        titles = [s["title"] for s in GUIDE_SECTIONS]
        pick = st.selectbox("Jump to section", titles, key="guide_sec")
        for sec in GUIDE_SECTIONS:
            with st.expander(sec["title"], expanded=(sec["title"] == pick)):
                st.markdown(sec["body"])

    with gtab3:
        st.markdown("**Step-by-step procedures**")
        for proc in PROCEDURES:
            with st.expander(proc["name"], expanded=False):
                for i, step in enumerate(proc["steps"], 1):
                    st.markdown(f"{i}. {step}")

        st.markdown("**Tab purpose index**")
        tab_help = pd.DataFrame([
            {"Tab": "Overview", "Purpose": "KPIs, stages, outcomes"},
            {"Tab": "Data panel", "Purpose": "Demo / upload multimodal tables"},
            {"Tab": "Run pipeline", "Purpose": "Align, engineer, window, train"},
            {"Tab": "Results", "Purpose": "Metrics + feature importance"},
            {"Tab": "Ablation", "Purpose": "Drop modality groups"},
            {"Tab": "EEG / Audio", "Purpose": "Neuro continuous tokens + viz"},
            {"Tab": "Optuna", "Purpose": "XGB hyperparameter search"},
            {"Tab": "Calibration", "Purpose": "Probability reliability"},
            {"Tab": "SHAP", "Purpose": "Global/local explanations"},
            {"Tab": "Interactions", "Purpose": "Pairwise SHAP effects"},
            {"Tab": "Playground", "Purpose": "What-if on top features"},
            {"Tab": "Dataset cards", "Purpose": "Manifest v2 + stress"},
            {"Tab": "Tokenizers", "Purpose": "Registry preview + YAML"},
            {"Tab": "Stimuli / trials", "Purpose": "Events, PEA, daily rollups"},
            {"Tab": "FM Probe", "Purpose": "Frozen encoder + linear head"},
            {"Tab": "Foundation Model", "Purpose": "Masks, sequences, SSL pairs"},
            {"Tab": "Export", "Purpose": "Downloads + FM bundle"},
            {"Tab": "Guide", "Purpose": "This documentation"},
            {"Tab": "Settings", "Purpose": "Session / paths"},
        ])
        st.dataframe(tab_help, hide_index=True, use_container_width=True)

    with gtab4:
        st.markdown("**Metrics**")
        st.dataframe(pd.DataFrame(METRIC_GLOSSARY), hide_index=True, use_container_width=True)
        st.markdown("**SHAP plots**")
        st.dataframe(pd.DataFrame(SHAP_GLOSSARY), hide_index=True, use_container_width=True)
        st.markdown("**Tokenizers**")
        st.dataframe(pd.DataFrame(TOKENIZER_TABLE), hide_index=True, use_container_width=True)
        st.markdown("**Contrastive strategies**")
        st.dataframe(pd.DataFrame(CONTRASTIVE_TABLE), hide_index=True, use_container_width=True)
        if HAS_PLOTLY:
            fig = px.timeline(
                pd.DataFrame({
                    "Task": ["Align", "Tokenize", "Train probe", "Explain", "FM export"],
                    "Start": [0, 1, 2, 3, 2],
                    "Finish": [1, 2, 4, 5, 4],
                    "Resource": ["data", "data", "model", "model", "fm"],
                }),
                x_start="Start", x_end="Finish", y="Task", color="Resource",
                title="Relative effort phases (illustrative)",
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    with gtab5:
        st.markdown("**Choose a path by goal**")
        st.dataframe(pd.DataFrame(DECISION_PATHS), hide_index=True, use_container_width=True)
        st.markdown("**Implemented vs frontiers**")
        gap = next((s for s in GUIDE_SECTIONS if s["id"] == "gaps"), None)
        if gap:
            st.markdown(gap["body"])
        st.info(
            "Tip: keep subject-wise splits for any claim about new people; "
            "record norm policy in the manifest before comparing probes to pretrain."
        )
