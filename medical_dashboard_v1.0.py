# ==============================================================
# Medical Insights Dashboard
# Full version with Authentication + Random Forest + Interactive UI
# Mental Health Analytics Studio — Full Application
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime, timedelta

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, mean_squared_error,
    precision_score, recall_score, confusion_matrix, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, make_scorer, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance


import matplotlib.pyplot as plt

from predictors import *
import xgboost as xgb
import shap

import warnings
warnings.filterwarnings("ignore")
from collections import Counter

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from optuna.terminator import (
    BaseTerminator,
    Terminator,
    MedianErrorEvaluator,
    RegretBoundEvaluator,
    BestValueStagnationEvaluator,
    )
    TERMINATOR_OK = True
except ImportError:
    TERMINATOR_OK = False
from optuna.terminator.erroreval import StaticErrorEvaluator

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
        plot_intermediate_values,
        plot_edf,
        plot_timeline,
        plot_contour,
        plot_rank,
        plot_terminator_improvement,
    )
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

# ==============================================================
# PAGE + THEME
# ==============================================================
st.set_page_config(
    page_title="Medical Insights Dashboard - Mental Health Analytics Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #f0f4f8 0%, #e8eef5 100%); }
.block-container { padding-top: 1.25rem; max-width: 1180px; }
h1 { color: #0f172a !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { color: #1e293b !important; font-weight: 600 !important; }
section[data-testid="stSidebar"] { background: #0f172a !important; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown strong { color: #f8fafc !important; }
div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);
}
div[data-testid="stMetric"] label {
    color: #64748b !important; font-size: 0.78rem !important;
    font-weight: 500 !important; text-transform: uppercase; letter-spacing: .04em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f172a !important; font-weight: 700 !important;
}
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important; border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; color: #fff !important;
}
button[data-baseweb="tab"] { font-weight: 600 !important; color: #64748b !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1d4ed8 !important; }
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #1d4ed8 100%);
    color: #f8fafc; padding: 1.35rem 1.5rem; border-radius: 14px;
    margin-bottom: 1.1rem; box-shadow: 0 8px 24px rgba(15,23,42,.18);
}
.hero-banner h1 { color: #fff !important; margin: 0 !important; font-size: 1.55rem !important; }
.hero-banner p { color: #cbd5e1; margin: .35rem 0 0 0; font-size: .92rem; }
.status-pill {
    display: inline-block; padding: .2rem .65rem; border-radius: 999px;
    font-size: .75rem; font-weight: 600; margin: 0 .3rem .25rem 0;
}
.status-done { background: #dcfce7; color: #166534; }
.status-todo { background: #334155; color: #cbd5e1; }
.footer-note {
    color: #94a3b8; font-size: .8rem; text-align: center;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
  <h1>Medical Insights Dashboard - Mental Health Analytics Studio</h1>
  <p>Survey scoring · Activity fusion · RF / XGBoost · Windows · SHAP suite · Interactions</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================
# HELPERS
# ==============================================================
def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ", "_").str.replace("-", "_")
    )
    return df


def find_col(cands, cols):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def score_phq9(df, item_cols):
    items = df[item_cols].apply(pd.to_numeric, errors="coerce").clip(0, 3)
    total = items.sum(axis=1, min_count=1)

    def sev(s):
        if pd.isna(s): return np.nan
        if s <= 4: return "Minimal"
        if s <= 9: return "Mild"
        if s <= 14: return "Moderate"
        if s <= 19: return "Moderately severe"
        return "Severe"

    return pd.DataFrame({
        "phq9_total": total,
        "phq9_severity": total.apply(sev),
        "phq9_elevated": (total >= 10).astype("Int64"),
    })


def score_gad7(df, item_cols):
    items = df[item_cols].apply(pd.to_numeric, errors="coerce").clip(0, 3)
    total = items.sum(axis=1, min_count=1)

    def sev(s):
        if pd.isna(s): return np.nan
        if s <= 4: return "Minimal"
        if s <= 9: return "Mild"
        if s <= 14: return "Moderate"
        return "Severe"

    return pd.DataFrame({
        "gad7_total": total,
        "gad7_severity": total.apply(sev),
        "gad7_elevated": (total >= 10).astype("Int64"),
    })


def score_energy(df, col, low_cut=4):
    level = pd.to_numeric(df[col], errors="coerce").clip(0, 10)

    def band(v):
        if pd.isna(v): return np.nan
        if v <= 3: return "Low"
        if v <= 6: return "Moderate"
        return "High"

    return pd.DataFrame({
        "energy_level": level,
        "energy_band": level.apply(band),
        "energy_low": (level <= low_cut).astype("Int64"),
    })


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def window_aggregates(vals, stats):
    vals = np.asarray(vals, dtype=float)
    out = {}
    if "mean" in stats: out["mean"] = np.nanmean(vals)
    if "std" in stats: out["std"] = np.nanstd(vals)
    if "min" in stats: out["min"] = np.nanmin(vals)
    if "max" in stats: out["max"] = np.nanmax(vals)
    if "last" in stats: out["last"] = vals[-1]
    if "median" in stats: out["median"] = np.nanmedian(vals)
    if "sum" in stats: out["sum"] = np.nansum(vals)
    if "range" in stats: out["range"] = np.nanmax(vals) - np.nanmin(vals)
    if "cv" in stats:
        m = np.nanmean(vals)
        out["cv"] = float(np.nanstd(vals) / (abs(m) + 1e-6))
    if "slope" in stats and len(vals) >= 2:
        x = np.arange(len(vals), dtype=float)
        y = np.nan_to_num(vals, nan=np.nanmean(vals))
        out["slope"] = float(np.polyfit(x, y, 1)[0])
    if "delta" in stats and len(vals) >= 2:
        out["delta"] = float(vals[-1] - vals[0])
    return out


def create_windowed_data_fe(
    df, id_col, date_col, feature_cols, target_col,
    window_size=7, sliding=True, predict_next_day=True,
    agg_stats=None, use_ratios=True, use_composites=True,
):
    if agg_stats is None:
        agg_stats = ["mean", "std", "min", "max", "last"]

    cols = set(feature_cols)
    ratio_pairs = []
    for a, b, name in [
        ("total_steps", "sedentary", "ratio_steps_sedentary"),
        ("steps", "sedentary", "ratio_steps_sedentary"),
        ("moderate", "vigorous", "ratio_moderate_vigorous"),
        ("light", "sedentary", "ratio_light_sedentary"),
    ]:
        if a in cols and b in cols:
            ratio_pairs.append((a, b, name))

    records = []
    for pid, group in df.groupby(id_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        max_start = len(group) - window_size - (1 if predict_next_day else 0)
        if max_start < 0:
            continue
        starts = (
            range(0, max_start + 1) if sliding
            else range(0, max_start + 1, window_size)
        )
        for start in starts:
            window = group.iloc[start: start + window_size]
            feat = {}
            for col in feature_cols:
                stats = window_aggregates(window[col].values, agg_stats)
                for s, v in stats.items():
                    feat[f"{col}_{s}"] = v

            if use_ratios:
                for num, den, name in ratio_pairs:
                    nk, dk = f"{num}_mean", f"{den}_mean"
                    if nk in feat and dk in feat:
                        feat[name] = _safe_div(feat[nk], feat[dk])

            if use_composites:
                def m(c):
                    return feat.get(f"{c}_mean")
                light, mod, vig = m("light"), m("moderate"), m("vigorous")
                sed = m("sedentary")
                if all(v is not None for v in [light, mod, vig]):
                    feat["active_minutes_mean"] = light + mod + vig
                    feat["intensity_index"] = light + 2 * mod + 3 * vig
                    if sed is not None:
                        feat["active_sedentary_ratio"] = _safe_div(
                            feat["active_minutes_mean"], sed
                        )

            if predict_next_day:
                row = group.iloc[start + window_size]
                feat["target"] = row[target_col]
                feat["target_date"] = row[date_col]
            else:
                feat["target"] = window[target_col].iloc[-1]
                feat["target_date"] = window[date_col].iloc[-1]
            feat[id_col] = pid
            feat["window_start"] = window[date_col].iloc[0]
            feat["window_end"] = window[date_col].iloc[-1]
            records.append(feat)
    return pd.DataFrame(records)


def remap_labels(y_train, y_test):
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        return None, None, None, None
    mapping = {c: i for i, c in enumerate(train_classes)}
    y_tr = np.array([mapping[v] for v in y_train], dtype=int)
    valid = np.isin(y_test, train_classes)
    y_te = np.array([mapping[v] for v in y_test[valid]], dtype=int)
    return y_tr, y_te, valid, mapping


def make_model(model_type, params, is_classification, y_train_mapped=None):
    """Build Random Forest or XGBoost model."""
    if model_type == "Random Forest":
        rf_params = dict(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=-1,
            random_state=42,
        )
        if rf_params["max_depth"] is not None:
            rf_params["max_depth"] = int(rf_params["max_depth"])
        if is_classification:
            return RandomForestClassifier(**rf_params, class_weight="balanced")
        return RandomForestRegressor(**rf_params)

    # XGBoost
    base = dict(
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 6)),
        min_child_weight=int(params.get("min_child_weight", 1)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        gamma=float(params.get("gamma", 0.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        n_jobs=-1, random_state=42, verbosity=0,
    )
    if not is_classification:
        return XGBRegressor(**base, objective="reg:squarederror")
    n_classes = len(np.unique(y_train_mapped))
    if n_classes == 2:
        return XGBClassifier(**base, objective="binary:logistic", eval_metric="logloss")
    return XGBClassifier(
        **base, objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss"
    )


def default_params(model_type):
    if model_type == "Random Forest":
        return {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        }
    return {
        "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6,
        "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
    }


def mean_sd(vals):
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return (float(np.mean(clean)), float(np.std(clean))) if clean else (np.nan, np.nan)


def safe_plotly(fn, *args, **kwargs):
    try:
        fig = fn(*args, **kwargs)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#334155", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Plot unavailable: {e}")


def normalize_shap_values(values, prefer_class=1):
    """Force SHAP values to (n_samples, n_features)."""
    values = np.array(values)
    if isinstance(values, list):
        # multi-class list of arrays
        idx = prefer_class if prefer_class < len(values) else 0
        values = np.array(values[idx])
    if values.ndim == 3:
        # (n, features, classes)
        cls = prefer_class if values.shape[-1] > prefer_class else 0
        values = values[:, :, cls]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def make_demo_surveys(n_participants=15, days=40, seed=42):
    rng = np.random.default_rng(seed)
    rows, start = [], datetime(2024, 1, 1)
    for p in range(1, n_participants + 1):
        pid = f"P{p:03d}"
        for d in range(days):
            if rng.random() < 0.35:
                continue
            row = {"participant_id": pid, "date": (start + timedelta(days=d)).strftime("%Y-%m-%d")}
            for i in range(1, 10):
                row[f"phq9_q{i}"] = int(rng.integers(0, 4))
            for i in range(1, 8):
                row[f"gad7_q{i}"] = int(rng.integers(0, 4))
            row["energy_level"] = int(rng.integers(0, 11))
            rows.append(row)
    return pd.DataFrame(rows)


def make_demo_activity(n_participants=15, days=40, seed=43):
    rng = np.random.default_rng(seed)
    rows, start = [], datetime(2024, 1, 1)
    for p in range(1, n_participants + 1):
        pid = f"P{p:03d}"
        for d in range(days):
            rows.append({
                "participant_id": pid,
                "date": (start + timedelta(days=d)).strftime("%Y-%m-%d"),
                "total_steps": int(rng.integers(1500, 14000)),
                "sedentary": int(rng.integers(300, 800)),
                "light": int(rng.integers(50, 250)),
                "moderate": int(rng.integers(10, 120)),
                "vigorous": int(rng.integers(0, 60)),
                "total_distance": float(rng.uniform(1.0, 12.0)),
                "duration_minutes": int(rng.integers(20, 180)),
            })
    return pd.DataFrame(rows)


def pill(label, done):
    cls = "status-done" if done else "status-todo"
    mark = "✓" if done else "·"
    return f'<span class="status-pill {cls}">{mark} {label}</span>'

def render_terminator_improvement(study):
    if study is None:
        st.info("No Optuna study available.")
        return
    
    if not TERMINATOR_OK:
        st.info("Optuna terminator not available.")
        return
    try:
        fig = plot_terminator_improvement(
            study,
            plot_error=False,
            improvement_evaluator=BestValueStagnationEvaluator(max_stagnation_trials=10),
            error_evaluator=StaticErrorEvaluator(constant=0),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter,sans-serif", color="#334155", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Terminator improvement plot unavailable: {e}")


# ==============================================================
# SIDEBAR
# ==============================================================
st.sidebar.markdown("## Data sources")
use_demo = st.sidebar.checkbox("Use built-in demo data", value=False)
survey_file = st.sidebar.file_uploader("Surveys CSV", type=["csv"])
activity_file = st.sidebar.file_uploader("Activity CSV", type=["csv"])

if use_demo and st.sidebar.button("Generate demo datasets"):
    st.session_state["demo_surveys"] = make_demo_surveys()
    st.session_state["demo_activity"] = make_demo_activity()
    st.sidebar.success("Demo data ready")

st.sidebar.markdown("---")
st.sidebar.markdown("## Model family")
model_type = st.sidebar.radio(
    "Algorithm",
    ["XGBoost", "Random Forest"],
    index=0,
    help="Used in CV, Optuna (XGB), and final model",
)
st.session_state["model_type"] = model_type

st.sidebar.markdown("---")
st.sidebar.markdown("## Pipeline status")
st.sidebar.markdown(
    pill("Surveys", "surveys_scored" in st.session_state)
    + pill("Merge", "analysis_table" in st.session_state)
    + pill("CV", "X" in st.session_state)
    + pill("Model", "final_model" in st.session_state),
    unsafe_allow_html=True,
)

if st.sidebar.button("Reset session"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("## Model I/O")
up_model = st.sidebar.file_uploader("Load model (.joblib)", type=["joblib", "pkl"])
if up_model is not None and JOBLIB_OK:
    st.session_state["final_model"] = joblib.load(up_model)
    st.sidebar.success("Model loaded")

tabs = st.tabs([
    "① Surveys",
    "② Merge",
    "③ Modeling",
    "④ Tuning",
    "⑤ Results",
    "⑥ Calibration",
    "⑦ Explainability",
    "⑧ Interactions",
    "⑨ Playground",
    "⑩ Export",
    "ML algorithms and tools",
    "About",
])

# ==============================================================
# TAB 1 — SURVEYS
# ==============================================================
with tabs[0]:
    st.subheader("Score clinical surveys")
    st.caption("PHQ-9 · GAD-7 · Energy level")

    if use_demo and "demo_surveys" in st.session_state:
        surveys = clean_columns(st.session_state["demo_surveys"].copy())
        st.info("Using demo surveys")
    elif survey_file is not None:
        surveys = clean_columns(pd.read_csv(survey_file))
    else:
        surveys = None
        st.info("Upload a surveys file or enable demo data in the sidebar.")

    if surveys is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(surveys):,}")
        c2.metric("Columns", surveys.shape[1])
        id_tmp = find_col(["participant_id", "id", "participant"], surveys.columns)
        c3.metric("Participants", surveys[id_tmp].nunique() if id_tmp else "—")

        with st.expander("Preview raw data", expanded=False):
            st.dataframe(surveys.head(15), use_container_width=True)

        id_g = find_col(["participant_id", "id", "participant", "user_id"], surveys.columns)
        date_g = find_col(["date", "datetime", "day", "timestamp"], surveys.columns)
        c1, c2 = st.columns(2)
        s_id = c1.selectbox(
            "Participant ID", list(surveys.columns),
            index=list(surveys.columns).index(id_g) if id_g in surveys.columns else 0, key="s_id",
        )
        s_date = c2.selectbox(
            "Date", list(surveys.columns),
            index=list(surveys.columns).index(date_g) if date_g in surveys.columns else 0, key="s_date",
        )
        surveys[s_date] = pd.to_datetime(surveys[s_date], errors="coerce")

        phq_def = [f"phq9_q{i}" for i in range(1, 10) if f"phq9_q{i}" in surveys.columns]
        gad_def = [f"gad7_q{i}" for i in range(1, 8) if f"gad7_q{i}" in surveys.columns]
        energy_def = find_col(["energy_level", "energy", "rate_your_energy"], surveys.columns)

        phq_items = st.multiselect("PHQ-9 items (9)", list(surveys.columns), default=phq_def)
        gad_items = st.multiselect("GAD-7 items (7)", list(surveys.columns), default=gad_def)
        energy_col = st.selectbox(
            "Energy column",
            ["(none)"] + list(surveys.columns),
            index=(1 + list(surveys.columns).index(energy_def)) if energy_def else 0,
        )
        low_cut = st.slider("Low-energy threshold", 0, 10, 4)

        if st.button("Score surveys", type="primary"):
            scored = surveys[[s_id, s_date]].copy()
            scored.columns = ["participant_id", "date"]
            done = []
            if len(phq_items) == 9:
                scored = pd.concat([scored, score_phq9(surveys, phq_items)], axis=1)
                done.append("PHQ-9")
            if len(gad_items) == 7:
                scored = pd.concat([scored, score_gad7(surveys, gad_items)], axis=1)
                done.append("GAD-7")
            if energy_col != "(none)":
                scored = pd.concat([scored, score_energy(surveys, energy_col, low_cut)], axis=1)
                done.append("Energy")
            if "phq9_total" in scored.columns and "gad7_total" in scored.columns:
                scored["sum_phq9_gad7"] = scored["phq9_total"] + scored["gad7_total"]
            scored = scored.dropna(subset=["date"])
            scored["participant_id"] = scored["participant_id"].astype(str)
            scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
            scored = (
                scored.sort_values(["participant_id", "date"])
                .drop_duplicates(["participant_id", "date"], keep="last")
                .reset_index(drop=True)
            )
            st.session_state["surveys_scored"] = scored
            st.success("Completed: " + ", ".join(done))

        if "surveys_scored" in st.session_state:
            scored = st.session_state["surveys_scored"]
            tcol = st.selectbox(
                "Explore target distribution",
                [c for c in scored.columns if c not in ("participant_id", "date")],
                key="exp_t",
            )
            if PLOTLY_OK:
                if pd.api.types.is_numeric_dtype(scored[tcol]):
                    fig = px.histogram(scored, x=tcol, nbins=20, title=tcol,
                                      color_discrete_sequence=["#2563eb"])
                else:
                    vc = scored[tcol].value_counts().reset_index()
                    vc.columns = [tcol, "count"]
                    fig = px.bar(vc, x=tcol, y="count", color_discrete_sequence=["#2563eb"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(scored, use_container_width=True, height=280)
            st.download_button(
                "Download scored surveys",
                scored.to_csv(index=False).encode(),
                "surveys_scored.csv", "text/csv",
            )

# ==============================================================
# TAB 2 — MERGE
# ==============================================================
with tabs[1]:
    st.subheader("Merge activity with survey targets")
    if "surveys_scored" not in st.session_state:
        st.warning("Score surveys in tab ① first.")
    else:
        if use_demo and "demo_activity" in st.session_state:
            activity = clean_columns(st.session_state["demo_activity"].copy())
            st.info("Using demo activity")
        elif activity_file is not None:
            activity = clean_columns(pd.read_csv(activity_file))
        else:
            activity = None
            st.info("Upload activity data or generate demo data.")

        if activity is not None:
            with st.expander("Preview activity data", expanded=False):
                st.dataframe(activity.head(20), use_container_width=True)
            scored = st.session_state["surveys_scored"].copy()
            id_g = find_col(["participant_id", "id", "participant"], activity.columns)
            date_g = find_col(["date", "datetime", "day", "timestamp"], activity.columns)
            c1, c2, c3 = st.columns(3)
            a_id = c1.selectbox(
                "Activity ID", list(activity.columns),
                index=list(activity.columns).index(id_g) if id_g in activity.columns else 0, key="a_id",
            )
            a_date = c2.selectbox(
                "Activity date", list(activity.columns),
                index=list(activity.columns).index(date_g) if date_g in activity.columns else 0, key="a_date",
            )
            how = c3.selectbox("Join type", ["inner", "left"])

            act = activity.rename(columns={a_id: "participant_id", a_date: "date"})
            act["participant_id"] = act["participant_id"].astype(str)
            act["date"] = pd.to_datetime(act["date"], errors="coerce").dt.normalize()
            act = act.dropna(subset=["date"])
            scored["participant_id"] = scored["participant_id"].astype(str)
            scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()

            feats = [c for c in act.select_dtypes("number").columns if c != "participant_id"]
            targets = [
                c for c in scored.columns
                if c not in ("participant_id", "date", "phq9_severity", "gad7_severity", "energy_band")
            ]

            if st.button("Merge datasets", type="primary"):
                merged = act.merge(scored, on=["participant_id", "date"], how=how)
                with st.expander("Preview merged data", expanded=False):
                    st.dataframe(merged.head(20), use_container_width=True)
                st.session_state["merged"] = merged
                st.session_state["candidate_features"] = feats
                st.session_state["target_candidates"] = targets
                st.success(f"Merged {merged.shape[0]:,} rows × {merged.shape[1]} columns")

            if "merged" in st.session_state:
                merged = st.session_state["merged"]
                feats = st.session_state["candidate_features"]
                targets = st.session_state["target_candidates"]
                target_col = st.selectbox("Target variable", targets, key="tgt")
                selected = st.multiselect("Activity features", feats, default=feats, key="feats")
                if not selected:
                    selected = feats
                holdout_n = st.slider("Holdout participants", 0, 5, 0)

                if target_col not in selected:
                    model_df = merged.dropna(subset=[target_col]).copy()
                    c1, c2 = st.columns(2)
                    c1.metric("Rows with target", f"{len(model_df):,}")
                    c2.metric("Participants", model_df["participant_id"].nunique())
                    if st.button("Save analysis table", type="primary"):
                        analysis = model_df[
                            ["participant_id", "date", target_col] + selected
                        ].rename(columns={target_col: "target"})
                        all_ids = analysis["participant_id"].unique().tolist()
                        rng = np.random.default_rng(42)
                        holdout_ids = list(
                            rng.choice(all_ids, size=min(holdout_n, len(all_ids)), replace=False)
                        ) if holdout_n else []
                        st.session_state["holdout_ids"] = holdout_ids
                        st.session_state["analysis_table"] = analysis
                        st.session_state["analysis_features"] = selected
                        st.session_state["analysis_target_name"] = target_col
                        st.success(f"Saved {len(analysis):,} rows · holdout={holdout_ids or 'none'}")

# ==============================================================
# TAB 3 — MODELING (with train error + RF/XGB)
# ==============================================================
with tabs[2]:
    st.subheader("Feature engineering · windows · cross-validation")
    st.caption(f"Active algorithm: **{st.session_state.get('model_type', 'XGBoost')}** (change in sidebar)")

    if "analysis_table" not in st.session_state:
        st.warning("Save an analysis table in tab ② first.")
    else:
        analysis = st.session_state["analysis_table"].copy()
        feat_base = st.session_state["analysis_features"]
        holdout_ids = st.session_state.get("holdout_ids", [])
        analysis_train = (
            analysis[~analysis["participant_id"].isin(holdout_ids)].copy()
            if holdout_ids else analysis
        )
        model_type = st.session_state.get("model_type", "XGBoost")

        st.markdown("**Window settings**")
        c1, c2, c3, c4 = st.columns(4)
        window_size = c1.select_slider("Window size", options=[2, 7, 14, 21, 30], value=7)
        sliding = c2.toggle("Sliding window", True)
        predict_next_day = c3.toggle("Next-day target", True)
        n_splits = c4.select_slider("Folds", options=list(range(3, 11)), value=5)

        st.markdown("**Feature engineering**")
        preset = st.radio("Preset", ["Minimal", "Default", "Rich", "Trend"], horizontal=True)
        preset_map = {
            "Minimal": ["mean", "std", "last"],
            "Default": ["mean", "std", "min", "max", "last"],
            "Rich": ["mean", "std", "min", "max", "last", "median", "range", "cv", "slope"],
            "Trend": ["mean", "last", "slope", "delta"],
        }
        agg_stats = st.multiselect(
            "Window statistics",
            ["mean", "std", "min", "max", "last", "median", "sum", "range", "cv", "slope", "delta"],
            default=preset_map[preset],
        )
        r1, r2 = st.columns(2)
        use_ratios = r1.toggle("Ratio features", True)
        use_composites = r2.toggle("Composite features", True)

        task_choice = st.radio("Task", ["Auto", "Classification", "Regression"], horizontal=True)
        n_unique = analysis_train["target"].nunique(dropna=True)
        is_classification = (
            n_unique <= 15 if task_choice == "Auto" else task_choice == "Classification"
        )
        st.info(
            f"Target **{st.session_state.get('analysis_target_name')}** · "
            f"{'Classification' if is_classification else 'Regression'} · "
            f"**{model_type}** · {len(feat_base)} base features"
        )

        if st.button("Run analysis", type="primary"):
            with st.spinner("Building windows and running GroupKFold…"):
                windowed = create_windowed_data_fe(
                    analysis_train, "participant_id", "date", feat_base, "target",
                    window_size, sliding, predict_next_day,
                    agg_stats=agg_stats,
                    use_ratios=use_ratios,
                    use_composites=use_composites,
                )
            if len(windowed) == 0:
                st.error("No windows created.")
                st.stop()

            meta = ["participant_id", "target", "target_date", "window_start", "window_end"]
            feat_names = [c for c in windowed.columns if c not in meta]
            X = windowed[feat_names].fillna(windowed[feat_names].median(numeric_only=True))
            y_raw = windowed["target"]
            groups = windowed["participant_id"].values

            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str)).astype(int)
                st.write("Classes:", list(le.classes_))
            else:
                le, y = None, y_raw.astype(float).values

            st.session_state.update({
                "windowed": windowed, "X": X, "y": y, "groups": groups,
                "feat_names": feat_names, "is_classification": is_classification,
                "label_encoder": le,
            })

            k1, k2, k3 = st.columns(3)
            k1.metric("Windows", f"{len(windowed):,}")
            k2.metric("Features", X.shape[1])
            k3.metric("Participants", pd.Series(groups).nunique())

            params = default_params(model_type)
            actual = min(n_splits, len(np.unique(groups)))
            cv = GroupKFold(n_splits=actual)
            metrics = {
                k: [] for k in [
                    "fold", "accuracy", "f1", "auc", "mse", "train_error"
                ]
            }
            oof_prob, oof_true = [], []
            prog = st.progress(0)

            for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y[tr], y[te]
                metrics["fold"].append(fold + 1)

                if is_classification:
                    y_tr_m, y_te_m, valid, mapping = remap_labels(y_tr, y_te)
                    if y_tr_m is None or valid.sum() == 0:
                        for k in ["accuracy", "f1", "auc", "mse", "train_error"]:
                            metrics[k].append(np.nan)
                        prog.progress((fold + 1) / actual)
                        continue
                    X_te_f = X_te.iloc[valid]
                    model = make_model(model_type, params, True, y_tr_m)
                    model.fit(X_tr, y_tr_m)
                    pred = model.predict(X_te_f)
                    tr_pred = model.predict(X_tr)
                    # train error = 1 - train accuracy
                    metrics["train_error"].append(1.0 - accuracy_score(y_tr_m, tr_pred))
                    metrics["accuracy"].append(accuracy_score(y_te_m, pred))
                    metrics["f1"].append(f1_score(y_te_m, pred, average="weighted"))
                    metrics["mse"].append(np.nan)
                    try:
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X_te_f)
                            pos = mapping.get(1, 1 if proba.shape[1] > 1 else 0)
                            pos = min(pos, proba.shape[1] - 1)
                            oof_prob.append(proba[:, pos])
                            oof_true.append(
                                (y_te[valid] == 1).astype(int) if 1 in mapping else y_te_m
                            )
                            if proba.shape[1] == 2:
                                metrics["auc"].append(roc_auc_score(y_te_m, proba[:, 1]))
                            else:
                                metrics["auc"].append(roc_auc_score(
                                    y_te_m, proba, multi_class="ovr", average="weighted"
                                ))
                        else:
                            metrics["auc"].append(np.nan)
                    except Exception:
                        metrics["auc"].append(np.nan)
                else:
                    model = make_model(model_type, params, False)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)
                    tr_pred = model.predict(X_tr)
                    # train error = train MSE
                    metrics["train_error"].append(mean_squared_error(y_tr, tr_pred))
                    metrics["mse"].append(mean_squared_error(y_te, pred))
                    metrics["accuracy"].append(np.nan)
                    metrics["f1"].append(np.nan)
                    metrics["auc"].append(np.nan)
                prog.progress((fold + 1) / actual)

            if oof_prob:
                st.session_state["oof_true"] = np.concatenate(
                    [np.asarray(x).reshape(-1) for x in oof_true]
                )
                st.session_state["oof_prob"] = np.concatenate(
                    [np.asarray(x).reshape(-1) for x in oof_prob]
                )

            fold_df = pd.DataFrame(metrics)
            st.session_state["cv_fold_df"] = fold_df
            mean_te, sd_te = mean_sd(metrics["train_error"])
            st.session_state["cv_summary"] = {
                "mean_acc": mean_sd(metrics["accuracy"])[0],
                "sd_acc": mean_sd(metrics["accuracy"])[1],
                "mean_auc": mean_sd(metrics["auc"])[0],
                "sd_auc": mean_sd(metrics["auc"])[1],
                "mean_f1": mean_sd(metrics["f1"])[0],
                "sd_f1": mean_sd(metrics["f1"])[1],
                "mse": mean_sd(metrics["mse"])[0],
                "train_error": mean_te,
                "sd_train_error": sd_te,
                "model_type": model_type,
            }
            s = st.session_state["cv_summary"]
            a, b, c, d, e = st.columns(5)
            a.metric("Mean Acc", f"{s['mean_acc']:.3f}" if s["mean_acc"] == s["mean_acc"] else "—")
            b.metric("Mean AUC", f"{s['mean_auc']:.3f}" if s["mean_auc"] == s["mean_auc"] else "—")
            c.metric("Mean F1", f"{s['mean_f1']:.3f}" if s["mean_f1"] == s["mean_f1"] else "—")
            d.metric("MSE", f"{s['mse']:.3f}" if s["mse"] == s["mse"] else "—")
            e.metric(
                "Train error",
                f"{s['train_error']:.3f}" if s["train_error"] == s["train_error"] else "—",
                help="Classification: 1 − train accuracy · Regression: train MSE",
            )
            if PLOTLY_OK:
                m = st.selectbox(
                    "Chart metric",
                    ["accuracy", "auc", "f1", "mse", "train_error"],
                )
                fig = px.bar(fold_df, x="fold", y=m, text=m, color_discrete_sequence=["#2563eb"])
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fold_df, use_container_width=True)

# ==============================================================
# TAB 4 — TUNING (XGB Optuna; RF simpler grid note)
# ==============================================================
with tabs[3]:
    st.subheader("Hyperparameter tuning")
    model_type = st.session_state.get("model_type", "XGBoost")

    if "X" not in st.session_state:
        st.warning("Run modeling tab first.")
    elif model_type == "Random Forest":
        st.info(
            "Optuna search below is tuned for **XGBoost**. "
            "For Random Forest, train a production RF with sidebar defaults or adjust params here."
        )
        X = st.session_state["X"]
        y = st.session_state["y"]
        is_classification = st.session_state["is_classification"]
        n_est = st.slider("RF n_estimators", 50, 500, 200)
        max_d = st.slider("RF max_depth", 3, 30, 12)
        min_leaf = st.slider("RF min_samples_leaf", 1, 20, 2)
        if st.button("Train production Random Forest", type="primary"):
            params = {
                "n_estimators": n_est,
                "max_depth": max_d,
                "min_samples_leaf": min_leaf,
                "max_features": "sqrt",
            }
            if is_classification:
                y_fit = np.asarray(y).astype(int)
                model = make_model("Random Forest", params, True, y_fit)
                model.fit(X, y_fit)
            else:
                model = make_model("Random Forest", params, False)
                model.fit(X, y)
            st.session_state["final_model"] = model
            st.session_state["best_params"] = params
            st.session_state["model_type"] = "Random Forest"
            st.success("Random Forest production model trained")
            if JOBLIB_OK:
                buf = io.BytesIO()
                joblib.dump(model, buf)
                st.download_button(
                    "Download model (.joblib)", buf.getvalue(),
                    "rf_final_model.joblib", "application/octet-stream",
                )
    elif not OPTUNA_OK:
        st.error("Install: pip install optuna plotly")
    else:
        X = st.session_state["X"]
        y = st.session_state["y"]
        groups = st.session_state["groups"]
        is_classification = st.session_state["is_classification"]
        c1, c2, c3 = st.columns(3)
        n_trials = c1.slider("Trials", 5, 40, 12)
        opt_splits = c2.slider("CV folds", 3, 8, 5)
        timeout = c3.number_input("Timeout (sec)", 0, 3600, 120, 30)
        use_terminator = st.checkbox("Use Optuna terminator", value=True)

        if st.button("Start Optuna search (XGBoost)", type="primary"):
            def objective(trial, use_pruning=True):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                }
                actual = min(opt_splits, len(np.unique(groups)))
                cv = GroupKFold(n_splits=actual)
                scores = []
                for tr, te in cv.split(X, y, groups=groups):
                    X_tr, X_te = X.iloc[tr], X.iloc[te]
                    y_tr, y_te = y[tr], y[te]
                    if is_classification:
                        y_tr_m, y_te_m, valid, _ = remap_labels(y_tr, y_te)
                        if y_tr_m is None or valid.sum() == 0:
                            continue
                        X_te = X_te.iloc[valid]
                        model = make_model("XGBoost", params, True, y_tr_m)
                        model.fit(X_tr, y_tr_m)
                        try:
                            proba = model.predict_proba(X_te)
                            if proba.shape[1] == 2:
                                scores.append(roc_auc_score(y_te_m, proba[:, 1]))
                            else:
                                scores.append(roc_auc_score(
                                    y_te_m, proba, multi_class="ovr", average="weighted"
                                ))
                        except Exception:
                            scores.append(f1_score(y_te_m, model.predict(X_te), average="weighted"))
                    else:
                        model = make_model("XGBoost", params, False)
                        model.fit(X_tr, y_tr)
                        scores.append(-mean_squared_error(y_te, model.predict(X_te)))

                
                # Optinal single intermediate report (enables intermediate plot if pruning is used)
                if use_pruning and scores:
                    trial.report(np.mean(scores), step=1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return float(np.mean(scores)) if scores else -1e9

            study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            callbacks = []
            if use_terminator and TERMINATOR_OK:
                terminator = Terminator(
                    improvement_evaluator=BestValueStagnationEvaluator(max_stagnation_trials=10),
                    error_evaluator=StaticErrorEvaluator(constant=0),
                )
                callbacks.append(terminator)
            prog, stat = st.progress(0), st.empty()

            def cb(study, trial):
                prog.progress(min(1.0, len(study.trials) / max(n_trials, 1)))
                best = study.best_value if study.best_trial else float("nan")
                stat.markdown(f"Trial **{len(study.trials)}/{n_trials}** · best `{best:.4f}`")

            callbacks.append(cb)
            study.optimize(
                objective, n_trials=n_trials,
                timeout=None if timeout == 0 else int(timeout),
                callbacks=[cb], show_progress_bar=False,
            )
            st.session_state["optuna_study"] = study
            st.success(f"Best CV score: {study.best_value:.4f}")

        study = st.session_state.get("optuna_study")

        if study and study.best_trial:
            st.metric("Best CV score", f"{study.best_value:.4f}")
            with st.expander("Best hyperparameters", expanded=True):
                st.json(study.best_params)
            vt = st.tabs(["History", "Importance", "Parallel", "Slice", "Intermediate", "EDF", "Timeline", "Contour", "Terminator", "Top trials"])
            with vt[0]: safe_plotly(plot_optimization_history, study)
            with vt[1]: safe_plotly(plot_param_importances, study)
            with vt[2]: safe_plotly(plot_parallel_coordinate, study, params=list(study.best_params.keys()))
            with vt[3]: safe_plotly(plot_slice, study)
            with vt[4]: safe_plotly(plot_intermediate_values, study)
            with vt[5]: safe_plotly(plot_edf, study)
            with vt[6]: safe_plotly(plot_timeline, study)
            with vt[7]: 
                params = list(study.best_params.keys())
                if len(params) >= 2:
                    c1, c2 = st.columns(2)
                    with c1:
                        p1 = st.selectbox(
                            "Contour X",
                            params, 
                            index=0,
                            key="cx"
                        )
                    with c2:
                        p2 = st.selectbox(
                            "Contour Y",
                            params,
                            index=min(1, len(params) - 1),
                            key="cy"
                        )
                    if p1 != p2:
                        safe_plotly(plot_contour, study, params=[p1, p2])
                    else:
                        st.info("Select two different parameters.")
                else:
                    st.info("Need >= 2 hyperparameters.")
            with vt[8]:
               st.subheader("Optuna terminator")
               render_terminator_improvement(study)
            with vt[9]:
                st.subheader("Top trials")
                top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -np.inf, reverse=True)[:10]
                top_df = pd.DataFrame([{
                    "Trial": t.number,
                    "Value": t.value,
                    **t.params
                } for t in top_trials])
                st.dataframe(top_df, use_container_width=True)


            # Rank plot
            st.plotly_chart(plot_rank(study), use_container_width=True)

            # Trials dataframe extras
            trials_df = study.trials_dataframe()
            st.subheader("Trials dataframe")
            cols = [c for c in trials_df.columns if not c.startswith("params_") or c in ["value", "number", "state", "duration"]]
            top = trials_df[trials_df["state"] == "COMPLETE"].sort_values("value", ascending=False).head(10)
            st.dataframe(top[cols], use_container_width=True, height=280)

            if "duration" in trials_df.columns and "value" in trials_df.columns:
                st.subheader("Trial duration vs. CV score")
                df = trials_df[trials_df["state"] == "COMPLETE"].copy()
                df["duration_seconds"] = df["duration"].dt.total_seconds()
                fig = px.scatter(
                    df, x="duration_seconds", y="value", color="state",
                    hover_data=["number"], color_discrete_sequence=["#2563eb", "#f97316", "#6b7280"],
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)



            if st.button("Train production XGBoost", type="primary"):
                best = study.best_params
                if is_classification:
                    y_fit = np.asarray(y).astype(int)
                    model = make_model("XGBoost", best, True, y_fit)
                    model.fit(X, y_fit)
                else:
                    model = make_model("XGBoost", best, False)
                    model.fit(X, y)
                st.session_state["final_model"] = model
                st.session_state["best_params"] = best
                st.session_state["model_type"] = "XGBoost"
                st.success("XGBoost production model trained")
                if JOBLIB_OK:
                    buf = io.BytesIO()
                    joblib.dump(model, buf)
                    st.download_button(
                        "Download model (.joblib)", buf.getvalue(),
                        "xgb_final_model.joblib", "application/octet-stream",
                    )

# ==============================================================
# TAB 5 — RESULTS (includes train error summary)
# ==============================================================
with tabs[4]:
    st.subheader("Model results")
    model = st.session_state.get("final_model")
    if model is None:
        st.warning("Train or load a model first.")
    else:
        X, y = st.session_state["X"], st.session_state["y"]
        is_classification = st.session_state["is_classification"]
        st.caption(f"Model: **{st.session_state.get('model_type', '—')}**")
        with st.expander("Hyperparameters"):
            st.json(st.session_state.get("best_params", {}))

        if st.session_state.get("cv_summary"):
            s = st.session_state["cv_summary"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CV Acc", f"{s.get('mean_acc', np.nan):.3f}")
            c2.metric("CV AUC", f"{s.get('mean_auc', np.nan):.3f}")
            c3.metric("CV F1", f"{s.get('mean_f1', np.nan):.3f}")
            c4.metric("CV MSE", f"{s.get('mse', np.nan):.3f}")
            c5.metric("CV Train error", f"{s.get('train_error', np.nan):.3f}")

        pred = model.predict(X)
        if is_classification and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                p1 = proba[:, 1]
                thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
                pred_t = (p1 >= thr).astype(int)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{accuracy_score(y, pred_t):.3f}")
                c2.metric("Precision", f"{precision_score(y, pred_t, zero_division=0):.3f}")
                c3.metric("Recall", f"{recall_score(y, pred_t, zero_division=0):.3f}")
                c4.metric("F1", f"{f1_score(y, pred_t, zero_division=0):.3f}")
                # in-sample train error diagnostic
                st.metric("In-sample train error (1−acc)", f"{1 - accuracy_score(y, pred):.3f}")
                if PLOTLY_OK:
                    cm = confusion_matrix(y, pred_t)
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                    title=f"Confusion matrix @ {thr:.2f}")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
                st.metric("Train error (1−acc)", f"{1 - accuracy_score(y, pred):.3f}")
        else:
            mse = mean_squared_error(y, pred)
            c1, c2, c3 = st.columns(3)
            c1.metric("MSE", f"{mse:.4f}")
            c2.metric("RMSE", f"{np.sqrt(mse):.4f}")
            c3.metric("Train error (MSE)", f"{mse:.4f}")

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({
                "feature": st.session_state["feat_names"],
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).head(20)
            if PLOTLY_OK:
                fig = px.bar(imp, x="importance", y="feature", orientation="h",
                             color_discrete_sequence=["#1d4ed8"], title="Feature importance")
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, height=480,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# TAB 6 — CALIBRATION
# ==============================================================
with tabs[5]:
    st.subheader("Probability calibration")
    if not st.session_state.get("is_classification"):
        st.info("Available for classification targets.")
    elif "oof_prob" not in st.session_state:
        st.warning("Run modeling tab first to collect out-of-fold probabilities.")
    else:
        y_true = np.asarray(st.session_state["oof_true"]).reshape(-1)
        p_raw = np.asarray(st.session_state["oof_prob"]).reshape(-1)
        n = min(len(y_true), len(p_raw))
        y_true, p_raw = y_true[:n], p_raw[:n]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_raw, y_true)
        p_cal = iso.predict(p_raw)
        st.session_state["iso_model"] = iso

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        for name, p, color in [("Raw", p_raw, "#2563eb"), ("Isotonic", p_cal, "#059669")]:
            frac, meanp = calibration_curve(y_true, p, n_bins=8, strategy="quantile")
            axes[0].plot(meanp, frac, "o-", label=name, color=color)
        axes[0].set_title("Reliability diagram")
        axes[0].legend(frameon=False)
        axes[0].grid(True, alpha=0.25)
        axes[1].hist(p_raw, bins=20, alpha=0.55, label="Raw", color="#2563eb")
        axes[1].hist(p_cal, bins=20, alpha=0.55, label="Isotonic", color="#059669")
        axes[1].legend(frameon=False)
        axes[1].set_title("Probability distribution")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier (raw)", f"{brier_score_loss(y_true, p_raw):.4f}")
        c2.metric("Brier (isotonic)", f"{brier_score_loss(y_true, p_cal):.4f}")
        try:
            c3.metric("AUC (raw)", f"{roc_auc_score(y_true, p_raw):.4f}")
            c4.metric("AUC (isotonic)", f"{roc_auc_score(y_true, p_cal):.4f}")
        except Exception:
            pass

# ==============================================================
# TAB 7 — SHAP SUITE (waterfall, beeswarm, bar, dependence)
# ==============================================================
with tabs[6]:
    st.subheader("SHAP explainability suite")
    model = st.session_state.get("final_model")
    if not SHAP_OK:
        st.error("Install: pip install shap")
    elif model is None or "X" not in st.session_state:
        st.warning("Train a model and run windowing first.")
    else:
        X = st.session_state["X"]
        is_classification = st.session_state["is_classification"]
        class_index = st.number_input("Class index (binary → 1)", 0, 20, 1) if is_classification else 0
        sample_n = st.slider("Sample size for global plots", 40, min(250, len(X)), min(100, len(X)))

        if st.button("Compute SHAP values", type="primary"):
            with st.spinner("TreeSHAP…"):
                Xs = X.sample(sample_n, random_state=42) if len(X) > sample_n else X.copy()
                explainer = shap.TreeExplainer(model)
                exp = explainer(Xs)
                values = normalize_shap_values(
                    exp.values if hasattr(exp, "values") else exp,
                    prefer_class=int(class_index),
                )
                st.session_state["shap_values"] = values
                st.session_state["shap_X"] = Xs
                st.session_state["shap_explainer"] = explainer
                st.session_state["shap_exp_obj"] = exp
            st.success(f"SHAP ready for {len(Xs)} rows × {values.shape[1]} features")

        if "shap_values" in st.session_state:
            values = st.session_state["shap_values"]
            Xs = st.session_state["shap_X"]
            names = list(Xs.columns)

            shap_tabs = st.tabs([
                "Waterfall (local)",
                "Beeswarm",
                "Bar (global)",
                "Dependence",
            ])

            with shap_tabs[0]:
                row_i = st.slider("Row in SHAP sample", 0, len(Xs) - 1, 0, key="wf_row")
                # rebuild single explanation
                base = st.session_state["shap_exp_obj"].base_values
                base = np.array(base)
                if base.ndim == 2:
                    cls = int(class_index) if base.shape[-1] > class_index else 0
                    base_val = float(base[row_i, cls])
                else:
                    base_val = float(np.ravel(base)[row_i] if len(np.ravel(base)) > row_i else np.ravel(base)[0])
                exp_plot = shap.Explanation(
                    values=values[row_i],
                    base_values=base_val,
                    data=Xs.values[row_i],
                    feature_names=names,
                )
                fig, ax = plt.subplots(figsize=(10, 5.5))
                try:
                    shap.plots.waterfall(exp_plot, max_display=12, show=False)
                except Exception:
                    order = np.argsort(np.abs(values[row_i]))[::-1][:12]
                    ax.barh([names[i] for i in order][::-1], [values[row_i, i] for i in order][::-1])
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

            with shap_tabs[1]:
                st.caption("Beeswarm: each dot is one sample; color = feature value")
                fig, ax = plt.subplots(figsize=(10, 6))
                try:
                    shap.summary_plot(values, Xs, feature_names=names, show=False, max_display=18)
                except Exception as e:
                    st.warning(str(e))
                    mean_abs = np.abs(values).mean(0)
                    order = np.argsort(mean_abs)[::-1][:18]
                    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1])
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

            with shap_tabs[2]:
                mean_abs = np.abs(values).mean(axis=0)
                gdf = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
                gdf = gdf.sort_values("mean_abs_shap", ascending=False)
                if PLOTLY_OK:
                    fig = px.bar(
                        gdf.head(20), x="mean_abs_shap", y="feature", orientation="h",
                        color_discrete_sequence=["#1d4ed8"], title="Global mean |SHAP|",
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"}, height=500,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(gdf.head(25), use_container_width=True)

            with shap_tabs[3]:
                feat = st.selectbox("Feature", names, key="dep_feat")
                color = st.selectbox("Color by (interaction hint)", ["auto"] + names, key="dep_color")
                fig, ax = plt.subplots(figsize=(9, 5))
                shap.dependence_plot(
                    feat, values, Xs, feature_names=names,
                    interaction_index=None if color == "auto" else color,
                    ax=ax, show=False,
                )
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

# ==============================================================
# TAB 8 — SHAP INTERACTIONS (combined feature effects)
# ==============================================================
with tabs[7]:
    st.subheader("SHAP interactions · combined feature effects")
    st.caption(
        "Shows which **pairs of features** most strongly interact in the model "
        "(extra effect beyond main effects alone)."
    )
    model = st.session_state.get("final_model")
    if not SHAP_OK:
        st.error("Install: pip install shap")
    elif model is None or "X" not in st.session_state:
        st.warning("Need trained model + windowed X.")
    else:
        X = st.session_state["X"]
        is_classification = st.session_state["is_classification"]
        class_index = st.number_input("Class index", 0, 20, 1, key="inter_cls") if is_classification else 0
        sample_n = st.slider("Sample size", 30, min(120, len(X)), min(60, len(X)), key="inter_n")

        if st.button("Compute interaction values", type="primary"):
            with st.spinner("SHAP interaction values (can take a minute)…"):
                Xs = X.sample(sample_n, random_state=42) if len(X) > sample_n else X.copy()
                explainer = shap.TreeExplainer(model)
                # main effects
                exp = explainer(Xs)
                main = normalize_shap_values(
                    exp.values if hasattr(exp, "values") else exp,
                    prefer_class=int(class_index),
                )
                # interactions
                inter = explainer.shap_interaction_values(Xs)
                inter = np.array(inter)
                if inter.ndim == 4:
                    cls = int(class_index) if inter.shape[-1] > class_index else 0
                    inter = inter[:, :, :, cls]
                elif isinstance(inter, list):
                    inter = np.array(inter[int(class_index) if class_index < len(inter) else 0])

                names = list(Xs.columns)
                mean_abs = np.abs(inter).mean(axis=0)  # (F, F)
                rows = []
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        rows.append({
                            "feature_1": names[i],
                            "feature_2": names[j],
                            "mean_abs_interaction": float(mean_abs[i, j]),
                        })
                pairs = (
                    pd.DataFrame(rows)
                    .sort_values("mean_abs_interaction", ascending=False)
                    .reset_index(drop=True)
                )
                st.session_state["inter_main"] = main
                st.session_state["inter_matrix"] = inter
                st.session_state["inter_X"] = Xs
                st.session_state["inter_pairs"] = pairs
                st.session_state["inter_names"] = names
            st.success(f"Top interactions ranked for {len(Xs)} samples")

        if "inter_pairs" in st.session_state:
            pairs = st.session_state["inter_pairs"]
            st.markdown("### Top interacting feature pairs")
            top_k = st.slider("Show top K pairs", 5, 30, 15)
            st.dataframe(
                pairs.head(top_k).style.format({"mean_abs_interaction": "{:.5f}"}),
                use_container_width=True,
            )
            if PLOTLY_OK:
                show = pairs.head(top_k).copy()
                show["pair"] = show["feature_1"] + " × " + show["feature_2"]
                fig = px.bar(
                    show, x="mean_abs_interaction", y="pair", orientation="h",
                    title="Strongest pairwise interactions",
                    color_discrete_sequence=["#7c3aed"],
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}, height=max(360, 22 * top_k),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Dependence with interaction coloring")
            names = st.session_state["inter_names"]
            # default to top pair
            default_f = pairs.iloc[0]["feature_1"] if len(pairs) else names[0]
            default_c = pairs.iloc[0]["feature_2"] if len(pairs) else names[min(1, len(names)-1)]
            f1 = st.selectbox(
                "Main feature", names,
                index=names.index(default_f) if default_f in names else 0, key="int_f1",
            )
            f2 = st.selectbox(
                "Interaction color feature", names,
                index=names.index(default_c) if default_c in names else 0, key="int_f2",
            )
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.dependence_plot(
                f1,
                st.session_state["inter_main"],
                st.session_state["inter_X"],
                feature_names=names,
                interaction_index=f2,
                ax=ax,
                show=False,
            )
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.download_button(
                "Download interaction pairs CSV",
                pairs.to_csv(index=False).encode(),
                "shap_interaction_pairs.csv",
                "text/csv",
            )

# ==============================================================
# TAB 9 — PLAYGROUND
# ==============================================================
with tabs[8]:
    st.subheader("What-if playground")
    model = st.session_state.get("final_model")
    if model is None or "X" not in st.session_state:
        st.warning("Train a model first.")
    else:
        X = st.session_state["X"]
        feat_names = st.session_state["feat_names"]
        is_classification = st.session_state["is_classification"]
        base = X.iloc[st.number_input("Template row", 0, len(X) - 1, 0)]
        edit = feat_names[: min(12, len(feat_names))]
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(edit):
            with cols[i % 3]:
                lo, hi = float(X[f].quantile(0.05)), float(X[f].quantile(0.95))
                if lo == hi:
                    hi = lo + 1.0
                default = float(np.clip(base[f], lo, hi))
                values[f] = st.slider(f, lo, hi, default, key=f"pg_{f}")
        x_in = base.copy()
        for f, v in values.items():
            x_in[f] = v
        x_df = pd.DataFrame([x_in])[feat_names]
        pred = model.predict(x_df)[0]
        if is_classification:
            st.metric("Predicted class", f"{int(pred)}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_df)[0]
                if PLOTLY_OK:
                    fig = px.bar(
                        x=list(range(len(proba))), y=proba,
                        labels={"x": "Class", "y": "Probability"},
                        color_discrete_sequence=["#2563eb"], range_y=[0, 1],
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Predicted value", f"{float(pred):.4f}")

# ==============================================================
# TAB 10 — EXPORT
# ==============================================================
with tabs[9]:
    st.subheader("Export experiment pack")
    notes = st.text_area("Notes", "Activity windows predicting mental-health targets")
    if st.button("Build export ZIP", type="primary"):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
            if "analysis_table" in st.session_state:
                z.writestr("analysis_table.csv", st.session_state["analysis_table"].to_csv(index=False))
            if "cv_fold_df" in st.session_state:
                z.writestr("cv_folds.csv", st.session_state["cv_fold_df"].to_csv(index=False))
            if "cv_summary" in st.session_state:
                z.writestr("cv_summary.csv", pd.DataFrame([st.session_state["cv_summary"]]).to_csv(index=False))
            if "best_params" in st.session_state:
                z.writestr("best_params.csv", pd.DataFrame([st.session_state["best_params"]]).to_csv(index=False))
            if "inter_pairs" in st.session_state:
                z.writestr("shap_interactions.csv", st.session_state["inter_pairs"].to_csv(index=False))
            z.writestr("notes.txt", notes)
            if JOBLIB_OK and "final_model" in st.session_state:
                buf = io.BytesIO()
                joblib.dump(st.session_state["final_model"], buf)
                z.writestr("final_model.joblib", buf.getvalue())
        st.download_button(
            "Download ZIP",
            mem.getvalue(),
            f"mh_studio_export_{datetime.now():%Y%m%d_%H%M}.zip",
            "application/zip",
        )


# ==============================================================
# TAB 11 — Machine Learning algorithms and tools
# ==============================================================
with tabs[10]:
    st.subheader("Machine Learning algorithms and tools")

    st.markdown(
        """
        Practical overview of suitable machine learning algorithms and tools.
        """
        )

    st.subheader("Core Supervised Algorithms")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Algorithms _ Best for _ Strengths in our case _ Weaknesses
        </div>
        """,
        unsafe_allow_html=True
        )
    

    st.markdown(
    """
    1. Random Forest - Classification & Regression - Robust, handles non-linear relations, good baseline, build-in importance -  Can be based toward high-cardinality features

    2. XGBoost - Classification & Regression - Usually strongest tabular performance, handles missing values well - Need more tuning, less interpretable than linear models

    3. LightGBM - Large datasets - Very fast, excellent accuracy, low memory - Slightly harder to tune

    4. CatBoost - Mixed feature types - Great with categorical data, lessoverfiltering - Heavier dependency

    5. Logical / Linear Regression - Simple baseline - Highly interpretable, fast - Cannot capture complex interactions

    6. Elastic Net / Lasso - Feature selection - Automatically select important activity features - Linear only

    7. HistGradienBoosting (sklearn) - Medium/large data - Fast, native missing-value support, no extra library - Slightly less accurate than XGBoost on average

    Random Forest & XGBoost are our main models, and LightGBM is optionally third strong candidate.
    """)

    st.subheader("Time-Series / Sequence-Aware Methods")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Method _ When to use _ Notes
        </div>
        """,
        unsafe_allow_html=True
        )


    st.markdown(
    """  
    1. LSTM /GRU (deep learning) - Longer sequences, complex temporal patterns - Needs more data, harder to interpret

    2. Temporal Convolutinal Networks (TCN) - Sequential activity patterns - Often better than LSTM on tabular time series

    3. TSFresh + classical ML - Automatic time-series feature extraction - Can replace natural mean/std/min/max features

    4. Rocket / MiniRocket - Fast time-series classification - Very strong baseline for activity sequences

    5. Prophet / classical forecasting - Pure next-day forecasting of energy/steps - Less suitable for PHQ-9/GAD-7 style targets

    6. This application is focused on windowedtabular features derived from daily summaries + tree models, so we do not implement deep sequence models here. 
    It is possible to use LSTM/GRU or TCN on the raw daily sequences, but it requires more data and more complex modeling.
    
    Activity over time is compressed into windowed engineered features (mean, std, min, max, etc.) and then used in tree-based models (Random Forest, XGBoost, LightGBM) for prediction.

    """)

    st.subheader("Specialized / Advanced Approaches")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Approach _ Purpose
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
    """   
    1. Hierarchical / Mixed effects models - Account for repeated measures per participant

    2. Multi-task learning - Predict PHQ-9, GAD-7 and energy score at the same time

    3. Original regression - When targets are severity categories (minimal -> Severe)

    4. Survival models (Cox, Random Suvival Forest) - Time-to-event (e.g. time until high PHQ-9 episode)

    5. Anomaly detection (Isolation Forest Autoencoders) - Detect unusual activity patterns that precede most changes

    6. Clustering (K-Means, HDBSCAN) - Discover participant behavioral phenotypes
    """)

    st.subheader("Feature Engineering & Selection Tools")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Tool _ Role
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
    """
    1. SHAP - Best explanation of feature impact + interactions

    2. Permutation Importance - Reliable model-agnostic ranking

    3. Boruta / Boruta-SHAP - Automatic all-relevant feature selection

    4. TSFresh or tsflex - Automatic extraction of many time-series features

    5. scikit-learn (SelectFromModel) - Single importance-based selection
    """)

    st.subheader("Evaluation & Validation Tools")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Tool _ Use
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
    """
    1. Stratified GroupKFold - Very important - keeps all data from one paticipant in the same fold (avoids leakage)

    2. TimeSeriesSplit - Respects temporal order

    3. Optuna / Hyperopt - Hyperparameter optimization

    - Optuna is more flexible and has better visualization tools, but Hyperopt is simpler to use for basic tasks.
    - Optuna can also be used to optimize multiple objectives (e.g. accuracy + AUC) at the same time, which is useful for multi-task learning.
    - Optuna also has a pruning feature that can stop unpromising trials early, saving time and resources.
    - Optuna tipically uses fANOVA-style importance ranking to show which hyperparameters are most important for the model performance, 
        which can help guide future tuning (via a surrogate model over completed trials).
        It's empirical from the study, not theoretical ranking of XGBoost hyperparameters. 
    For example, it may find that learning_rate is more important than max_depth for a given dataset, 
    even though max_depth is often considered a more important hyperparameter in general.
    - Optuna narrows the search through widen or refine ranges of hyperparameters based on previous trials or shrink low-importance ones, 
    which can lead to better performance than a fixed grid search.
    If everything is equal, Optuna is more efficient than Hyperopt because it uses a tree-structured Parzen estimator (TPE) to model the objective function,
    which can find good hyperparameters faster than random search or grid search.
    Terminator plot shows how many trials were pruned and how many completed, which can help diagnose if the search space is too large or if the model is overfitting.
    It estimates remaining improvement based on the best trial so far, which can help decide when to stop the optimization early.
    Practical tip: Pattern shows 'High values early' that the serch still has room to improve, while 'Flat' pattern shows that the search is converging and may be close to the best solution.
    - Optuna visualization tools include:
    - Hyperparameter importance plot: Shows which hyperparameters are most important for the model performance,
    - Parallel coordinate plot: Shows how different hyperparameter combinations affect the model performance,
    - Contour plot: Shows the relationship between two hyperparameters and the model performance,
    - Slice plot: Shows how the model performance changes as one hyperparameter is varied while keeping the others fixed.
    - Optimization history plot: Shows how the model performance changes over the course of the optimization trials,
    - Pruning plot: Shows how many trials were pruned and how many completed, which can help diagnose if the search space is too large or if the model is overfitting.
    - Remaining improvement plot: Shows the estimated remaining improvement based on the best trial so far.
    - Terminator plot: Shows how many trials were pruned and how many completed, which can help diagnose if the search space is too large or if the model is overfitting.


    4. scikit-learn metrics - Accuracy, F1, AUC, MSE, MAE, calibration

    Classification metrics: Accuracy, F1, AUC, Precision, Recall, Confusion matrix
    - Accuracy is the proportion of correct predictions out of all predictions made. It is a simple and intuitive metric, but it can be misleading if the classes are imbalanced.
    - F1 score is the harmonic mean of precision and recall. It is a better metric for imbalanced classes, as it takes both false positives and false negatives into account.
    - AUC (Area Under the Curve) is a metric that measures the ability of the model to distinguish between classes. It is a good metric for imbalanced classes, as it is not affected by the class distribution.
    - Precision is the proportion of true positives out of all positive predictions made. It is a good metric for imbalanced classes, as it focuses on the positive class.
    - Recall is the proportion of true positives out of all actual positives. It is a good metric for imbalanced classes, as it focuses on the positive class.
    - Confusion matrix is a table that shows the number of true positives, true negatives, false positives, and false negatives. It is a good way to visualize the performance of the model and identify areas for improvement.
    - ROC AUC curve is a plot that shows the trade-off between true positive rate and false positive rate at different classification thresholds. It is a good way to visualize the performance of the model and identify the optimal threshold for classification.
    - Brier score is a metric that measures the accuracy of probabilistic predictions. It is a good metric for imbalanced classes, as it takes into account the predicted probabilities and not just the predicted classes.

    Regression metrics: MSE, RMSE, MAE, R2
    - MSE (Mean Squared Error) is the average of the squared differences between the predicted and actual values. It is a good metric for regression tasks, as it penalizes larger errors more heavily than smaller errors.
    - RMSE (Root Mean Squared Error) is the square root of the MSE. It is a good metric for regression tasks, as it is in the same units as the predicted and actual values, making it easier to interpret.
    - MAE (Mean Absolute Error) is the average of the absolute differences between the predicted and actual values. It is a good metric for regression tasks, as it is less sensitive to outliers than MSE and RME.
    - R2 (Coefficient of Determination) is a metric that measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It is a good metric for
    regression tasks, as it provides a measure of how well the model fits the data.


    As we have multiple rows per participant, GroupKFold (grouped by participant_id) method is more preferable instead of ordinary KFold in order to get realistic performance estimates
    """)

    st.subheader("Deployment & Interface Tools")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Tool _ Role
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
    """ 
    1. Streamlit - Interactive dashboard

    2. Plotly - Interactive charts

    3. SHAP + matplotlib - Dependence plots, waterfall, beeswarm

    SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of machine learning models. 
    It connects optimal credit allocation with local explanations using the classic Shapley values from cooperative game theory 
    and their related extensions.
    - Positive SHAP values indicate that a feature is pushing the prediction higher, while negative SHAP values indicate that a feature is pushing the prediction lower.
    - SHAP values can be used to explain individual predictions (local explanations) or to understand the overall behavior of the model (global explanations).
    - SHAP values can be computed for any machine learning model, but they are particularly useful for tree-based models like Random Forest and XGBoost.
    - SHAP values can be visualized using various plots, such as waterfall plots, beeswarm plots, and dependence plots, which help to understand the impact of each feature on the model's predictions.
    - SHAP is model-agnostic, meaning it can be applied to any machine learning model, but it is especially efficient for tree-based models due to the TreeSHAP algorithm.
    

    4. joblib / pickle - Save trained models

    5. FastAPI (optinal) - Serve predictions as an API 

    """)


    st.subheader("SHAP plotted for local or global understanding")

    st.markdown(
        """
        <div style="
        background-color: #eaf2f8;
        padding: 12px 16px;
        border-radius: 8px;
        color: #1a5276;
        front-size: 700px;
        front-weight: bold;
        ">
        Global _ Use
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown(
    """ 
    1. Explain one prediction - Waterfall / force

    2. Rank features globally - Bar / beeswarm

    3. See shape of effect - Dependence

    4. See feature pairs - Interaction values + dependence color

    5. Algorithm for RF / XGB - TreeSHAP

    """)

# ==============================================================
# TAB 12 — ABOUT
# ==============================================================
with tabs[11]:
    st.subheader("About This Dashboard")
    st.image("/Users/alice/Downloads/IMG_5558.jpeg", caption="Dashboard Image")
    st.markdown("""
        This dashboard was developed to provide insights into medical data using machine learning techniques, specifically Random Forest models.
        It includes features for data exploration, prediction, and visualization.
    

    ** Main capabilities of the dashboard include:
    - **Data Exploration**: Explore participant data and visualize trends.
    - **Prediction Tool**: Build and evaluate Random Forest models for health outcome predictions.  
    - **Feature Importance**: Analyze which features are most influential in the model's predictions.


    ** Technologies Used:
    - **Streamlit**: For building the interactive web application.
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical computations.
    - **Plotly**: For interactive visualizations.


    ** Machine Learning:
    - **Random Forest**: An ensemble learning method used for classification and regression tasks.


    **Typical data sources include:
    - Electronic Health Records (EHR)
    - Medical records and patient surveys
    - Daily health monitoring data (steps, heart rate, exercise data from wearable devices)
    - Sleep patterns and quality data
    - Blood pressure, blood tests, and glucose level readings
    - Dietary and nutrition information

    """)

st.markdown("""
<div class="footer-note">
  Mental Health Analytics Studio · Research and decision-support tool · Not a medical diagnostic device
</div>
""", unsafe_allow_html=True)
