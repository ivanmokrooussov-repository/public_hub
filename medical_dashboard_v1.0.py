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
