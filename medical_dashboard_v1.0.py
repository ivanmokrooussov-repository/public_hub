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
