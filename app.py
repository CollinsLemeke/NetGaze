import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    RandomizedSearchCV, StratifiedKFold, KFold, learning_curve
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, PowerTransformer
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score,
    explained_variance_score, max_error, median_absolute_error,
    roc_auc_score, log_loss, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import warnings
import io
import json
import time

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGaze ML Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS — Dark scientific theme with teal/amber accents
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-card-hover: #1f2b3d;
    --border-color: #2a3a50;
    --text-primary: #e8ecf1;
    --text-secondary: #8899aa;
    --accent-teal: #00d4aa;
    --accent-teal-dim: rgba(0, 212, 170, 0.15);
    --accent-amber: #f59e0b;
    --accent-amber-dim: rgba(245, 158, 11, 0.15);
    --accent-rose: #f43f5e;
    --accent-rose-dim: rgba(244, 63, 94, 0.15);
    --accent-blue: #3b82f6;
    --accent-blue-dim: rgba(59, 130, 246, 0.15);
    --accent-violet: #8b5cf6;
    --accent-violet-dim: rgba(139, 92, 246, 0.15);
    --accent-emerald: #10b981;
    --accent-cyan: #06b6d4;
    --accent-pink: #ec4899;
    --accent-lime: #84cc16;
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Global ── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111827 100%) !important;
    border-right: 1px solid var(--border-color) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-teal) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
    margin-top: 1.2rem;
}

/* ── Cards ── */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 1rem;
}
div[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ── Metric Cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent-teal) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-teal), #00b894) !important;
    color: #0a0f1a !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 12px rgba(0, 212, 170, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 212, 170, 0.5) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-amber), #d97706) !important;
    color: #0a0f1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(245, 158, 11, 0.3) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}
.stSlider > div > div > div { background: var(--accent-teal) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-teal-dim) !important;
    color: var(--accent-teal) !important;
}

/* ── Dataframes ── */
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden; }

/* ── Checkbox ── */
.stCheckbox label span { color: var(--text-primary) !important; }

/* ── Success / Warning / Info ── */
.stSuccess { background: rgba(0, 212, 170, 0.1) !important; border-left: 4px solid var(--accent-teal) !important; }
.stWarning { background: rgba(245, 158, 11, 0.1) !important; border-left: 4px solid var(--accent-amber) !important; }
.stInfo { background: rgba(59, 130, 246, 0.1) !important; border-left: 4px solid var(--accent-blue) !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1525 0%, #1a2332 50%, #0d2818 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    color: #fff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-banner .accent { color: var(--accent-teal); }
.hero-banner p {
    color: var(--text-secondary);
    font-size: 1rem;
    margin: 0;
    max-width: 600px;
}

/* ── Step indicator ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--accent-teal-dim);
    color: var(--accent-teal);
    padding: 6px 14px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}
.step-badge.amber { background: var(--accent-amber-dim); color: var(--accent-amber); }
.step-badge.rose { background: var(--accent-rose-dim); color: var(--accent-rose); }
.step-badge.blue { background: var(--accent-blue-dim); color: var(--accent-blue); }
.step-badge.violet { background: var(--accent-violet-dim); color: var(--accent-violet); }

/* ── Section headers ── */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.5rem 0 1rem 0;
}

/* ── Stat grid ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1.2rem;
    text-align: center;
}
.stat-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent-teal);
}
.stat-card .label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border-color);
    margin: 2rem 0;
}

/* ── Quality cards ── */
.quality-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.quality-card .title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95rem;
    margin-bottom: 0.3rem;
}
.quality-card .desc {
    color: var(--text-secondary);
    font-size: 0.82rem;
    line-height: 1.4;
}

/* ── Progress bar ── */
.progress-container {
    background: var(--bg-secondary);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.progress-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* ── Insight card ── */
.insight-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(0,212,170,0.05) 100%);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.insight-card .icon { font-size: 1.2rem; margin-right: 0.5rem; }
.insight-card .text { color: var(--text-primary); font-size: 0.9rem; }

/* ── Hint banner ── */
.hint-banner {
    background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.hint-banner .hint-icon {
    font-size: 1.1rem;
    margin-top: 1px;
    flex-shrink: 0;
}
.hint-banner .hint-body {
    color: #c0cde0;
    font-size: 0.84rem;
    line-height: 1.55;
}
.hint-banner .hint-body strong {
    color: #e8ecf1;
}

/* ── Summary recommendation card ── */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: rgba(0,212,170,0.4); }
.rec-card .rec-icon { font-size: 1.5rem; flex-shrink: 0; margin-top: 2px; }
.rec-card .rec-content { flex: 1; }
.rec-card .rec-title {
    color: #e8ecf1;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.rec-card .rec-type {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 10px;
    margin-bottom: 6px;
}
.rec-card .rec-desc {
    color: #8899aa;
    font-size: 0.82rem;
    line-height: 1.5;
}
.rec-card .rec-action {
    color: #00d4aa;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Hide default streamlit elements ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,35,50,0.6)",
    font=dict(family="DM Sans", color="#e8ecf1", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#2a3a50", zerolinecolor="#2a3a50"),
    yaxis=dict(gridcolor="#2a3a50", zerolinecolor="#2a3a50"),
    colorway=["#00d4aa", "#f59e0b", "#3b82f6", "#f43f5e", "#8b5cf6",
              "#06b6d4", "#ec4899", "#84cc16", "#10b981", "#f97316"],
    hoverlabel=dict(bgcolor="#1a2332", font_color="#e8ecf1", bordercolor="#2a3a50"),
)

PALETTE_VIVID = ["#00d4aa", "#f59e0b", "#3b82f6", "#f43f5e", "#8b5cf6",
                 "#06b6d4", "#ec4899", "#84cc16", "#10b981", "#f97316"]
PALETTE_GRADIENT_TEAL = ["#064e3b", "#065f46", "#047857", "#059669",
                         "#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#00d4aa"]
PALETTE_SUNSET = ["#1e1b4b", "#312e81", "#4338ca", "#6366f1", "#818cf8",
                  "#f59e0b", "#f97316", "#ef4444", "#f43f5e"]
PALETTE_OCEAN = ["#0c4a6e", "#075985", "#0369a1", "#0284c7", "#0ea5e9",
                 "#38bdf8", "#7dd3fc", "#00d4aa", "#06b6d4"]

HEATMAP_SCALE = [[0, "#0a0f1a"], [0.25, "#1e3a5f"], [0.5, "#0284c7"],
                 [0.75, "#00d4aa"], [1, "#f59e0b"]]
DIVERGING_SCALE = [[0, "#f43f5e"], [0.5, "#1a2332"], [1, "#00d4aa"]]
CORRELATION_SCALE = [[0, "#f43f5e"], [0.25, "#b91c4c"], [0.5, "#1a2332"],
                     [0.75, "#047857"], [1, "#00d4aa"]]


CLASSIFICATION_MODELS = {
    "Logistic Regression": "Linear model for binary/multi-class classification. Fast, interpretable, good baseline.",
    "Random Forest": "Ensemble of decision trees. Robust, handles non-linear data, provides feature importance.",
    "XGBoost": "Gradient boosted trees. High accuracy, handles missing data, great for competitions.",
    "Decision Tree": "Single tree model. Highly interpretable, visualisable, prone to overfitting.",
    "Gradient Boosting": "Sequential ensemble method. Strong accuracy, slower to train than RF.",
    "AdaBoost": "Adaptive boosting. Combines weak learners, good for balanced datasets.",
    "Extra Trees": "Extremely Randomised Trees. Faster than RF, reduces variance further.",
    "SVM": "Support Vector Machine. Effective in high-dimensional spaces, works well with clear margins.",
    "KNN": "K-Nearest Neighbours. Instance-based, no training phase, sensitive to feature scaling.",
    "Naive Bayes": "Probabilistic classifier. Very fast, works well with text data and small datasets.",
    "Neural Network": "Deep learning model. Flexible architecture, learns complex patterns, needs more data."
}

REGRESSION_MODELS = {
    "Linear Regression": "Simple linear model. Fast, interpretable, assumes linear relationship.",
    "Ridge Regression": "Linear with L2 regularisation. Handles multicollinearity, prevents overfitting.",
    "Lasso Regression": "Linear with L1 regularisation. Feature selection built-in, sparse solutions.",
    "Elastic Net": "Combines L1 and L2 regularisation. Balanced between Ridge and Lasso.",
    "Random Forest": "Ensemble of decision trees. Handles non-linear relationships, robust to outliers.",
    "XGBoost": "Gradient boosted trees. Top accuracy, handles complex patterns and missing data.",
    "Decision Tree": "Single tree regressor. Interpretable, captures non-linear patterns, can overfit.",
    "Gradient Boosting": "Sequential boosting. Strong performance, slower training, many hyperparameters.",
    "AdaBoost": "Adaptive boosting for regression. Good ensemble method, less prone to overfitting.",
    "Extra Trees": "Extremely Randomised Trees. Fast, handles noise well.",
    "SVR": "Support Vector Regression. Works well in high dimensions, flexible kernel choices.",
    "KNN": "K-Nearest Neighbours. Simple, non-parametric, sensitive to feature scaling.",
    "Neural Network": "Deep learning model. Learns complex non-linear mappings, needs more data."
}


# ─────────────────────────────────────────────────────────
# UI HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def badge(text, color="teal"):
    st.markdown(f'<div class="step-badge {color}">{text}</div>', unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def insight_card(icon, text):
    st.markdown(f'''<div class="insight-card">
        <span class="icon">{icon}</span>
        <span class="text">{text}</span>
    </div>''', unsafe_allow_html=True)


def hint_banner(icon, text):
    """Render a contextual hint/guide banner."""
    st.markdown(f'''<div class="hint-banner">
        <span class="hint-icon">{icon}</span>
        <div class="hint-body">{text}</div>
    </div>''', unsafe_allow_html=True)


def recommendation_card(icon, title, type_label, type_color, desc, action):
    """Render a preprocessing recommendation card."""
    st.markdown(f'''<div class="rec-card">
        <div class="rec-icon">{icon}</div>
        <div class="rec-content">
            <div class="rec-title">{title}</div>
            <div class="rec-type" style="background:rgba({_hex_to_rgb(type_color)},0.15); color:{type_color};">{type_label}</div>
            <div class="rec-desc">{desc}</div>
            <div class="rec-action">➤ {action}</div>
        </div>
    </div>''', unsafe_allow_html=True)


def _hex_to_rgb(hex_color):
    """Convert #RRGGBB to 'R,G,B' string."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ─────────────────────────────────────────────────────────
# DATA HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def get_dtype_summary(df):
    numeric = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    boolean = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return numeric, categorical, boolean, datetime


def auto_detect_task(y):
    if y.dtype == "object" or y.dtype.name == "category":
        return "Classification"
    nunique = y.nunique()
    if nunique <= 20 and y.dtype in ["int64", "int32"]:
        return "Classification"
    return "Regression"


def compute_data_quality_score(df):
    scores = []
    completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    scores.append(completeness * 40)
    dup_ratio = 1 - df.duplicated().sum() / max(len(df), 1)
    scores.append(dup_ratio * 20)
    n_types = df.dtypes.nunique()
    scores.append(min(n_types / 3, 1) * 10)
    card_scores = []
    for col in df.columns:
        ratio = df[col].nunique() / max(len(df), 1)
        if 0.001 < ratio < 0.95:
            card_scores.append(1)
        elif ratio <= 0.001:
            card_scores.append(0.5)
        else:
            card_scores.append(0.3)
    scores.append(np.mean(card_scores) * 15 if card_scores else 0)
    row_score = min(len(df) / 1000, 1)
    scores.append(row_score * 15)
    return min(sum(scores), 100)


def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((series < lower) | (series > upper)).sum(), lower, upper


# ─────────────────────────────────────────────────────────
# SMART COLUMN DETECTION & TRANSFORM ENGINE
# ─────────────────────────────────────────────────────────
SMART_TYPE_ICONS = {
    "pay_range": "💰", "currency_value": "💲", "percentage": "📊",
    "date_string": "📅", "text_description": "📝", "url": "🔗",
    "email": "📧", "numeric": "🔢", "categorical": "🏷️",
    "boolean": "✅", "id_column": "🔑",
    "high_cardinality_categorical": "🎯",
    "low_cardinality_categorical": "📦",
}

SMART_TYPE_COLORS = {
    "pay_range": "#f59e0b", "currency_value": "#f59e0b", "percentage": "#3b82f6",
    "date_string": "#8b5cf6", "text_description": "#06b6d4", "url": "#ec4899",
    "email": "#ec4899", "numeric": "#00d4aa", "categorical": "#10b981",
    "boolean": "#84cc16", "id_column": "#8899aa",
    "high_cardinality_categorical": "#f97316",
    "low_cardinality_categorical": "#10b981",
}

SMART_TYPE_DESCRIPTIONS = {
    "pay_range": "Salary/price ranges with currency symbols (e.g. £70,000 – £120,000)",
    "currency_value": "Single monetary values with currency symbols (e.g. $50,000)",
    "percentage": "Percentage values stored as text (e.g. 45%)",
    "date_string": "Date/time values stored as text strings",
    "text_description": "Free-text descriptions or sentences (avg >4 words)",
    "url": "Web URLs or links",
    "email": "Email addresses",
    "numeric": "Standard numeric data ready for ML",
    "categorical": "Standard categorical data",
    "boolean": "True/False binary values",
    "id_column": "Unique identifier — likely not useful for prediction",
    "high_cardinality_categorical": "Categorical with many unique values (>15) — needs special encoding",
    "low_cardinality_categorical": "Categorical with few unique values (≤15) — ideal for one-hot encoding",
}


def detect_column_smart_type(series, col_name=""):
    """Detect semantic column type beyond basic dtype."""
    if series.dtype in ["int64", "float64", "int32", "float32"]:
        if series.nunique() / max(len(series), 1) > 0.95 and len(series) > 20:
            return "id_column"
        return "numeric"
    if series.dtype == "bool":
        return "boolean"
    sample = series.dropna().astype(str)
    if len(sample) == 0:
        return "categorical"
    sample_vals = sample.head(100)
    col_lower = col_name.lower().strip()

    # 1) Date detection FIRST
    try:
        parsed = pd.to_datetime(sample_vals.head(30), infer_datetime_format=True, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return "date_string"
    except Exception:
        pass
    date_kw = ["date", "time", "created", "updated", "timestamp", "born", "started", "ended", "posted", "published"]
    if any(kw in col_lower for kw in date_kw):
        try:
            parsed = pd.to_datetime(sample_vals.head(10), infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.5:
                return "date_string"
        except Exception:
            pass

    # 2) Pay/Price range
    pay_pat = re.compile(r"[\$£€]?\s*[\d,]+\.?\d*\s*[kK]?\s*[-–—]\s*[\$£€]?\s*[\d,]+\.?\d*\s*[kK]?\+?")
    pay_matches = sample_vals.apply(lambda x: bool(pay_pat.search(str(x)))).mean()
    pay_kw = ["pay", "salary", "wage", "price", "cost", "range", "compensation", "income", "budget", "fee"]
    if pay_matches > 0.5 or (any(kw in col_lower for kw in pay_kw) and pay_matches > 0.2):
        return "pay_range"

    # 3) Single currency value
    if sample_vals.apply(lambda x: bool(re.match(r"^[\$£€]\s*[\d,]+\.?\d*\s*[kK]?\+?$", str(x).strip()))).mean() > 0.5:
        return "currency_value"

    # 4) Percentage
    if sample_vals.apply(lambda x: bool(re.match(r"^\d+\.?\d*\s*%$", str(x).strip()))).mean() > 0.5:
        return "percentage"

    # 5) Text description
    avg_words = sample_vals.str.split().str.len().mean()
    if avg_words > 4 and sample_vals.str.len().mean() > 25:
        return "text_description"

    # 6) URL / Email
    if sample_vals.apply(lambda x: bool(re.search(r"https?://|www\.", str(x)))).mean() > 0.5:
        return "url"
    if sample_vals.apply(lambda x: bool(re.match(r"[\w.+-]+@[\w-]+\.[\w.]+", str(x).strip()))).mean() > 0.5:
        return "email"

    # 7) Cardinality-aware categorical
    n_unique = series.nunique()
    if n_unique > 15:
        return "high_cardinality_categorical"
    return "low_cardinality_categorical"


def parse_pay_range(value):
    if pd.isna(value):
        return None, None, None
    s = str(value).strip()
    currency = "GBP" if "£" in s else "USD" if "$" in s else "EUR" if "€" in s else None
    clean = re.sub(r"[\$£€]", "", s).strip()
    clean = re.sub(r"(\d)\s*[kK]", lambda m: str(int(m.group(1)) * 1000), clean)
    clean = clean.rstrip("+").strip()
    parts = re.split(r"\s*[-–—]\s*|\s+to\s+", clean)
    numbers = []
    for part in parts:
        for n in re.findall(r"[\d,]+\.?\d*", part):
            try:
                numbers.append(float(n.replace(",", "")))
            except ValueError:
                pass
    if len(numbers) >= 2:
        return min(numbers), max(numbers), currency
    elif len(numbers) == 1:
        return numbers[0], numbers[0], currency
    return None, None, currency


def parse_currency_value(value):
    if pd.isna(value):
        return None
    s = re.sub(r"[\$£€,\s]", "", str(value).strip()).rstrip("+")
    s = re.sub(r"(\d)\s*[kK]$", lambda m: str(int(m.group(1)) * 1000), s)
    try:
        return float(s)
    except ValueError:
        return None


def parse_percentage(value):
    if pd.isna(value):
        return None
    try:
        return float(str(value).strip().rstrip("%")) / 100.0
    except ValueError:
        return None


def extract_url_features(series, col_name):
    """Extract features from URL columns: domain, path length, has query params, protocol."""
    feats = pd.DataFrame(index=series.index)
    text = series.fillna("").astype(str)
    feats[f"{col_name}_has_https"] = text.str.startswith("https").astype(int)
    feats[f"{col_name}_url_length"] = text.str.len()
    feats[f"{col_name}_path_depth"] = text.str.count("/").clip(upper=10)
    feats[f"{col_name}_has_query"] = text.str.contains(r"\?", regex=True).astype(int)
    # Extract domain as categorical
    domain = text.str.extract(r"(?:https?://)?(?:www\.)?([^/\?]+)", expand=False).fillna("unknown")
    feats[f"{col_name}_domain"] = domain
    return feats


def extract_email_features(series, col_name):
    """Extract features from email columns: domain, local part length."""
    feats = pd.DataFrame(index=series.index)
    text = series.fillna("").astype(str)
    feats[f"{col_name}_local_len"] = text.str.split("@").str[0].str.len()
    domain = text.str.split("@").str[-1].fillna("unknown")
    feats[f"{col_name}_domain"] = domain
    feats[f"{col_name}_is_gmail"] = domain.str.contains("gmail", case=False).astype(int)
    feats[f"{col_name}_is_corporate"] = (~domain.str.contains("gmail|yahoo|hotmail|outlook", case=False)).astype(int)
    return feats


def extract_date_features(series, col_name):
    parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
    feats = pd.DataFrame(index=series.index)
    feats[f"{col_name}_year"] = parsed.dt.year
    feats[f"{col_name}_month"] = parsed.dt.month
    feats[f"{col_name}_day"] = parsed.dt.day
    feats[f"{col_name}_dayofweek"] = parsed.dt.dayofweek
    feats[f"{col_name}_quarter"] = parsed.dt.quarter
    min_d = parsed.min()
    if pd.notna(min_d):
        feats[f"{col_name}_days_since_start"] = (parsed - min_d).dt.days
    return feats


def extract_text_features(series, col_name, method="basic", max_features=15):
    feats = pd.DataFrame(index=series.index)
    text = series.fillna("").astype(str)
    feats[f"{col_name}_word_count"] = text.str.split().str.len().fillna(0).astype(int)
    feats[f"{col_name}_char_count"] = text.str.len().fillna(0).astype(int)
    feats[f"{col_name}_avg_word_len"] = text.apply(
        lambda x: round(np.mean([len(w) for w in x.split()]), 2) if x.strip() else 0
    )
    if "tfidf" in method:
        try:
            tfidf = TfidfVectorizer(max_features=max_features, stop_words="english",
                                     ngram_range=(1, 2), min_df=2 if len(series) > 10 else 1)
            mat = tfidf.fit_transform(text)
            tdf = pd.DataFrame(mat.toarray(),
                                columns=[f"{col_name}_tfidf_{w}" for w in tfidf.get_feature_names_out()],
                                index=series.index)
            feats = pd.concat([feats, tdf], axis=1)
        except Exception:
            pass
    if "keyword" in method:
        stopwords = {"the","a","an","is","are","was","were","and","or","of","to","in","for","with","on","at","by","from","as","it","this","that","be","has","have","had"}
        all_words = " ".join(text).lower().split()
        filtered = [w for w in all_words if w not in stopwords and len(w) > 2]
        top = [w for w, _ in Counter(filtered).most_common(max_features)]
        for word in top:
            feats[f"{col_name}_has_{word}"] = text.str.lower().str.contains(word, regex=False).astype(int)
    return feats


def build_smart_type_report(df):
    rows = []
    for col in df.columns:
        st_type = detect_column_smart_type(df[col], col)
        sample = df[col].dropna().head(3).astype(str).tolist()
        rows.append({
            "Column": col, "Dtype": str(df[col].dtype), "Smart Type": st_type,
            "Icon": SMART_TYPE_ICONS.get(st_type, "❓"), "Unique": df[col].nunique(),
            "Sample": ", ".join(sample)[:80],
        })
    return pd.DataFrame(rows)


def build_preprocessing_summary(df):
    """Build a list of preprocessing recommendation dicts for each column."""
    recs = []
    for col in df.columns:
        stype = detect_column_smart_type(df[col], col)
        n_unique = df[col].nunique()
        n_rows = len(df)
        icon = SMART_TYPE_ICONS.get(stype, "❓")
        color = SMART_TYPE_COLORS.get(stype, "#8899aa")

        if stype == "low_cardinality_categorical":
            recs.append(dict(
                col=col, icon=icon, type_label=f"CATEGORICAL · {n_unique} classes", type_color=color,
                desc=f"Low-cardinality categorical column with {n_unique} unique values. This is ideal for one-hot encoding, which creates binary features for each category.",
                action="One-Hot Encoding (recommended) or Label Encoding"
            ))
        elif stype == "high_cardinality_categorical":
            recs.append(dict(
                col=col, icon="🎯", type_label=f"HIGH-CARD CATEGORICAL · {n_unique} classes", type_color="#f97316",
                desc=f"High-cardinality column with {n_unique} unique values. One-hot encoding would create too many features. Target Encoding or Frequency Encoding preserves information without dimensional explosion.",
                action="Target Encoding (preferred), Frequency Encoding, or Label Encoding"
            ))
        elif stype == "text_description":
            avg_words = df[col].dropna().astype(str).str.split().str.len().mean()
            recs.append(dict(
                col=col, icon=icon, type_label=f"TEXT · avg {avg_words:.0f} words", type_color=color,
                desc=f"Free-text column containing descriptive sentences. ML models cannot process raw text — it must be vectorised into numeric features using TF-IDF, keyword flags, or basic statistics (word count, char count).",
                action="TF-IDF Vectorization + basic stats, or Keyword presence flags"
            ))
        elif stype == "pay_range":
            sample = df[col].dropna().head(2).astype(str).tolist()
            recs.append(dict(
                col=col, icon=icon, type_label="PAY / PRICE RANGE", type_color=color,
                desc=f"Semi-structured range values like \"{sample[0] if sample else 'N/A'}\". Extracting min, max, and midpoint as separate numeric features makes this data usable for regression or feature analysis.",
                action="Extract min, max, midpoint → numeric features (+ optional currency flag)"
            ))
        elif stype == "currency_value":
            recs.append(dict(
                col=col, icon=icon, type_label="CURRENCY VALUE", type_color=color,
                desc="Monetary values with currency symbols (£, $, €). Stripping symbols and converting to float makes this a clean numeric feature.",
                action="Strip symbols → numeric float"
            ))
        elif stype == "percentage":
            recs.append(dict(
                col=col, icon=icon, type_label="PERCENTAGE", type_color=color,
                desc="Values like '45%' stored as text. Converting to decimal (0.45) or raw number (45.0) makes this ML-ready.",
                action="Convert to decimal (0–1) or numeric (0–100)"
            ))
        elif stype == "date_string":
            recs.append(dict(
                col=col, icon=icon, type_label="DATE / DATETIME", type_color=color,
                desc="Date values stored as text. Extracting year, month, day, day-of-week, quarter, and days-since-start creates temporal features that capture seasonality and trends.",
                action="Extract temporal features (year, month, day, quarter, day_of_week)"
            ))
        elif stype == "url":
            recs.append(dict(
                col=col, icon=icon, type_label="URL / LINK", type_color=color,
                desc="Web URLs contain structural information. Extracting domain, path depth, protocol, and query-parameter presence creates useful categorical and numeric features.",
                action="Extract domain, path depth, HTTPS flag, query-param flag"
            ))
        elif stype == "email":
            recs.append(dict(
                col=col, icon=icon, type_label="EMAIL ADDRESS", type_color=color,
                desc="Email addresses encode domain information (corporate vs personal). Extracting the domain, local-part length, and provider flags creates structured features.",
                action="Extract domain, provider flags (Gmail, corporate), local-part length"
            ))
        elif stype == "id_column":
            recs.append(dict(
                col=col, icon=icon, type_label="ID / UNIQUE KEY", type_color=color,
                desc=f"This column has {n_unique} unique values across {n_rows} rows ({n_unique/n_rows*100:.0f}% unique). Identifiers carry no predictive signal and should be dropped before training.",
                action="Drop column (recommended) — no predictive value"
            ))
        elif stype == "numeric":
            sk = df[col].skew() if df[col].dtype in ["int64","float64"] else 0
            n_outliers, _, _ = detect_outliers_iqr(df[col].dropna()) if df[col].dtype in ["int64","float64"] else (0,0,0)
            notes = []
            if abs(sk) > 1:
                notes.append(f"heavily skewed ({sk:.1f})")
            if n_outliers > 0:
                notes.append(f"{n_outliers} outliers")
            extra = f" Note: {', '.join(notes)}." if notes else ""
            recs.append(dict(
                col=col, icon=icon, type_label="NUMERIC", type_color=color,
                desc=f"Standard numeric feature, already ML-compatible.{extra} Consider scaling (Standard/MinMax) if using distance-based models (KNN, SVM).",
                action="Optional: scaling, outlier treatment, or log transform if skewed"
            ))
        elif stype == "boolean":
            recs.append(dict(
                col=col, icon=icon, type_label="BOOLEAN", type_color=color,
                desc="Binary True/False values. Already ML-compatible as 0/1.",
                action="No action needed — already numeric (0/1)"
            ))
    return recs


# ─────────────────────────────────────────────────────────
# ROBUST TARGET ENCODING & NN CONFIG HELPERS
# ─────────────────────────────────────────────────────────
def encode_target_safely(y_train, y_test, task_type):
    """
    Encode target labels for ML training with safety for unseen test labels.

    Rules enforced:
    - Classification + text target → LabelEncoder (NEVER OneHotEncoder for target)
    - Classification + numeric target → passthrough (already encoded)
    - Regression → passthrough (NEVER scale target here)

    Why LabelEncoder and not OHE?
    OHE creates multiple target columns (e.g. 39 columns for 39 classes),
    which breaks sklearn classifiers that expect a single y column.
    For Neural Networks, we use sparse_categorical_crossentropy which
    works with integer labels — no OHE needed.

    Fits on union of train+test labels to prevent unseen-label crashes.
    """
    info = {"method": "none", "n_classes": 0, "class_names": [], "warnings": []}

    if task_type == "Classification":
        if y_train.dtype == "object" or y_train.dtype.name == "category":
            le = LabelEncoder()
            # Fit on ALL unique labels from both sets to avoid transform errors
            all_labels = pd.concat([y_train, y_test]).unique()
            le.fit(all_labels)
            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)
            info["method"] = "label_encoder"
            info["n_classes"] = len(le.classes_)
            info["class_names"] = [str(c) for c in le.classes_]

            # Warn about test-only labels the model will never learn
            train_labels = set(y_train.unique())
            test_only = set(y_test.unique()) - train_labels
            if test_only:
                info["warnings"].append(
                    f"{len(test_only)} class(es) appear only in test set: "
                    f"{list(test_only)[:5]}{'...' if len(test_only) > 5 else ''}. "
                    f"The model will never see these during training."
                )
            return y_train_enc, y_test_enc, le, info
        else:
            # Already numeric (user may have label-encoded manually in Step ③)
            y_train_enc = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
            y_test_enc = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
            info["method"] = "numeric_passthrough"
            info["n_classes"] = len(np.unique(np.concatenate([y_train_enc, y_test_enc])))
            info["class_names"] = [str(c) for c in sorted(np.unique(y_train_enc))]
            return y_train_enc, y_test_enc, None, info
    else:
        # Regression — never encode, never scale here
        y_train_enc = (y_train.values if hasattr(y_train, 'values') else np.array(y_train)).astype(float)
        y_test_enc = (y_test.values if hasattr(y_test, 'values') else np.array(y_test)).astype(float)
        info["method"] = "regression_passthrough"
        return y_train_enc, y_test_enc, None, info


def determine_nn_config(task_type, n_classes, class_subtype=None):
    """
    Determine Neural Network output layer configuration.

    Classification:
      Binary (2 classes)  → 1 unit, sigmoid, binary_crossentropy
      Multiclass (>2)     → n_classes units, softmax, sparse_categorical_crossentropy

    Why sparse_categorical_crossentropy (not categorical_crossentropy)?
    - sparse_ works with integer-encoded targets (0, 1, 2, ...)
    - plain categorical_crossentropy requires one-hot encoded targets
    - Since we use LabelEncoder (integers), sparse_ is correct and memory-efficient

    Regression:
      1 unit, linear, mse
    """
    if task_type == "Classification":
        if class_subtype == "Binary Classification" or n_classes == 2:
            return {
                "output_units": 1,
                "output_activation": "sigmoid",
                "loss_fn": "binary_crossentropy",
                "metrics": ["accuracy"],
                "desc": (
                    "Binary → sigmoid + binary_crossentropy. "
                    "Target is 0/1 integers."
                ),
            }
        else:
            return {
                "output_units": n_classes,
                "output_activation": "softmax",
                "loss_fn": "sparse_categorical_crossentropy",
                "metrics": ["accuracy"],
                "desc": (
                    f"Multiclass ({n_classes} classes) → softmax + sparse_categorical_crossentropy. "
                    f"'Sparse' means target stays as integers (0,1,2,...) from LabelEncoder — "
                    f"no one-hot encoding needed. This is memory-efficient for many classes."
                ),
            }
    else:
        return {
            "output_units": 1,
            "output_activation": "linear",
            "loss_fn": "mse",
            "metrics": ["mae"],
            "desc": "Regression → linear + MSE. Target is continuous numeric.",
        }


# ─────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────
defaults = {
    "df": None, "df_original": None,
    "X_train": None, "X_test": None, "y_train": None, "y_test": None,
    "model": None, "model_name": None, "task_type": None,
    "trained": False, "preds": None,
    "y_train_enc": None, "y_test_enc": None,
    "label_encoder": None,
    "training_history": None,
    "current_step": 1,
    "class_subtype": None,
    "cv_results": None,
    "tuning_results": None,
    "all_models_results": None,
    "scaler_used": None,
    "feature_cols": None,
    "_target_col": None,
    "y_scaler": None,
    "target_info": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🔬 Net<span class="accent">Gaze</span> ML Pipeline</h1>
    <p>A complete machine learning and deep learning pipeline — from data upload to model evaluation — all in one beautiful interface.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR — PIPELINE STEPS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 NetGaze Pipeline")
    st.caption("Follow the steps below to build your ML model.")
    st.markdown("---")

    st.markdown("### ① Upload Data")
    uploaded_file = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset to begin the pipeline."
    )

    if uploaded_file:
        new_name = uploaded_file.name
        if st.session_state.get("_uploaded_filename") != new_name:
            try:
                if new_name.endswith(".csv"):
                    df_loaded = pd.read_csv(uploaded_file)
                else:
                    df_loaded = pd.read_excel(uploaded_file)
                st.session_state.df = df_loaded.copy()
                st.session_state.df_original = df_loaded.copy()
                st.session_state._uploaded_filename = new_name
                for key in ["X_train", "X_test", "y_train", "y_test",
                            "model", "trained", "preds", "training_history",
                            "cv_results", "tuning_results", "all_models_results"]:
                    st.session_state[key] = None
                st.session_state.trained = False
                st.success(f"✓ Loaded {df_loaded.shape[0]:,} rows × {df_loaded.shape[1]} cols")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.success(f"✓ Working with: {new_name} ({st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols)")

    step_options = []
    if st.session_state.df is not None:
        step_options = [
            "② Explore & Visualise",
            "③ Clean & Preprocess",
            "④ Feature Selection & Split",
            "⑤ Model Training",
            "⑥ Evaluation & Interpretation",
        ]
    active_step = st.radio("Pipeline Stage", step_options, index=0) if step_options else None


# ─────────────────────────────────────────────────────────
# NO DATA STATE
# ─────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 📁 Upload
        Start by uploading a CSV or Excel file in the sidebar.
        """)
    with col2:
        st.markdown("""
        #### 🔍 Explore
        Visualise distributions, correlations, and patterns in your data.
        """)
    with col3:
        st.markdown("""
        #### 🤖 Train
        Select from 10+ ML/DL algorithms and evaluate performance.
        """)
    st.stop()


# ─────────────────────────────────────────────────────────
# ② EXPLORE & VISUALISE
# ─────────────────────────────────────────────────────────
if active_step == "② Explore & Visualise":
    df = st.session_state.df
    badge("② EXPLORE & VISUALISE", "teal")
    section_header("Dataset Overview")

    hint_banner("💡", "<strong>How to use this step:</strong> Start by reviewing the data quality metrics and preview below. Check for missing values, duplicates, and data types. Then explore the <strong>Visualisations</strong> tabs to understand distributions, correlations, and outliers. These insights will guide your preprocessing decisions in the next step.")

    numeric_cols, cat_cols, bool_cols, dt_cols = get_dtype_summary(df)

    quality_score = compute_data_quality_score(df)
    quality_color = "#00d4aa" if quality_score >= 75 else "#f59e0b" if quality_score >= 50 else "#f43f5e"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    dup_count = df.duplicated().sum()
    c4.metric("Duplicates", f"{dup_count:,}")
    c5.metric("Quality Score", f"{quality_score:.0f}/100")

    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.caption(f"Memory usage: {mem_usage:.2f} MB  |  Numeric: {len(numeric_cols)}  |  Categorical: {len(cat_cols)}  |  Boolean: {len(bool_cols)}  |  DateTime: {len(dt_cols)}")

    # ── Smart Preprocessing Summary (the new analysis card) ──
    with st.expander("🧠 Column Analysis & Preprocessing Summary", expanded=True):
        hint_banner("📊", "<strong>What is this?</strong> Each column has been automatically analysed to detect its semantic type — pay ranges, dates, text descriptions, URLs, emails, IDs, and more. Below you'll find a <strong>recommended preprocessing strategy</strong> for every column, so you know exactly what to do in the Clean & Preprocess step.")

        recs = build_preprocessing_summary(df)
        type_report = build_smart_type_report(df)

        # Quick stats banner
        transformable_count = sum(1 for r in recs if r["type_label"] not in ["NUMERIC", "BOOLEAN"])
        needs_action = sum(1 for r in recs if "Drop" in r["action"] or "Extract" in r["action"] or "Encoding" in r["action"] or "TF-IDF" in r["action"] or "Strip" in r["action"] or "Convert" in r["action"])

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Columns", len(recs))
        sc2.metric("Already ML-Ready", sum(1 for r in recs if r["type_label"] in ["NUMERIC", "BOOLEAN"]))
        sc3.metric("Need Transformation", needs_action)
        sc4.metric("Smart Types Found", len(set(r["type_label"].split(" · ")[0] for r in recs)))

        st.markdown("---")

        # Render each recommendation card
        for rec in recs:
            recommendation_card(
                rec["icon"], rec["col"], rec["type_label"],
                rec["type_color"], rec["desc"], rec["action"]
            )

    # Data preview
    with st.expander("📋 Data Preview", expanded=False):
        tab_head, tab_tail, tab_sample, tab_info, tab_types = st.tabs(
            ["Head", "Tail", "Random Sample", "Column Info", "Type Distribution"]
        )
        with tab_head:
            st.dataframe(df.head(20), use_container_width=True)
        with tab_tail:
            st.dataframe(df.tail(20), use_container_width=True)
        with tab_sample:
            n_sample = min(20, len(df))
            st.dataframe(df.sample(n_sample), use_container_width=True)
        with tab_info:
            info_data = []
            for col in df.columns:
                outliers = 0
                if col in numeric_cols:
                    outliers, _, _ = detect_outliers_iqr(df[col].dropna())
                info_data.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": df[col].notna().sum(),
                    "Null": df[col].isnull().sum(),
                    "Null %": f"{df[col].isnull().mean()*100:.1f}%",
                    "Unique": df[col].nunique(),
                    "Outliers (IQR)": outliers if col in numeric_cols else "—",
                })
            st.dataframe(pd.DataFrame(info_data), use_container_width=True)
        with tab_types:
            type_counts = df.dtypes.astype(str).value_counts()
            fig = px.pie(
                values=type_counts.values, names=type_counts.index,
                title="Column Type Distribution",
                color_discrete_sequence=PALETTE_VIVID,
                hole=0.45
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            fig.update_traces(textinfo="label+percent+value", textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Descriptive Statistics ──
    with st.expander("📊 Descriptive Statistics"):
        if numeric_cols:
            st.markdown("**Numerical Columns**")
            desc = df[numeric_cols].describe().T
            desc["skewness"] = df[numeric_cols].skew()
            desc["kurtosis"] = df[numeric_cols].kurtosis()
            desc["CV %"] = (desc["std"] / desc["mean"].abs() * 100).round(2)
            st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)

            st.markdown("**Distribution Insights:**")
            for col in numeric_cols[:8]:
                sk = df[col].skew()
                if abs(sk) > 1:
                    direction = "right" if sk > 0 else "left"
                    insight_card("📐", f"**{col}** is heavily skewed {direction} (skewness={sk:.2f}). Consider log or Box-Cox transform.")
                elif abs(sk) > 0.5:
                    direction = "right" if sk > 0 else "left"
                    insight_card("📐", f"**{col}** is moderately skewed {direction} (skewness={sk:.2f}).")

        if cat_cols:
            st.markdown("**Categorical Columns**")
            st.dataframe(df[cat_cols].describe().T, use_container_width=True)

    divider()
    section_header("Visualisations")

    vis_tab1, vis_tab2, vis_tab3, vis_tab4, vis_tab5, vis_tab6, vis_tab7 = st.tabs([
        "📈 Distributions", "🔗 Correlations", "📦 Box & Violin",
        "🔄 Pairwise", "📋 Value Counts", "🔍 Outlier Analysis", "📊 Statistical Tests"
    ])

    with vis_tab1:
        hint_banner("📈", "<strong>Distributions</strong> show how values are spread across each feature. Look for <strong>skewness</strong> (lopsided data), <strong>multimodality</strong> (multiple peaks), and <strong>outliers</strong> (extreme values far from the centre).")
        if numeric_cols:
            sel_dist_cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:3], key="dist_cols")
            chart_type = st.selectbox("Chart type", ["Histogram + KDE", "Histogram + Box", "Ridge Plot"], key="dist_chart_type")

            if sel_dist_cols:
                if chart_type == "Ridge Plot":
                    fig = go.Figure()
                    for i, col_name in enumerate(sel_dist_cols):
                        data = df[col_name].dropna()
                        fig.add_trace(go.Violin(
                            x=data, name=col_name, side='positive',
                            line_color=PALETTE_VIVID[i % len(PALETTE_VIVID)],
                            fillcolor=PALETTE_VIVID[i % len(PALETTE_VIVID)],
                            opacity=0.6, meanline_visible=True
                        ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=400, title="Ridge Distribution Plot",
                                      showlegend=True, violingap=0, violinmode='overlay')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    n_cols = min(3, len(sel_dist_cols))
                    for i in range(0, len(sel_dist_cols), n_cols):
                        cols = st.columns(n_cols)
                        for j, col_name in enumerate(sel_dist_cols[i:i+n_cols]):
                            with cols[j]:
                                color = PALETTE_VIVID[(i+j) % len(PALETTE_VIVID)]
                                marginal = "box" if chart_type == "Histogram + Box" else "violin"
                                fig = px.histogram(
                                    df, x=col_name, nbins=40, marginal=marginal,
                                    title=f"{col_name}",
                                    color_discrete_sequence=[color],
                                    opacity=0.8
                                )
                                fig.update_layout(**PLOTLY_LAYOUT, height=350, title_font_size=14)
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for distribution plots.")

    with vis_tab2:
        hint_banner("🔗", "<strong>Correlation analysis</strong> reveals linear relationships between numeric features. Values near <strong>+1</strong> or <strong>−1</strong> indicate strong relationships. High correlation between features may indicate <strong>multicollinearity</strong> — consider dropping one of the pair.")
        if len(numeric_cols) >= 2:
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
            corr = df[numeric_cols].corr(method=corr_method)

            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=CORRELATION_SCALE,
                aspect="auto", title=f"Feature Correlation Matrix ({corr_method.title()})",
                zmin=-1, zmax=1
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=max(500, len(numeric_cols)*30))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Top Correlations (absolute):**")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({
                        "Feature A": corr.columns[i],
                        "Feature B": corr.columns[j],
                        "Correlation": corr.iloc[i, j]
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values("Correlation", key=abs, ascending=False).head(15)
            fig = px.bar(
                corr_df, x="Correlation", y=[f"{r['Feature A']} ↔ {r['Feature B']}" for _, r in corr_df.iterrows()],
                orientation="h", title="Top 15 Feature Correlations",
                color="Correlation", color_continuous_scale=DIVERGING_SCALE,
                color_continuous_midpoint=0
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=max(350, len(corr_df)*28))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for a correlation matrix.")

    with vis_tab3:
        hint_banner("📦", "<strong>Box plots</strong> show the median, quartiles, and outliers. <strong>Violin plots</strong> add probability density. Use the optional <strong>group by</strong> to compare distributions across categories.")
        if numeric_cols:
            sel_box_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:4], key="box_cols")
            plot_type = st.selectbox("Plot type", ["Box Plot", "Violin Plot", "Box + Strip"], key="box_type")
            group_by = st.selectbox("Optional: group by categorical column", ["None"] + cat_cols, key="box_group")

            if sel_box_cols:
                for col_name in sel_box_cols:
                    color_idx = sel_box_cols.index(col_name)
                    if group_by != "None":
                        if plot_type == "Violin Plot":
                            fig = px.violin(df, x=group_by, y=col_name, color=group_by,
                                            box=True, points="outliers",
                                            title=f"{col_name} by {group_by}",
                                            color_discrete_sequence=PALETTE_VIVID)
                        elif plot_type == "Box + Strip":
                            fig = px.strip(df, x=group_by, y=col_name, color=group_by,
                                           title=f"{col_name} by {group_by}",
                                           color_discrete_sequence=PALETTE_VIVID)
                            fig.update_traces(marker=dict(size=3, opacity=0.4))
                        else:
                            fig = px.box(df, x=group_by, y=col_name, color=group_by,
                                         title=f"{col_name} by {group_by}",
                                         color_discrete_sequence=PALETTE_VIVID)
                    else:
                        color = PALETTE_VIVID[color_idx % len(PALETTE_VIVID)]
                        if plot_type == "Violin Plot":
                            fig = px.violin(df, y=col_name, box=True, points="outliers",
                                            title=f"{col_name}",
                                            color_discrete_sequence=[color])
                        elif plot_type == "Box + Strip":
                            fig = px.strip(df, y=col_name, title=f"{col_name}",
                                           color_discrete_sequence=[color])
                            fig.update_traces(marker=dict(size=3, opacity=0.4))
                        else:
                            fig = px.box(df, y=col_name, title=f"{col_name}",
                                         color_discrete_sequence=[color])
                    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with vis_tab4:
        hint_banner("🔄", "<strong>Scatter matrix</strong> plots every pair of selected features against each other. Great for spotting <strong>clusters</strong>, <strong>linear trends</strong>, and <strong>class separability</strong>. Use the colour option to highlight different target classes.")
        if len(numeric_cols) >= 2:
            sel_pair_cols = st.multiselect("Select columns (2–5 recommended)", numeric_cols, default=numeric_cols[:3], key="pair_cols")
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="pair_color")
            if len(sel_pair_cols) >= 2:
                color_arg = color_col if color_col != "None" else None
                fig = px.scatter_matrix(
                    df, dimensions=sel_pair_cols, color=color_arg,
                    title="Pairwise Scatter Matrix",
                    color_discrete_sequence=PALETTE_VIVID
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=600)
                fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.6))
                st.plotly_chart(fig, use_container_width=True)

    with vis_tab5:
        hint_banner("📋", "<strong>Value counts</strong> show category frequencies. Use this to check <strong>class balance</strong> (important for classification targets) and identify <strong>rare categories</strong> that may need grouping.")
        all_cols = df.columns.tolist()
        sel_vc_col = st.selectbox("Select column", all_cols, key="vc_col")
        vc_chart = st.selectbox("Chart type", ["Horizontal Bar", "Treemap", "Donut"], key="vc_chart")
        if sel_vc_col:
            vc = df[sel_vc_col].value_counts().head(25)
            if vc_chart == "Treemap":
                fig = px.treemap(
                    names=vc.index.astype(str), parents=["" for _ in vc],
                    values=vc.values, title=f"Treemap — {sel_vc_col}",
                    color=vc.values, color_continuous_scale=PALETTE_GRADIENT_TEAL
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=450)
                fig.update_coloraxes(showscale=False)
            elif vc_chart == "Donut":
                fig = px.pie(
                    values=vc.values, names=vc.index.astype(str),
                    title=f"Donut — {sel_vc_col}", hole=0.5,
                    color_discrete_sequence=PALETTE_VIVID
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=420)
                fig.update_traces(textinfo="label+percent", textfont_size=11)
            else:
                fig = px.bar(
                    x=vc.values, y=vc.index.astype(str), orientation="h",
                    title=f"Top {len(vc)} values — {sel_vc_col}",
                    labels={"x": "Count", "y": sel_vc_col},
                    color=vc.values, color_continuous_scale=HEATMAP_SCALE
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=max(300, len(vc)*25), showlegend=False)
                fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with vis_tab6:
        hint_banner("🔍", "<strong>Outlier analysis</strong> uses the IQR method (1.5× interquartile range). Columns with high outlier percentages may need <strong>capping</strong> (Winsorisation), <strong>removal</strong>, or <strong>log transformation</strong> in the preprocessing step.")
        if numeric_cols:
            outlier_data = []
            for col in numeric_cols:
                data = df[col].dropna()
                n_outliers, lower, upper = detect_outliers_iqr(data)
                outlier_data.append({
                    "Column": col,
                    "Outliers": n_outliers,
                    "Outlier %": round(n_outliers / len(data) * 100, 2) if len(data) > 0 else 0,
                    "Lower Bound": round(lower, 4),
                    "Upper Bound": round(upper, 4),
                    "Min": round(data.min(), 4),
                    "Max": round(data.max(), 4),
                })
            outlier_df = pd.DataFrame(outlier_data).sort_values("Outlier %", ascending=False)
            st.dataframe(outlier_df, use_container_width=True)

            fig = px.bar(
                outlier_df, x="Outlier %", y="Column", orientation="h",
                title="Outlier Percentage by Feature",
                color="Outlier %",
                color_continuous_scale=[[0, "#00d4aa"], [0.5, "#f59e0b"], [1, "#f43f5e"]]
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=max(300, len(outlier_df)*28))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with vis_tab7:
        hint_banner("📊", "<strong>Statistical tests</strong> check whether your data follows a normal (bell-curve) distribution. Many ML algorithms assume normality — if your data fails the test, consider applying a <strong>log</strong>, <strong>Box-Cox</strong>, or <strong>Yeo-Johnson</strong> transform.")
        if numeric_cols:
            test_col = st.selectbox("Select column for normality test", numeric_cols, key="stat_test_col")
            if test_col:
                data = df[test_col].dropna()
                sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data

                col_a, col_b = st.columns(2)
                with col_a:
                    sorted_data = np.sort(sample)
                    theoretical = np.sort(np.random.normal(sample.mean(), sample.std(), len(sample)))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=theoretical, y=sorted_data, mode="markers",
                        marker=dict(color="#00d4aa", size=3, opacity=0.6), name="Data"
                    ))
                    min_v = min(theoretical.min(), sorted_data.min())
                    max_v = max(theoretical.max(), sorted_data.max())
                    fig.add_trace(go.Scatter(
                        x=[min_v, max_v], y=[min_v, max_v],
                        mode="lines", line=dict(color="#f43f5e", dash="dash"), name="Normal"
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, title=f"Q-Q Plot — {test_col}", height=380,
                                      xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    st.markdown("**Normality Test Results:**")
                    try:
                        if len(sample) <= 5000:
                            stat_s, p_s = shapiro(sample)
                            st.markdown(f"**Shapiro-Wilk:** W={stat_s:.4f}, p={p_s:.4e}")
                            if p_s < 0.05:
                                st.warning("Data is NOT normally distributed (p < 0.05)")
                            else:
                                st.success("Data appears normally distributed (p ≥ 0.05)")
                    except Exception:
                        pass
                    try:
                        stat_d, p_d = normaltest(sample)
                        st.markdown(f"**D'Agostino-Pearson:** stat={stat_d:.4f}, p={p_d:.4e}")
                    except Exception:
                        pass
                    st.markdown(f"**Skewness:** {skew(sample):.4f}")
                    st.markdown(f"**Kurtosis:** {kurtosis(sample):.4f}")
                    st.markdown(f"**Sample size:** {len(sample):,}")


# ─────────────────────────────────────────────────────────
# ③ CLEAN & PREPROCESS — FIXED TAB INDEXING + ENHANCED
# ─────────────────────────────────────────────────────────
elif active_step == "③ Clean & Preprocess":
    df = st.session_state.df
    badge("③ CLEAN & PREPROCESS", "amber")
    section_header("Data Cleaning & Preprocessing")

    hint_banner("🛠️", "<strong>How to use this step:</strong> Work through the tabs from left to right. Start with <strong>Missing Values</strong>, then handle <strong>Duplicates</strong>, apply <strong>Smart Transforms</strong> for special column types (pay ranges, dates, text), <strong>Encode</strong> categorical columns, <strong>Scale</strong> numeric features, and treat <strong>Outliers</strong>. Finally, <strong>Preview</strong> your cleaned data before moving on.")

    clean_tab1, clean_tab2, clean_tab3, clean_tab4, clean_tab5, clean_tab6, clean_tab7 = st.tabs([
       "🔍 Missing Values", "🗑️ Duplicates & Drops", "🧠 Smart Transform",
       "🏷️ Encoding", "📏 Scaling", "🔧 Outlier Treatment", "✅ Preview & Confirm"
   ])

    # ── TAB 1: Missing Values ──
    with clean_tab1:
        hint_banner("🔍", "<strong>Missing values</strong> can break ML models or reduce accuracy. Choose a strategy based on the column type: <strong>median</strong> for skewed numeric data, <strong>mean</strong> for normally distributed data, <strong>mode</strong> for categorical columns. Drop rows only if missing data is minimal (<5%).")
        missing = df.isnull().sum()
        cols_with_missing = missing[missing > 0].index.tolist()
        if not cols_with_missing:
            st.success("No missing values found in the dataset!")
        else:
            miss_df = pd.DataFrame({
                "Column": cols_with_missing,
                "Missing": [missing[c] for c in cols_with_missing],
                "Percentage": [missing[c]/len(df)*100 for c in cols_with_missing]
            }).sort_values("Missing", ascending=True)

            fig = px.bar(
                miss_df, x="Missing", y="Column", orientation="h",
                text="Percentage", title="Missing Values Overview",
                color="Percentage",
                color_continuous_scale=[[0, "#00d4aa"], [0.5, "#f59e0b"], [1, "#f43f5e"]]
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(**PLOTLY_LAYOUT, height=max(300, len(cols_with_missing)*35))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Quick Actions:**")
            qa1, qa2, qa3 = st.columns(3)
            with qa1:
                if st.button("Drop all rows with missing", key="drop_all_miss"):
                    df = df.dropna()
                    st.session_state.df = df
                    st.success(f"Dropped rows. New shape: {df.shape}")
                    st.rerun()
            with qa2:
                if st.button("Fill numeric with median", key="fill_med_all"):
                    for col in cols_with_missing:
                        if df[col].dtype in ["int64", "float64"]:
                            df[col] = df[col].fillna(df[col].median())
                    st.session_state.df = df
                    st.success("Filled numeric columns with median.")
                    st.rerun()
            with qa3:
                if st.button("Fill categorical with mode", key="fill_mode_all"):
                    for col in cols_with_missing:
                        if df[col].dtype == "object":
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                    st.session_state.df = df
                    st.success("Filled categorical columns with mode.")
                    st.rerun()

            st.markdown("---")
            st.markdown("**Per-Column Strategy:**")
            sel_missing_cols = st.multiselect("Select columns to handle individually", cols_with_missing, key="miss_cols")
            strategies = {}
            if sel_missing_cols:
                for col in sel_missing_cols:
                    dtype = df[col].dtype
                    if dtype in ["int64", "float64"]:
                        options = ["Drop Rows", "Drop Column", "Fill with Mean", "Fill with Median",
                                   "Fill with Mode", "Fill with 0", "Forward Fill", "Backward Fill",
                                   "Interpolate (Linear)"]
                    else:
                        options = ["Drop Rows", "Drop Column", "Fill with Mode",
                                   "Fill with 'Unknown'", "Forward Fill", "Backward Fill"]
                    strategies[col] = st.selectbox(
                        f"Strategy for `{col}` ({dtype}, {missing[col]} missing)",
                        options, key=f"miss_strat_{col}"
                    )

                if st.button("Apply Missing Value Fixes", key="apply_missing"):
                    for col, strat in strategies.items():
                        if strat == "Drop Rows":
                            df = df.dropna(subset=[col])
                        elif strat == "Drop Column":
                            df = df.drop(columns=[col])
                        elif strat == "Fill with Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif strat == "Fill with Median":
                            df[col] = df[col].fillna(df[col].median())
                        elif strat == "Fill with Mode":
                            df[col] = df[col].fillna(df[col].mode()[0])
                        elif strat == "Fill with 0":
                            df[col] = df[col].fillna(0)
                        elif strat == "Fill with 'Unknown'":
                            df[col] = df[col].fillna("Unknown")
                        elif strat == "Forward Fill":
                            df[col] = df[col].ffill()
                        elif strat == "Backward Fill":
                            df[col] = df[col].bfill()
                        elif strat == "Interpolate (Linear)":
                            df[col] = df[col].interpolate(method="linear")
                    st.session_state.df = df
                    st.success("Missing values handled!")
                    st.rerun()

    # ── TAB 2: Duplicates & Drops ──
    with clean_tab2:
        hint_banner("🗑️", "<strong>Duplicate rows</strong> can bias your model by overrepresenting certain samples. <strong>Low-variance columns</strong> (where nearly all values are the same) add noise without predictive signal. <strong>ID columns</strong> should always be dropped before training.")
        dup_count = df.duplicated().sum()
        st.info(f"Found **{dup_count}** duplicate rows in the dataset.")
        if dup_count > 0:
            if st.button("Remove All Duplicates", key="rm_dup"):
                df = df.drop_duplicates()
                st.session_state.df = df
                st.success(f"Removed {dup_count} duplicates.")
                st.rerun()

        st.markdown("---")

        # Auto-detect ID columns and recommend dropping
        type_report = build_smart_type_report(df)
        id_cols = type_report[type_report["Smart Type"] == "id_column"]["Column"].tolist()
        if id_cols:
            st.markdown("**🔑 ID Columns Detected (recommended to drop):**")
            for col in id_cols:
                st.caption(f"`{col}` — {df[col].nunique()} unique values ({df[col].nunique()/len(df)*100:.0f}% unique)")
            if st.button("Drop All ID Columns", key="drop_id_cols"):
                df = df.drop(columns=id_cols)
                st.session_state.df = df
                st.success(f"Dropped {len(id_cols)} ID column(s): {id_cols}")
                st.rerun()
            st.markdown("---")

        st.markdown("**Drop Specific Columns**")
        cols_to_drop = st.multiselect("Select columns to remove", df.columns.tolist(), key="drop_cols")
        if cols_to_drop and st.button("Drop Selected Columns", key="drop_cols_btn"):
            df = df.drop(columns=cols_to_drop)
            st.session_state.df = df
            st.success(f"Dropped {len(cols_to_drop)} columns.")
            st.rerun()

        st.markdown("---")
        st.markdown("**Drop Low-Variance Columns**")
        st.caption("Columns where nearly all values are the same may not add predictive value.")
        numeric_cols_curr = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols_curr:
            threshold = st.slider("Variance threshold", 0.0, 1.0, 0.01, 0.01, key="var_thresh")
            low_var = [c for c in numeric_cols_curr if df[c].var() < threshold]
            if low_var:
                st.warning(f"Low-variance columns (< {threshold}): {low_var}")
                if st.button("Drop Low-Variance Columns", key="drop_lowvar"):
                    df = df.drop(columns=low_var)
                    st.session_state.df = df
                    st.success(f"Dropped {len(low_var)} low-variance columns.")
                    st.rerun()
            else:
                st.success("No columns below the variance threshold.")

    # ── TAB 3: Smart Transform ──
    with clean_tab3:
        hint_banner("🧠", "<strong>Smart Transform</strong> automatically detects columns that need special processing — pay ranges (£70k–£120k), dates, text descriptions, percentages, URLs, and emails. Select transform actions below and click <strong>Apply</strong> to convert them into ML-ready numeric features.")

        type_report = build_smart_type_report(df)

        st.markdown("#### 🧠 Column Analysis")
        st.caption("Each column has been analysed for its semantic type. Highlighted columns can be automatically transformed into ML-ready features.")

        # Build styled report table
        report_html = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-family: \'DM Sans\', sans-serif; font-size: 0.85rem;">'
        report_html += '<tr style="border-bottom: 2px solid #2a3a50;">'
        for h in ["", "Column", "Dtype", "Smart Type", "Unique", "Sample Values"]:
            report_html += f'<th style="text-align:left; padding:10px 12px; color:#8899aa; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">{h}</th>'
        report_html += '</tr>'

        transformable_types = ("pay_range", "currency_value", "percentage", "date_string", "text_description", "url", "email", "id_column")
        transformable_cols = {}
        for _, row in type_report.iterrows():
            stype = row["Smart Type"]
            color = SMART_TYPE_COLORS.get(stype, "#8899aa")
            is_t = stype in transformable_types
            bg = f"rgba({_hex_to_rgb(color)},0.06)" if is_t else "transparent"
            bl = f"3px solid {color}" if is_t else "3px solid transparent"
            if is_t:
                transformable_cols[row["Column"]] = stype
            report_html += f'<tr style="border-bottom:1px solid #1a2332; background:{bg}; border-left:{bl};">'
            report_html += f'<td style="padding:8px 12px; font-size:1.1rem;">{row["Icon"]}</td>'
            report_html += f'<td style="padding:8px 12px; color:#e8ecf1; font-weight:600;">{row["Column"]}</td>'
            report_html += f'<td style="padding:8px 12px; color:#8899aa; font-family:\'JetBrains Mono\',monospace; font-size:0.8rem;">{row["Dtype"]}</td>'
            report_html += f'<td style="padding:8px 12px;"><span style="background:rgba({_hex_to_rgb(color)},0.15); color:{color}; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600;">{stype}</span></td>'
            report_html += f'<td style="padding:8px 12px; color:#8899aa; font-family:\'JetBrains Mono\',monospace;">{row["Unique"]}</td>'
            report_html += f'<td style="padding:8px 12px; color:#8899aa; font-size:0.8rem; max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{row["Sample"]}</td>'
            report_html += '</tr>'
        report_html += '</table></div>'
        st.markdown(report_html, unsafe_allow_html=True)

        if not transformable_cols:
            st.info("No columns detected that need smart transformation. Your data appears already ML-ready.")
        else:
            st.markdown("---")
            st.markdown(f"#### 🔧 Transform {len(transformable_cols)} Detected Column{'s' if len(transformable_cols) > 1 else ''}")

            by_type = {}
            for col, stype in transformable_cols.items():
                by_type.setdefault(stype, []).append(col)

            transform_actions = {}

            # ── ID Column ──
            if "id_column" in by_type:
                st.markdown('''<div style="background:rgba(136,153,170,0.08); border:1px solid rgba(136,153,170,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">🔑</span><span style="color:#8899aa; font-weight:700;">ID / Unique Key Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem;">Unique identifiers carry no predictive signal. Dropping them before training prevents the model from memorising individual records.</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["id_column"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Drop column (recommended)", "Skip"], key=f"smart_{col}")

            # ── Pay Range ──
            if "pay_range" in by_type:
                st.markdown('''<div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">💰</span><span style="color:#f59e0b; font-weight:700;">Pay / Price Range Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem; line-height:1.5;">Extracts <strong>min</strong>, <strong>max</strong>, and <strong>midpoint</strong> numeric features from range strings like "£70,000 – £120,000". Detects mixed currencies automatically and adds a currency flag column if multiple are found.</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["pay_range"]:
                    sample = df[col].dropna().head(3).tolist()
                    st.markdown(f"**`{col}`** — e.g., `{sample[0] if sample else 'N/A'}`")
                    c1, c2 = st.columns(2)
                    with c1:
                        transform_actions[col] = st.selectbox(f"Action", ["Extract min, max, midpoint", "Extract min & max only", "Extract midpoint only", "Skip"], key=f"smart_{col}")
                    with c2:
                        transform_actions[f"{col}__keep"] = st.checkbox(f"Keep original", value=False, key=f"keep_{col}")
                    if transform_actions[col] != "Skip":
                        prev = df[col].head(5).apply(parse_pay_range)
                        prev_df = pd.DataFrame(prev.tolist(), columns=["Min", "Max", "Currency"])
                        prev_df["Midpoint"] = ((prev_df["Min"].fillna(0) + prev_df["Max"].fillna(0)) / 2).round(0)
                        prev_df.insert(0, "Original", df[col].head(5).values)
                        st.dataframe(prev_df, use_container_width=True, height=180)

            # ── Currency Value ──
            if "currency_value" in by_type:
                st.markdown('''<div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">💲</span><span style="color:#f59e0b; font-weight:700;">Currency Value Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem;">Strips currency symbols (£, $, €) and converts to numeric. "$50,000" → 50000.0</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["currency_value"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Convert to numeric", "Skip"], key=f"smart_{col}")

            # ── Percentage ──
            if "percentage" in by_type:
                st.markdown('''<div style="background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">📊</span><span style="color:#3b82f6; font-weight:700;">Percentage Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem;">"45%" → 0.45 (decimal) or 45.0 (raw number).</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["percentage"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Convert to decimal (0-1)", "Convert to number (0-100)", "Skip"], key=f"smart_{col}")

            # ── Date String ──
            if "date_string" in by_type:
                st.markdown('''<div style="background:rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">📅</span><span style="color:#8b5cf6; font-weight:700;">Date / DateTime Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem; line-height:1.5;">Extracts <strong>year</strong>, <strong>month</strong>, <strong>day</strong>, <strong>day of week</strong>, <strong>quarter</strong>, and <strong>days since earliest</strong> as numeric features for ML.</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["date_string"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Extract all date features", "Extract year & month only", "Skip"], key=f"smart_{col}")
                    if transform_actions[col] != "Skip":
                        st.dataframe(extract_date_features(df[col].head(5), col), use_container_width=True, height=180)

            # ── Text Description ──
            if "text_description" in by_type:
                st.markdown('''<div style="background:rgba(6,182,212,0.08); border:1px solid rgba(6,182,212,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">📝</span><span style="color:#06b6d4; font-weight:700;">Text Description Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem; line-height:1.5;">
                        • <strong>Basic stats</strong> — word count, character count, avg word length<br>
                        • <strong>TF-IDF</strong> — term frequency vectors capturing key terms (best for diverse text)<br>
                        • <strong>Keyword flags</strong> — binary indicators for most common meaningful words
                    </div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["text_description"]:
                    c1, c2 = st.columns(2)
                    with c1:
                        transform_actions[col] = st.selectbox(f"Method for `{col}`", ["Basic stats only", "TF-IDF + basic stats", "Keyword presence + basic stats", "Skip"], key=f"smart_{col}")
                    with c2:
                        if "TF-IDF" in transform_actions.get(col, "") or "Keyword" in transform_actions.get(col, ""):
                            transform_actions[f"{col}__n"] = st.slider(f"Max features", 5, 50, 15, key=f"nf_{col}")
                    transform_actions[f"{col}__keep"] = st.checkbox(f"Keep original `{col}`", value=False, key=f"keep_{col}")

            # ── URL ──
            if "url" in by_type:
                st.markdown('''<div style="background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">🔗</span><span style="color:#ec4899; font-weight:700;">URL / Link Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem; line-height:1.5;">Extracts structural features: <strong>domain</strong>, <strong>path depth</strong>, <strong>HTTPS flag</strong>, <strong>query parameter flag</strong>, and <strong>URL length</strong>.</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["url"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Extract URL features", "Skip"], key=f"smart_{col}")
                    transform_actions[f"{col}__keep"] = st.checkbox(f"Keep original `{col}`", value=False, key=f"keep_{col}")

            # ── Email ──
            if "email" in by_type:
                st.markdown('''<div style="background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.2); border-radius:10px; padding:1rem 1.2rem; margin:0.8rem 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.5rem;"><span style="font-size:1.2rem;">📧</span><span style="color:#ec4899; font-weight:700;">Email Address Columns</span></div>
                    <div style="color:#8899aa; font-size:0.82rem; line-height:1.5;">Extracts <strong>domain</strong>, <strong>local-part length</strong>, and <strong>provider flags</strong> (Gmail, corporate).</div>
                </div>''', unsafe_allow_html=True)
                for col in by_type["email"]:
                    transform_actions[col] = st.selectbox(f"Action for `{col}`", ["Extract email features", "Skip"], key=f"smart_{col}")
                    transform_actions[f"{col}__keep"] = st.checkbox(f"Keep original `{col}`", value=False, key=f"keep_{col}")

            # ── Apply Button ──
            st.markdown("---")
            active = {k: v for k, v in transform_actions.items() if not k.endswith("__keep") and not k.endswith("__n") and v != "Skip"}
            if active:
                st.markdown(f"**{len(active)} transform{'s' if len(active) > 1 else ''} ready.**")
                if st.button("⚡ Apply Smart Transforms", type="primary", key="apply_smart"):
                    drops = []
                    for col, action in active.items():
                        stype = transformable_cols.get(col)
                        if stype == "id_column":
                            drops.append(col)
                        elif stype == "pay_range":
                            parsed = df[col].apply(parse_pay_range)
                            pdf = pd.DataFrame(parsed.tolist(), columns=["_mn","_mx","_cur"], index=df.index)
                            if "midpoint" in action.lower() or "min, max" in action.lower():
                                df[f"{col}_min"] = pdf["_mn"]
                                df[f"{col}_max"] = pdf["_mx"]
                                if "midpoint" in action.lower():
                                    df[f"{col}_mid"] = ((pdf["_mn"].fillna(0)+pdf["_mx"].fillna(0))/2).round(0)
                            elif "midpoint only" in action.lower():
                                df[f"{col}_mid"] = ((pdf["_mn"].fillna(0)+pdf["_mx"].fillna(0))/2).round(0)
                            if pdf["_cur"].nunique() > 1:
                                df[f"{col}_currency"] = pdf["_cur"]
                            if not transform_actions.get(f"{col}__keep", False):
                                drops.append(col)
                        elif stype == "currency_value":
                            df[col] = df[col].apply(parse_currency_value)
                        elif stype == "percentage":
                            if "decimal" in action.lower():
                                df[col] = df[col].apply(parse_percentage)
                            else:
                                df[col] = df[col].apply(lambda x: parse_percentage(x)*100 if parse_percentage(x) is not None else None)
                        elif stype == "date_string":
                            feats = extract_date_features(df[col], col)
                            if "year & month" in action.lower():
                                feats = feats[[c for c in feats.columns if "year" in c or "month" in c]]
                            for c in feats.columns:
                                df[c] = feats[c]
                            drops.append(col)
                        elif stype == "text_description":
                            mth = "tfidf+basic" if "tfidf" in action.lower() else "keywords+basic" if "keyword" in action.lower() else "basic"
                            nf = transform_actions.get(f"{col}__n", 15)
                            feats = extract_text_features(df[col], col, method=mth, max_features=nf)
                            for c in feats.columns:
                                df[c] = feats[c]
                            if not transform_actions.get(f"{col}__keep", False):
                                drops.append(col)
                        elif stype == "url":
                            feats = extract_url_features(df[col], col)
                            for c in feats.columns:
                                df[c] = feats[c]
                            if not transform_actions.get(f"{col}__keep", False):
                                drops.append(col)
                        elif stype == "email":
                            feats = extract_email_features(df[col], col)
                            for c in feats.columns:
                                df[c] = feats[c]
                            if not transform_actions.get(f"{col}__keep", False):
                                drops.append(col)
                    drops = [c for c in drops if c in df.columns]
                    if drops:
                        df = df.drop(columns=drops)
                    st.session_state.df = df
                    st.success(f"✅ Smart transforms applied! New shape: {df.shape}")
                    st.rerun()

    # ── TAB 4: Encoding ──
    with clean_tab4:
        hint_banner("🏷️", "<strong>Encoding</strong> converts categorical text into numbers. Use <strong>One-Hot</strong> for low-cardinality columns (≤15 unique values), <strong>Label Encoding</strong> or <strong>Frequency Encoding</strong> for high-cardinality columns (>15 unique). Avoid one-hot encoding columns with hundreds of categories — it creates too many features.")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.success("No categorical columns to encode. All columns are already numeric!")
        else:
            # Enhanced categorical summary
            cat_summary = []
            for c in cat_cols:
                n_unique = df[c].nunique()
                cat_summary.append({
                    "Column": c,
                    "Unique Values": n_unique,
                    "Cardinality": "🟢 Low" if n_unique <= 15 else "🟡 Medium" if n_unique <= 50 else "🔴 High",
                    "Recommended": "One-Hot" if n_unique <= 15 else "Frequency/Label" if n_unique <= 50 else "Label Encoding",
                    "Sample Values": ", ".join(df[c].dropna().unique()[:5].astype(str))
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)

            sel_enc_cols = st.multiselect("Select columns to encode", cat_cols, key="enc_cols")
            enc_methods = {}
            if sel_enc_cols:
                for col in sel_enc_cols:
                    n_unique = df[col].nunique()
                    default_method = "Label Encoding" if n_unique > 10 else "One-Hot Encoding"
                    enc_methods[col] = st.selectbox(
                        f"`{col}` ({n_unique} unique) — Encoding method",
                        ["Label Encoding", "One-Hot Encoding", "Frequency Encoding"],
                        index=0 if default_method == "Label Encoding" else 1,
                        key=f"enc_method_{col}"
                    )

                if st.button("Apply Encoding", key="apply_enc"):
                    for col, method in enc_methods.items():
                        if method == "Label Encoding":
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                        elif method == "One-Hot Encoding":
                            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                            encoded = ohe.fit_transform(df[[col]])
                            enc_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([col]), index=df.index)
                            df = pd.concat([df.drop(columns=[col]), enc_df], axis=1)
                        elif method == "Frequency Encoding":
                            freq_map = df[col].value_counts(normalize=True).to_dict()
                            df[col] = df[col].map(freq_map)
                    st.session_state.df = df
                    st.success("Encoding applied!")
                    st.rerun()

    # ── TAB 5: Scaling ──
    with clean_tab5:
        hint_banner("📏", "<strong>Scaling</strong> normalises numeric features to a common range. Essential for <strong>distance-based models</strong> (KNN, SVM) and <strong>gradient-based models</strong> (Neural Networks, Logistic Regression). Tree-based models (Random Forest, XGBoost) generally don't need scaling.")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns to scale.")
        else:
            sel_scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols, key="scale_cols")
            scale_method = st.selectbox(
                "Scaling method",
                [
                    "Standard Scaler (zero mean, unit variance)",
                    "Min-Max Scaler (0 to 1)",
                    "Robust Scaler (median, IQR — outlier resistant)",
                    "Power Transform (Yeo-Johnson — normalise skewed data)"
                ],
                key="scale_method"
            )
            if sel_scale_cols and st.button("Apply Scaling", key="apply_scale"):
                for col in sel_scale_cols:
                    if "Standard" in scale_method:
                        scaler = StandardScaler()
                    elif "Min-Max" in scale_method:
                        scaler = MinMaxScaler()
                    elif "Robust" in scale_method:
                        scaler = RobustScaler()
                    else:
                        scaler = PowerTransformer(method="yeo-johnson")
                    df[col] = scaler.fit_transform(df[[col]])
                st.session_state.df = df
                st.success("Scaling applied!")
                st.rerun()

    # ── TAB 6: Outlier Treatment ──
    with clean_tab6:
        hint_banner("🔧", "<strong>Outliers</strong> are extreme values that can distort model training. <strong>Capping</strong> (Winsorisation) limits values to IQR bounds without losing rows. <strong>Log transform</strong> reduces the impact of large values on skewed data. Only treat outliers if they represent noise, not genuine rare events.")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            sel_outlier_cols = st.multiselect("Select columns for outlier treatment", numeric_cols, key="outlier_cols")
            outlier_method = st.selectbox(
                "Treatment method",
                ["Cap at IQR bounds (Winsorisation)", "Remove outlier rows",
                 "Replace with median", "Log transform (positive values only)"],
                key="outlier_method"
            )
            if sel_outlier_cols:
                for col in sel_outlier_cols:
                    n_out, lower, upper = detect_outliers_iqr(df[col].dropna())
                    st.caption(f"`{col}`: {n_out} outliers | bounds [{lower:.2f}, {upper:.2f}]")

                if st.button("Apply Outlier Treatment", key="apply_outlier"):
                    for col in sel_outlier_cols:
                        _, lower, upper = detect_outliers_iqr(df[col].dropna())
                        if "Cap" in outlier_method:
                            df[col] = df[col].clip(lower=lower, upper=upper)
                        elif "Remove" in outlier_method:
                            df = df[(df[col] >= lower) & (df[col] <= upper)]
                        elif "Replace" in outlier_method:
                            median_val = df[col].median()
                            df.loc[(df[col] < lower) | (df[col] > upper), col] = median_val
                        elif "Log" in outlier_method:
                            if (df[col] > 0).all():
                                df[col] = np.log1p(df[col])
                            else:
                                st.warning(f"Cannot log-transform `{col}` — contains non-positive values.")
                    st.session_state.df = df
                    st.success("Outlier treatment applied!")
                    st.rerun()

    # ── TAB 7: Preview & Confirm ──
    with clean_tab7:
        hint_banner("✅", "<strong>Review your cleaned data</strong> before moving to feature selection. Check that missing values are handled, categorical columns are encoded, and the shape looks right. You can <strong>download</strong> the processed data or <strong>reset</strong> to the original if needed.")

        if st.session_state.df_original is not None:
            orig = st.session_state.df_original
            c1, c2, c3 = st.columns(3)
            row_diff = orig.shape[0] - df.shape[0]
            col_diff = orig.shape[1] - df.shape[1]
            miss_diff = orig.isnull().sum().sum() - df.isnull().sum().sum()
            c1.metric("Rows", f"{df.shape[0]:,}", delta=f"{-row_diff}" if row_diff else "0")
            c2.metric("Columns", f"{df.shape[1]}", delta=f"{-col_diff}" if col_diff else "0")
            c3.metric("Missing", f"{df.isnull().sum().sum()}", delta=f"{-miss_diff}" if miss_diff else "0")

        # Show updated smart type summary
        remaining_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if remaining_cat:
            st.warning(f"⚠️ {len(remaining_cat)} columns still need encoding before ML: {remaining_cat}")
        else:
            st.success("✅ All columns are numeric — ready for feature selection and training!")

        st.dataframe(df.head(30), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Processed Data", csv, "processed_data.csv", "text/csv")
        with col_b:
            if st.button("🔄 Reset to Original Data", key="reset_data"):
                st.session_state.df = st.session_state.df_original.copy()
                st.success("Data reset to original.")
                st.rerun()


# ─────────────────────────────────────────────────────────
# ④ FEATURE SELECTION & SPLIT
# ─────────────────────────────────────────────────────────
elif active_step == "④ Feature Selection & Split":
    df = st.session_state.df
    badge("④ FEATURE SELECTION & SPLIT", "blue")
    section_header("Select Features and Split Data")

    hint_banner("🎯", "<strong>How to use this step:</strong> Select your <strong>target column</strong> (what you want to predict) and <strong>feature columns</strong> (input variables). The system auto-detects whether it's a classification or regression task. Adjust the <strong>test set size</strong> (20% is standard) and click <strong>Split Data</strong>.")

    target_col = st.selectbox("🎯 Select Target Column (y)", df.columns.tolist(), key="target_col")

    available_features = [c for c in df.columns if c != target_col]
    feature_cols = st.multiselect(
        "📊 Select Feature Columns (X)",
        available_features,
        default=available_features,
        key="feature_cols_widget"
    )

    if target_col and feature_cols:
        detected_task = auto_detect_task(df[target_col])
        st.info(f"🔎 Auto-detected task type: **{detected_task}** (based on target column properties)")

        task_override = st.selectbox(
            "Task Type (override if needed)",
            ["Classification", "Regression"],
            index=0 if detected_task == "Classification" else 1,
            key="task_override"
        )
        st.session_state.task_type = task_override

        if task_override == "Classification":
            n_classes = df[target_col].nunique()
            if n_classes == 2:
                detected_subtype = "Binary Classification (2 classes)"
            else:
                detected_subtype = f"Multiclass Classification ({n_classes} classes)"
            st.info(f"🔎 Detected: **{detected_subtype}**")

            class_subtype = st.selectbox(
                "Classification Sub-Type",
                ["Binary Classification", "Multiclass Classification"],
                index=0 if n_classes == 2 else 1,
                key="class_subtype_widget"
            )
            st.session_state.class_subtype = class_subtype
        else:
            st.session_state.class_subtype = None

        # Target distribution preview
        st.markdown("**Target Distribution:**")
        if task_override == "Classification":
            vc = df[target_col].value_counts()
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Class Distribution — {target_col}",
                         color=vc.index.astype(str), color_discrete_sequence=PALETTE_VIVID)
            fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            min_class = vc.min()
            max_class = vc.max()
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            singleton_classes = vc[vc < 2]
            n_singletons = len(singleton_classes)

            if n_singletons > 0:
                st.warning(
                    f"⚠️ **{n_singletons} class{'es' if n_singletons > 1 else ''} ha{'ve' if n_singletons > 1 else 's'} only 1 sample** — "
                    f"stratified splitting will be disabled automatically. These rare classes cannot be guaranteed in both train and test sets. "
                    f"Consider collecting more data or merging rare classes."
                )
                with st.expander(f"Show {n_singletons} singleton classes"):
                    st.dataframe(
                        pd.DataFrame({"Class": singleton_classes.index, "Count": singleton_classes.values}),
                        use_container_width=True
                    )
            elif imbalance_ratio > 3:
                st.warning(f"⚠️ Class imbalance detected (ratio {imbalance_ratio:.1f}:1). Consider using balanced class weights in training.")

            # Hint about target encoding
            if df[target_col].dtype == "object" or df[target_col].dtype.name == "category":
                hint_banner("🏷️",
                    "<strong>Target column is text-based</strong> — this is fine at the split stage. "
                    "The target will be <strong>automatically label-encoded</strong> (text → numbers) in Step ⑤ Model Training. "
                    "You do <strong>not</strong> need to encode it manually. Do <strong>not</strong> scale the target — "
                    "only feature columns (X) should be scaled/encoded in preprocessing."
                )
        else:
            fig = px.histogram(df, x=target_col, nbins=50, marginal="violin",
                               color_discrete_sequence=["#00d4aa"])
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)
 
            # Large target value hint for NN regression
            if df[target_col].dtype in ["int64", "float64"]:
                tgt_range = df[target_col].max() - df[target_col].min()
                if tgt_range > 10000:
                    hint_banner("📏",
                        f"<strong>Target has a large range ({tgt_range:,.0f})</strong>. "
                        "Tree-based models handle this fine. For <strong>Neural Networks</strong>, "
                        "large target values can cause gradient instability — optional target "
                        "scaling will be available in Step ⑤ when you select Neural Network."
                    )

        divider()

        # Feature correlation with target
        st.markdown("**Feature-Target Correlation:**")
        numeric_features = [c for c in feature_cols if df[c].dtype in ["int64", "float64"]]
        if numeric_features and df[target_col].dtype in ["int64", "float64"]:
            corr_with_target = df[numeric_features].corrwith(df[target_col]).abs().sort_values(ascending=True)
            fig = px.bar(
                x=corr_with_target.values, y=corr_with_target.index,
                orientation="h", title=f"Absolute Correlation with {target_col}",
                color=corr_with_target.values,
                color_continuous_scale=HEATMAP_SCALE
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=max(300, len(corr_with_target)*22))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif df[target_col].dtype == "object":
            st.caption("Feature-target correlation is not available for text-based targets. This will work after target encoding in Step ⑤.")

        divider()

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            test_size = st.slider("Test set size", 0.10, 0.50, 0.20, 0.05, key="test_size")
        with col_b:
            random_state = st.number_input("Random state", value=42, key="rand_state")
        with col_c:
            shuffle = st.checkbox("Shuffle before split", value=True, key="shuffle_split")

        if st.button("✂️ Split Data", key="split_btn"):
            X = df[feature_cols]
            y = df[target_col]

            remaining_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
            if remaining_cat:
                st.warning(f"⚠️ These feature columns are still categorical and need encoding first: {remaining_cat}")
                st.stop()

            # Smart stratification: only stratify if ALL classes have ≥2 samples
            stratify_val = None
            if task_override == "Classification":
                class_counts = y.value_counts()
                min_count = class_counts.min()
                n_classes = y.nunique()
                min_test_samples = max(1, int(len(y) * test_size))

                if min_count >= 2 and n_classes <= 50:
                    stratify_val = y
                    st.caption("✓ Using stratified split (all classes have ≥2 samples).")
                else:
                    stratify_val = None
                    if min_count < 2:
                        singleton_count = (class_counts < 2).sum()
                        st.info(
                            f"ℹ️ Stratification disabled — {singleton_count} class{'es have' if singleton_count > 1 else ' has'} "
                            f"only 1 sample. Using random shuffle split instead."
                        )
                    else:
                        st.info(f"ℹ️ Stratification disabled — too many classes ({n_classes}). Using random shuffle split.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state),
                stratify=stratify_val, shuffle=shuffle
            )

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_cols = feature_cols
            st.session_state["_target_col"] = target_col

            st.success("Data split successfully!")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("X_train", f"{X_train.shape}")
            c2.metric("X_test", f"{X_test.shape}")
            c3.metric("y_train", f"{y_train.shape[0]}")
            c4.metric("y_test", f"{y_test.shape[0]}")
    else:
        st.warning("Please select both a target column and at least one feature column.")


# ─────────────────────────────────────────────────────────
# ⑤ MODEL TRAINING
# ─────────────────────────────────────────────────────────
elif active_step == "⑤ Model Training":
    badge("⑤ MODEL TRAINING", "rose")
    section_header("Select and Train Your Model")

    hint_banner("🚀", "<strong>How to use this step:</strong> Choose between <strong>Single Model</strong> (pick one algorithm, tune its hyperparameters), <strong>Model Comparison</strong> (train multiple algorithms side-by-side), or <strong>Auto-Tune</strong> (automated hyperparameter search). Start with Model Comparison to find the best algorithm, then fine-tune it with Auto-Tune.")

    if st.session_state.X_train is None:
        st.warning("⚠️ Please split your data first in Step ④.")
        st.stop()

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    task_type = st.session_state.task_type

    # ── ROBUST TARGET ENCODING ──
    y_train_enc, y_test_enc, label_encoder, target_info = encode_target_safely(y_train, y_test, task_type)
    st.session_state.y_train_enc = y_train_enc
    st.session_state.y_test_enc = y_test_enc
    st.session_state.label_encoder = label_encoder
    st.session_state.target_info = target_info
 
    # Show what happened with target encoding
    if target_info["method"] == "label_encoder":
        hint_banner("🏷️",
            f"<strong>Target auto-encoded:</strong> LabelEncoder → <strong>{target_info['n_classes']} classes</strong> "
            f"mapped to integers (0 to {target_info['n_classes']-1}). Original class names are preserved "
            f"for display in evaluation. <em>Never use OneHotEncoder for the target</em> — it creates "
            f"multiple columns which breaks classification."
        )
    elif target_info["method"] == "numeric_passthrough":
        hint_banner("🔢",
            f"<strong>Target already numeric</strong> — {target_info['n_classes']} unique values detected. "
            f"No encoding applied. Using values as-is."
        )
    elif target_info["method"] == "regression_passthrough":
        hint_banner("📈",
            "<strong>Regression target</strong> — using raw numeric values. No encoding or scaling applied. "
            "Target scaling is <strong>never</strong> done by default. For Neural Networks with very large "
            "target values, an optional scaling option will appear below."
        )
 
    # Show any warnings from the encoding process
    for w in target_info.get("warnings", []):
        st.warning(f"⚠️ {w}")

    train_tab1, train_tab2, train_tab3 = st.tabs([
        "🎯 Single Model", "🏆 Model Comparison", "⚡ Auto-Tune (Hyperparameter Search)"
    ])

    with train_tab1:
        available_models = CLASSIFICATION_MODELS if task_type == "Classification" else REGRESSION_MODELS
        class_subtype = st.session_state.get("class_subtype", None)
        if class_subtype:
            st.markdown(f"**Task: {task_type}** ({class_subtype})")
        else:
            st.markdown(f"**Task: {task_type}**")

        model_name = st.selectbox("Select Algorithm", list(available_models.keys()), key="model_select")
        st.caption(available_models[model_name])

        st.markdown("---")
        st.markdown("**⚙️ Hyperparameters**")

        model = None
        is_neural_net = model_name == "Neural Network"

        if model_name == "Logistic Regression":
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                max_iter = st.slider("Max iterations", 100, 5000, 1000, key="lr_maxiter")
            with hp2:
                C = st.slider("Regularisation (C)", 0.01, 10.0, 1.0, key="lr_c")
            with hp3:
                penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], key="lr_penalty")
            solver = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"
            l1_ratio = None
            if penalty == "elasticnet":
                l1_ratio = st.slider("L1 Ratio (for ElasticNet)", 0.0, 1.0, 0.5, key="lr_l1")
            model = LogisticRegression(max_iter=max_iter, C=C, penalty=penalty if penalty != "none" else None,
                                       solver=solver, l1_ratio=l1_ratio)

        elif model_name == "Linear Regression":
            model = LinearRegression()
            st.info("Linear Regression has no major hyperparameters to tune.")

        elif model_name == "Ridge Regression":
            alpha = st.slider("Alpha (regularisation)", 0.01, 100.0, 1.0, key="ridge_alpha")
            model = Ridge(alpha=alpha)

        elif model_name == "Lasso Regression":
            alpha = st.slider("Alpha (regularisation)", 0.01, 100.0, 1.0, key="lasso_alpha")
            model = Lasso(alpha=alpha)

        elif model_name == "Elastic Net":
            hp1, hp2 = st.columns(2)
            with hp1:
                alpha = st.slider("Alpha", 0.01, 10.0, 1.0, key="en_alpha")
            with hp2:
                l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, key="en_l1")
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        elif model_name == "Random Forest":
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                n_est = st.slider("Number of trees", 10, 500, 100, key="rf_n")
            with hp2:
                max_depth = st.slider("Max depth (0=unlimited)", 0, 50, 0, key="rf_depth")
            with hp3:
                min_samples_split = st.slider("Min samples split", 2, 20, 2, key="rf_mss")
            hp4, hp5 = st.columns(2)
            with hp4:
                min_samples_leaf = st.slider("Min samples leaf", 1, 20, 1, key="rf_msl")
            with hp5:
                max_features = st.selectbox("Max features", ["sqrt", "log2", "None"], key="rf_mf")
            md = None if max_depth == 0 else max_depth
            mf = None if max_features == "None" else max_features
            if task_type == "Classification":
                class_weight = st.selectbox("Class weight", ["None", "balanced"], key="rf_cw")
                cw = None if class_weight == "None" else "balanced"
                model = RandomForestClassifier(n_estimators=n_est, max_depth=md, min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf, max_features=mf,
                                               class_weight=cw, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_est, max_depth=md, min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf, max_features=mf,
                                              random_state=42, n_jobs=-1)

        elif model_name == "XGBoost":
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                n_est = st.slider("Boosting rounds", 50, 1000, 200, key="xgb_n")
            with hp2:
                lr = st.slider("Learning rate", 0.01, 0.5, 0.1, key="xgb_lr")
            with hp3:
                max_depth = st.slider("Max depth", 2, 15, 6, key="xgb_depth")
            hp4, hp5, hp6 = st.columns(3)
            with hp4:
                subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05, key="xgb_sub")
            with hp5:
                colsample = st.slider("Colsample by tree", 0.5, 1.0, 0.8, 0.05, key="xgb_col")
            with hp6:
                reg_alpha = st.slider("L1 reg (alpha)", 0.0, 10.0, 0.0, key="xgb_alpha")
            reg_lambda = st.slider("L2 reg (lambda)", 0.0, 10.0, 1.0, key="xgb_lambda")
            if task_type == "Classification":
                model = XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                      subsample=subsample, colsample_bytree=colsample,
                                      reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                      use_label_encoder=False, eval_metric='logloss', random_state=42)
            else:
                model = XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                     subsample=subsample, colsample_bytree=colsample,
                                     reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=42)

        elif model_name == "Decision Tree":
            hp1, hp2 = st.columns(2)
            with hp1:
                max_depth = st.slider("Max depth (0=unlimited)", 0, 30, 0, key="dt_depth")
            with hp2:
                criterion = st.selectbox("Criterion",
                                         ["gini", "entropy"] if task_type == "Classification" else ["squared_error", "friedman_mse", "absolute_error"],
                                         key="dt_crit")
            min_samples_split = st.slider("Min samples split", 2, 20, 2, key="dt_mss")
            md = None if max_depth == 0 else max_depth
            if task_type == "Classification":
                model = DecisionTreeClassifier(max_depth=md, criterion=criterion, min_samples_split=min_samples_split, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=md, criterion=criterion, min_samples_split=min_samples_split, random_state=42)

        elif model_name == "Gradient Boosting":
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                n_est = st.slider("Number of estimators", 50, 500, 100, key="gb_n")
            with hp2:
                lr = st.slider("Learning rate", 0.01, 0.5, 0.1, key="gb_lr")
            with hp3:
                max_depth = st.slider("Max depth", 2, 15, 3, key="gb_depth")
            subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.05, key="gb_sub")
            if task_type == "Classification":
                model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                                   subsample=subsample, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=max_depth,
                                                  subsample=subsample, random_state=42)

        elif model_name == "AdaBoost":
            hp1, hp2 = st.columns(2)
            with hp1:
                n_est = st.slider("Number of estimators", 10, 300, 50, key="ada_n")
            with hp2:
                lr = st.slider("Learning rate", 0.01, 2.0, 1.0, key="ada_lr")
            if task_type == "Classification":
                model = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
            else:
                model = AdaBoostRegressor(n_estimators=n_est, learning_rate=lr, random_state=42)

        elif model_name == "Extra Trees":
            hp1, hp2 = st.columns(2)
            with hp1:
                n_est = st.slider("Number of trees", 10, 500, 100, key="et_n")
            with hp2:
                max_depth = st.slider("Max depth (0=unlimited)", 0, 50, 0, key="et_depth")
            md = None if max_depth == 0 else max_depth
            if task_type == "Classification":
                model = ExtraTreesClassifier(n_estimators=n_est, max_depth=md, random_state=42, n_jobs=-1)
            else:
                model = ExtraTreesRegressor(n_estimators=n_est, max_depth=md, random_state=42, n_jobs=-1)

        elif model_name == "SVM" or model_name == "SVR":
            hp1, hp2 = st.columns(2)
            with hp1:
                C = st.slider("C (Regularisation)", 0.01, 10.0, 1.0, key="svm_c")
            with hp2:
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
            gamma = st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma")
            if task_type == "Classification":
                model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
            else:
                model = SVR(C=C, kernel=kernel, gamma=gamma)

        elif model_name == "KNN":
            hp1, hp2, hp3 = st.columns(3)
            with hp1:
                k = st.slider("Number of neighbours (K)", 1, 50, 5, key="knn_k")
            with hp2:
                weights = st.selectbox("Weights", ["uniform", "distance"], key="knn_w")
            with hp3:
                metric = st.selectbox("Distance metric", ["minkowski", "euclidean", "manhattan", "chebyshev"], key="knn_metric")
            if task_type == "Classification":
                model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
            else:
                model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric)

        elif model_name == "Naive Bayes":
            var_smoothing = st.slider("Variance smoothing (log scale)", -12, -6, -9, key="nb_smooth")
            model = GaussianNB(var_smoothing=10**var_smoothing)

        elif model_name == "Neural Network":
            st.markdown("**🧠 Neural Network Architecture**")

            n_layers = st.slider("Number of hidden layers (depth)", 1, 10, 2, key="nn_layers")
            use_dropout = st.checkbox("Add Dropout between layers", value=True, key="nn_dropout_toggle")
            if use_dropout:
                dropout_rate = st.slider("Dropout rate", 0.05, 0.5, 0.2, 0.05, key="nn_dropout_rate")
            else:
                dropout_rate = 0.0

            use_batchnorm = st.checkbox("Add BatchNormalization", value=True, key="nn_bn")

            st.markdown("**Configure each layer:**")
            neurons = []
            activations = []
            for row_start in range(0, n_layers, 4):
                row_end = min(row_start + 4, n_layers)
                cols = st.columns(row_end - row_start)
                for idx, i in enumerate(range(row_start, row_end)):
                    with cols[idx]:
                        st.markdown(f"**Layer {i+1}**")
                        neurons.append(st.number_input(f"Neurons", 4, 2048, 64, key=f"nn_n_{i}"))
                        activations.append(st.selectbox(f"Activation", ["relu", "tanh", "sigmoid", "elu", "selu", "leaky_relu"], key=f"nn_a_{i}"))

            st.markdown("---")
            st.markdown("**Training Configuration:**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                epochs = st.slider("Epochs", 10, 1000, 50, key="nn_epochs")
            with tc2:
                batch_size = st.slider("Batch size", 8, 512, 32, key="nn_batch")
            with tc3:
                learning_rate = st.select_slider("Learning rate",
                    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05], value=0.001, key="nn_lr")

            tc4, tc5 = st.columns(2)
            with tc4:
                early_stop_patience = st.slider("Early stopping patience", 3, 50, 10, key="nn_patience")
            with tc5:
                optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"], key="nn_opt")

            # ── NN Output Configuration (using helper function) ──
            class_subtype = st.session_state.get("class_subtype", None)
            n_classes_nn = target_info["n_classes"] if task_type == "Classification" else 0
            nn_config = determine_nn_config(task_type, n_classes_nn, class_subtype)
 
            output_units = nn_config["output_units"]
            output_activation = nn_config["output_activation"]
            loss_fn = nn_config["loss_fn"]
            metrics = nn_config["metrics"]
 
            # Show the configuration explanation to the user
            st.markdown("**🔧 Output Layer Configuration:**")
            hint_banner("⚙️", f"<strong>{nn_config['desc']}</strong>")
 
            # ── Optional target scaling for NN regression ──
            scale_target_nn = False
            if task_type == "Regression":
                tgt_range = float(y_train_enc.max()) - float(y_train_enc.min())
                if tgt_range > 1000:
                    scale_target_nn = st.checkbox(
                        f"📏 Scale target for training (range: {tgt_range:,.0f})",
                        value=True, key="nn_scale_y",
                        help="Large target values can cause gradient instability in neural networks. "
                             "This normalises the target during training using StandardScaler, "
                             "then inverse-transforms predictions back to original units."
                    )
 
            # Architecture preview
            arch_str = f"Input ({X_train.shape[1]})"
            for i, (n, a) in enumerate(zip(neurons, activations)):
                arch_str += f"  →  Dense({n}, {a})"
                if use_batchnorm:
                    arch_str += f"  →  BN"
                if use_dropout and dropout_rate > 0:
                    arch_str += f"  →  Dropout({dropout_rate})"
            arch_str += f"  →  Output({output_units}, {output_activation})"
            arch_str += f"\n\nLoss: {loss_fn}  |  Optimizer: {optimizer_choice}(lr={learning_rate})"
            arch_str += f"\nLayers: {n_layers}  |  Epochs: {epochs}  |  Batch: {batch_size}  |  Early Stop: {early_stop_patience}"
            if scale_target_nn:
                arch_str += f"\nTarget Scaling: StandardScaler (inverse-transform on predictions)"
            st.code(arch_str, language=None)

        # Cross-validation option
        divider()
        use_cv = st.checkbox("Enable Cross-Validation", value=False, key="use_cv")
        if use_cv:
            cv_folds = st.slider("Number of folds", 3, 10, 5, key="cv_folds")

        # Train button
        if st.button("🚀 Train Model", type="primary", key="train_btn"):
            with st.spinner("Training in progress..."):
                start_time = time.time()

                if is_neural_net:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
                    from tensorflow.keras.optimizers import Adam, SGD, RMSprop
                    try:
                        from tensorflow.keras.optimizers import AdamW
                    except ImportError:
                        AdamW = Adam
                    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

                    nn_model = Sequential()
                    nn_model.add(Dense(neurons[0], activation=activations[0], input_dim=X_train.shape[1]))
                    if use_batchnorm:
                        nn_model.add(BatchNormalization())
                    if use_dropout and dropout_rate > 0:
                        nn_model.add(Dropout(dropout_rate))
                    for i in range(1, len(neurons)):
                        nn_model.add(Dense(neurons[i], activation=activations[i]))
                        if use_batchnorm:
                            nn_model.add(BatchNormalization())
                        if use_dropout and dropout_rate > 0:
                            nn_model.add(Dropout(dropout_rate))
                    nn_model.add(Dense(output_units, activation=output_activation))

                    opt_map = {"Adam": Adam, "SGD": SGD, "RMSprop": RMSprop, "AdamW": AdamW}
                    optimizer = opt_map.get(optimizer_choice, Adam)(learning_rate=learning_rate)
                    nn_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

                    # ── Optional target scaling for NN regression ──
                    y_scaler = None
                    y_train_nn = y_train_enc
                    y_test_nn = y_test_enc
                    if task_type == "Regression" and scale_target_nn:
                        y_scaler = StandardScaler()
                        y_train_nn = y_scaler.fit_transform(y_train_enc.reshape(-1, 1)).flatten()
                        y_test_nn = y_scaler.transform(y_test_enc.reshape(-1, 1)).flatten()
                        st.session_state.y_scaler = y_scaler
                    else:
                        st.session_state.y_scaler = None

                    # Live training display
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a2332, #0d1525);
                                border: 1px solid #2a3a50; border-radius: 12px;
                                padding: 1.2rem 1.5rem; margin-bottom: 1rem;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 1.3rem;">🧠</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;
                                         color: #00d4aa; font-weight: 600;">NEURAL NETWORK TRAINING LOG</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    progress_bar_nn = st.progress(0, text="Initialising training...")
                    epoch_log_container = st.container()
                    live_chart_placeholder = st.empty()

                    class StreamlitEpochLogger(Callback):
                        def __init__(self, total_epochs, progress_bar, log_container, chart_placeholder):
                            super().__init__()
                            self.total_epochs = total_epochs
                            self.progress_bar = progress_bar
                            self.log_container = log_container
                            self.chart_placeholder = chart_placeholder
                            self.epoch_logs_html = []
                            self.history_data = {"loss": [], "val_loss": [], "epoch": []}
                            self.metric_key = None
                            self.best_val_loss = float("inf")
                            self.best_epoch = 0
                            self.epoch_start_time = None

                        def on_epoch_begin(self, epoch, logs=None):
                            self.epoch_start_time = time.time()

                        def on_epoch_end(self, epoch, logs=None):
                            epoch_time = time.time() - self.epoch_start_time
                            progress = (epoch + 1) / self.total_epochs
                            self.progress_bar.progress(progress, text=f"Training epoch {epoch+1}/{self.total_epochs}  •  {progress*100:.0f}%  •  ~{epoch_time:.1f}s/epoch")

                            loss = logs.get("loss", 0)
                            val_loss = logs.get("val_loss", 0)
                            self.history_data["loss"].append(loss)
                            self.history_data["val_loss"].append(val_loss)
                            self.history_data["epoch"].append(epoch + 1)

                            if self.metric_key is None:
                                for k in logs.keys():
                                    if k not in ["loss", "val_loss"] and not k.startswith("val_"):
                                        self.metric_key = k
                                        self.history_data[k] = []
                                        self.history_data[f"val_{k}"] = []
                                        break

                            metric_val = logs.get(self.metric_key, None) if self.metric_key else None
                            val_metric_val = logs.get(f"val_{self.metric_key}", None) if self.metric_key else None
                            if self.metric_key and self.metric_key in self.history_data:
                                self.history_data[self.metric_key].append(metric_val)
                                self.history_data[f"val_{self.metric_key}"].append(val_metric_val)

                            is_best = val_loss < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_loss
                                self.best_epoch = epoch + 1

                            loss_ratio = val_loss / max(self.history_data["val_loss"][0], 1e-8)
                            loss_color = "#00d4aa" if loss_ratio <= 0.5 else "#10b981" if loss_ratio <= 0.85 else "#f59e0b" if loss_ratio <= 1.0 else "#f43f5e"

                            if epoch > 0:
                                prev_val_loss = self.history_data["val_loss"][-2]
                                if val_loss < prev_val_loss:
                                    arrow, arrow_color = "▼", "#00d4aa"
                                    delta = f"-{((prev_val_loss - val_loss)/prev_val_loss*100):.1f}%"
                                elif val_loss > prev_val_loss:
                                    arrow, arrow_color = "▲", "#f43f5e"
                                    delta = f"+{((val_loss - prev_val_loss)/prev_val_loss*100):.1f}%"
                                else:
                                    arrow, arrow_color, delta = "━", "#8899aa", "0.0%"
                            else:
                                arrow, arrow_color, delta = "●", "#3b82f6", "start"

                            best_badge = '<span style="background:#00d4aa; color:#0a0f1a; padding:1px 6px; border-radius:4px; font-size:0.65rem; font-weight:700; margin-left:6px;">★ BEST</span>' if is_best else ""

                            metric_str = ""
                            if metric_val is not None:
                                metric_name = self.metric_key.replace("_", " ").title()
                                metric_str = f'<span style="color:#8899aa;">  │  {metric_name}: </span><span style="color:#3b82f6; font-weight:600;">{metric_val:.4f}</span><span style="color:#8899aa;"> → val: </span><span style="color:#8b5cf6; font-weight:600;">{val_metric_val:.4f}</span>'

                            epoch_html = (
                                f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.78rem; '
                                f'padding: 6px 12px; margin: 2px 0; border-radius: 6px; '
                                f'background: {"rgba(0,212,170,0.06)" if is_best else "rgba(26,35,50,0.4)"}; '
                                f'border-left: 3px solid {loss_color}; display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">'
                                f'<span style="color:#8899aa; min-width: 75px;">Epoch {epoch+1:>4}/{self.total_epochs}</span>'
                                f'<span style="color:{arrow_color}; font-size:0.85rem; min-width: 16px;">{arrow}</span>'
                                f'<span style="color:#8899aa;">loss: </span><span style="color:#f59e0b; font-weight:600;">{loss:.4f}</span>'
                                f'<span style="color:#8899aa;">  │  val_loss: </span><span style="color:{loss_color}; font-weight:600;">{val_loss:.4f}</span>'
                                f'<span style="color:{arrow_color}; font-size:0.7rem;"> ({delta})</span>{metric_str}'
                                f'<span style="color:#8899aa; font-size:0.7rem; margin-left:auto;">{epoch_time:.2f}s</span>{best_badge}</div>'
                            )
                            self.epoch_logs_html.append(epoch_html)

                            visible_logs = self.epoch_logs_html[-15:]
                            log_html = '<div style="background: #0a0f1a; border: 1px solid #2a3a50; border-radius: 10px; padding: 10px; max-height: 420px; overflow-y: auto;">' + "".join(visible_logs) + '</div>'
                            if len(self.epoch_logs_html) > 15:
                                log_html = f'<div style="color:#8899aa; font-size:0.72rem; font-family: \'JetBrains Mono\', monospace; padding: 4px 12px; margin-bottom: 4px;">... {len(self.epoch_logs_html)-15} earlier epochs hidden</div>' + log_html
                            self.log_container.markdown(log_html, unsafe_allow_html=True)

                            if (epoch + 1) % max(1, min(3, self.total_epochs // 20)) == 0 or epoch == self.total_epochs - 1 or is_best:
                                with self.chart_placeholder.container():
                                    chart_fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss Curve", f"{self.metric_key.replace('_',' ').title() if self.metric_key else 'Metric'} Curve"], horizontal_spacing=0.1)
                                    chart_fig.add_trace(go.Scatter(x=self.history_data["epoch"], y=self.history_data["loss"], name="Train Loss", mode="lines", line=dict(color="#00d4aa", width=2.5)), row=1, col=1)
                                    chart_fig.add_trace(go.Scatter(x=self.history_data["epoch"], y=self.history_data["val_loss"], name="Val Loss", mode="lines", line=dict(color="#f43f5e", width=2.5, dash="dot")), row=1, col=1)
                                    chart_fig.add_trace(go.Scatter(x=[self.best_epoch], y=[self.best_val_loss], name="Best", mode="markers", marker=dict(color="#f59e0b", size=10, symbol="star", line=dict(width=1.5, color="#fff"))), row=1, col=1)
                                    if self.metric_key and self.metric_key in self.history_data:
                                        chart_fig.add_trace(go.Scatter(x=self.history_data["epoch"], y=self.history_data[self.metric_key], name=f"Train {self.metric_key}", mode="lines", line=dict(color="#3b82f6", width=2.5)), row=1, col=2)
                                        chart_fig.add_trace(go.Scatter(x=self.history_data["epoch"], y=self.history_data[f"val_{self.metric_key}"], name=f"Val {self.metric_key}", mode="lines", line=dict(color="#8b5cf6", width=2.5, dash="dot")), row=1, col=2)
                                    chart_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,35,50,0.6)", font=dict(family="DM Sans", color="#e8ecf1", size=11), margin=dict(l=40, r=20, t=45, b=35), height=280, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, font_size=10))
                                    chart_fig.update_xaxes(gridcolor="#2a3a50", zerolinecolor="#2a3a50", title_text="Epoch", title_font_size=10)
                                    chart_fig.update_yaxes(gridcolor="#2a3a50", zerolinecolor="#2a3a50")
                                    st.plotly_chart(chart_fig, use_container_width=True)

                        def on_train_end(self, logs=None):
                            self.progress_bar.progress(1.0, text="✅ Training complete!")

                    streamlit_logger = StreamlitEpochLogger(epochs, progress_bar_nn, epoch_log_container, live_chart_placeholder)
                    callbacks = [
                        EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True),
                        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, early_stop_patience//2), min_lr=1e-6),
                        streamlit_logger
                    ]

                    history = nn_model.fit(
                        X_train, y_train_nn, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test_nn), callbacks=callbacks, verbose=0
                    )

                    total_epochs_run = len(history.history["loss"])
                    best_val = min(history.history["val_loss"])
                    best_ep = history.history["val_loss"].index(best_val) + 1
                    stopped_early = total_epochs_run < epochs

                    summary_color = "#00d4aa" if best_val < history.history["val_loss"][0] * 0.5 else "#f59e0b"
                    early_stop_text = f'<span style="color:#f59e0b;">Early stopped at epoch {total_epochs_run}</span>' if stopped_early else f'<span style="color:#8899aa;">Completed all {epochs} epochs</span>'

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0d2818, #1a2332); border: 1px solid rgba(0,212,170,0.3); border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;">
                            <span style="font-size: 1.2rem;">✅</span>
                            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #00d4aa; font-weight: 700;">TRAINING COMPLETE</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
                            <div><div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase;">Best Val Loss</div><div style="color: {summary_color}; font-size: 1.1rem; font-weight: 700;">{best_val:.4f}</div></div>
                            <div><div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase;">Best Epoch</div><div style="color: #3b82f6; font-size: 1.1rem; font-weight: 700;">{best_ep}/{total_epochs_run}</div></div>
                            <div><div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase;">Loss Reduction</div><div style="color: #00d4aa; font-size: 1.1rem; font-weight: 700;">{((history.history['val_loss'][0] - best_val) / history.history['val_loss'][0] * 100):.1f}%</div></div>
                            <div><div style="color: #8899aa; font-size: 0.7rem; text-transform: uppercase;">Status</div><div>{early_stop_text}</div></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    st.session_state.model = nn_model
                    st.session_state.training_history = history.history
                    st.session_state.model_name = "Neural Network"
                    st.session_state.trained = True

                    raw_preds = nn_model.predict(X_test)
                    if task_type == "Classification":
                        preds = (raw_preds > 0.5).astype(int).flatten() if output_units == 1 else np.argmax(raw_preds, axis=1)
                    else:
                        preds = raw_preds.flatten()
                        # Inverse-transform if target was scaled
                        if y_scaler is not None:
                            preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                    st.session_state.preds = preds
                else:
                    model.fit(X_train, y_train_enc)
                    preds = model.predict(X_test)
                    st.session_state.model = model
                    st.session_state.model_name = model_name
                    st.session_state.trained = True
                    st.session_state.preds = preds

                    if use_cv and not is_neural_net:
                        scoring = "accuracy" if task_type == "Classification" else "r2"
                        try:
                            # Use StratifiedKFold for classification, but fall back to KFold
                            # if any class has fewer samples than the number of folds
                            if task_type == "Classification":
                                min_class_count = pd.Series(y_train_enc).value_counts().min()
                                if min_class_count >= cv_folds:
                                    cv_strategy = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
                                else:
                                    cv_strategy = KFold(cv_folds, shuffle=True, random_state=42)
                                    st.caption(f"⚠️ Using KFold instead of StratifiedKFold — some classes have fewer than {cv_folds} samples")
                            else:
                                cv_strategy = cv_folds  # sklearn handles int for regression
 
                            cv_scores = cross_val_score(model, X_train, y_train_enc, cv=cv_strategy, scoring=scoring)
                            st.session_state.cv_results = {
                                "scores": cv_scores, "mean": cv_scores.mean(),
                                "std": cv_scores.std(), "folds": cv_folds, "metric": scoring
                            }
                        except Exception as e:
                            st.warning(f"Cross-validation failed: {e}")
                            st.session_state.cv_results = None

                elapsed = time.time() - start_time

            st.success(f"✅ {model_name if not is_neural_net else 'Neural Network'} trained in {elapsed:.2f}s!")

            if task_type == "Classification":
                acc = accuracy_score(y_test_enc, st.session_state.preds)
                f1 = f1_score(y_test_enc, st.session_state.preds, average="weighted")
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc:.4f}")
                c2.metric("F1 Score", f"{f1:.4f}")
                c3.metric("Train Time", f"{elapsed:.2f}s")
            else:
                mse_val = mean_squared_error(y_test_enc, st.session_state.preds)
                r2_val = r2_score(y_test_enc, st.session_state.preds)
                c1, c2, c3 = st.columns(3)
                c1.metric("MSE", f"{mse_val:.4f}")
                c2.metric("R² Score", f"{r2_val:.4f}")
                c3.metric("Train Time", f"{elapsed:.2f}s")

            if st.session_state.cv_results:
                cv = st.session_state.cv_results
                st.markdown(f"**Cross-Validation ({cv['folds']}-fold):** Mean {cv['metric']}={cv['mean']:.4f} ± {cv['std']:.4f}")
                fig = px.bar(x=[f"Fold {i+1}" for i in range(cv['folds'])], y=cv['scores'], title=f"Cross-Validation Scores ({cv['metric']})", color=cv['scores'], color_continuous_scale=HEATMAP_SCALE)
                fig.add_hline(y=cv['mean'], line_dash="dash", line_color="#f43f5e", annotation_text=f"Mean: {cv['mean']:.4f}")
                fig.update_layout(**PLOTLY_LAYOUT, height=300)
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            st.info("→ Go to **⑥ Evaluation & Interpretation** for detailed results.")

    with train_tab2:
        hint_banner("🏆", "<strong>Model Comparison</strong> trains multiple algorithms on your data and ranks them side-by-side. This is the fastest way to find the best-performing model for your dataset. Select at least 3-5 models for a meaningful comparison.")

        if task_type == "Classification":
            model_options = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "SVM": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }
        else:
            model_options = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(random_state=42),
            }

        selected_models = st.multiselect("Select models to compare", list(model_options.keys()), default=list(model_options.keys())[:5], key="compare_models")

        if selected_models and st.button("🏆 Run Comparison", key="compare_btn"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, name in enumerate(selected_models):
                status_text.text(f"Training {name}...")
                progress_bar.progress((i) / len(selected_models))
                try:
                    m = model_options[name]
                    start = time.time()
                    m.fit(X_train, y_train_enc)
                    elapsed = time.time() - start
                    preds = m.predict(X_test)

                    if task_type == "Classification":
                        results.append({"Model": name, "Accuracy": accuracy_score(y_test_enc, preds), "Precision": precision_score(y_test_enc, preds, average="weighted", zero_division=0), "Recall": recall_score(y_test_enc, preds, average="weighted", zero_division=0), "F1 Score": f1_score(y_test_enc, preds, average="weighted", zero_division=0), "Time (s)": round(elapsed, 3)})
                    else:
                        results.append({"Model": name, "R² Score": r2_score(y_test_enc, preds), "MSE": mean_squared_error(y_test_enc, preds), "RMSE": np.sqrt(mean_squared_error(y_test_enc, preds)), "MAE": mean_absolute_error(y_test_enc, preds), "Time (s)": round(elapsed, 3)})
                except Exception as e:
                    st.warning(f"Failed to train {name}: {e}")

            progress_bar.progress(1.0)
            status_text.text("Comparison complete!")

            if results:
                results_df = pd.DataFrame(results)
                st.session_state.all_models_results = results_df
                sort_col = "F1 Score" if task_type == "Classification" else "R² Score"
                results_df = results_df.sort_values(sort_col, ascending=False)
                st.dataframe(results_df.style.format("{:.4f}", subset=[c for c in results_df.columns if c not in ["Model", "Time (s)"]]), use_container_width=True)

                if task_type == "Classification":
                    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
                else:
                    metric_cols = ["R² Score"]

                fig = go.Figure()
                for idx, row in results_df.iterrows():
                    color = PALETTE_VIVID[list(results_df.index).index(idx) % len(PALETTE_VIVID)]
                    if task_type == "Classification":
                        fig.add_trace(go.Bar(name=row["Model"], x=metric_cols, y=[row[m] for m in metric_cols], marker_color=color, opacity=0.85))
                    else:
                        fig.add_trace(go.Bar(name=row["Model"], x=["R²", "1-MSE", "1-MAE"],
                                             y=[row["R² Score"], max(0, 1-row["MSE"]/max(results_df["MSE"])), max(0, 1-row["MAE"]/max(results_df["MAE"]))],
                                             marker_color=color, opacity=0.85))
                fig.update_layout(**PLOTLY_LAYOUT, title="Model Comparison", barmode="group", height=400)
                st.plotly_chart(fig, use_container_width=True)

                best_model_name = results_df.iloc[0]["Model"]
                st.success(f"🥇 Best model: **{best_model_name}** ({sort_col}: {results_df.iloc[0][sort_col]:.4f})")
                if st.button(f"Use {best_model_name} as trained model", key="use_best"):
                    best_m = model_options[best_model_name]
                    best_m.fit(X_train, y_train_enc)
                    st.session_state.model = best_m
                    st.session_state.model_name = best_model_name
                    st.session_state.trained = True
                    st.session_state.preds = best_m.predict(X_test)
                    st.success(f"✅ {best_model_name} set as the active model!")

    with train_tab3:
        hint_banner("⚡", "<strong>Auto-Tune</strong> uses Randomized Search to find the best hyperparameter combination. Select a model, set the number of iterations (more = better results but slower), and let the search run. The best configuration is automatically set as your active model.")

        tune_model = st.selectbox("Select model to tune", ["Random Forest", "XGBoost", "Gradient Boosting", "KNN", "SVM/SVR", "Logistic Regression"], key="tune_model")
        n_iter = st.slider("Number of search iterations", 10, 100, 30, key="tune_iter")
        tune_cv = st.slider("CV folds for tuning", 3, 7, 3, key="tune_cv")

        param_grids = {
            "Random Forest": {"n_estimators": [50, 100, 200, 300, 500], "max_depth": [None, 5, 10, 15, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2", None]},
            "XGBoost": {"n_estimators": [50, 100, 200, 300, 500], "max_depth": [3, 5, 7, 9, 12], "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0], "reg_alpha": [0, 0.1, 1, 10], "reg_lambda": [0.1, 1, 5, 10]},
            "Gradient Boosting": {"n_estimators": [50, 100, 200, 300], "max_depth": [3, 5, 7, 10], "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "min_samples_split": [2, 5, 10]},
            "KNN": {"n_neighbors": [3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"], "metric": ["minkowski", "euclidean", "manhattan"]},
            "SVM/SVR": {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]},
            "Logistic Regression": {"C": [0.01, 0.1, 0.5, 1, 5, 10], "max_iter": [500, 1000, 2000], "solver": ["lbfgs", "saga"]},
        }

        if st.button("⚡ Start Hyperparameter Search", key="tune_btn"):
            with st.spinner(f"Running {n_iter} iterations of RandomizedSearchCV..."):
                if tune_model == "Random Forest":
                    base = RandomForestClassifier(random_state=42, n_jobs=-1) if task_type == "Classification" else RandomForestRegressor(random_state=42, n_jobs=-1)
                elif tune_model == "XGBoost":
                    base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42) if task_type == "Classification" else XGBRegressor(random_state=42)
                elif tune_model == "Gradient Boosting":
                    base = GradientBoostingClassifier(random_state=42) if task_type == "Classification" else GradientBoostingRegressor(random_state=42)
                elif tune_model == "KNN":
                    base = KNeighborsClassifier() if task_type == "Classification" else KNeighborsRegressor()
                elif tune_model == "SVM/SVR":
                    base = SVC(probability=True, random_state=42) if task_type == "Classification" else SVR()
                else:
                    base = LogisticRegression()

                scoring = "f1_weighted" if task_type == "Classification" else "r2"
                search = RandomizedSearchCV(base, param_grids[tune_model], n_iter=n_iter, cv=tune_cv, scoring=scoring, n_jobs=-1, random_state=42, verbose=0)
                search.fit(X_train, y_train_enc)

                st.session_state.tuning_results = {"best_params": search.best_params_, "best_score": search.best_score_, "cv_results": pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(20)}
                st.session_state.model = search.best_estimator_
                st.session_state.model_name = f"{tune_model} (Tuned)"
                st.session_state.trained = True
                st.session_state.preds = search.best_estimator_.predict(X_test)

            st.success(f"✅ Best {scoring}: {search.best_score_:.4f}")
            st.markdown("**Best Parameters:**")
            st.json(search.best_params_)

            top_results = st.session_state.tuning_results["cv_results"][["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10)
            st.dataframe(top_results, use_container_width=True)

            cv_res = st.session_state.tuning_results["cv_results"]
            fig = px.scatter(cv_res.head(50), x=cv_res.head(50).index, y="mean_test_score", error_y="std_test_score", title="Hyperparameter Search Results", color="mean_test_score", color_continuous_scale=HEATMAP_SCALE, labels={"x": "Configuration", "mean_test_score": scoring})
            fig.update_layout(**PLOTLY_LAYOUT, height=350)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# ⑥ EVALUATION & INTERPRETATION
# ─────────────────────────────────────────────────────────
elif active_step == "⑥ Evaluation & Interpretation":
    badge("⑥ EVALUATION & INTERPRETATION", "violet")
    section_header("Model Evaluation & Interpretation")

    hint_banner("📈", "<strong>How to use this step:</strong> Review the <strong>Metrics</strong> tab for overall performance numbers. Use <strong>Visualisations</strong> to inspect the confusion matrix, ROC curve, or residual plots. <strong>Feature Importance</strong> reveals which features drive predictions. The <strong>Prediction Explorer</strong> lets you inspect individual test samples.")

    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first in Step ⑤.")
        st.stop()

    model = st.session_state.model
    model_name = st.session_state.model_name
    task_type = st.session_state.task_type
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_test_enc = st.session_state.y_test_enc
    y_train_enc = st.session_state.y_train_enc
    preds = st.session_state.preds

    st.markdown(f"**Model:** `{model_name}` &nbsp;|&nbsp; **Task:** `{task_type}`")

    eval_tab1, eval_tab2, eval_tab3, eval_tab4, eval_tab5 = st.tabs([
        "📊 Metrics", "📈 Visualisations", "🧠 Model Structure",
        "🔍 Feature Importance", "📋 Prediction Explorer"
    ])

    with eval_tab1:
        if task_type == "Classification":
            acc = accuracy_score(y_test_enc, preds)
            prec = precision_score(y_test_enc, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test_enc, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test_enc, preds, average="weighted", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test_enc, preds)
            mcc = matthews_corrcoef(y_test_enc, preds)
            kappa = cohen_kappa_score(y_test_enc, preds)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc:.4f}")
            c2.metric("Precision", f"{prec:.4f}")
            c3.metric("Recall", f"{rec:.4f}")
            c4.metric("F1 Score", f"{f1:.4f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Balanced Accuracy", f"{bal_acc:.4f}")
            c6.metric("MCC", f"{mcc:.4f}")
            c7.metric("Cohen's Kappa", f"{kappa:.4f}")

            try:
                if model_name == "Neural Network":
                    proba = model.predict(X_test)
                    ll = log_loss(y_test_enc, proba.flatten() if proba.shape[-1] == 1 else proba)
                elif hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    ll = log_loss(y_test_enc, proba)
                else:
                    ll = None
                if ll is not None:
                    st.metric("Log Loss", f"{ll:.4f}")
            except Exception:
                pass

            st.markdown("---")
            st.markdown("**Classification Report**")
            le = st.session_state.label_encoder
            target_names = [str(c) for c in le.classes_] if le else [str(c) for c in sorted(np.unique(y_test_enc))]
            report = classification_report(y_test_enc, preds, target_names=target_names, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}", na_rep=""), use_container_width=True)
        else:
            mse_val = mean_squared_error(y_test_enc, preds)
            rmse_val = np.sqrt(mse_val)
            mae_val = mean_absolute_error(y_test_enc, preds)
            r2_val = r2_score(y_test_enc, preds)
            ev = explained_variance_score(y_test_enc, preds)
            me = max_error(y_test_enc, preds)
            medae = median_absolute_error(y_test_enc, preds)

            try:
                train_preds = model.predict(X_train)
                if hasattr(train_preds, 'flatten'):
                    train_preds = train_preds.flatten()
                r2_train = r2_score(y_train_enc, train_preds)
            except Exception:
                r2_train = None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MSE", f"{mse_val:.4f}")
            c2.metric("RMSE", f"{rmse_val:.4f}")
            c3.metric("MAE", f"{mae_val:.4f}")
            c4.metric("R² Score", f"{r2_val:.4f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Explained Variance", f"{ev:.4f}")
            c6.metric("Max Error", f"{me:.4f}")
            c7.metric("Median Abs Error", f"{medae:.4f}")

            if r2_train is not None:
                gap = r2_train - r2_val
                if gap > 0.1:
                    st.warning(f"⚠️ Potential overfitting: Train R²={r2_train:.4f} vs Test R²={r2_val:.4f} (gap: {gap:.4f})")
                else:
                    st.success(f"✓ Good generalisation: Train R²={r2_train:.4f} vs Test R²={r2_val:.4f}")

    with eval_tab2:
        if task_type == "Classification":
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_test_enc, preds)
            le = st.session_state.label_encoder
            labels = [str(c) for c in le.classes_] if le else [str(c) for c in sorted(np.unique(y_test_enc))]

            cm_type = st.selectbox("Display", ["Counts", "Normalised (row %)", "Both"], key="cm_type")
            if cm_type == "Normalised (row %)":
                cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
                text_fmt = ".1f"
            else:
                cm_display = cm
                text_fmt = "d"

            fig = px.imshow(cm_display, text_auto=text_fmt, x=labels, y=labels, color_continuous_scale=HEATMAP_SCALE, title="Confusion Matrix")
            fig.update_layout(**PLOTLY_LAYOUT, height=450)
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual")
            st.plotly_chart(fig, use_container_width=True)

            n_classes = len(np.unique(y_test_enc))
            if n_classes == 2:
                try:
                    if model_name == "Neural Network":
                        proba = model.predict(X_test).flatten()
                    elif hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        proba = model.decision_function(X_test)
                    else:
                        proba = None

                    if proba is not None:
                        col_roc, col_pr = st.columns(2)
                        with col_roc:
                            fpr, tpr, _ = roc_curve(y_test_enc, proba)
                            roc_auc = auc(fpr, tpr)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})", line=dict(color="#00d4aa", width=2.5), fill="tozeroy", fillcolor="rgba(0,212,170,0.1)"))
                            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(color="#8899aa", dash="dash")))
                            fig.update_layout(**PLOTLY_LAYOUT, title="ROC Curve", height=400, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                            st.plotly_chart(fig, use_container_width=True)
                        with col_pr:
                            precision_vals, recall_vals, _ = precision_recall_curve(y_test_enc, proba)
                            pr_auc = auc(recall_vals, precision_vals)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode="lines", name=f"PR (AUC={pr_auc:.3f})", line=dict(color="#f59e0b", width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"))
                            fig.update_layout(**PLOTLY_LAYOUT, title="Precision-Recall Curve", height=400, xaxis_title="Recall", yaxis_title="Precision")
                            st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            if len(labels) > 2:
                st.markdown("**Per-Class Metrics:**")
                report_dict = classification_report(y_test_enc, preds, target_names=labels, output_dict=True)
                class_metrics = {k: v for k, v in report_dict.items() if k in labels}
                if class_metrics:
                    cm_df = pd.DataFrame(class_metrics).T
                    fig = go.Figure()
                    for metric, color in zip(["precision", "recall", "f1-score"], ["#00d4aa", "#f59e0b", "#3b82f6"]):
                        if metric in cm_df.columns:
                            fig.add_trace(go.Bar(name=metric.title(), x=cm_df.index, y=cm_df[metric], marker_color=color, opacity=0.85))
                    fig.update_layout(**PLOTLY_LAYOUT, title="Per-Class Performance", barmode="group", height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test_enc, y=preds, mode="markers", marker=dict(color="#00d4aa", size=5, opacity=0.5, line=dict(width=0.5, color="#047857")), name="Predictions"))
                min_val = min(y_test_enc.min(), preds.min())
                max_val = max(y_test_enc.max(), preds.max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", line=dict(color="#f43f5e", dash="dash", width=2), name="Perfect"))
                fig.update_layout(**PLOTLY_LAYOUT, title="Actual vs Predicted", height=450, xaxis_title="Actual", yaxis_title="Predicted")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                residuals = y_test_enc - preds
                fig = px.histogram(residuals, nbins=50, title="Residual Distribution", color_discrete_sequence=["#f59e0b"], marginal="box")
                fig.update_layout(**PLOTLY_LAYOUT, height=450)
                st.plotly_chart(fig, use_container_width=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=preds, y=residuals, mode="markers", marker=dict(color="#3b82f6", size=4, opacity=0.5), name="Residuals"))
            fig.add_hline(y=0, line_dash="dash", line_color="#f43f5e")
            fig.update_layout(**PLOTLY_LAYOUT, title="Residuals vs Predicted", height=350, xaxis_title="Predicted", yaxis_title="Residual")
            st.plotly_chart(fig, use_container_width=True)

        if model_name == "Neural Network" and st.session_state.training_history:
            history = st.session_state.training_history
            st.markdown("**Training History**")
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Metric"])
            fig.add_trace(go.Scatter(y=history["loss"], name="Train Loss", line=dict(color="#00d4aa", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss", line=dict(color="#f43f5e", width=2, dash="dot")), row=1, col=1)
            metric_key = [k for k in history.keys() if k not in ["loss", "val_loss"] and not k.startswith("val_")][0]
            fig.add_trace(go.Scatter(y=history[metric_key], name=f"Train {metric_key}", line=dict(color="#00d4aa", width=2)), row=1, col=2)
            fig.add_trace(go.Scatter(y=history[f"val_{metric_key}"], name=f"Val {metric_key}", line=dict(color="#f43f5e", width=2, dash="dot")), row=1, col=2)
            fig.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)
            best_epoch = np.argmin(history["val_loss"])
            st.info(f"Best validation loss at epoch {best_epoch+1}/{len(history['loss'])} (val_loss={history['val_loss'][best_epoch]:.4f})")

    with eval_tab3:
        st.markdown("**Model Architecture / Structure**")
        if model_name == "Neural Network":
            st.markdown("**Neural Network Summary:**")
            summary_buf = io.StringIO()
            model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))
            st.code(summary_buf.getvalue(), language=None)
            layers_info = []
            for layer in model.layers:
                config = layer.get_config()
                layers_info.append({"Layer": layer.name, "Type": layer.__class__.__name__, "Units/Filters": config.get("units", "—"), "Activation": config.get("activation", "—"), "Parameters": layer.count_params()})
            st.dataframe(pd.DataFrame(layers_info), use_container_width=True)
            st.metric("Total Parameters", f"{model.count_params():,}")
        elif "Decision Tree" in model_name:
            st.markdown("**Decision Tree Structure (text):**")
            tree_text = export_text(model, feature_names=list(X_train.columns), max_depth=5)
            st.code(tree_text, language=None)
            st.markdown("**Tree Visualisation:**")
            from sklearn.tree import plot_tree
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(model, feature_names=list(X_train.columns), filled=True, rounded=True, ax=ax, max_depth=4, fontsize=8, class_names=True if task_type == "Classification" else False)
            ax.set_facecolor("#1a2332")
            fig.patch.set_facecolor("#0a0f1a")
            st.pyplot(fig)
        elif any(n in model_name for n in ["Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees"]):
            st.info(f"{model_name} is an ensemble. Showing first estimator tree.")
            try:
                first_tree = model.estimators_[0]
                if hasattr(first_tree, 'tree_'):
                    from sklearn.tree import plot_tree
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(first_tree, feature_names=list(X_train.columns), filled=True, rounded=True, ax=ax, max_depth=3, fontsize=8)
                    ax.set_facecolor("#1a2332")
                    fig.patch.set_facecolor("#0a0f1a")
                    st.pyplot(fig)
            except Exception as e:
                st.caption(f"Could not visualise tree: {e}")
            if hasattr(model, "n_estimators"):
                st.metric("Number of Estimators", model.n_estimators)
        else:
            st.info(f"`{model_name}` does not have a tree structure. Check Feature Importance for insights.")
            if hasattr(model, "get_params"):
                st.json(model.get_params())

    with eval_tab4:
        st.markdown("**Feature Importance** — which features had the most impact?")
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title=f"Feature Importance — {model_name}", color="Importance", color_continuous_scale=HEATMAP_SCALE)
            fig.update_layout(**PLOTLY_LAYOUT, height=max(350, len(imp_df)*25))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, "coef_"):
            coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            if len(coefs) == len(X_train.columns):
                imp_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": coefs}).sort_values("Coefficient", key=abs, ascending=True)
                fig = px.bar(imp_df, x="Coefficient", y="Feature", orientation="h", title=f"Feature Coefficients — {model_name}", color="Coefficient", color_continuous_scale=DIVERGING_SCALE, color_continuous_midpoint=0)
                fig.update_layout(**PLOTLY_LAYOUT, height=max(350, len(imp_df)*25))
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This model doesn't provide direct feature importance.")

        divider()
        st.markdown("**Permutation Importance** — measures how much each feature affects predictions when its values are shuffled.")
        if st.button("Compute Permutation Importance", key="perm_imp_btn"):
            with st.spinner("Computing permutation importance..."):
                sample_size = min(500, len(X_test))
                X_sample = X_test.iloc[:sample_size].copy()
                y_sample = y_test_enc[:sample_size]
                is_nn = model_name == "Neural Network"

                if is_nn:
                    def _score_fn(X_in, y_in):
                        raw = model.predict(X_in, verbose=0)
                        if task_type == "Classification":
                            p = (raw > 0.5).astype(int).flatten() if raw.shape[-1] == 1 else np.argmax(raw, axis=1)
                            return accuracy_score(y_in, p)
                        else:
                            return r2_score(y_in, raw.flatten())
                    baseline = _score_fn(X_sample, y_sample)
                    imp_means, imp_stds = [], []
                    for col_idx in range(X_sample.shape[1]):
                        scores_drop = []
                        for _ in range(5):
                            X_perm = X_sample.copy()
                            X_perm.iloc[:, col_idx] = np.random.permutation(X_perm.iloc[:, col_idx].values)
                            scores_drop.append(baseline - _score_fn(X_perm, y_sample))
                        imp_means.append(np.mean(scores_drop))
                        imp_stds.append(np.std(scores_drop))
                    perm_df = pd.DataFrame({"Feature": X_train.columns, "Importance Mean": imp_means, "Importance Std": imp_stds}).sort_values("Importance Mean", ascending=True)
                else:
                    scoring = "accuracy" if task_type == "Classification" else "r2"
                    perm_result = permutation_importance(model, X_sample, y_sample, n_repeats=10, random_state=42, scoring=scoring)
                    perm_df = pd.DataFrame({"Feature": X_train.columns, "Importance Mean": perm_result.importances_mean, "Importance Std": perm_result.importances_std}).sort_values("Importance Mean", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=perm_df["Importance Mean"], y=perm_df["Feature"], orientation="h", marker_color="#8b5cf6",
                                     error_x=dict(type="data", array=perm_df["Importance Std"], color="#e8ecf1", thickness=1), name="Permutation Importance"))
                fig.update_layout(**PLOTLY_LAYOUT, title="Permutation Importance", height=max(350, len(perm_df)*25))
                st.plotly_chart(fig, use_container_width=True)

        divider()
        st.markdown("**SHAP Analysis** (SHapley Additive exPlanations)")
        st.caption("SHAP values explain the contribution of each feature to individual predictions.")
        if st.button("Run SHAP Analysis", key="shap_btn"):
            try:
                import shap
                with st.spinner("Computing SHAP values..."):
                    sample_size = min(200, len(X_test))
                    X_sample = X_test.iloc[:sample_size]
                    if any(n in model_name for n in ["Random Forest", "XGBoost", "Gradient Boosting", "Decision Tree", "AdaBoost", "Extra Trees"]):
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_sample)
                    else:
                        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
                        shap_values = explainer.shap_values(X_sample)

                    st.markdown("**SHAP Summary Plot:**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values[0], X_sample, show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample, show=False)
                    fig = plt.gcf()
                    fig.patch.set_facecolor("#0a0f1a")
                    for ax in fig.axes:
                        ax.set_facecolor("#1a2332")
                        ax.tick_params(colors="#e8ecf1")
                        ax.xaxis.label.set_color("#e8ecf1")
                        ax.yaxis.label.set_color("#e8ecf1")
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"SHAP analysis failed: {e}")

    with eval_tab5:
        st.markdown("**Prediction Explorer** — inspect individual predictions and see where the model succeeds or fails.")
        pred_df = X_test.copy()
        pred_df["Actual"] = y_test_enc
        pred_df["Predicted"] = preds
 
        if task_type == "Classification":
            # Show original class names alongside numeric labels
            le = st.session_state.label_encoder
            if le is not None:
                try:
                    pred_df["Actual_Label"] = le.inverse_transform(y_test_enc)
                    pred_df["Predicted_Label"] = le.inverse_transform(preds)
                except Exception:
                    pass  # Fallback: just show numeric
 
            pred_df["Correct"] = (pred_df["Actual"] == pred_df["Predicted"])
            correct_pct = pred_df["Correct"].mean() * 100
            st.markdown(f"**{correct_pct:.1f}%** of test predictions are correct ({pred_df['Correct'].sum()}/{len(pred_df)})")
            view_filter = st.selectbox("Filter", ["All", "Correct Only", "Errors Only"], key="pred_filter")
            if view_filter == "Correct Only":
                pred_df = pred_df[pred_df["Correct"]]
            elif view_filter == "Errors Only":
                pred_df = pred_df[~pred_df["Correct"]]
        else:
            pred_df["Error"] = pred_df["Actual"] - pred_df["Predicted"]
            pred_df["Abs Error"] = abs(pred_df["Error"])
            view_filter = st.selectbox("Sort by", ["Default", "Highest Error", "Lowest Error"], key="pred_sort")
            if view_filter == "Highest Error":
                pred_df = pred_df.sort_values("Abs Error", ascending=False)
            elif view_filter == "Lowest Error":
                pred_df = pred_df.sort_values("Abs Error", ascending=True)

        st.dataframe(pred_df.head(100), use_container_width=True)
        csv_pred = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv_pred, "predictions.csv", "text/csv")