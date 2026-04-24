---
title: NetGaze ML Pipeline
emoji: 🔬
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
license: mit
short_description: Complete ML & Deep Learning Pipeline Dashboard
---

# 🔬 NetGaze ML Pipeline

> **A complete, no-code machine learning and deep learning pipeline — from raw CSV to evaluated model — in one beautiful interface.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006400)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Deployed](https://img.shields.io/badge/🤗-HF%20Spaces-yellow)](https://huggingface.co/spaces)

---

## Table of Contents

- [Overview](#overview)
- [What Makes NetGaze Different](#what-makes-netgaze-different)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [The NetGaze Pipeline (End-to-End)](#the-netgaze-pipeline-end-to-end)
  - [① Upload Data](#-upload-data)
  - [② Explore & Visualise](#-explore--visualise)
  - [③ Clean & Preprocess](#-clean--preprocess)
  - [④ Feature Selection & Split](#-feature-selection--split)
  - [⑤ Model Training](#-model-training)
  - [⑥ Evaluation & Interpretation](#-evaluation--interpretation)
- [Supported Algorithms](#supported-algorithms)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deployment on Hugging Face Spaces](#deployment-on-hugging-face-spaces)
- [How to Use NetGaze (Practical Walkthrough)](#how-to-use-netgaze-practical-walkthrough)
- [Design Philosophy](#design-philosophy)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

**NetGaze** is a complete, single-page Streamlit application that takes a user through every stage of a supervised machine learning workflow without writing a single line of code. Upload a CSV, and the app guides you through data exploration, cleaning, feature engineering, model selection, training, and evaluation, with interactive Plotly visualisations and production-grade ML logic at every step.

It is built for three audiences:

1. **Students and learners** who want to see an ML pipeline end-to-end with contextual hints at every stage.
2. **Analysts and domain experts** who need ML insights without learning Python.
3. **ML practitioners** who want a fast, interactive sandbox for quick model comparison, hyperparameter tuning, and dataset profiling before moving to production code.

Under the hood, NetGaze wires together **scikit-learn**, **XGBoost**, **TensorFlow/Keras**, **Plotly**, and **SHAP** behind a dark scientific UI that guides the user through six clearly-labelled pipeline stages.

---

## What Makes NetGaze Different

Most no-code ML tools either oversimplify (hiding everything useful) or overwhelm (surfacing every hyperparameter without guidance). NetGaze tries to thread that needle with four deliberate design choices:

1. **Smart column analysis.** Each column in your dataset is automatically profiled to detect its *semantic* type, not just "object" or "int64", but `pay_range`, `date_string`, `email`, `url`, `high_cardinality_categorical`, `id_column`, and so on. Each detected type gets a tailored preprocessing recommendation card explaining what to do and why.

2. **Contextual hint banners.** Every tab and sub-section begins with a blue hint banner that explains what this step is, why it matters, and how to use it. No one gets stuck staring at a page wondering what "Winsorisation" means.

3. **Hardened training logic.** Silent ML failures (non-numeric targets crashing XGBoost, singleton classes crashing stratified splits, Keras rejecting object columns) are caught upfront with clear, actionable error messages that tell the user exactly which previous step to revisit.

4. **Live neural network training log.** When training a Keras model, every epoch is streamed to the UI as a styled log entry with loss, validation loss, delta from previous epoch, per-epoch time, and a live-updating loss curve, so you can watch the model converge in real time.

---

## Key Features

| Capability | Details |
|------------|---------|
| **Data Upload** | CSV and Excel (.xlsx, .xls) support |
| **Data Quality Score** | 0 to 100 composite score from completeness, duplicates, type diversity, cardinality, and row count |
| **Smart Type Detection** | 13 semantic types detected automatically with tailored transform suggestions |
| **Interactive EDA** | 7 visualisation tabs: distributions, correlations, box/violin, pairwise, value counts, outlier analysis, statistical tests (Shapiro-Wilk, D'Agostino-Pearson, Q-Q plots) |
| **Preprocessing** | 7-tab cleaning suite: missing values, duplicates/drops, smart transforms, encoding, scaling, outlier treatment, preview |
| **Feature Engineering** | Pay range parsing, currency extraction, date feature expansion, TF-IDF vectorisation, keyword flags, URL feature extraction, email domain extraction |
| **Smart Splitting** | Stratified train/test split with automatic fallback to random split when singleton classes are detected |
| **Model Training** | 3 modes: Single Model, Model Comparison, Auto-Tune (RandomizedSearchCV) |
| **11+ Algorithms** | Logistic/Linear, Ridge, Lasso, ElasticNet, Random Forest, XGBoost, Decision Tree, Gradient Boosting, AdaBoost, Extra Trees, SVM/SVR, KNN, Naive Bayes, Neural Network |
| **Neural Network Builder** | Configurable depth (1 to 10 layers), per-layer neurons and activations, optional BatchNorm and Dropout, 4 optimisers, early stopping, learning rate reduction |
| **Live Training Log** | Per-epoch streaming UI with colour-coded delta tracking and live loss/metric charts |
| **Cross-Validation** | Stratified K-Fold for classification (automatic KFold fallback on low-count classes), K-Fold for regression |
| **Hyperparameter Search** | RandomizedSearchCV over curated param grids for 6 model families |
| **Evaluation Metrics** | Classification: accuracy, precision, recall, F1, balanced accuracy, MCC, Cohen's Kappa, log loss. Regression: MSE, RMSE, MAE, R², explained variance, max error, median abs error |
| **Model Interpretation** | Feature importance, coefficients, permutation importance, SHAP summary plots, decision tree visualisation |
| **Prediction Explorer** | Filter test predictions by correct/incorrect, sort by error magnitude, downloadable as CSV |
| **Dark Scientific UI** | Custom CSS theme with teal/amber accents, JetBrains Mono for numerics, DM Sans for body text |

---

## Tech Stack

```
Frontend & Orchestration:  Streamlit 1.56
Visualisation:             Plotly (interactive), Matplotlib + Seaborn (tree plots, SHAP)
Classical ML:              scikit-learn 1.4+
Gradient Boosting:         XGBoost 2.x
Deep Learning:             TensorFlow / Keras
Explainability:            SHAP
Data Handling:             Pandas, NumPy, SciPy
Statistical Tests:         SciPy (Shapiro-Wilk, D'Agostino-Pearson)
Text Vectorisation:        scikit-learn TfidfVectorizer
Fonts:                     JetBrains Mono, DM Sans (Google Fonts)
```

---

## The NetGaze Pipeline (End-to-End)

The app is structured as six sequential pipeline stages selected from the sidebar radio. Each stage builds on the previous one, and session state carries the dataset, splits, model, and results across stages.

---

### ① Upload Data

The entry point. Users drop a CSV or Excel file into the sidebar uploader. NetGaze reads it into a pandas DataFrame, stores both the working copy and an immutable original (for the reset button in Step ③), and clears any stale training artifacts from previous sessions.

The landing page before upload shows three quick-start cards explaining the pipeline.

---

### ② Explore & Visualise

The first full stage after upload. This is where the user learns what's actually in their data before touching it.

**Dataset Overview (top metrics strip)**

Five headline metrics: rows, columns, missing cells, duplicates, and a composite **Data Quality Score** out of 100.

The quality score is weighted as follows:
- 40 points for completeness (non-null ratio)
- 20 points for duplicate-freeness
- 10 points for type diversity (more than one dtype present)
- 15 points for healthy cardinality (not too sparse, not too unique)
- 15 points for row count sufficiency (target of 1,000+ rows)

**🧠 Column Analysis & Preprocessing Summary**

Every column is analysed by `detect_column_smart_type()` which looks beyond pandas dtypes to detect 13 semantic types:

| Smart Type | Icon | What It Detects |
|------------|------|-----------------|
| `pay_range` | 💰 | "£70,000 – £120,000" style salary/price ranges |
| `currency_value` | 💲 | Single monetary values like "$50,000" |
| `percentage` | 📊 | "45%" style percentage strings |
| `date_string` | 📅 | Dates stored as text, parsed with `pd.to_datetime()` |
| `text_description` | 📝 | Free text with more than 4 words on average |
| `url` | 🔗 | Web URLs and links |
| `email` | 📧 | Email addresses |
| `id_column` | 🔑 | More than 95% unique numeric columns |
| `high_cardinality_categorical` | 🎯 | Categorical with more than 15 unique values |
| `low_cardinality_categorical` | 📦 | Categorical with 15 or fewer unique values |
| `numeric` | 🔢 | Standard numeric, ML-ready |
| `boolean` | ✅ | True/False |
| `categorical` | 🏷️ | Fallback categorical |

For each column, a recommendation card is rendered with a tailored preprocessing action. For example, a `pay_range` column gets "Extract min, max, midpoint → numeric features (+ optional currency flag)", while a `high_cardinality_categorical` gets "Target Encoding (preferred), Frequency Encoding, or Label Encoding".

**Data Preview tabs:** Head, Tail, Random Sample, Column Info (per-column null counts, unique counts, outlier counts), and Type Distribution (pie chart).

**Descriptive Statistics:** Numeric columns get mean, std, quartiles, skewness, kurtosis, and coefficient of variation. Distribution insight cards flag heavily skewed columns with transform suggestions.

**Visualisation tabs (7 total):**

1. **📈 Distributions** — Histograms with KDE/box/violin marginals or ridge plots across multiple columns
2. **🔗 Correlations** — Pearson/Spearman/Kendall correlation matrix with diverging heatmap plus ranked top-15 feature pair correlations
3. **📦 Box & Violin** — Box plots, violin plots, or box+strip with optional grouping by categorical
4. **🔄 Pairwise** — Scatter matrix of selected features with optional colour-by-category
5. **📋 Value Counts** — Horizontal bar, treemap, or donut chart for any column
6. **🔍 Outlier Analysis** — IQR-based outlier count per feature with ranked bar chart and bounds table
7. **📊 Statistical Tests** — Q-Q plot, Shapiro-Wilk test, D'Agostino-Pearson test, skewness, kurtosis

---

### ③ Clean & Preprocess

A seven-tab workflow that runs left to right. Every tab opens with a hint banner explaining what it does and when to use it.

**🔍 Missing Values**

Visualised per column with a horizontal bar chart coloured by severity. Three quick actions (drop all rows, fill numeric with median, fill categorical with mode) for one-click cleaning, plus per-column strategies including linear interpolation, forward/backward fill, and custom fill values.

**🗑️ Duplicates & Drops**

One-click duplicate removal, automatic detection and removal of ID columns (flagged by smart type analysis), arbitrary column dropping, and a low-variance column detector with adjustable threshold.

**🧠 Smart Transform** (the flagship feature)

This tab is where NetGaze earns its name. Every column flagged with a non-trivial smart type gets its own tailored transform section:

- **Pay/Price Range:** Parses strings like `"£70,000 – £120,000"` into `min`, `max`, and `midpoint` numeric columns, with automatic currency detection and optional currency flag column when multiple currencies are mixed.
- **Currency Value:** Strips `$`, `£`, `€` and converts `"$50k"` into `50000.0`.
- **Percentage:** Converts `"45%"` to either decimal `0.45` or raw `45.0`.
- **Date String:** Expands dates into `year`, `month`, `day`, `day_of_week`, `quarter`, and `days_since_start` numeric features.
- **Text Description:** Three methods. Basic statistics only (word count, char count, avg word length), TF-IDF vectorisation (configurable max features, 1 to 2 gram), or keyword presence flags for the most common meaningful words.
- **URL:** Extracts `has_https`, `url_length`, `path_depth`, `has_query`, and `domain`.
- **Email:** Extracts `local_len`, `domain`, `is_gmail`, `is_corporate`.
- **ID Column:** Recommended for dropping. Carries no predictive signal.

Each transform has a preview table showing the first 5 rows of the new columns before committing, and a "Keep original" checkbox.

**🏷️ Encoding**

Cardinality-aware categorical encoding. An enhanced summary table shows each categorical column with a cardinality badge (🟢 Low, 🟡 Medium, 🔴 High) and a recommended encoding method. Three encoding methods available:

- **One-Hot Encoding** — recommended for low cardinality (15 or fewer unique values)
- **Label Encoding** — recommended for high cardinality (more than 50 unique values)
- **Frequency Encoding** — replaces category with its relative frequency

**📏 Scaling**

Four scaling methods with explanations of when to use each:

- **Standard Scaler** — zero mean, unit variance (best for normally distributed data)
- **Min-Max Scaler** — 0 to 1 range (best for neural networks)
- **Robust Scaler** — median and IQR (best for data with outliers)
- **Power Transform (Yeo-Johnson)** — normalises skewed data

**🔧 Outlier Treatment**

IQR-based outlier handling with four strategies: cap at IQR bounds (Winsorisation), remove outlier rows, replace with median, or log transform (for positive values only).

**✅ Preview & Confirm**

Side-by-side comparison of original vs current shape, a warning if any categorical columns still need encoding, download button for the processed CSV, and a one-click reset to original.

---

### ④ Feature Selection & Split

The user selects a target column and one or more feature columns. NetGaze automatically:

- Detects whether the task is **Classification** or **Regression** via `auto_detect_task()` (object/category goes to classification, numeric with 20 or fewer unique ints goes to classification, otherwise regression)
- For classification, detects whether the sub-type is **binary** (2 classes) or **multiclass** (3 or more classes) and flags it
- Flags class imbalance when the majority-to-minority ratio exceeds 3 to 1
- Detects **singleton classes** (classes with only 1 sample) that would break stratified splitting, and warns the user upfront
- Shows the target distribution (bar chart for classification, histogram plus violin for regression)
- For regression targets with large ranges (more than 10,000), hints that NN target scaling will be available in Step ⑤
- Computes and visualises feature-target correlation (when both are numeric)

**Smart stratification logic:** Stratified train/test split is used when every class has 2 or more samples. Otherwise the app automatically falls back to a random shuffle split and explains why in an info banner.

---

### ⑤ Model Training

The core of the app. Before any training logic runs, `prepare_training_data()` hardens the inputs against three common failure modes:

1. **Non-numeric feature columns** — XGBoost and Keras both reject object dtypes. NetGaze hard-stops with a clear error message listing the offending columns and pointing the user back to the Encoding tab in Step ③.
2. **Non-numeric target** — Automatically label-encoded to integers (fit on the union of train+test labels to prevent unseen-label crashes at prediction time). The original class names are preserved for display in evaluation.
3. **Wrong dtypes** — Features coerced to `float32`, classification targets to `int64`, regression targets to `float32`.

Training is then offered in three modes via tabs.

**🎯 Single Model**

Pick one algorithm from the dropdown. Every hyperparameter relevant to that model is exposed with sensible defaults and sliders. Examples:

- **Random Forest:** n_estimators, max_depth, min_samples_split/leaf, max_features, class_weight (balanced/none)
- **XGBoost:** n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda
- **SVM:** C, kernel (rbf/linear/poly/sigmoid), gamma
- **Neural Network:** See below

Optional cross-validation checkbox runs Stratified K-Fold (classification) or K-Fold (regression) with configurable folds (3 to 10), with automatic fallback to plain K-Fold when any class has fewer samples than folds.

**Neural Network Builder**

The most feature-rich single-model interface in the app:

- **Depth:** 1 to 10 hidden layers
- **Per-layer configuration:** neurons (4 to 2048), activation (relu, tanh, sigmoid, elu, selu, leaky_relu)
- **Regularisation:** Optional BatchNormalization, optional Dropout (0.05 to 0.5)
- **Training:** 10 to 1000 epochs, batch size 8 to 512, learning rate selector (0.0001 to 0.05), early stopping patience (3 to 50 epochs)
- **Optimiser:** Adam, SGD, RMSprop, AdamW
- **Output layer auto-configuration** via `determine_nn_config()`:
  - Binary classification goes to 1 unit, sigmoid, `binary_crossentropy`
  - Multiclass goes to N units, softmax, `sparse_categorical_crossentropy` (integer targets from LabelEncoder, memory-efficient)
  - Regression goes to 1 unit, linear, `mse`
- **Optional target scaling** for regression with large target ranges, using StandardScaler with automatic inverse-transform on predictions

Before training starts, a code block renders the full architecture preview:

```
Input (42)  →  Dense(128, relu)  →  BN  →  Dropout(0.2)
            →  Dense(64, relu)   →  BN  →  Dropout(0.2)
            →  Output(3, softmax)

Loss: sparse_categorical_crossentropy  |  Optimizer: Adam(lr=0.001)
Layers: 2  |  Epochs: 50  |  Batch: 32  |  Early Stop: 10
```

**Live training log.** A custom `StreamlitEpochLogger` Keras callback streams each epoch to the UI as a styled log entry:

```
Epoch  12/50  ▼  loss: 0.4213  │  val_loss: 0.3891 (-8.4%)  │  Accuracy: 0.8523 → val: 0.8701   0.67s  ★ BEST
```

Every few epochs, a live-updating Plotly loss curve is redrawn showing training vs validation loss and metric, with a gold star marker on the best-so-far epoch.

**🏆 Model Comparison**

Select any subset of algorithms and train them all on the same data. Results are ranked by F1 (classification) or R² (regression) and displayed as a sortable dataframe plus a grouped bar chart. The winning model can be set as the active model with one click.

**⚡ Auto-Tune**

RandomizedSearchCV over curated param grids for six model families (Random Forest, XGBoost, Gradient Boosting, KNN, SVM/SVR, Logistic Regression). Configurable iterations (10 to 100) and CV folds (3 to 7). The best estimator becomes the active model automatically, and the top 10 configurations are shown in a scatter chart with error bars.

---

### ⑥ Evaluation & Interpretation

Five evaluation tabs, each addressing a different question about the trained model.

**📊 Metrics**

Classification dashboard: accuracy, precision, recall, F1 (weighted) as the headline row. Balanced accuracy, Matthews Correlation Coefficient, and Cohen's Kappa as the secondary row. Log loss when applicable. Full per-class classification report.

Regression dashboard: MSE, RMSE, MAE, R² as headline. Explained variance, max error, median absolute error as secondary. Automatic **overfitting check** comparing train vs test R² with a warning if the gap exceeds 0.1.

**📈 Visualisations**

- **Classification:** Confusion matrix (counts and/or row-normalised percentages), ROC curve with AUC (binary), Precision-Recall curve with AUC (binary), per-class precision/recall/F1 grouped bar chart (multiclass)
- **Regression:** Actual vs predicted scatter with perfect-prediction diagonal, residual distribution histogram with box marginal, residuals vs predicted scatter
- **Neural Network:** Full training history with train/val loss and train/val metric over epochs, annotated with the best epoch

**🧠 Model Structure**

- **Neural Network:** Full Keras `model.summary()` output plus a per-layer table (name, type, units, activation, parameters) and total parameter count
- **Decision Tree:** Text representation via `export_text()` plus a visual plot via `sklearn.tree.plot_tree`
- **Tree Ensembles** (Random Forest, Gradient Boosting, AdaBoost, Extra Trees): First estimator visualised as a representative tree
- **Linear / other models:** All hyperparameters dumped as JSON

**🔍 Feature Importance**

Three interpretability methods available:

1. **Built-in importance** — `feature_importances_` for tree-based models, `coef_` for linear models (shown with diverging colour scale centred at zero)
2. **Permutation Importance** — Model-agnostic. Measures drop in test score when each feature is shuffled. Works with Keras NNs via a custom scoring wrapper
3. **SHAP Analysis** — Uses `TreeExplainer` for tree models, `KernelExplainer` for everything else. Summary plot shows feature impact distribution across samples

**📋 Prediction Explorer**

An interactive table of test predictions with:

- Classification: Actual vs Predicted side-by-side, with original class names restored via the stored LabelEncoder, plus a Correct/Incorrect column. Filter to see only errors, only correct predictions, or all
- Regression: Actual vs Predicted with Error and Absolute Error columns. Sort by highest/lowest error to find the most problematic predictions
- One-click CSV download of the prediction table

---

## Supported Algorithms

### Classification (11 models)

| Model | Use Case |
|-------|----------|
| **Logistic Regression** | Fast, interpretable baseline. Supports L1/L2/ElasticNet penalties |
| **Random Forest** | Robust all-rounder. Provides feature importance. Handles non-linearity |
| **XGBoost** | High accuracy. Handles missing data. Strong for tabular competitions |
| **Decision Tree** | Highly interpretable. Visualisable. Prone to overfitting |
| **Gradient Boosting** | Strong accuracy. Slower than RF but often better |
| **AdaBoost** | Combines weak learners. Good for balanced data |
| **Extra Trees** | Faster than RF. Reduces variance further |
| **SVM** | Effective in high-dimensional spaces. Flexible kernels |
| **KNN** | Instance-based. No training phase. Needs feature scaling |
| **Naive Bayes** | Very fast. Strong on text and small datasets |
| **Neural Network** | Learns complex patterns. Fully configurable architecture |

### Regression (12 models)

| Model | Use Case |
|-------|----------|
| **Linear Regression** | Fast interpretable baseline for linear relationships |
| **Ridge** | Linear with L2 regularisation. Handles multicollinearity |
| **Lasso** | Linear with L1 regularisation. Built-in feature selection |
| **Elastic Net** | Balanced L1/L2. Best of Ridge and Lasso |
| **Random Forest** | Robust to outliers. Handles non-linearity |
| **XGBoost** | Top accuracy. Strong for structured data |
| **Decision Tree** | Interpretable. Captures non-linear patterns |
| **Gradient Boosting** | Strong performance with many hyperparameters |
| **AdaBoost** | Adaptive ensemble. Resistant to overfitting |
| **Extra Trees** | Fast ensemble method. Handles noise well |
| **SVR** | Support Vector Regression. Flexible kernel choices |
| **KNN** | Simple non-parametric regressor |
| **Neural Network** | Learns complex non-linear mappings |

---

## Key Engineering Decisions

A handful of deliberate choices make NetGaze robust in practice rather than falling over at the first surprising input.

| Decision | Choice | Why |
|----------|--------|-----|
| **Target encoding** | `LabelEncoder` (not `OneHotEncoder`) | Classifiers expect a 1D integer vector. OHE produces a matrix which breaks sklearn classifiers and Keras `sparse_categorical_crossentropy` |
| **Non-numeric X guard** | Hard-stop with explicit error | Silent auto-encoding of free-text or high-cardinality columns produces meaningless models. Better to send the user back to the Encoding tab with a clear list of offenders |
| **NN loss function** | `sparse_categorical_crossentropy` for multiclass | Works directly with LabelEncoded integer targets. Memory-efficient and avoids the OHE-target footgun |
| **Stratification fallback** | Auto-detect singleton classes and fall back to random split | Stratified splits crash when any class has fewer samples than the split ratio requires. NetGaze detects this and explains the fallback |
| **CV strategy** | StratifiedKFold for classification with KFold fallback | Same logic as split fallback. Don't crash on rare classes |
| **LabelEncoder fit scope** | Fit on union of train+test labels | Prevents "unseen label at predict time" crashes when test contains a class missing from train |
| **NN target scaling** | Opt-in only, for large-range regression | Tree models don't need it. For NNs, large target values can cause gradient instability, but scaling always carries an inverse-transform cost. Only offered when the target range exceeds 1000 |
| **Dynamic data prep** | `float32` for features, `int64`/`float32` for targets | Matches XGBoost and Keras expectations. Prevents silent dtype coercion bugs |
| **Session state keys** | Typed defaults dict | Prevents `KeyError` on first-load when components try to read state before it's been set |
| **Single-file app** | Everything in `app.py` | One file to read, modify, and deploy. Keeps the project accessible to learners |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda
- Approximately 2 GB free disk space for dependencies (TensorFlow is the heavy one)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/[your-username]/netgaze-ml-pipeline.git
cd netgaze-ml-pipeline

# Install dependencies
pip install -r requirements.txt
```

---

## Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

To run on a specific port:

```bash
streamlit run app.py --server.port 8080
```

To expose it over your network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## Deployment on Hugging Face Spaces

NetGaze is deployed on Hugging Face Spaces using the Streamlit SDK. The configuration at the top of this README (the YAML frontmatter) is what Spaces reads to launch the app.

To deploy your own copy:

1. Create a new Space on Hugging Face, selecting **Streamlit** as the SDK
2. Clone the Space repository locally
3. Copy `app.py` and `requirements.txt` into the repo root
4. Keep the YAML frontmatter at the top of `README.md` intact
5. Push to the Space's remote. HF will build and deploy automatically

Build time is typically 4 to 8 minutes due to TensorFlow installation.

---

## How to Use NetGaze (Practical Walkthrough)

Here is a typical session, start to finish.

**Step 1: Upload.** Drop `your_dataset.csv` into the sidebar uploader. The app confirms the shape and switches to Step ② automatically.

**Step 2: Explore.** Read the Quality Score. Expand the 🧠 Column Analysis card and scan the recommendation for every column. Note which ones say "Drop column" (IDs), "Extract min, max, midpoint" (pay ranges), "TF-IDF vectorisation" (free text), and so on. Check the Correlations tab to see if any features are obvious duplicates of the target.

**Step 3: Clean.** Work through the tabs left to right. Handle missing values first. Drop any ID columns via the Duplicates & Drops tab. Go to Smart Transform and apply the transforms the app suggested in Step ②. Encode any remaining categorical columns. Scale numeric columns if you plan to use KNN, SVM, or a Neural Network.

**Step 4: Split.** Pick your target column. Confirm the auto-detected task type and sub-type are correct. Adjust test size if needed (20% is a good default). Click Split Data.

**Step 5a: Compare models.** Go to the 🏆 Model Comparison tab and select 5 to 7 algorithms including Random Forest, XGBoost, and Logistic Regression. Click Run Comparison. The ranked results tell you which algorithm family is a good fit for this data.

**Step 5b: Tune the winner.** Go to the ⚡ Auto-Tune tab, select the winning algorithm, set iterations to 50, and click Start. This is where you get your best production-ready model.

**Step 6: Evaluate.** Review the Metrics tab for headline numbers. Check the Confusion Matrix for class-level weaknesses (classification) or the Residuals tab (regression). Use Feature Importance to explain the model's decisions. Use the Prediction Explorer to sanity-check the worst errors.

**Step 7: Export.** Download the prediction CSV from the Prediction Explorer, and if you want the cleaned dataset, grab it from the Preview & Confirm tab in Step ③.

---

## Design Philosophy

NetGaze follows four principles that shape every UI decision.

**1. Hints first, advanced controls second.** Every tab starts with a blue hint banner explaining what this step does, when to use it, and what the common pitfalls are. Advanced users can ignore them. Beginners rely on them.

**2. Catch failures upfront, not during training.** Non-numeric targets, object-dtype features, singleton classes, and unseen test labels are all detected before training begins and either auto-fixed or reported with actionable error messages pointing back to the right step.

**3. Explain every automatic decision.** When NetGaze auto-encodes a target, falls back from stratified to random split, or configures an NN output layer, it surfaces a hint banner explaining exactly what it did and why. No silent magic.

**4. Visualise relentlessly.** Every metric, every distribution, every model comparison, every training run has an interactive Plotly chart attached to it. Text tables are reserved for cases where the visual would be less readable.

---

## Repository Structure

```
.
├── README.md                 # This file (with HF Spaces YAML frontmatter)
├── app.py                    # Complete NetGaze pipeline (~3,100 lines)
├── requirements.txt          # Python dependencies
└── LICENSE                   # MIT License
```

NetGaze is intentionally a single-file app. Every helper function, every CSS rule, every pipeline stage lives in `app.py`. This keeps the project accessible. One file to read, one file to modify, one file to deploy.

---

## Dependencies

```
streamlit>=1.56.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0
scikit-learn>=1.4.0
xgboost>=2.0.0
tensorflow>=2.15.0
scipy>=1.12.0
shap>=0.44.0
openpyxl>=3.1.0
```

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Roadmap

Features that may land in future versions:

- **LightGBM and CatBoost** alongside XGBoost
- **Time-series cross-validation** for temporal data
- **SMOTE and class balancing options** in the split step
- **Automated feature engineering** (polynomial features, interaction terms)
- **Model export** — download trained models as `.pkl`, `.joblib`, or `.h5`
- **Custom metric plug-in** — let users define their own scoring functions
- **Multi-target regression / multi-label classification** support
- **Stacking and voting ensembles** through the UI
- **MLflow integration** for experiment tracking

---

## Author

**Collins Lemeke**  

NetGaze was built to make end-to-end ML pipelines accessible to students, researchers, and domain experts who shouldn't need to learn Streamlit, scikit-learn, and Keras just to test an idea.

For questions, feedback, or feature requests, open a GitHub issue or reach out via Hugging Face.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

---

> *Built with Streamlit, scikit-learn, XGBoost, TensorFlow, Plotly, and SHAP. Designed for learners and practitioners who want to ship ML fast without cutting corners.*
