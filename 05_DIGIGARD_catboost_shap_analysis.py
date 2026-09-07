#!/usr/bin/env python3
"""
05_DIGIGARD_catboost_shap_analysis.py

DIGIGARD Stage 05: thorough CatBoost-only SHAP and feature analysis.

This stage runs AFTER:
    01_DIGIGARD_merge_and_clean_project_histories.py
    02_DIGIGARD_create_train_test_splits.py
    03_DIGIGARD_train_classifier.py

Recommended before/alongside:
    04_DIGIGARD_evaluate_catboost_classifier.py

This stage performs NO training and NO hyperparameter optimization.

It loads ONE saved Stage-03 CatBoostClassifier for ONE target and analyzes:
    1) full_dataset      = Stage-02 train + test
    2) training_dataset  = Stage-02 train
    3) testing_dataset   = Stage-02 test

The saved CatBoost model and exact Stage-03 feature contract are reused.

===============================================================================
SHAP INTERPRETATION FOR BINARY CATBOOST CLASSIFIERS
===============================================================================

CatBoost native SHAP values are additive in RAW model-output space:

    expected_value + sum_j SHAP_j = RawFormulaVal

For binary classification, RawFormulaVal is the raw margin / logit.

Probability is obtained separately:

    P(class=1) = sigmoid(RawFormulaVal)

Therefore this script explicitly stores and verifies BOTH:
    - SHAP raw-output reconstruction
    - probability reconstruction after applying the sigmoid

It never claims that native CatBoost SHAP contributions add directly to
predict_proba(X)[:, 1].

===============================================================================
ANALYSES SAVED FOR EACH DATASET
===============================================================================

A) COMPLETE LOCAL SHAP
    - one SHAP value per feature per row
    - expected raw value
    - SHAP sum
    - reconstructed raw output
    - actual CatBoost raw output
    - raw reconstruction error
    - reconstructed probability
    - actual CatBoost probability
    - probability reconstruction error

B) NORMALIZED LOCAL SHAP
    For each row and feature j:

        normalized_abs_shap_j =
            abs(SHAP_j) / sum_k abs(SHAP_k)

    For non-zero rows the normalized absolute contributions sum to 1.

C) TOP-K LOCAL EXPLANATIONS
    For every row:
    - rank
    - feature
    - feature value
    - signed SHAP
    - absolute SHAP
    - normalized absolute SHAP
    - compact JSON representation

D) GLOBAL SHAP FEATURE ANALYSIS
    - mean absolute SHAP
    - relative mean absolute SHAP
    - mean / std / min / max signed SHAP
    - median absolute SHAP
    - q25 / q75 / q90 / q95 absolute SHAP
    - fraction of positive / negative / zero contributions
    - mean absolute SHAP by true class
    - mean absolute SHAP by predicted class
    - global ranking

E) FEATURE-LEVEL ANALYSIS
    - feature descriptive statistics
    - feature missingness
    - Pearson and Spearman relation with target
    - Pearson and Spearman relation with predicted probability
    - Pearson and Spearman relation between feature value and its own SHAP value
    - CatBoost native feature importance
    - CatBoost-vs-SHAP importance comparison

F) PLOTS
    - global mean-|SHAP| importance
    - normalized global SHAP importance
    - mean signed SHAP
    - SHAP importance by true class
    - SHAP importance by predicted class
    - CatBoost native importance
    - CatBoost-vs-SHAP importance scatter
    - SHAP beeswarm-style plot
    - feature-value-vs-SHAP dependence plots for top features
    - feature-value distribution plots for top features
    - local explanation plots for selected representative rows

G) CROSS-DATASET COMPARISON
    - full/train/test SHAP importance side-by-side
    - rank comparison
    - pairwise rank-correlation matrix
    - pairwise importance-correlation matrix
    - feature-rank stability

Dependencies
------------
Required:
    numpy
    pandas
    matplotlib
    catboost

Optional:
    shap
        If installed, standard SHAP summary/bar plots are also produced.
        The native CatBoost SHAP analysis does NOT depend on the shap package.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier, Pool


# =============================================================================
# CONFIG: ONE TARGET + ONE CATBOOST MODEL RUN PER EXECUTION
# =============================================================================

TARGET = "isBugPresent"

# Alternatives:
# TARGET = "isBugfix"
# TARGET = "isSZZBugIntroducer"


# Exact Stage-03 CatBoost run folder.
MODEL_RUN = "catboost_default_rs42"

# Alternative:
# MODEL_RUN = "catboost_optuna_rs42"


# =============================================================================
# PIPELINE PATHS
# =============================================================================

RANDOM_STATE = 42

STAGE02_ROOT = Path("DIGIGARD_02_train_test_splits")
STAGE03_ROOT = Path("DIGIGARD_03_models")
STAGE05_ROOT = Path("DIGIGARD_05_catboost_shap")

STAGE02_TARGET_ROOT = STAGE02_ROOT / TARGET
SPLIT_DIR = STAGE02_TARGET_ROOT / f"splits_{RANDOM_STATE}"

TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

MODEL_DIR = STAGE03_ROOT / TARGET / MODEL_RUN
MODEL_MANIFEST = MODEL_DIR / "model_manifest.json"
MODEL_FEATURES_JSON = MODEL_DIR / "features.json"
MODEL_PATH = MODEL_DIR / "best_model.cbm"

OUTPUT_DIR = STAGE05_ROOT / TARGET / MODEL_RUN


# =============================================================================
# SHAP COMPUTATION
# =============================================================================

# Compute native CatBoost SHAP in batches to avoid one enormous Pool calculation.
SHAP_BATCH_SIZE = 5000

# float32 greatly reduces large CSV/matrix storage while retaining sufficient
# precision for the analysis. Reconstruction checks themselves use float64.
SHAP_STORAGE_DTYPE = np.float32

# Save one full SHAP contribution column per model feature per row.
SAVE_FULL_LOCAL_SHAP_CSV = True

# Save one normalized absolute SHAP contribution per feature per row.
SAVE_NORMALIZED_LOCAL_SHAP_CSV = True

# Local strongest-feature summary.
LOCAL_SHAP_TOP_K = 10


# =============================================================================
# GLOBAL / PLOT SETTINGS
# =============================================================================

GLOBAL_PLOT_TOP_N = 30
BEESWARM_TOP_N = 20
DEPENDENCE_TOP_N = 12
FEATURE_DISTRIBUTION_TOP_N = 12

# To keep dense plots readable, plotting can use a deterministic subset while all
# CSV/statistical analyses continue to use every row.
MAX_ROWS_FOR_BEESWARM_PLOT: Optional[int] = 5000
MAX_ROWS_FOR_DEPENDENCE_PLOT: Optional[int] = 10000
MAX_ROWS_FOR_FEATURE_DISTRIBUTION_PLOT: Optional[int] = 10000

# Optional standard `shap` package plots.
ENABLE_OPTIONAL_SHAP_PACKAGE_PLOTS = True
SHAP_PACKAGE_MAX_ROWS: Optional[int] = 5000


# =============================================================================
# LOCAL EXPLANATION PLOTS
# =============================================================================

SAVE_SELECTED_LOCAL_PLOTS = True

# In addition to representative rows, optionally plot first N rows.
LOCAL_FIRST_N_ROWS = 0

# Automatically select:
#   - highest predicted probabilities
#   - lowest predicted probabilities
#   - highest total |SHAP|
#   - largest false-positive scores
#   - lowest false-negative scores
N_REPRESENTATIVE_ROWS_PER_CATEGORY = 5


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

SAVE_PNG = True
SAVE_EPS = True
PLOT_DPI = 220

OVERWRITE_EXISTING_ANALYSIS = True


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================

DATASET_ORDER = [
    "full_dataset",
    "training_dataset",
    "testing_dataset",
]

DATASET_DISPLAY_NAMES = {
    "full_dataset": "Full target dataset (train + test)",
    "training_dataset": "Training dataset",
    "testing_dataset": "Testing dataset",
}

DATASET_INTERPRETATION = {
    "full_dataset": (
        "Descriptive SHAP analysis of the union of Stage-02 train and test rows. "
        "Because it includes training observations, it is not an independent "
        "generalization explanation set."
    ),
    "training_dataset": (
        "In-sample SHAP analysis of Stage-02 training observations. Useful for "
        "understanding what the fitted model learned on its training domain."
    ),
    "testing_dataset": (
        "Held-out SHAP analysis of Stage-02 test observations. This is the most "
        "important explanation set for assessing whether learned feature effects "
        "persist on unseen held-out data."
    ),
}


# =============================================================================
# COLUMN NAMES
# =============================================================================

SHAP_PREFIX = "SHAP__"
NORM_SHAP_PREFIX = "SHAP_NORMALIZED_ABS__"

SHAP_EXPECTED_RAW_COL = "SHAP__expected_raw_value"
SHAP_SUM_COL = "SHAP__sum"
SHAP_RECONSTRUCTED_RAW_COL = "SHAP__reconstructed_raw_output"
MODEL_RAW_COL = "MODEL__raw_output"
SHAP_RAW_ERROR_COL = "SHAP__raw_reconstruction_error"

MODEL_PROBABILITY_COL = f"{TARGET}__predicted_probability_class1"
MODEL_PREDICTED_CLASS_COL = f"{TARGET}__predicted_class"

SHAP_RECONSTRUCTED_PROBABILITY_COL = (
    "SHAP__reconstructed_probability_class1"
)
SHAP_PROBABILITY_ERROR_COL = (
    "SHAP__probability_reconstruction_error"
)

NORMALIZED_ROW_SUM_COL = "SHAP__normalized_abs_contributions_sum"
TOTAL_ABS_SHAP_COL = "SHAP__total_absolute_contribution"
TOPK_JSON_COL = f"SHAP__top_{LOCAL_SHAP_TOP_K}_local_explanation_json"


# =============================================================================
# VALIDATE CONFIG
# =============================================================================

if SHAP_BATCH_SIZE <= 0:
    raise ValueError("SHAP_BATCH_SIZE must be >= 1.")

if LOCAL_SHAP_TOP_K <= 0:
    raise ValueError("LOCAL_SHAP_TOP_K must be >= 1.")

for name, value in [
    ("GLOBAL_PLOT_TOP_N", GLOBAL_PLOT_TOP_N),
    ("BEESWARM_TOP_N", BEESWARM_TOP_N),
    ("DEPENDENCE_TOP_N", DEPENDENCE_TOP_N),
    ("FEATURE_DISTRIBUTION_TOP_N", FEATURE_DISTRIBUTION_TOP_N),
]:
    if value <= 0:
        raise ValueError(f"{name} must be >= 1.")


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def safe_package_version(package_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package_name)
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable logistic sigmoid.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)

    positive = x >= 0
    negative = ~positive

    out[positive] = 1.0 / (
        1.0 + np.exp(-x[positive])
    )

    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (
        1.0 + exp_x
    )

    return out


def validate_binary_target(
    series: pd.Series,
    dataset_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{dataset_name}: target '{TARGET}' contains "
            f"{int(numeric.isna().sum())} missing/non-numeric values."
        )

    observed = sorted(
        float(v)
        for v in pd.unique(numeric)
    )

    unexpected = [
        v
        for v in observed
        if v not in {0.0, 1.0}
    ]

    if unexpected:
        raise ValueError(
            f"{dataset_name}: target is not binary 0/1. "
            f"Unexpected values: {unexpected}"
        )

    return numeric.astype(int)


def preprocess_numeric_features(
    df: pd.DataFrame,
    features: List[str],
    dataset_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: required Stage-03 features are missing:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    coercion_losses: Dict[str, int] = {}

    for feature in features:
        original = X[feature]
        before = int(original.notna().sum())

        if pd.api.types.is_numeric_dtype(original):
            converted = pd.to_numeric(
                original,
                errors="coerce",
            )
        else:
            text = (
                original
                .astype("string")
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            converted = pd.to_numeric(
                text,
                errors="coerce",
            )

        after = int(converted.notna().sum())
        loss = max(0, before - after)

        if loss > 0:
            coercion_losses[feature] = loss

        X[feature] = converted

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    report = {
        "dataset": dataset_name,
        "rows": int(len(X)),
        "feature_count": len(features),
        "numeric_coercions_to_nan": coercion_losses,
        "total_missing_cells": int(
            X.isna().sum().sum()
        ),
        "rows_with_any_missing_feature": int(
            X.isna().any(axis=1).sum()
        ),
        "rows_with_all_features_missing": int(
            X.isna().all(axis=1).sum()
        ),
        "all_missing_features": [
            feature
            for feature in features
            if X[feature].isna().all()
        ],
    }

    return X, report


def deterministic_plot_subset(
    n_rows: int,
    max_rows: Optional[int],
    seed: int,
) -> np.ndarray:
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows, dtype=int)

    rng = np.random.default_rng(seed)

    selected = np.sort(
        rng.choice(
            np.arange(n_rows, dtype=int),
            size=max_rows,
            replace=False,
        )
    )

    return selected


# =============================================================================
# LOAD MODEL / CONTRACT
# =============================================================================

def load_model_manifest() -> Dict[str, Any]:
    if not MODEL_MANIFEST.exists():
        raise SystemExit(
            "Stage-03 model_manifest.json not found:\n"
            f"  {MODEL_MANIFEST.resolve()}"
        )

    with MODEL_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    if manifest.get("target") != TARGET:
        raise ValueError(
            "Target mismatch between Stage-05 configuration and "
            "Stage-03 model manifest."
        )

    model_type = str(
        manifest.get("model_type", "")
    ).strip().upper()

    if model_type != "CATBOOST":
        raise ValueError(
            "Stage 05 is CatBoost-only. "
            f"Loaded model_type={model_type!r}."
        )

    if manifest.get("run_name") != MODEL_RUN:
        raise ValueError(
            "MODEL_RUN mismatch between Stage-05 configuration "
            "and Stage-03 manifest."
        )

    return manifest


def load_features() -> List[str]:
    if not MODEL_FEATURES_JSON.exists():
        raise SystemExit(
            "Stage-03 features.json not found:\n"
            f"  {MODEL_FEATURES_JSON.resolve()}"
        )

    with MODEL_FEATURES_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        features = list(payload)

    elif isinstance(payload, dict):
        features = list(
            payload.get("features", [])
        )

    else:
        raise ValueError(
            "Unsupported Stage-03 features.json format."
        )

    if not features:
        raise ValueError(
            "No features found in Stage-03 feature contract."
        )

    if len(features) != len(set(features)):
        raise ValueError(
            "Duplicate feature names in Stage-03 feature contract."
        )

    return features


def load_model() -> CatBoostClassifier:
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Stage-03 CatBoost model not found:\n"
            f"  {MODEL_PATH.resolve()}"
        )

    model = CatBoostClassifier()
    model.load_model(
        str(MODEL_PATH)
    )

    return model


def resolve_probability_threshold(
    manifest: Dict[str, Any],
) -> float:
    threshold = float(
        manifest.get(
            "default_probability_threshold",
            0.50,
        )
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid model probability threshold: {threshold}"
        )

    return threshold


# =============================================================================
# NATIVE CATBOOST SHAP
# =============================================================================

def compute_native_catboost_shap(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CatBoost returns n_features SHAP contributions plus one final expected value.
    """
    n_rows = int(len(X))
    batches: List[np.ndarray] = []

    for start in range(
        0,
        n_rows,
        SHAP_BATCH_SIZE,
    ):
        stop = min(
            start + SHAP_BATCH_SIZE,
            n_rows,
        )

        print(
            f"[SHAP] rows {start:,}:{stop:,} "
            f"of {n_rows:,}"
        )

        X_batch = X.iloc[start:stop]

        pool = Pool(
            X_batch,
            feature_names=features,
        )

        shap_batch = np.asarray(
            model.get_feature_importance(
                data=pool,
                type="ShapValues",
                model_output="Raw",
                verbose=False,
            ),
            dtype=float,
        )

        if shap_batch.ndim != 2:
            raise ValueError(
                "Expected binary CatBoost SHAP output to be 2D; "
                f"got shape {shap_batch.shape}."
            )

        expected_width = len(features) + 1

        if shap_batch.shape[1] != expected_width:
            raise ValueError(
                "Unexpected CatBoost SHAP width.\n"
                f"Expected {expected_width}; "
                f"got {shap_batch.shape[1]}."
            )

        batches.append(shap_batch)

    if not batches:
        raise ValueError(
            "No SHAP rows were computed."
        )

    shap_all = np.concatenate(
        batches,
        axis=0,
    )

    if shap_all.shape[0] != n_rows:
        raise RuntimeError(
            "SHAP row count mismatch."
        )

    contributions = shap_all[
        :,
        :-1,
    ].astype(
        SHAP_STORAGE_DTYPE,
        copy=False,
    )

    expected_values = shap_all[
        :,
        -1,
    ].astype(
        SHAP_STORAGE_DTYPE,
        copy=False,
    )

    return contributions, expected_values


def compute_model_outputs(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    threshold: float,
) -> Dict[str, np.ndarray]:
    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 2
    ):
        raise ValueError(
            "Expected binary predict_proba output with shape (n,2); "
            f"got {probabilities.shape}."
        )

    p1 = probabilities[:, 1]

    raw = np.asarray(
        model.predict(
            X,
            prediction_type="RawFormulaVal",
        ),
        dtype=float,
    ).reshape(-1)

    predicted_class = (
        p1 >= threshold
    ).astype(int)

    return {
        "probability_class0": probabilities[:, 0],
        "probability_class1": p1,
        "raw_output": raw,
        "predicted_class": predicted_class,
    }


# =============================================================================
# GLOBAL SHAP ANALYSIS
# =============================================================================

def class_conditional_mean_abs(
    shap64: np.ndarray,
    labels: np.ndarray,
    label_value: int,
) -> np.ndarray:
    mask = (
        np.asarray(labels, dtype=int)
        == int(label_value)
    )

    if not np.any(mask):
        return np.full(
            shap64.shape[1],
            np.nan,
            dtype=float,
        )

    return np.mean(
        np.abs(
            shap64[mask]
        ),
        axis=0,
    )


def build_global_shap_importance(
    shap_values: np.ndarray,
    features: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    shap64 = np.asarray(
        shap_values,
        dtype=float,
    )

    abs_shap = np.abs(
        shap64
    )

    mean_abs = np.mean(
        abs_shap,
        axis=0,
    )

    total_mean_abs = float(
        np.sum(mean_abs)
    )

    relative = (
        mean_abs / total_mean_abs
        if total_mean_abs > 0
        else np.zeros_like(mean_abs)
    )

    positive_fraction = np.mean(
        shap64 > 0,
        axis=0,
    )

    negative_fraction = np.mean(
        shap64 < 0,
        axis=0,
    )

    zero_fraction = np.mean(
        shap64 == 0,
        axis=0,
    )

    df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap":
            mean_abs,
        "relative_mean_abs_shap":
            relative,
        "mean_signed_shap":
            np.mean(
                shap64,
                axis=0,
            ),
        "std_signed_shap":
            np.std(
                shap64,
                axis=0,
                ddof=0,
            ),
        "min_signed_shap":
            np.min(
                shap64,
                axis=0,
            ),
        "max_signed_shap":
            np.max(
                shap64,
                axis=0,
            ),
        "median_abs_shap":
            np.median(
                abs_shap,
                axis=0,
            ),
        "q25_abs_shap":
            np.quantile(
                abs_shap,
                0.25,
                axis=0,
            ),
        "q75_abs_shap":
            np.quantile(
                abs_shap,
                0.75,
                axis=0,
            ),
        "q90_abs_shap":
            np.quantile(
                abs_shap,
                0.90,
                axis=0,
            ),
        "q95_abs_shap":
            np.quantile(
                abs_shap,
                0.95,
                axis=0,
            ),
        "positive_shap_fraction":
            positive_fraction,
        "negative_shap_fraction":
            negative_fraction,
        "zero_shap_fraction":
            zero_fraction,
        "mean_abs_shap_true_0":
            class_conditional_mean_abs(
                shap64,
                y_true,
                0,
            ),
        "mean_abs_shap_true_1":
            class_conditional_mean_abs(
                shap64,
                y_true,
                1,
            ),
        "mean_abs_shap_pred_0":
            class_conditional_mean_abs(
                shap64,
                y_pred,
                0,
            ),
        "mean_abs_shap_pred_1":
            class_conditional_mean_abs(
                shap64,
                y_pred,
                1,
            ),
    })

    df = df.sort_values(
        [
            "mean_abs_shap",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    df.insert(
        0,
        "global_shap_rank",
        np.arange(
            1,
            len(df) + 1,
            dtype=int,
        ),
    )

    return df


def get_catboost_native_importance(
    model: CatBoostClassifier,
    features: List[str],
) -> pd.DataFrame:
    values = np.asarray(
        model.get_feature_importance(),
        dtype=float,
    )

    if len(values) != len(features):
        raise ValueError(
            "CatBoost native feature-importance length does not "
            "match feature contract."
        )

    df = pd.DataFrame({
        "feature": features,
        "catboost_native_importance": values,
    })

    total = float(
        df[
            "catboost_native_importance"
        ].sum()
    )

    if total > 0:
        df[
            "catboost_native_importance_normalized"
        ] = (
            df[
                "catboost_native_importance"
            ] / total
        )
    else:
        df[
            "catboost_native_importance_normalized"
        ] = 0.0

    df = df.sort_values(
        [
            "catboost_native_importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    df.insert(
        0,
        "catboost_native_rank",
        np.arange(
            1,
            len(df) + 1,
            dtype=int,
        ),
    )

    return df


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def safe_corr(
    a: pd.Series,
    b: pd.Series,
    method: str,
) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(
                a,
                errors="coerce",
            ),
            pd.to_numeric(
                b,
                errors="coerce",
            ),
        ],
        axis=1,
    ).dropna()

    if len(pair) < 3:
        return float("nan")

    if (
        pair.iloc[:, 0].nunique()
        < 2
        or pair.iloc[:, 1].nunique()
        < 2
    ):
        return float("nan")

    try:
        return float(
            pair.iloc[:, 0].corr(
                pair.iloc[:, 1],
                method=method,
            )
        )
    except Exception:
        return float("nan")


def build_feature_analysis(
    X: pd.DataFrame,
    y_true: np.ndarray,
    probability_class1: np.ndarray,
    shap_values: np.ndarray,
    features: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    y_series = pd.Series(
        np.asarray(
            y_true,
            dtype=float,
        ),
        index=X.index,
    )

    probability_series = pd.Series(
        np.asarray(
            probability_class1,
            dtype=float,
        ),
        index=X.index,
    )

    shap64 = np.asarray(
        shap_values,
        dtype=float,
    )

    for feature_index, feature in enumerate(
        features
    ):
        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        ).astype(float)

        shap_series = pd.Series(
            shap64[
                :,
                feature_index,
            ],
            index=X.index,
        )

        valid_values = values.dropna()

        rows.append({
            "feature":
                feature,
            "n_rows":
                int(len(values)),
            "n_missing":
                int(values.isna().sum()),
            "missing_fraction":
                float(values.isna().mean()),
            "n_unique_non_missing":
                int(valid_values.nunique()),
            "mean":
                float(valid_values.mean())
                if len(valid_values)
                else float("nan"),
            "std":
                float(valid_values.std(ddof=1))
                if len(valid_values) > 1
                else float("nan"),
            "min":
                float(valid_values.min())
                if len(valid_values)
                else float("nan"),
            "q25":
                float(valid_values.quantile(0.25))
                if len(valid_values)
                else float("nan"),
            "median":
                float(valid_values.median())
                if len(valid_values)
                else float("nan"),
            "q75":
                float(valid_values.quantile(0.75))
                if len(valid_values)
                else float("nan"),
            "max":
                float(valid_values.max())
                if len(valid_values)
                else float("nan"),
            "pearson_feature_vs_target":
                safe_corr(
                    values,
                    y_series,
                    "pearson",
                ),
            "spearman_feature_vs_target":
                safe_corr(
                    values,
                    y_series,
                    "spearman",
                ),
            "pearson_feature_vs_probability":
                safe_corr(
                    values,
                    probability_series,
                    "pearson",
                ),
            "spearman_feature_vs_probability":
                safe_corr(
                    values,
                    probability_series,
                    "spearman",
                ),
            "pearson_feature_vs_own_shap":
                safe_corr(
                    values,
                    shap_series,
                    "pearson",
                ),
            "spearman_feature_vs_own_shap":
                safe_corr(
                    values,
                    shap_series,
                    "spearman",
                ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# LOCAL SHAP TABLES
# =============================================================================

def build_local_shap_tables(
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    features: List[str],
    shap_values: np.ndarray,
    expected_values: np.ndarray,
    model_outputs: Dict[str, np.ndarray],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, float],
]:
    n_rows = int(len(source_df))

    shap64 = np.asarray(
        shap_values,
        dtype=float,
    )

    expected64 = np.asarray(
        expected_values,
        dtype=float,
    )

    model_raw = np.asarray(
        model_outputs[
            "raw_output"
        ],
        dtype=float,
    )

    model_p1 = np.asarray(
        model_outputs[
            "probability_class1"
        ],
        dtype=float,
    )

    predicted_class = np.asarray(
        model_outputs[
            "predicted_class"
        ],
        dtype=int,
    )

    shap_sum = np.sum(
        shap64,
        axis=1,
    )

    reconstructed_raw = (
        expected64
        + shap_sum
    )

    raw_error = (
        reconstructed_raw
        - model_raw
    )

    reconstructed_probability = stable_sigmoid(
        reconstructed_raw
    )

    probability_error = (
        reconstructed_probability
        - model_p1
    )

    abs_shap = np.abs(
        shap64
    )

    total_abs = np.sum(
        abs_shap,
        axis=1,
    )

    normalized_abs = np.divide(
        abs_shap,
        total_abs[:, None],
        out=np.zeros_like(
            abs_shap,
            dtype=float,
        ),
        where=(
            total_abs[:, None]
            > 0.0
        ),
    )

    normalized_row_sum = np.sum(
        normalized_abs,
        axis=1,
    )

    nonzero_rows = (
        total_abs > 0.0
    )

    if np.any(nonzero_rows):
        max_norm_error = float(
            np.max(
                np.abs(
                    normalized_row_sum[
                        nonzero_rows
                    ]
                    - 1.0
                )
            )
        )
    else:
        max_norm_error = 0.0

    diagnostics = {
        "max_abs_raw_reconstruction_error":
            float(
                np.max(
                    np.abs(
                        raw_error
                    )
                )
            ),
        "mean_abs_raw_reconstruction_error":
            float(
                np.mean(
                    np.abs(
                        raw_error
                    )
                )
            ),
        "max_abs_probability_reconstruction_error":
            float(
                np.max(
                    np.abs(
                        probability_error
                    )
                )
            ),
        "mean_abs_probability_reconstruction_error":
            float(
                np.mean(
                    np.abs(
                        probability_error
                    )
                )
            ),
        "max_nonzero_normalized_row_sum_error":
            max_norm_error,
        "expected_value_mean":
            float(
                np.mean(
                    expected64
                )
            ),
        "expected_value_std":
            float(
                np.std(
                    expected64
                )
            ),
    }

    # ---------------------------------------------------------------------
    # Common metadata/prediction scaffold.
    # ---------------------------------------------------------------------
    local = pd.DataFrame({
        "__row_index__":
            np.arange(
                n_rows,
                dtype=int,
            )
    })

    metadata_candidates = [
        "DIGIGARD_split_origin",
        "source_project",
        "source_file",
        "project",
        "hash",
        "date",
        "fileId",
        "fileName",
        "revision",
    ]

    for column in metadata_candidates:
        if column in source_df.columns:
            local[column] = (
                source_df[
                    column
                ]
                .reset_index(drop=True)
            )

    if TARGET in source_df.columns:
        local[TARGET] = (
            source_df[
                TARGET
            ]
            .reset_index(drop=True)
        )

    local[
        MODEL_PROBABILITY_COL
    ] = model_p1

    local[
        MODEL_PREDICTED_CLASS_COL
    ] = predicted_class

    local[
        SHAP_EXPECTED_RAW_COL
    ] = expected64.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        SHAP_SUM_COL
    ] = shap_sum.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        SHAP_RECONSTRUCTED_RAW_COL
    ] = reconstructed_raw.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        MODEL_RAW_COL
    ] = model_raw.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        SHAP_RAW_ERROR_COL
    ] = raw_error.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        SHAP_RECONSTRUCTED_PROBABILITY_COL
    ] = reconstructed_probability.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        SHAP_PROBABILITY_ERROR_COL
    ] = probability_error.astype(
        SHAP_STORAGE_DTYPE
    )

    local[
        TOTAL_ABS_SHAP_COL
    ] = total_abs.astype(
        SHAP_STORAGE_DTYPE
    )

    # ---------------------------------------------------------------------
    # Full raw local SHAP table.
    # ---------------------------------------------------------------------
    full_local = local.copy()

    for feature_index, feature in enumerate(
        features
    ):
        full_local[
            f"{SHAP_PREFIX}{feature}"
        ] = shap_values[
            :,
            feature_index,
        ]

    # ---------------------------------------------------------------------
    # Full normalized absolute local SHAP table.
    # ---------------------------------------------------------------------
    normalized_local = local.copy()

    for feature_index, feature in enumerate(
        features
    ):
        normalized_local[
            f"{NORM_SHAP_PREFIX}{feature}"
        ] = normalized_abs[
            :,
            feature_index,
        ].astype(
            SHAP_STORAGE_DTYPE
        )

    normalized_local[
        NORMALIZED_ROW_SUM_COL
    ] = normalized_row_sum.astype(
        SHAP_STORAGE_DTYPE
    )

    # ---------------------------------------------------------------------
    # Top-K local explanations.
    # ---------------------------------------------------------------------
    k = min(
        LOCAL_SHAP_TOP_K,
        len(features),
    )

    ranked_indices = np.argsort(
        -abs_shap,
        axis=1,
        kind="stable",
    )[
        :,
        :k,
    ]

    topk = local.copy()
    X_reset = X.reset_index(drop=True)

    json_cells: List[str] = []

    for row_index in range(
        n_rows
    ):
        entries: List[
            Dict[str, Any]
        ] = []

        for rank_zero, feature_index in enumerate(
            ranked_indices[
                row_index
            ]
        ):
            feature_index = int(
                feature_index
            )

            rank = rank_zero + 1
            feature = features[
                feature_index
            ]

            feature_value = (
                X_reset.iloc[
                    row_index
                ][
                    feature
                ]
            )

            signed_shap = float(
                shap64[
                    row_index,
                    feature_index,
                ]
            )

            abs_value = float(
                abs_shap[
                    row_index,
                    feature_index,
                ]
            )

            norm_value = float(
                normalized_abs[
                    row_index,
                    feature_index,
                ]
            )

            topk.loc[
                row_index,
                f"SHAP_TOP_{rank:02d}__feature",
            ] = feature

            topk.loc[
                row_index,
                f"SHAP_TOP_{rank:02d}__feature_value",
            ] = feature_value

            topk.loc[
                row_index,
                f"SHAP_TOP_{rank:02d}__signed_shap",
            ] = signed_shap

            topk.loc[
                row_index,
                f"SHAP_TOP_{rank:02d}__abs_shap",
            ] = abs_value

            topk.loc[
                row_index,
                f"SHAP_TOP_{rank:02d}__normalized_abs_shap",
            ] = norm_value

            entries.append({
                "rank":
                    rank,
                "feature":
                    feature,
                "feature_value":
                    json_safe(
                        feature_value
                    ),
                "signed_shap":
                    signed_shap,
                "abs_shap":
                    abs_value,
                "normalized_abs_shap":
                    norm_value,
            })

        json_cells.append(
            json.dumps(
                entries,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    topk[
        TOPK_JSON_COL
    ] = json_cells

    return (
        full_local,
        normalized_local,
        topk,
        diagnostics,
    )


# =============================================================================
# PLOT HELPERS
# =============================================================================

def save_figure(
    fig: plt.Figure,
    base_path: Path,
) -> None:
    ensure_dir(
        base_path.parent
    )

    if SAVE_PNG:
        fig.savefig(
            base_path.with_suffix(".png"),
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    if SAVE_EPS:
        fig.savefig(
            base_path.with_suffix(".eps"),
            bbox_inches="tight",
        )

    plt.close(fig)


def save_horizontal_bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    xlabel: str,
    base_path: Path,
) -> None:
    labels = list(labels)
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(labels)

    fig_height = max(
        5.0,
        min(
            22.0,
            0.34 * n + 2.5,
        ),
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            fig_height,
        )
    )

    y_pos = np.arange(
        n
    )

    ax.barh(
        y_pos,
        values,
    )

    ax.set_yticks(
        y_pos,
        labels=labels,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_title(
        title
    )

    ax.axvline(
        0.0,
        linewidth=0.8,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        base_path,
    )


def plot_global_shap_suite(
    global_df: pd.DataFrame,
    native_df: pd.DataFrame,
    dataset_dir: Path,
    display_name: str,
) -> None:
    plot_dir = (
        dataset_dir
        / "plots"
        / "global"
    )

    ensure_dir(
        plot_dir
    )

    top = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=top["feature"],
        values=top["mean_abs_shap"],
        title=(
            f"{display_name}\n"
            "Global CatBoost SHAP importance"
        ),
        xlabel="Mean absolute SHAP contribution (raw margin space)",
        base_path=(
            plot_dir
            / "global_mean_abs_shap"
        ),
    )

    save_horizontal_bar_plot(
        labels=top["feature"],
        values=top["relative_mean_abs_shap"],
        title=(
            f"{display_name}\n"
            "Normalized global SHAP importance"
        ),
        xlabel="Relative mean absolute SHAP contribution",
        base_path=(
            plot_dir
            / "global_relative_mean_abs_shap"
        ),
    )

    signed = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_signed_shap",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=signed["feature"],
        values=signed["mean_signed_shap"],
        title=(
            f"{display_name}\n"
            "Mean signed SHAP contribution"
        ),
        xlabel="Mean signed SHAP contribution (raw margin space)",
        base_path=(
            plot_dir
            / "global_mean_signed_shap"
        ),
    )

    class0 = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_abs_shap_true_0",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=class0["feature"],
        values=class0["mean_abs_shap_true_0"],
        title=(
            f"{display_name}\n"
            "Mean |SHAP| for true class 0"
        ),
        xlabel="Mean absolute SHAP contribution",
        base_path=(
            plot_dir
            / "mean_abs_shap_true_class_0"
        ),
    )

    class1 = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_abs_shap_true_1",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=class1["feature"],
        values=class1["mean_abs_shap_true_1"],
        title=(
            f"{display_name}\n"
            "Mean |SHAP| for true class 1"
        ),
        xlabel="Mean absolute SHAP contribution",
        base_path=(
            plot_dir
            / "mean_abs_shap_true_class_1"
        ),
    )

    pred0 = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_abs_shap_pred_0",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=pred0["feature"],
        values=pred0["mean_abs_shap_pred_0"],
        title=(
            f"{display_name}\n"
            "Mean |SHAP| for predicted class 0"
        ),
        xlabel="Mean absolute SHAP contribution",
        base_path=(
            plot_dir
            / "mean_abs_shap_predicted_class_0"
        ),
    )

    pred1 = (
        global_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(global_df),
            )
        )
        .sort_values(
            "mean_abs_shap_pred_1",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=pred1["feature"],
        values=pred1["mean_abs_shap_pred_1"],
        title=(
            f"{display_name}\n"
            "Mean |SHAP| for predicted class 1"
        ),
        xlabel="Mean absolute SHAP contribution",
        base_path=(
            plot_dir
            / "mean_abs_shap_predicted_class_1"
        ),
    )

    native_top = (
        native_df
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(native_df),
            )
        )
        .sort_values(
            "catboost_native_importance",
            ascending=True,
        )
    )

    save_horizontal_bar_plot(
        labels=native_top["feature"],
        values=native_top["catboost_native_importance"],
        title=(
            f"{display_name}\n"
            "CatBoost native feature importance"
        ),
        xlabel="CatBoost native importance",
        base_path=(
            plot_dir
            / "catboost_native_feature_importance"
        ),
    )

    comparison = (
        global_df[
            [
                "feature",
                "relative_mean_abs_shap",
            ]
        ]
        .merge(
            native_df[
                [
                    "feature",
                    "catboost_native_importance_normalized",
                ]
            ],
            on="feature",
            how="inner",
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            7.0,
        )
    )

    ax.scatter(
        comparison[
            "catboost_native_importance_normalized"
        ],
        comparison[
            "relative_mean_abs_shap"
        ],
        alpha=0.75,
    )

    ax.set_xlabel(
        "Normalized CatBoost native importance"
    )

    ax.set_ylabel(
        "Normalized mean |SHAP|"
    )

    ax.set_title(
        f"{display_name}\n"
        "CatBoost native importance vs SHAP"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        plot_dir
        / "catboost_native_vs_shap_importance"
    )


def plot_beeswarm_style(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    global_df: pd.DataFrame,
    dataset_dir: Path,
    display_name: str,
    seed: int,
) -> None:
    top_features = (
        global_df[
            "feature"
        ]
        .head(
            min(
                BEESWARM_TOP_N,
                len(global_df),
            )
        )
        .tolist()
    )

    if not top_features:
        return

    feature_to_index = {
        feature: index
        for index, feature
        in enumerate(X.columns)
    }

    selected_rows = (
        deterministic_plot_subset(
            n_rows=len(X),
            max_rows=
                MAX_ROWS_FOR_BEESWARM_PLOT,
            seed=seed,
        )
    )

    rng = np.random.default_rng(
        seed + 999
    )

    fig_height = max(
        6.0,
        0.48 * len(top_features)
        + 2.5,
    )

    fig, ax = plt.subplots(
        figsize=(
            11.0,
            fig_height,
        )
    )

    y_tick_positions: List[float] = []

    for row_pos, feature in enumerate(
        reversed(
            top_features
        )
    ):
        feature_index = feature_to_index[
            feature
        ]

        shap_feature = np.asarray(
            shap_values[
                selected_rows,
                feature_index,
            ],
            dtype=float,
        )

        feature_values = pd.to_numeric(
            X.iloc[
                selected_rows
            ][
                feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        # Normalize feature values solely for the colormap.
        finite = np.isfinite(
            feature_values
        )

        normalized_values = np.full(
            len(feature_values),
            0.5,
            dtype=float,
        )

        if np.any(finite):
            finite_values = feature_values[
                finite
            ]

            lo = float(
                np.nanquantile(
                    finite_values,
                    0.05,
                )
            )

            hi = float(
                np.nanquantile(
                    finite_values,
                    0.95,
                )
            )

            if hi > lo:
                normalized_values[
                    finite
                ] = np.clip(
                    (
                        finite_values
                        - lo
                    )
                    / (
                        hi - lo
                    ),
                    0.0,
                    1.0,
                )

        jitter = rng.uniform(
            -0.20,
            0.20,
            size=len(
                selected_rows
            ),
        )

        y = (
            np.full(
                len(selected_rows),
                row_pos,
                dtype=float,
            )
            + jitter
        )

        ax.scatter(
            shap_feature,
            y,
            c=normalized_values,
            cmap="viridis",
            s=10,
            alpha=0.65,
            linewidths=0,
        )

        y_tick_positions.append(
            float(row_pos)
        )

    ax.axvline(
        0.0,
        linewidth=0.9,
    )

    ax.set_yticks(
        y_tick_positions,
        labels=list(
            reversed(
                top_features
            )
        ),
    )

    ax.set_xlabel(
        "SHAP contribution (raw margin space)"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        f"{display_name}\n"
        "SHAP beeswarm-style summary"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        dataset_dir
        / "plots"
        / "summary"
        / "shap_beeswarm_style"
    )


def plot_dependence_suite(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    global_df: pd.DataFrame,
    dataset_dir: Path,
    display_name: str,
    seed: int,
) -> None:
    dependence_dir = (
        dataset_dir
        / "plots"
        / "dependence"
    )

    ensure_dir(
        dependence_dir
    )

    selected_rows = (
        deterministic_plot_subset(
            n_rows=len(X),
            max_rows=
                MAX_ROWS_FOR_DEPENDENCE_PLOT,
            seed=seed,
        )
    )

    feature_to_index = {
        feature: index
        for index, feature
        in enumerate(X.columns)
    }

    top_features = (
        global_df[
            "feature"
        ]
        .head(
            min(
                DEPENDENCE_TOP_N,
                len(global_df),
            )
        )
        .tolist()
    )

    for rank, feature in enumerate(
        top_features,
        start=1,
    ):
        feature_index = feature_to_index[
            feature
        ]

        x_values = pd.to_numeric(
            X.iloc[
                selected_rows
            ][
                feature
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        y_values = np.asarray(
            shap_values[
                selected_rows,
                feature_index,
            ],
            dtype=float,
        )

        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
        )

        fig, ax = plt.subplots(
            figsize=(
                8.0,
                6.0,
            )
        )

        ax.scatter(
            x_values[
                valid
            ],
            y_values[
                valid
            ],
            s=12,
            alpha=0.55,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "SHAP contribution (raw margin space)"
        )

        ax.set_title(
            f"{display_name}\n"
            f"SHAP dependence — rank {rank}: {feature}"
        )

        ax.grid(
            alpha=0.20
        )

        fig.tight_layout()

        safe_name = (
            f"{rank:02d}_"
            + "".join(
                ch
                if ch.isalnum()
                or ch in {
                    "_",
                    "-",
                    ".",
                }
                else "_"
                for ch in feature
            )
        )

        save_figure(
            fig,
            dependence_dir
            / safe_name
        )


def plot_feature_distribution_suite(
    X: pd.DataFrame,
    global_df: pd.DataFrame,
    dataset_dir: Path,
    display_name: str,
    seed: int,
) -> None:
    out_dir = (
        dataset_dir
        / "plots"
        / "feature_distributions"
    )

    ensure_dir(
        out_dir
    )

    selected_rows = deterministic_plot_subset(
        n_rows=len(X),
        max_rows=
            MAX_ROWS_FOR_FEATURE_DISTRIBUTION_PLOT,
        seed=seed,
    )

    top_features = (
        global_df[
            "feature"
        ]
        .head(
            min(
                FEATURE_DISTRIBUTION_TOP_N,
                len(global_df),
            )
        )
        .tolist()
    )

    for rank, feature in enumerate(
        top_features,
        start=1,
    ):
        values = pd.to_numeric(
            X.iloc[
                selected_rows
            ][
                feature
            ],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(
                8.0,
                5.5,
            )
        )

        ax.hist(
            values.to_numpy(
                dtype=float
            ),
            bins=40,
            alpha=0.80,
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Count"
        )

        ax.set_title(
            f"{display_name}\n"
            f"Feature distribution — rank {rank}: {feature}"
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

        fig.tight_layout()

        safe_name = (
            f"{rank:02d}_"
            + "".join(
                ch
                if ch.isalnum()
                or ch in {
                    "_",
                    "-",
                    ".",
                }
                else "_"
                for ch in feature
            )
        )

        save_figure(
            fig,
            out_dir
            / safe_name
        )


# =============================================================================
# OPTIONAL STANDARD SHAP PACKAGE PLOTS
# =============================================================================

def save_optional_shap_package_plots(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    expected_values: np.ndarray,
    global_df: pd.DataFrame,
    dataset_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    if not ENABLE_OPTIONAL_SHAP_PACKAGE_PLOTS:
        return {
            "enabled": False,
            "reason": "disabled_by_config",
        }

    try:
        import shap
    except Exception as exc:
        return {
            "enabled": False,
            "reason": (
                "shap package not available"
            ),
            "error": str(exc),
        }

    selected_rows = (
        deterministic_plot_subset(
            n_rows=len(X),
            max_rows=
                SHAP_PACKAGE_MAX_ROWS,
            seed=seed,
        )
    )

    X_plot = X.iloc[
        selected_rows
    ].reset_index(
        drop=True
    )

    shap_plot = np.asarray(
        shap_values[
            selected_rows
        ],
        dtype=float,
    )

    expected_plot = np.asarray(
        expected_values[
            selected_rows
        ],
        dtype=float,
    )

    explanation = shap.Explanation(
        values=shap_plot,
        base_values=expected_plot,
        data=X_plot.to_numpy(
            dtype=float
        ),
        feature_names=list(
            X.columns
        ),
    )

    output_dir = (
        dataset_dir
        / "plots"
        / "shap_package"
    )

    ensure_dir(
        output_dir
    )

    saved: List[str] = []

    try:
        shap.plots.bar(
            explanation,
            max_display=min(
                GLOBAL_PLOT_TOP_N,
                X.shape[1],
            ),
            show=False,
        )

        fig = plt.gcf()
        fig.tight_layout()

        base_path = (
            output_dir
            / "shap_standard_bar"
        )

        if SAVE_PNG:
            fig.savefig(
                base_path.with_suffix(".png"),
                dpi=PLOT_DPI,
                bbox_inches="tight",
            )

        if SAVE_EPS:
            fig.savefig(
                base_path.with_suffix(".eps"),
                bbox_inches="tight",
            )

        plt.close(fig)
        saved.append(
            str(base_path)
        )

    except Exception as exc:
        print(
            f"[WARN] optional SHAP bar plot failed: {exc}"
        )

    try:
        shap.plots.beeswarm(
            explanation,
            max_display=min(
                BEESWARM_TOP_N,
                X.shape[1],
            ),
            show=False,
        )

        fig = plt.gcf()
        fig.tight_layout()

        base_path = (
            output_dir
            / "shap_standard_beeswarm"
        )

        if SAVE_PNG:
            fig.savefig(
                base_path.with_suffix(".png"),
                dpi=PLOT_DPI,
                bbox_inches="tight",
            )

        if SAVE_EPS:
            fig.savefig(
                base_path.with_suffix(".eps"),
                bbox_inches="tight",
            )

        plt.close(fig)
        saved.append(
            str(base_path)
        )

    except Exception as exc:
        print(
            f"[WARN] optional SHAP beeswarm plot failed: {exc}"
        )

    return {
        "enabled": True,
        "package_version":
            safe_package_version(
                "shap"
            ),
        "rows_used":
            int(
                len(
                    selected_rows
                )
            ),
        "saved_plot_bases":
            saved,
    }


# =============================================================================
# REPRESENTATIVE LOCAL EXPLANATIONS
# =============================================================================

def representative_row_indices(
    y_true: np.ndarray,
    probability: np.ndarray,
    predicted_class: np.ndarray,
    total_abs_shap: np.ndarray,
) -> Dict[str, List[int]]:
    n = len(y_true)
    k = min(
        N_REPRESENTATIVE_ROWS_PER_CATEGORY,
        n,
    )

    result: Dict[
        str,
        List[int]
    ] = {}

    result[
        "highest_probability"
    ] = np.argsort(
        -probability,
        kind="stable",
    )[
        :k
    ].astype(
        int
    ).tolist()

    result[
        "lowest_probability"
    ] = np.argsort(
        probability,
        kind="stable",
    )[
        :k
    ].astype(
        int
    ).tolist()

    result[
        "highest_total_absolute_shap"
    ] = np.argsort(
        -total_abs_shap,
        kind="stable",
    )[
        :k
    ].astype(
        int
    ).tolist()

    false_positive_mask = (
        (y_true == 0)
        & (predicted_class == 1)
    )

    fp_indices = np.flatnonzero(
        false_positive_mask
    )

    if len(fp_indices) > 0:
        order = fp_indices[
            np.argsort(
                -probability[
                    fp_indices
                ],
                kind="stable",
            )
        ]

        result[
            "strongest_false_positives"
        ] = order[
            :k
        ].astype(
            int
        ).tolist()

    false_negative_mask = (
        (y_true == 1)
        & (predicted_class == 0)
    )

    fn_indices = np.flatnonzero(
        false_negative_mask
    )

    if len(fn_indices) > 0:
        order = fn_indices[
            np.argsort(
                probability[
                    fn_indices
                ],
                kind="stable",
            )
        ]

        result[
            "strongest_false_negatives"
        ] = order[
            :k
        ].astype(
            int
        ).tolist()

    if LOCAL_FIRST_N_ROWS > 0:
        result[
            "first_rows"
        ] = list(
            range(
                min(
                    LOCAL_FIRST_N_ROWS,
                    n,
                )
            )
        )

    return result


def save_selected_local_plots(
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    features: List[str],
    shap_values: np.ndarray,
    expected_values: np.ndarray,
    model_outputs: Dict[str, np.ndarray],
    dataset_dir: Path,
    display_name: str,
) -> Dict[str, Any]:
    if not SAVE_SELECTED_LOCAL_PLOTS:
        return {
            "enabled": False,
        }

    y_true = validate_binary_target(
        source_df[
            TARGET
        ],
        display_name,
    ).to_numpy(
        dtype=int
    )

    probability = np.asarray(
        model_outputs[
            "probability_class1"
        ],
        dtype=float,
    )

    predicted_class = np.asarray(
        model_outputs[
            "predicted_class"
        ],
        dtype=int,
    )

    shap64 = np.asarray(
        shap_values,
        dtype=float,
    )

    total_abs = np.sum(
        np.abs(
            shap64
        ),
        axis=1,
    )

    categories = representative_row_indices(
        y_true=y_true,
        probability=probability,
        predicted_class=predicted_class,
        total_abs_shap=total_abs,
    )

    local_dir = (
        dataset_dir
        / "plots"
        / "local_explanations"
    )

    ensure_dir(
        local_dir
    )

    saved_rows: List[
        Dict[str, Any]
    ] = []

    for category, indices in categories.items():
        category_dir = (
            local_dir
            / category
        )

        ensure_dir(
            category_dir
        )

        for local_rank, row_index in enumerate(
            indices,
            start=1,
        ):
            row_shap = shap64[
                row_index
            ]

            abs_row = np.abs(
                row_shap
            )

            k = min(
                LOCAL_SHAP_TOP_K,
                len(features),
            )

            top_indices = np.argsort(
                -abs_row,
                kind="stable",
            )[
                :k
            ]

            # Plot smallest at bottom / largest at top.
            top_indices = top_indices[
                ::-1
            ]

            labels = [
                features[
                    int(index)
                ]
                for index in top_indices
            ]

            values = [
                float(
                    row_shap[
                        int(index)
                    ]
                )
                for index in top_indices
            ]

            fig_height = max(
                5.0,
                0.48
                * len(labels)
                + 2.8,
            )

            fig, ax = plt.subplots(
                figsize=(
                    10.0,
                    fig_height,
                )
            )

            y_pos = np.arange(
                len(labels)
            )

            ax.barh(
                y_pos,
                values,
            )

            ax.set_yticks(
                y_pos,
                labels=labels,
            )

            ax.axvline(
                0.0,
                linewidth=0.9,
            )

            ax.set_xlabel(
                "Local SHAP contribution (raw margin space)"
            )

            ax.set_title(
                f"{display_name}\n"
                f"{category} — row {row_index} | "
                f"true={y_true[row_index]} | "
                f"pred={predicted_class[row_index]} | "
                f"P(1)={probability[row_index]:.5f}"
            )

            ax.grid(
                axis="x",
                alpha=0.20,
            )

            fig.tight_layout()

            base_path = (
                category_dir
                / (
                    f"{local_rank:02d}"
                    f"_row_{row_index:08d}"
                )
            )

            save_figure(
                fig,
                base_path,
            )

            saved_rows.append({
                "category":
                    category,
                "category_rank":
                    local_rank,
                "row_index":
                    int(row_index),
                "true_class":
                    int(
                        y_true[
                            row_index
                        ]
                    ),
                "predicted_class":
                    int(
                        predicted_class[
                            row_index
                        ]
                    ),
                "probability_class1":
                    float(
                        probability[
                            row_index
                        ]
                    ),
                "total_abs_shap":
                    float(
                        total_abs[
                            row_index
                        ]
                    ),
                "expected_raw_value":
                    float(
                        expected_values[
                            row_index
                        ]
                    ),
                "plot_base":
                    str(
                        base_path
                    ),
            })

    rows_df = pd.DataFrame(
        saved_rows
    )

    rows_path = (
        local_dir
        / "selected_local_rows.csv"
    )

    rows_df.to_csv(
        rows_path,
        index=False,
    )

    return {
        "enabled": True,
        "n_saved":
            int(
                len(
                    saved_rows
                )
            ),
        "selected_rows_csv":
            str(
                rows_path
            ),
        "categories":
            categories,
    }


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_one_dataset(
    dataset_key: str,
    source_df: pd.DataFrame,
    model: CatBoostClassifier,
    features: List[str],
    native_importance_df: pd.DataFrame,
    probability_threshold: float,
    seed: int,
) -> Dict[str, Any]:
    display_name = DATASET_DISPLAY_NAMES[
        dataset_key
    ]

    interpretation = DATASET_INTERPRETATION[
        dataset_key
    ]

    dataset_dir = (
        OUTPUT_DIR
        / dataset_key
    )

    ensure_dir(
        dataset_dir
    )

    print()
    print("=" * 88)
    print(
        f"STAGE 05 SHAP: {display_name}"
    )
    print("=" * 88)

    if TARGET not in source_df.columns:
        raise ValueError(
            f"{dataset_key}: target '{TARGET}' missing."
        )

    y_true = validate_binary_target(
        source_df[
            TARGET
        ],
        dataset_key,
    ).to_numpy(
        dtype=int
    )

    X, preprocessing_report = preprocess_numeric_features(
        source_df,
        features,
        dataset_key,
    )

    print(
        f"[DATA] rows={len(X):,}, "
        f"features={len(features)}"
    )

    # ---------------------------------------------------------------------
    # Model outputs + native CatBoost SHAP.
    # ---------------------------------------------------------------------
    model_outputs = compute_model_outputs(
        model=model,
        X=X,
        threshold=
            probability_threshold,
    )

    shap_values, expected_values = (
        compute_native_catboost_shap(
            model=model,
            X=X,
            features=features,
        )
    )

    y_pred = np.asarray(
        model_outputs[
            "predicted_class"
        ],
        dtype=int,
    )

    # ---------------------------------------------------------------------
    # Global feature analysis.
    # ---------------------------------------------------------------------
    global_df = build_global_shap_importance(
        shap_values=shap_values,
        features=features,
        y_true=y_true,
        y_pred=y_pred,
    )

    global_csv = (
        dataset_dir
        / "global_shap_importance.csv"
    )

    global_df.to_csv(
        global_csv,
        index=False,
    )

    feature_analysis_df = build_feature_analysis(
        X=X,
        y_true=y_true,
        probability_class1=
            model_outputs[
                "probability_class1"
            ],
        shap_values=shap_values,
        features=features,
    )

    importance_comparison_df = (
        global_df[
            [
                "global_shap_rank",
                "feature",
                "mean_abs_shap",
                "relative_mean_abs_shap",
            ]
        ]
        .merge(
            native_importance_df,
            on="feature",
            how="left",
        )
        .merge(
            feature_analysis_df,
            on="feature",
            how="left",
        )
    )

    importance_comparison_df[
        "rank_difference_shap_minus_native"
    ] = (
        importance_comparison_df[
            "global_shap_rank"
        ]
        - importance_comparison_df[
            "catboost_native_rank"
        ]
    )

    importance_comparison_path = (
        dataset_dir
        / "feature_analysis_complete.csv"
    )

    importance_comparison_df.to_csv(
        importance_comparison_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Local SHAP outputs.
    # ---------------------------------------------------------------------
    (
        full_local_df,
        normalized_local_df,
        topk_local_df,
        reconstruction_diagnostics,
    ) = build_local_shap_tables(
        source_df=source_df,
        X=X,
        features=features,
        shap_values=shap_values,
        expected_values=expected_values,
        model_outputs=model_outputs,
    )

    full_local_path = (
        dataset_dir
        / "local_shap_full_all_features.csv"
    )

    normalized_local_path = (
        dataset_dir
        / "local_shap_normalized_absolute_all_features.csv"
    )

    topk_local_path = (
        dataset_dir
        / (
            f"local_shap_top_{LOCAL_SHAP_TOP_K}_features.csv"
        )
    )

    if SAVE_FULL_LOCAL_SHAP_CSV:
        full_local_df.to_csv(
            full_local_path,
            index=False,
        )

    if SAVE_NORMALIZED_LOCAL_SHAP_CSV:
        normalized_local_df.to_csv(
            normalized_local_path,
            index=False,
        )

    topk_local_df.to_csv(
        topk_local_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Global plot suite.
    # ---------------------------------------------------------------------
    plot_global_shap_suite(
        global_df=global_df,
        native_df=native_importance_df,
        dataset_dir=dataset_dir,
        display_name=display_name,
    )

    plot_beeswarm_style(
        X=X,
        shap_values=shap_values,
        global_df=global_df,
        dataset_dir=dataset_dir,
        display_name=display_name,
        seed=seed,
    )

    plot_dependence_suite(
        X=X,
        shap_values=shap_values,
        global_df=global_df,
        dataset_dir=dataset_dir,
        display_name=display_name,
        seed=seed + 100,
    )

    plot_feature_distribution_suite(
        X=X,
        global_df=global_df,
        dataset_dir=dataset_dir,
        display_name=display_name,
        seed=seed + 200,
    )

    optional_shap_package_report = (
        save_optional_shap_package_plots(
            X=X,
            shap_values=shap_values,
            expected_values=
                expected_values,
            global_df=global_df,
            dataset_dir=dataset_dir,
            seed=seed + 300,
        )
    )

    local_plot_report = save_selected_local_plots(
        source_df=source_df,
        X=X,
        features=features,
        shap_values=shap_values,
        expected_values=expected_values,
        model_outputs=model_outputs,
        dataset_dir=dataset_dir,
        display_name=display_name,
    )

    # ---------------------------------------------------------------------
    # Report.
    # ---------------------------------------------------------------------
    top_features = (
        global_df
        .head(
            min(
                25,
                len(global_df),
            )
        )
        .to_dict(
            orient="records"
        )
    )

    dataset_report = {
        "pipeline":
            "DIGIGARD",
        "stage":
            5,
        "stage_name":
            "catboost_shap_analysis",
        "dataset":
            dataset_key,
        "dataset_display_name":
            display_name,
        "interpretation":
            interpretation,
        "target":
            TARGET,
        "model_run":
            MODEL_RUN,
        "rows":
            int(
                len(
                    source_df
                )
            ),
        "feature_count":
            len(features),
        "probability_threshold":
            probability_threshold,
        "preprocessing":
            preprocessing_report,
        "shap": {
            "method":
                "CatBoost native ShapValues",
            "model_output":
                "Raw",
            "additive_space":
                "raw binary-classifier margin/logit",
            "probability_mapping":
                "sigmoid(expected_value + sum(SHAP))",
            "batch_size":
                SHAP_BATCH_SIZE,
            "storage_dtype":
                str(
                    SHAP_STORAGE_DTYPE
                ),
            "reconstruction_diagnostics":
                reconstruction_diagnostics,
        },
        "top_global_features":
            top_features,
        "optional_shap_package":
            optional_shap_package_report,
        "selected_local_plots":
            local_plot_report,
        "artifacts": {
            "global_shap_importance_csv":
                str(
                    global_csv
                ),
            "feature_analysis_complete_csv":
                str(
                    importance_comparison_path
                ),
            "full_local_shap_csv":
                str(
                    full_local_path
                )
                if SAVE_FULL_LOCAL_SHAP_CSV
                else None,
            "normalized_local_shap_csv":
                str(
                    normalized_local_path
                )
                if SAVE_NORMALIZED_LOCAL_SHAP_CSV
                else None,
            "topk_local_shap_csv":
                str(
                    topk_local_path
                ),
        },
    }

    write_json(
        dataset_dir
        / "shap_analysis_report.json",
        dataset_report,
    )

    # ---------------------------------------------------------------------
    # Human-readable report.
    # ---------------------------------------------------------------------
    report_lines: List[str] = []

    report_lines.append(
        f"DIGIGARD STAGE 05 - {display_name}"
    )

    report_lines.append(
        "=" * 78
    )

    report_lines.append(
        f"TARGET: {TARGET}"
    )

    report_lines.append(
        f"MODEL_RUN: {MODEL_RUN}"
    )

    report_lines.append(
        f"ROWS: {len(source_df)}"
    )

    report_lines.append(
        f"N_FEATURES: {len(features)}"
    )

    report_lines.append("")

    report_lines.append(
        "INTERPRETATION"
    )

    report_lines.append(
        interpretation
    )

    report_lines.append("")

    report_lines.append(
        "SHAP ADDITIVITY"
    )

    report_lines.append(
        "Native CatBoost binary-classifier SHAP values are additive in "
        "raw margin/logit space, not probability space."
    )

    report_lines.append(
        "expected_value + sum(SHAP) = RawFormulaVal"
    )

    report_lines.append(
        "sigmoid(RawFormulaVal) = P(class=1)"
    )

    report_lines.append("")

    report_lines.append(
        "RECONSTRUCTION CHECKS"
    )

    for key, value in reconstruction_diagnostics.items():
        report_lines.append(
            f"  {key}: {value:.12g}"
        )

    report_lines.append("")

    report_lines.append(
        "TOP GLOBAL FEATURES"
    )

    for _, row in global_df.head(
        min(
            25,
            len(global_df),
        )
    ).iterrows():
        report_lines.append(
            f"  {int(row['global_shap_rank']):>2}. "
            f"{row['feature']}: "
            f"mean|SHAP|={row['mean_abs_shap']:.8g}, "
            f"relative={row['relative_mean_abs_shap']:.6g}, "
            f"mean_signed={row['mean_signed_shap']:.8g}"
        )

    report_lines.append("")

    report_lines.append(
        "LOCAL OUTPUTS"
    )

    report_lines.append(
        f"  Full per-feature SHAP CSV: {full_local_path}"
    )

    report_lines.append(
        f"  Normalized absolute SHAP CSV: {normalized_local_path}"
    )

    report_lines.append(
        f"  Top-{LOCAL_SHAP_TOP_K} local explanation CSV: {topk_local_path}"
    )

    (
        dataset_dir
        / "SHAP_REPORT.txt"
    ).write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    print(
        "[SHAP CHECK] "
        f"max raw error="
        f"{reconstruction_diagnostics['max_abs_raw_reconstruction_error']:.3e} | "
        f"max probability error="
        f"{reconstruction_diagnostics['max_abs_probability_reconstruction_error']:.3e} | "
        f"max normalized row-sum error="
        f"{reconstruction_diagnostics['max_nonzero_normalized_row_sum_error']:.3e}"
    )

    print(
        "[TOP SHAP] "
        + ", ".join(
            global_df[
                "feature"
            ]
            .head(5)
            .tolist()
        )
    )

    return {
        "dataset":
            dataset_key,
        "display_name":
            display_name,
        "rows":
            int(
                len(
                    source_df
                )
            ),
        "global_shap_df":
            global_df,
        "report":
            dataset_report,
    }


# =============================================================================
# CROSS-DATASET COMPARISON
# =============================================================================

def rank_correlation_from_vectors(
    a: pd.Series,
    b: pd.Series,
) -> float:
    a_num = pd.to_numeric(
        a,
        errors="coerce",
    )

    b_num = pd.to_numeric(
        b,
        errors="coerce",
    )

    pair = pd.concat(
        [
            a_num,
            b_num,
        ],
        axis=1,
    ).dropna()

    if len(pair) < 3:
        return float("nan")

    if (
        pair.iloc[:, 0].nunique() < 2
        or pair.iloc[:, 1].nunique() < 2
    ):
        return float("nan")

    return float(
        pair.iloc[:, 0].corr(
            pair.iloc[:, 1],
            method="pearson",
        )
    )


def build_cross_dataset_comparison(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> Dict[str, Any]:
    comparison_dir = (
        OUTPUT_DIR
        / "cross_dataset_comparison"
    )

    ensure_dir(
        comparison_dir
    )

    merged: Optional[pd.DataFrame] = None

    for dataset_key in DATASET_ORDER:
        df = results[
            dataset_key
        ][
            "global_shap_df"
        ][
            [
                "feature",
                "global_shap_rank",
                "mean_abs_shap",
                "relative_mean_abs_shap",
                "mean_signed_shap",
            ]
        ].copy()

        df = df.rename(
            columns={
                "global_shap_rank":
                    f"{dataset_key}__rank",
                "mean_abs_shap":
                    f"{dataset_key}__mean_abs_shap",
                "relative_mean_abs_shap":
                    f"{dataset_key}__relative_mean_abs_shap",
                "mean_signed_shap":
                    f"{dataset_key}__mean_signed_shap",
            }
        )

        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df,
                on="feature",
                how="outer",
            )

    assert merged is not None

    rank_columns = [
        f"{dataset_key}__rank"
        for dataset_key in DATASET_ORDER
    ]

    relative_columns = [
        f"{dataset_key}__relative_mean_abs_shap"
        for dataset_key in DATASET_ORDER
    ]

    merged[
        "rank_mean"
    ] = merged[
        rank_columns
    ].mean(
        axis=1
    )

    merged[
        "rank_std"
    ] = merged[
        rank_columns
    ].std(
        axis=1,
        ddof=0,
    )

    merged[
        "rank_range"
    ] = (
        merged[
            rank_columns
        ].max(
            axis=1
        )
        - merged[
            rank_columns
        ].min(
            axis=1
        )
    )

    merged[
        "relative_importance_mean"
    ] = merged[
        relative_columns
    ].mean(
        axis=1
    )

    merged[
        "relative_importance_std"
    ] = merged[
        relative_columns
    ].std(
        axis=1,
        ddof=0,
    )

    merged = merged.sort_values(
        [
            "rank_mean",
            "feature",
        ],
        ascending=[
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    comparison_path = (
        comparison_dir
        / "global_shap_full_train_test_comparison.csv"
    )

    merged.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Pairwise rank correlations.
    # Since ranks are already ranks, Pearson correlation of the rank vectors
    # is Spearman rank correlation of the underlying importance values.
    # ---------------------------------------------------------------------
    rank_corr = pd.DataFrame(
        index=DATASET_ORDER,
        columns=DATASET_ORDER,
        dtype=float,
    )

    importance_corr = pd.DataFrame(
        index=DATASET_ORDER,
        columns=DATASET_ORDER,
        dtype=float,
    )

    feature_indexed = merged.set_index(
        "feature"
    )

    for left in DATASET_ORDER:
        for right in DATASET_ORDER:
            rank_corr.loc[
                left,
                right,
            ] = rank_correlation_from_vectors(
                feature_indexed[
                    f"{left}__rank"
                ],
                feature_indexed[
                    f"{right}__rank"
                ],
            )

            importance_corr.loc[
                left,
                right,
            ] = rank_correlation_from_vectors(
                feature_indexed[
                    f"{left}__relative_mean_abs_shap"
                ],
                feature_indexed[
                    f"{right}__relative_mean_abs_shap"
                ],
            )

    rank_corr_path = (
        comparison_dir
        / "shap_rank_correlation_matrix.csv"
    )

    importance_corr_path = (
        comparison_dir
        / "shap_relative_importance_correlation_matrix.csv"
    )

    rank_corr.to_csv(
        rank_corr_path
    )

    importance_corr.to_csv(
        importance_corr_path
    )

    # ---------------------------------------------------------------------
    # Plot side-by-side relative importance for strongest globally stable
    # features.
    # ---------------------------------------------------------------------
    top_features = (
        merged[
            "feature"
        ]
        .head(
            min(
                GLOBAL_PLOT_TOP_N,
                len(merged),
            )
        )
        .tolist()
    )

    plot_df = (
        merged[
            merged[
                "feature"
            ].isin(
                top_features
            )
        ]
        .set_index(
            "feature"
        )
        .loc[
            top_features
        ]
    )

    fig_height = max(
        6.0,
        0.40
        * len(
            top_features
        )
        + 2.8,
    )

    fig, ax = plt.subplots(
        figsize=(
            12.0,
            fig_height,
        )
    )

    y_positions = np.arange(
        len(
            top_features
        )
    )

    n_datasets = len(
        DATASET_ORDER
    )

    height = 0.80 / n_datasets

    for offset_index, dataset_key in enumerate(
        DATASET_ORDER
    ):
        offset = (
            offset_index
            - (
                n_datasets - 1
            ) / 2.0
        ) * height

        ax.barh(
            y_positions + offset,
            plot_df[
                f"{dataset_key}__relative_mean_abs_shap"
            ].to_numpy(
                dtype=float
            ),
            height=height,
            label=
                DATASET_DISPLAY_NAMES[
                    dataset_key
                ],
        )

    ax.set_yticks(
        y_positions,
        labels=top_features,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Relative mean absolute SHAP contribution"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "DIGIGARD CatBoost SHAP\n"
        "Full vs training vs testing feature importance"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        comparison_dir
        / "global_shap_full_train_test_comparison"
    )

    report = {
        "comparison_csv":
            str(
                comparison_path
            ),
        "rank_correlation_matrix_csv":
            str(
                rank_corr_path
            ),
        "relative_importance_correlation_matrix_csv":
            str(
                importance_corr_path
            ),
        "rank_correlation_matrix":
            rank_corr.to_dict(),
        "relative_importance_correlation_matrix":
            importance_corr.to_dict(),
        "most_rank_stable_features":
            merged.sort_values(
                [
                    "rank_std",
                    "rank_mean",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .head(
                min(
                    25,
                    len(merged),
                )
            )
            .to_dict(
                orient="records"
            ),
    }

    write_json(
        comparison_dir
        / "cross_dataset_shap_report.json",
        report,
    )

    return report


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = time.time()

    for required_path in [
        TRAIN_CSV,
        TEST_CSV,
        MODEL_MANIFEST,
        MODEL_FEATURES_JSON,
        MODEL_PATH,
    ]:
        if not required_path.exists():
            raise SystemExit(
                "Required upstream artifact not found:\n"
                f"  {required_path.resolve()}"
            )

    if (
        OUTPUT_DIR.exists()
        and not OVERWRITE_EXISTING_ANALYSIS
    ):
        raise SystemExit(
            "Stage-05 output directory exists and "
            "OVERWRITE_EXISTING_ANALYSIS=False:\n"
            f"  {OUTPUT_DIR.resolve()}"
        )

    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 88)
    print(
        "DIGIGARD STAGE 05 - CATBOOST SHAP + FEATURE ANALYSIS"
    )
    print("=" * 88)
    print(
        f"TARGET       : {TARGET}"
    )
    print(
        f"MODEL_RUN    : {MODEL_RUN}"
    )
    print(
        f"MODEL_PATH   : {MODEL_PATH.resolve()}"
    )
    print(
        f"TRAIN_CSV    : {TRAIN_CSV.resolve()}"
    )
    print(
        f"TEST_CSV     : {TEST_CSV.resolve()}"
    )
    print(
        f"OUTPUT_DIR   : {OUTPUT_DIR.resolve()}"
    )
    print()

    manifest = load_model_manifest()
    features = load_features()
    model = load_model()

    probability_threshold = (
        resolve_probability_threshold(
            manifest
        )
    )

    native_importance_df = (
        get_catboost_native_importance(
            model,
            features,
        )
    )

    native_importance_path = (
        OUTPUT_DIR
        / "catboost_native_feature_importance.csv"
    )

    native_importance_df.to_csv(
        native_importance_path,
        index=False,
    )

    train_df = pd.read_csv(
        TRAIN_CSV,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    train_for_full = train_df.copy()
    test_for_full = test_df.copy()

    train_for_full[
        "DIGIGARD_split_origin"
    ] = "train"

    test_for_full[
        "DIGIGARD_split_origin"
    ] = "test"

    full_df = pd.concat(
        [
            train_for_full,
            test_for_full,
        ],
        ignore_index=True,
        sort=False,
    )

    datasets = {
        "full_dataset":
            full_df,
        "training_dataset":
            train_df,
        "testing_dataset":
            test_df,
    }

    seeds = {
        "full_dataset":
            RANDOM_STATE + 1000,
        "training_dataset":
            RANDOM_STATE + 2000,
        "testing_dataset":
            RANDOM_STATE + 3000,
    }

    results: Dict[
        str,
        Dict[str, Any]
    ] = {}

    for dataset_key in DATASET_ORDER:
        results[
            dataset_key
        ] = analyze_one_dataset(
            dataset_key=
                dataset_key,
            source_df=
                datasets[
                    dataset_key
                ],
            model=
                model,
            features=
                features,
            native_importance_df=
                native_importance_df,
            probability_threshold=
                probability_threshold,
            seed=
                seeds[
                    dataset_key
                ],
        )

    cross_dataset_report = (
        build_cross_dataset_comparison(
            results
        )
    )

    elapsed_seconds = float(
        time.time()
        - start_time
    )

    # ---------------------------------------------------------------------
    # Compact dataset summary.
    # ---------------------------------------------------------------------
    summary_rows: List[
        Dict[str, Any]
    ] = []

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        report = result[
            "report"
        ]

        diagnostics = (
            report[
                "shap"
            ][
                "reconstruction_diagnostics"
            ]
        )

        top = result[
            "global_shap_df"
        ].iloc[
            0
        ]

        summary_rows.append({
            "dataset":
                dataset_key,
            "dataset_display_name":
                result[
                    "display_name"
                ],
            "rows":
                result[
                    "rows"
                ],
            "top_shap_feature":
                top[
                    "feature"
                ],
            "top_mean_abs_shap":
                float(
                    top[
                        "mean_abs_shap"
                    ]
                ),
            "top_relative_mean_abs_shap":
                float(
                    top[
                        "relative_mean_abs_shap"
                    ]
                ),
            "max_abs_raw_reconstruction_error":
                diagnostics[
                    "max_abs_raw_reconstruction_error"
                ],
            "max_abs_probability_reconstruction_error":
                diagnostics[
                    "max_abs_probability_reconstruction_error"
                ],
            "max_nonzero_normalized_row_sum_error":
                diagnostics[
                    "max_nonzero_normalized_row_sum_error"
                ],
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "shap_analysis_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Master JSON.
    # ---------------------------------------------------------------------
    master_report = {
        "pipeline":
            "DIGIGARD",
        "stage":
            5,
        "stage_name":
            "catboost_shap_and_feature_analysis",
        "target":
            TARGET,
        "model_run":
            MODEL_RUN,
        "model_type":
            "CATBOOST",
        "model_path":
            str(
                MODEL_PATH.resolve()
            ),
        "feature_count":
            len(features),
        "features":
            features,
        "probability_threshold":
            probability_threshold,
        "shap_method":
            "CatBoost native ShapValues",
        "shap_additive_space":
            "raw binary-classifier margin/logit",
        "probability_mapping":
            "sigmoid(expected_value + sum(SHAP))",
        "datasets": {
            dataset_key:
                results[
                    dataset_key
                ][
                    "report"
                ]
            for dataset_key in DATASET_ORDER
        },
        "cross_dataset_comparison":
            cross_dataset_report,
        "artifacts": {
            "catboost_native_feature_importance_csv":
                str(
                    native_importance_path.resolve()
                ),
            "shap_analysis_summary_csv":
                str(
                    summary_path.resolve()
                ),
        },
        "environment": {
            "python":
                sys.version,
            "platform":
                platform.platform(),
            "numpy":
                safe_package_version(
                    "numpy"
                ),
            "pandas":
                safe_package_version(
                    "pandas"
                ),
            "matplotlib":
                safe_package_version(
                    "matplotlib"
                ),
            "catboost":
                safe_package_version(
                    "catboost"
                ),
            "shap":
                safe_package_version(
                    "shap"
                ),
        },
        "elapsed_seconds":
            elapsed_seconds,
    }

    master_report_path = (
        OUTPUT_DIR
        / "shap_analysis_report.json"
    )

    write_json(
        master_report_path,
        master_report,
    )

    # ---------------------------------------------------------------------
    # Human-readable master report.
    # ---------------------------------------------------------------------
    lines: List[str] = []

    lines.append(
        "DIGIGARD STAGE 05 - CATBOOST SHAP + FEATURE ANALYSIS"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        f"TARGET: {TARGET}"
    )

    lines.append(
        f"MODEL_RUN: {MODEL_RUN}"
    )

    lines.append(
        f"MODEL: {MODEL_PATH.resolve()}"
    )

    lines.append(
        f"N_FEATURES: {len(features)}"
    )

    lines.append("")

    lines.append(
        "SHAP INTERPRETATION"
    )

    lines.append(
        "CatBoost native binary-classifier SHAP contributions are additive "
        "in raw margin/logit space."
    )

    lines.append(
        "expected_value + sum(SHAP) = RawFormulaVal"
    )

    lines.append(
        "sigmoid(RawFormulaVal) = predict_proba(X)[:, 1]"
    )

    lines.append("")

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        diagnostics = (
            result[
                "report"
            ][
                "shap"
            ][
                "reconstruction_diagnostics"
            ]
        )

        lines.append(
            "=" * 80
        )

        lines.append(
            result[
                "display_name"
            ]
        )

        lines.append(
            "=" * 80
        )

        lines.append(
            f"Rows: {result['rows']}"
        )

        lines.append(
            DATASET_INTERPRETATION[
                dataset_key
            ]
        )

        lines.append("")

        lines.append(
            "Reconstruction diagnostics:"
        )

        for key, value in diagnostics.items():
            lines.append(
                f"  {key}: {value:.12g}"
            )

        lines.append("")

        lines.append(
            "Top 20 global SHAP features:"
        )

        for _, row in (
            result[
                "global_shap_df"
            ]
            .head(
                min(
                    20,
                    len(
                        result[
                            "global_shap_df"
                        ]
                    ),
                )
            )
            .iterrows()
        ):
            lines.append(
                f"  {int(row['global_shap_rank']):>2}. "
                f"{row['feature']}: "
                f"mean|SHAP|={row['mean_abs_shap']:.8g}, "
                f"relative={row['relative_mean_abs_shap']:.6g}"
            )

        lines.append("")

    lines.append(
        "CROSS-DATASET SHAP STABILITY"
    )

    lines.append(
        "Rank-correlation and relative-importance correlation matrices are "
        "saved under cross_dataset_comparison/."
    )

    lines.append("")

    human_report_path = (
        OUTPUT_DIR
        / "SHAP_ANALYSIS_REPORT.txt"
    )

    human_report_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Run configuration.
    # ---------------------------------------------------------------------
    config = {
        "pipeline":
            "DIGIGARD",
        "stage":
            5,
        "target":
            TARGET,
        "model_run":
            MODEL_RUN,
        "model_type":
            "CATBOOST",
        "random_state":
            RANDOM_STATE,
        "train_csv":
            str(
                TRAIN_CSV
            ),
        "test_csv":
            str(
                TEST_CSV
            ),
        "model_path":
            str(
                MODEL_PATH
            ),
        "output_dir":
            str(
                OUTPUT_DIR
            ),
        "shap_batch_size":
            SHAP_BATCH_SIZE,
        "shap_storage_dtype":
            str(
                SHAP_STORAGE_DTYPE
            ),
        "local_shap_top_k":
            LOCAL_SHAP_TOP_K,
        "global_plot_top_n":
            GLOBAL_PLOT_TOP_N,
        "beeswarm_top_n":
            BEESWARM_TOP_N,
        "dependence_top_n":
            DEPENDENCE_TOP_N,
        "feature_distribution_top_n":
            FEATURE_DISTRIBUTION_TOP_N,
        "optional_shap_package_plots":
            ENABLE_OPTIONAL_SHAP_PACKAGE_PLOTS,
        "save_full_local_shap_csv":
            SAVE_FULL_LOCAL_SHAP_CSV,
        "save_normalized_local_shap_csv":
            SAVE_NORMALIZED_LOCAL_SHAP_CSV,
        "save_selected_local_plots":
            SAVE_SELECTED_LOCAL_PLOTS,
    }

    write_json(
        OUTPUT_DIR
        / "shap_analysis_config.json",
        config,
    )

    print()
    print("=" * 88)
    print(
        "DIGIGARD STAGE 05 COMPLETE"
    )
    print("=" * 88)

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        top = result[
            "global_shap_df"
        ].iloc[
            0
        ]

        print(
            f"{result['display_name']}: "
            f"top feature={top['feature']} | "
            f"relative mean|SHAP|="
            f"{float(top['relative_mean_abs_shap']):.6f}"
        )

    print()
    print(
        f"[OK] Output directory -> {OUTPUT_DIR}"
    )
    print(
        f"[OK] Summary          -> {summary_path}"
    )
    print(
        f"[OK] Master JSON      -> {master_report_path}"
    )
    print(
        f"[OK] Master TXT       -> {human_report_path}"
    )


if __name__ == "__main__":
    main()
