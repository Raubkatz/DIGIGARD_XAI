#!/usr/bin/env python3
"""
03_DIGIGARD_train_classifier.py

DIGIGARD Stage 03: train ONE binary classifier for ONE ground truth per run.

This stage runs AFTER:
    01_DIGIGARD_merge_and_clean_project_histories.py
    02_DIGIGARD_create_train_test_splits.py

Supported classifiers
---------------------
    MODEL_TYPE = "CATBOOST"
    MODEL_TYPE = "LIGHTGBM"

Hyperparameter optimization
---------------------------
    USE_HYPERPARAMETER_OPTIMIZATION = False
        -> use the DIGIGARD default parameter profile

    USE_HYPERPARAMETER_OPTIMIZATION = True
        -> run Optuna/TPE hyperparameter optimization on the INTERNAL
           training portion only

Core methodological rule
------------------------
This script ALWAYS runs exactly ONE TARGET per execution.

The Stage-02 test.csv is NEVER used for:
    - hyperparameter optimization
    - early stopping
    - model selection
    - choosing the number of boosting iterations

Workflow
--------
Stage-02 train.csv
    -> internal train/validation split
    -> balanced internal training subset
    -> optional Optuna CV optimization on internal training only
    -> early stopping against internal validation
    -> freeze hyperparameters + best boosting iteration
    -> refit a fresh final model on the FULL Stage-02 train.csv
       (after the configured training balancing)
    -> evaluate exactly once on untouched Stage-02 test.csv
    -> save model, reports, probabilities and feature importance

Important preprocessing
-----------------------
The Stage-01 DIGIGARD contract already defines all model features as numeric.
Therefore:
    - no standardization/scaling is performed
    - no learned imputation is performed
    - values are defensively coerced to numeric
    - +/- infinity is converted to NaN
    - NaN is left as NaN because CatBoost and LightGBM natively handle
      missing numeric values

Dependencies
------------
Required:
    pandas
    numpy
    scikit-learn
    joblib

Model packages:
    catboost
    lightgbm

Only when hyperparameter optimization is enabled:
    optuna
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# =============================================================================
# CONFIG: EDIT THESE FOR EACH RUN
# =============================================================================

# -------------------------------------------------------------------------
# ONE TARGET PER RUN.
# Must correspond to one Stage-02 target directory.
# -------------------------------------------------------------------------
TARGET = "isBugPresent"

# Other valid DIGIGARD ground truths created in Stage 02:
#
# TARGET = "isBugfix"
# TARGET = "isSZZBugIntroducer"


# -------------------------------------------------------------------------
# ONE MODEL PER RUN.
# Allowed:
#     "CATBOOST"
#     "LIGHTGBM"
# -------------------------------------------------------------------------
MODEL_TYPE = "CATBOOST"

# MODEL_TYPE = "LIGHTGBM"


# -------------------------------------------------------------------------
# Hyperparameter optimization switch.
#
# False:
#     train the selected model with the DIGIGARD default parameter profile.
#
# True:
#     optimize the selected model with Optuna/TPE using CV on the INTERNAL
#     training data only.
# -------------------------------------------------------------------------
USE_HYPERPARAMETER_OPTIMIZATION = False


# =============================================================================
# PIPELINE PATHS
# =============================================================================

RANDOM_STATE = 42

STAGE02_ROOT = Path("DIGIGARD_02_train_test_splits")
STAGE02_CONFIG = STAGE02_ROOT / "pipeline_config.json"

TARGET_STAGE02_ROOT = STAGE02_ROOT / TARGET
SPLIT_DIR = TARGET_STAGE02_ROOT / f"splits_{RANDOM_STATE}"

TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
FEATURES_JSON = TARGET_STAGE02_ROOT / "features.json"

STAGE03_ROOT = Path("DIGIGARD_03_models")


# =============================================================================
# INTERNAL VALIDATION
# =============================================================================

VAL_SIZE = 0.20
STRATIFY_INTERNAL_VALIDATION = True


# =============================================================================
# CLASS BALANCING
# =============================================================================

# This retains the experimental philosophy of the previous pipeline:
# use a balanced training set by keeping a configurable fraction of the
# minority class and the same number of majority examples.
BALANCE_TRAINING = True
TRAIN_MINORITY_FRACTION = 0.90

# Validation/test are additionally described using repeated balanced excerpts.
N_BALANCED_EVAL_EXCERPTS = 100
EVAL_MINORITY_FRACTION = 0.90


# =============================================================================
# CLASSIFICATION / PROBABILITY SETTINGS
# =============================================================================

PROBABILITY_THRESHOLD = 0.50

# Positive ground-truth class.
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0


# =============================================================================
# EARLY STOPPING
# =============================================================================

EARLY_STOPPING_ROUNDS = 100

# Maximum boosting rounds used by the DIGIGARD default models before early
# stopping determines the actual selected number.
DEFAULT_MAX_BOOSTING_ROUNDS = 3000


# =============================================================================
# FINAL REFIT
# =============================================================================

# Recommended:
# after tuning / early stopping on the internal split, refit a fresh final
# model on the complete Stage-02 train.csv using the frozen best settings.
REFIT_ON_FULL_STAGE02_TRAIN = True


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

OPTUNA_N_TRIALS = 50
OPTUNA_CV_FOLDS = 3
OPTUNA_TIMEOUT_SECONDS: Optional[int] = None

# Allowed:
#     "f1"
#     "balanced_accuracy"
#     "roc_auc"
#     "average_precision"
OPTIMIZATION_METRIC = "f1"

# Optuna output verbosity.
OPTUNA_SHOW_PROGRESS_BAR = True


# =============================================================================
# CATBOOST SETTINGS
# =============================================================================

CATBOOST_USE_GPU = False
CATBOOST_GPU_DEVICES = "0"
CATBOOST_THREAD_COUNT = -1
CATBOOST_VERBOSE = False


# =============================================================================
# LIGHTGBM SETTINGS
# =============================================================================

# CPU is the portable default.
# Typical values supported by LightGBM installations include "cpu", "gpu",
# and in some builds "cuda". Keep "cpu" unless your installation is configured.
LIGHTGBM_DEVICE_TYPE = "cpu"
LIGHTGBM_N_JOBS = -1
LIGHTGBM_VERBOSITY = -1


# =============================================================================
# OUTPUT / FAILURE BEHAVIOR
# =============================================================================

OVERWRITE_EXISTING_RUN = True

SAVE_JOBLIB_MODEL = True
SAVE_NATIVE_MODEL = True
SAVE_TEST_PREDICTIONS = True
SAVE_FEATURE_IMPORTANCE = True


# =============================================================================
# VALIDATION OF CONFIG
# =============================================================================

MODEL_TYPE = str(MODEL_TYPE).strip().upper()

if MODEL_TYPE not in {"CATBOOST", "LIGHTGBM"}:
    raise ValueError(
        "MODEL_TYPE must be 'CATBOOST' or 'LIGHTGBM'. "
        f"Got: {MODEL_TYPE!r}"
    )

if not 0.0 < VAL_SIZE < 1.0:
    raise ValueError("VAL_SIZE must be strictly between 0 and 1.")

if not 0.0 < TRAIN_MINORITY_FRACTION <= 1.0:
    raise ValueError("TRAIN_MINORITY_FRACTION must be in (0, 1].")

if not 0.0 < EVAL_MINORITY_FRACTION <= 1.0:
    raise ValueError("EVAL_MINORITY_FRACTION must be in (0, 1].")

if not 0.0 < PROBABILITY_THRESHOLD < 1.0:
    raise ValueError("PROBABILITY_THRESHOLD must be strictly between 0 and 1.")

if OPTUNA_CV_FOLDS < 2:
    raise ValueError("OPTUNA_CV_FOLDS must be >= 2.")

if OPTUNA_N_TRIALS < 1:
    raise ValueError("OPTUNA_N_TRIALS must be >= 1.")

if OPTIMIZATION_METRIC not in {
    "f1",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
}:
    raise ValueError(
        "OPTIMIZATION_METRIC must be one of: "
        "f1, balanced_accuracy, roc_auc, average_precision"
    )


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def safe_package_version(package: str) -> Optional[str]:
    try:
        return importlib_metadata.version(package)
    except Exception:
        return None


def class_counts(y: pd.Series | np.ndarray) -> Dict[str, int]:
    values = pd.Series(np.asarray(y)).value_counts().sort_index()
    return {
        str(int(key)): int(value)
        for key, value in values.items()
    }


def validate_binary_target(y: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(y, errors="coerce")

    if numeric.isna().any():
        raise ValueError(
            f"{name}: target contains {int(numeric.isna().sum())} missing/"
            "non-numeric values."
        )

    observed = sorted(float(v) for v in pd.unique(numeric))

    unexpected = [
        value
        for value in observed
        if value not in {0.0, 1.0}
    ]

    if unexpected:
        raise ValueError(
            f"{name}: target is not binary 0/1. "
            f"Unexpected values: {unexpected}"
        )

    out = numeric.astype(int)

    if out.nunique() != 2:
        raise ValueError(
            f"{name}: both classes 0 and 1 are required. "
            f"Counts: {class_counts(out)}"
        )

    return out


def preprocess_numeric_features(
    df: pd.DataFrame,
    features: List[str],
    dataset_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministic preprocessing only.

    No fitted transform is learned here, so there is no preprocessing leakage.
    """
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required model features:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    coercion_losses: Dict[str, int] = {}
    missing_before: Dict[str, int] = {}
    missing_after: Dict[str, int] = {}

    for column in features:
        original = X[column]
        missing_before[column] = int(original.isna().sum())

        if pd.api.types.is_numeric_dtype(original):
            converted = pd.to_numeric(original, errors="coerce")
        else:
            text = original.astype("string").str.strip()
            text = text.str.replace(",", ".", regex=False)
            converted = pd.to_numeric(text, errors="coerce")

        before_non_missing = int(original.notna().sum())
        after_non_missing = int(converted.notna().sum())

        loss = max(0, before_non_missing - after_non_missing)
        if loss > 0:
            coercion_losses[column] = loss

        X[column] = converted
        missing_after[column] = int(converted.isna().sum())

    X = X.replace([np.inf, -np.inf], np.nan)

    all_missing_features = [
        column
        for column in features
        if X[column].isna().all()
    ]

    report = {
        "dataset": dataset_name,
        "rows": int(len(X)),
        "feature_count": len(features),
        "numeric_coercions_to_nan": coercion_losses,
        "all_missing_features": all_missing_features,
        "total_missing_cells": int(X.isna().sum().sum()),
        "rows_with_any_missing_feature": int(X.isna().any(axis=1).sum()),
        "rows_with_all_features_missing": int(X.isna().all(axis=1).sum()),
    }

    return X, report


# =============================================================================
# BALANCING
# =============================================================================

def undersample_balanced_fraction(
    X: pd.DataFrame,
    y: pd.Series,
    fraction_minority: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Keep fraction_minority of the minority class and the same number of
    majority-class examples.

    This returns a strict 50/50 binary dataset.
    """
    y = pd.Series(
        np.asarray(y, dtype=int),
        index=X.index,
        name=y.name,
    )

    counts = y.value_counts()

    if len(counts) != 2:
        raise ValueError(
            f"Balanced undersampling requires two classes. Counts: "
            f"{counts.to_dict()}"
        )

    minority_class = counts.idxmin()
    majority_class = counts.idxmax()

    minority_idx = y.index[y == minority_class].to_numpy()
    majority_idx = y.index[y == majority_class].to_numpy()

    n_minority_available = len(minority_idx)
    n_pick = int(round(n_minority_available * fraction_minority))
    n_pick = max(1, min(n_pick, n_minority_available, len(majority_idx)))

    rng = np.random.default_rng(seed)

    chosen_minority = rng.choice(
        minority_idx,
        size=n_pick,
        replace=False,
    )
    chosen_majority = rng.choice(
        majority_idx,
        size=n_pick,
        replace=False,
    )

    selected = np.concatenate([
        chosen_minority,
        chosen_majority,
    ])
    rng.shuffle(selected)

    X_out = X.loc[selected].copy()
    y_out = y.loc[selected].copy()

    return X_out, y_out


def maybe_balance_training(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    before = class_counts(y)

    if not BALANCE_TRAINING:
        return X.copy(), y.copy(), {
            "enabled": False,
            "class_counts_before": before,
            "class_counts_after": before,
            "minority_fraction": None,
        }

    X_b, y_b = undersample_balanced_fraction(
        X=X,
        y=y,
        fraction_minority=TRAIN_MINORITY_FRACTION,
        seed=seed,
    )

    return X_b, y_b, {
        "enabled": True,
        "class_counts_before": before,
        "class_counts_after": class_counts(y_b),
        "minority_fraction": TRAIN_MINORITY_FRACTION,
    }


# =============================================================================
# METRICS
# =============================================================================

def probability_to_class(
    probability_class1: np.ndarray,
    threshold: float = PROBABILITY_THRESHOLD,
) -> np.ndarray:
    p = np.asarray(probability_class1, dtype=float)
    return (p >= threshold).astype(int)


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probability_class1: np.ndarray,
    threshold: float = PROBABILITY_THRESHOLD,
) -> Tuple[Dict[str, float], np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability_class1, dtype=float)

    if len(y) != len(p):
        raise ValueError("y_true and probability array length mismatch.")

    pred = probability_to_class(p, threshold)

    cm = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else float("nan")
    )

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y, pred, zero_division=0)
        ),
        "specificity": specificity,
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y, pred)
        ),
        "predicted_positive_fraction": float(pred.mean()),
        "true_positive_fraction": float(y.mean()),
        "probability_mean": float(np.mean(p)),
        "probability_std": float(np.std(p)),
        "brier_score": float(
            brier_score_loss(y, p)
        ),
    }

    try:
        out["roc_auc"] = float(
            roc_auc_score(y, p)
        )
    except Exception:
        out["roc_auc"] = float("nan")

    try:
        out["average_precision"] = float(
            average_precision_score(y, p)
        )
    except Exception:
        out["average_precision"] = float("nan")

    try:
        out["log_loss"] = float(
            log_loss(
                y,
                np.column_stack([1.0 - p, p]),
                labels=[0, 1],
            )
        )
    except Exception:
        out["log_loss"] = float("nan")

    return out, cm


def optimization_score(
    y_true: np.ndarray,
    probability_class1: np.ndarray,
) -> float:
    if OPTIMIZATION_METRIC == "f1":
        pred = probability_to_class(
            probability_class1,
            PROBABILITY_THRESHOLD,
        )
        return float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        )

    if OPTIMIZATION_METRIC == "balanced_accuracy":
        pred = probability_to_class(
            probability_class1,
            PROBABILITY_THRESHOLD,
        )
        return float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        )

    if OPTIMIZATION_METRIC == "roc_auc":
        return float(
            roc_auc_score(
                y_true,
                probability_class1,
            )
        )

    if OPTIMIZATION_METRIC == "average_precision":
        return float(
            average_precision_score(
                y_true,
                probability_class1,
            )
        )

    raise RuntimeError(
        f"Unsupported optimization metric: {OPTIMIZATION_METRIC}"
    )


def repeated_balanced_evaluation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    seed0: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, float]] = []
    cms: List[np.ndarray] = []

    for i in range(N_BALANCED_EVAL_EXCERPTS):
        X_b, y_b = undersample_balanced_fraction(
            X=X,
            y=y,
            fraction_minority=EVAL_MINORITY_FRACTION,
            seed=seed0 + i,
        )

        p = predict_probability_class1(
            model,
            X_b,
        )

        metrics, cm = binary_metrics(
            y_b,
            p,
            threshold=PROBABILITY_THRESHOLD,
        )

        rows.append(metrics)
        cms.append(cm.astype(float))

    metric_df = pd.DataFrame(rows)

    return {
        "iterations": N_BALANCED_EVAL_EXCERPTS,
        "minority_fraction": EVAL_MINORITY_FRACTION,
        "rows_per_excerpt": int(
            2
            * round(
                min(class_counts(y).values())
                * EVAL_MINORITY_FRACTION
            )
        ),
        "metrics_mean": {
            column: float(metric_df[column].mean())
            for column in metric_df.columns
        },
        "metrics_std": {
            column: float(metric_df[column].std(ddof=1))
            for column in metric_df.columns
        },
        "mean_confusion_matrix": (
            np.mean(
                np.stack(cms, axis=0),
                axis=0,
            ).tolist()
        ),
    }


# =============================================================================
# MODEL IMPORTS / CONSTRUCTION
# =============================================================================

def import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception as exc:
        raise ImportError(
            "CatBoost is required for MODEL_TYPE='CATBOOST'. "
            "Install package 'catboost'."
        ) from exc


def import_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier
        return lgb, LGBMClassifier
    except Exception as exc:
        raise ImportError(
            "LightGBM is required for MODEL_TYPE='LIGHTGBM'. "
            "Install package 'lightgbm'."
        ) from exc


def common_catboost_params(seed: int) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": seed,
        "allow_writing_files": False,
        "verbose": CATBOOST_VERBOSE,
        "thread_count": CATBOOST_THREAD_COUNT,
    }

    if CATBOOST_USE_GPU:
        params["task_type"] = "GPU"
        params["devices"] = CATBOOST_GPU_DEVICES
    else:
        params["task_type"] = "CPU"

    return params


def default_catboost_params(seed: int) -> Dict[str, Any]:
    params = common_catboost_params(seed)

    params.update({
        "iterations": DEFAULT_MAX_BOOSTING_ROUNDS,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
    })

    return params


def common_lightgbm_params(seed: int) -> Dict[str, Any]:
    return {
        "objective": "binary",
        "random_state": seed,
        "n_jobs": LIGHTGBM_N_JOBS,
        "verbosity": LIGHTGBM_VERBOSITY,
        "device_type": LIGHTGBM_DEVICE_TYPE,
        "importance_type": "gain",
    }


def default_lightgbm_params(seed: int) -> Dict[str, Any]:
    params = common_lightgbm_params(seed)

    params.update({
        "n_estimators": DEFAULT_MAX_BOOSTING_ROUNDS,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    })

    return params


def build_model(
    params: Dict[str, Any],
) -> Any:
    if MODEL_TYPE == "CATBOOST":
        CatBoostClassifier = import_catboost()
        return CatBoostClassifier(**params)

    lgb, LGBMClassifier = import_lightgbm()
    return LGBMClassifier(**params)


def fit_with_early_stopping(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Any:
    if MODEL_TYPE == "CATBOOST":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=CATBOOST_VERBOSE,
        )
        return model

    lgb, _ = import_lightgbm()

    callbacks = [
        lgb.early_stopping(
            stopping_rounds=EARLY_STOPPING_ROUNDS,
            first_metric_only=True,
            verbose=False,
        ),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )

    return model


def fit_without_validation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    if MODEL_TYPE == "CATBOOST":
        model.fit(
            X_train,
            y_train,
            verbose=CATBOOST_VERBOSE,
        )
        return model

    lgb, _ = import_lightgbm()

    model.fit(
        X_train,
        y_train,
        callbacks=[
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def predict_probability_class1(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    proba = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            "Expected binary predict_proba output of shape (n, 2), "
            f"got {proba.shape}."
        )

    return proba[:, 1]


def best_boosting_iteration(
    model: Any,
) -> int:
    if MODEL_TYPE == "CATBOOST":
        # tree_count_ already reflects use_best_model after early stopping.
        value = int(getattr(model, "tree_count_", 0) or 0)

        if value <= 0:
            best_iteration = getattr(model, "get_best_iteration", lambda: -1)()
            if best_iteration is not None and int(best_iteration) >= 0:
                value = int(best_iteration) + 1

        return max(1, value)

    # LightGBM best_iteration_ is the number of boosting iterations selected by
    # the early-stopping callback.
    value = int(getattr(model, "best_iteration_", 0) or 0)

    if value <= 0:
        value = int(getattr(model, "n_estimators_", 0) or 0)

    if value <= 0:
        value = int(
            model.get_params().get(
                "n_estimators",
                DEFAULT_MAX_BOOSTING_ROUNDS,
            )
        )

    return max(1, value)


def params_for_final_refit(
    selected_params: Dict[str, Any],
    selected_boosting_rounds: int,
    seed: int,
) -> Dict[str, Any]:
    params = dict(selected_params)

    if MODEL_TYPE == "CATBOOST":
        params.update(common_catboost_params(seed))
        params["iterations"] = int(selected_boosting_rounds)
    else:
        params.update(common_lightgbm_params(seed))
        params["n_estimators"] = int(selected_boosting_rounds)

    return params


# =============================================================================
# OPTUNA
# =============================================================================

def import_optuna():
    try:
        import optuna
        return optuna
    except Exception as exc:
        raise ImportError(
            "USE_HYPERPARAMETER_OPTIMIZATION=True requires package 'optuna'."
        ) from exc


def sample_catboost_params(
    trial: Any,
    seed: int,
) -> Dict[str, Any]:
    params = common_catboost_params(seed)

    params.update({
        "iterations": trial.suggest_int(
            "iterations",
            500,
            5000,
            step=100,
        ),
        "depth": trial.suggest_int(
            "depth",
            4,
            10,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True,
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1.0,
            30.0,
            log=True,
        ),
        "random_strength": trial.suggest_float(
            "random_strength",
            1e-3,
            10.0,
            log=True,
        ),
        "border_count": trial.suggest_int(
            "border_count",
            32,
            255,
        ),
    })

    return params


def sample_lightgbm_params(
    trial: Any,
    seed: int,
) -> Dict[str, Any]:
    max_depth = trial.suggest_int(
        "max_depth",
        3,
        12,
    )

    max_leaves = min(
        255,
        2 ** max_depth,
    )

    params = common_lightgbm_params(seed)

    params.update({
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            5000,
            step=100,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True,
        ),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int(
            "num_leaves",
            15,
            max_leaves,
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            5,
            100,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
        ),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            10.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            10.0,
            log=True,
        ),
    })

    return params


def run_hyperparameter_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    optuna = import_optuna()

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=(
            f"DIGIGARD_{TARGET}_{MODEL_TYPE}_{OPTIMIZATION_METRIC}"
        ),
    )

    cv = StratifiedKFold(
        n_splits=OPTUNA_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objective(trial: Any) -> float:
        fold_scores: List[float] = []

        for fold_i, (train_idx, val_idx) in enumerate(
            cv.split(X, y)
        ):
            fold_seed = RANDOM_STATE + 1000 + fold_i

            if MODEL_TYPE == "CATBOOST":
                params = sample_catboost_params(
                    trial,
                    seed=fold_seed,
                )
            else:
                params = sample_lightgbm_params(
                    trial,
                    seed=fold_seed,
                )

            model = build_model(params)

            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]

            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]

            model = fit_with_early_stopping(
                model=model,
                X_train=X_fold_train,
                y_train=y_fold_train,
                X_val=X_fold_val,
                y_val=y_fold_val,
            )

            p = predict_probability_class1(
                model,
                X_fold_val,
            )

            fold_scores.append(
                optimization_score(
                    y_fold_val.to_numpy(),
                    p,
                )
            )

        return float(np.mean(fold_scores))

    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS,
        timeout=OPTUNA_TIMEOUT_SECONDS,
        show_progress_bar=OPTUNA_SHOW_PROGRESS_BAR,
    )

    if len(study.trials) == 0:
        raise RuntimeError("Optuna produced no trials.")

    best_trial_params = dict(
        study.best_trial.params
    )

    # Reconstruct the complete model parameter dictionary because Optuna stores
    # only parameters explicitly suggested by the trial.
    if MODEL_TYPE == "CATBOOST":
        selected_params = common_catboost_params(
            RANDOM_STATE
        )
        selected_params.update(best_trial_params)
    else:
        selected_params = common_lightgbm_params(
            RANDOM_STATE
        )
        selected_params.update(best_trial_params)
        selected_params["subsample_freq"] = 1

    trials_df = study.trials_dataframe()
    trials_path = output_dir / "hyperparameter_trials.csv"
    trials_df.to_csv(
        trials_path,
        index=False,
    )

    optimization_report = {
        "enabled": True,
        "backend": "Optuna TPE",
        "metric": OPTIMIZATION_METRIC,
        "n_trials_requested": OPTUNA_N_TRIALS,
        "n_trials_completed": int(len(study.trials)),
        "cv_folds": OPTUNA_CV_FOLDS,
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_params": best_trial_params,
        "full_selected_params_before_early_stopping":
            selected_params,
        "trials_csv": str(trials_path),
    }

    return selected_params, optimization_report


# =============================================================================
# MODEL ARTIFACTS
# =============================================================================

def save_native_model(
    model: Any,
    output_dir: Path,
) -> Optional[Path]:
    if not SAVE_NATIVE_MODEL:
        return None

    if MODEL_TYPE == "CATBOOST":
        path = output_dir / "best_model.cbm"
        model.save_model(str(path))
        return path

    path = output_dir / "best_model.txt"
    model.booster_.save_model(str(path))
    return path


def save_feature_importance(
    model: Any,
    features: List[str],
    output_dir: Path,
) -> Optional[Path]:
    if not SAVE_FEATURE_IMPORTANCE:
        return None

    if MODEL_TYPE == "CATBOOST":
        values = np.asarray(
            model.get_feature_importance(),
            dtype=float,
        )
        importance_type = "CatBoost native feature importance"
    else:
        values = np.asarray(
            model.booster_.feature_importance(
                importance_type="gain",
            ),
            dtype=float,
        )
        importance_type = "LightGBM gain"

    if len(values) != len(features):
        raise ValueError(
            "Feature-importance length does not match feature contract."
        )

    importance = pd.DataFrame({
        "feature": features,
        "importance": values,
    }).sort_values(
        "importance",
        ascending=False,
        kind="stable",
    )

    total = float(importance["importance"].sum())

    if total > 0:
        importance["importance_normalized"] = (
            importance["importance"] / total
        )
    else:
        importance["importance_normalized"] = 0.0

    importance["importance_type"] = importance_type

    path = output_dir / "feature_importance.csv"
    importance.to_csv(path, index=False)

    return path


# =============================================================================
# REPORT FORMATTING
# =============================================================================

def format_metrics(metrics: Dict[str, float]) -> str:
    preferred = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
    ]

    lines: List[str] = []

    for key in preferred:
        if key in metrics:
            value = metrics[key]
            lines.append(
                f"    {key}: {value:.6g}"
            )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start_time = time.time()

    # -------------------------------------------------------------------------
    # Validate Stage-02 configuration and ensure this run targets exactly one
    # ground truth known to the pipeline.
    # -------------------------------------------------------------------------
    if not STAGE02_CONFIG.exists():
        raise SystemExit(
            "Stage-02 pipeline_config.json not found:\n"
            f"  {STAGE02_CONFIG.resolve()}\n\n"
            "Run 02_DIGIGARD_create_train_test_splits.py first."
        )

    with STAGE02_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as handle:
        stage02_config = json.load(handle)

    allowed_targets = list(
        stage02_config.get(
            "targets",
            [],
        )
    )

    if TARGET not in allowed_targets:
        raise SystemExit(
            f"TARGET={TARGET!r} is not a Stage-02 DIGIGARD target.\n"
            f"Allowed targets: {allowed_targets}"
        )

    for required_path in [
        TRAIN_CSV,
        TEST_CSV,
        FEATURES_JSON,
    ]:
        if not required_path.exists():
            raise SystemExit(
                "Required Stage-02 artifact not found:\n"
                f"  {required_path.resolve()}"
            )

    with FEATURES_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        features_payload = json.load(handle)

    if isinstance(features_payload, list):
        features = list(features_payload)
    elif isinstance(features_payload, dict):
        features = list(
            features_payload.get(
                "features",
                [],
            )
        )
    else:
        raise ValueError(
            f"Unsupported features.json format: "
            f"{type(features_payload).__name__}"
        )

    if not features:
        raise ValueError("No features loaded from Stage-02 features.json.")

    # -------------------------------------------------------------------------
    # Output namespace.
    # -------------------------------------------------------------------------
    optimization_tag = (
        "optuna"
        if USE_HYPERPARAMETER_OPTIMIZATION
        else "default"
    )

    model_slug = (
        "catboost"
        if MODEL_TYPE == "CATBOOST"
        else "lightgbm"
    )

    run_name = (
        f"{model_slug}_{optimization_tag}_rs{RANDOM_STATE}"
    )

    output_dir = (
        STAGE03_ROOT
        / TARGET
        / run_name
    )

    if output_dir.exists() and not OVERWRITE_EXISTING_RUN:
        raise SystemExit(
            "Output directory already exists and "
            "OVERWRITE_EXISTING_RUN=False:\n"
            f"  {output_dir.resolve()}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("DIGIGARD STAGE 03 - BINARY CLASSIFIER TRAINING")
    print("=" * 88)
    print(f"TARGET                  : {TARGET}")
    print(f"MODEL_TYPE              : {MODEL_TYPE}")
    print(
        "HYPERPARAMETER_OPTIM.  : "
        f"{USE_HYPERPARAMETER_OPTIMIZATION}"
    )
    print(f"RANDOM_STATE            : {RANDOM_STATE}")
    print(f"TRAIN_CSV               : {TRAIN_CSV.resolve()}")
    print(f"TEST_CSV                : {TEST_CSV.resolve()}")
    print(f"N_FEATURES              : {len(features)}")
    print(f"OUTPUT_DIR              : {output_dir.resolve()}")
    print()

    # -------------------------------------------------------------------------
    # Load Stage-02 data.
    # -------------------------------------------------------------------------
    train_df = pd.read_csv(
        TRAIN_CSV,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    if TARGET not in train_df.columns:
        raise ValueError(
            f"TARGET '{TARGET}' missing from train.csv."
        )

    if TARGET not in test_df.columns:
        raise ValueError(
            f"TARGET '{TARGET}' missing from test.csv."
        )

    y_stage02_train = validate_binary_target(
        train_df[TARGET],
        "train.csv",
    )

    y_test = validate_binary_target(
        test_df[TARGET],
        "test.csv",
    )

    X_stage02_train, train_preprocess_report = (
        preprocess_numeric_features(
            train_df,
            features,
            "train.csv",
        )
    )

    X_test, test_preprocess_report = (
        preprocess_numeric_features(
            test_df,
            features,
            "test.csv",
        )
    )

    # -------------------------------------------------------------------------
    # Internal train / validation split.
    # -------------------------------------------------------------------------
    stratify_values = (
        y_stage02_train
        if STRATIFY_INTERNAL_VALIDATION
        else None
    )

    (
        X_internal_train,
        X_internal_val,
        y_internal_train,
        y_internal_val,
    ) = train_test_split(
        X_stage02_train,
        y_stage02_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify_values,
    )

    # Preserve original indices so balancing selects the same X/y rows.
    X_internal_train = X_internal_train.copy()
    X_internal_val = X_internal_val.copy()

    y_internal_train = y_internal_train.loc[
        X_internal_train.index
    ].copy()

    y_internal_val = y_internal_val.loc[
        X_internal_val.index
    ].copy()

    # -------------------------------------------------------------------------
    # Balanced internal training data.
    # -------------------------------------------------------------------------
    (
        X_internal_train_bal,
        y_internal_train_bal,
        internal_balance_report,
    ) = maybe_balance_training(
        X=X_internal_train,
        y=y_internal_train,
        seed=RANDOM_STATE + 10_000,
    )

    # Balanced validation set used for early stopping.
    (
        X_internal_val_bal,
        y_internal_val_bal,
    ) = undersample_balanced_fraction(
        X=X_internal_val,
        y=y_internal_val,
        fraction_minority=EVAL_MINORITY_FRACTION,
        seed=RANDOM_STATE + 20_000,
    )

    print("[INTERNAL SPLIT]")
    print(
        f"  internal train raw : "
        f"{len(X_internal_train):,} "
        f"{class_counts(y_internal_train)}"
    )
    print(
        f"  internal train used: "
        f"{len(X_internal_train_bal):,} "
        f"{class_counts(y_internal_train_bal)}"
    )
    print(
        f"  internal val raw   : "
        f"{len(X_internal_val):,} "
        f"{class_counts(y_internal_val)}"
    )
    print(
        f"  internal val eval  : "
        f"{len(X_internal_val_bal):,} "
        f"{class_counts(y_internal_val_bal)}"
    )
    print()

    # -------------------------------------------------------------------------
    # Hyperparameter optimization OR default profile.
    # -------------------------------------------------------------------------
    if USE_HYPERPARAMETER_OPTIMIZATION:
        print(
            f"[HYPEROPT] Running Optuna: "
            f"{OPTUNA_N_TRIALS} trials, "
            f"{OPTUNA_CV_FOLDS}-fold CV, "
            f"metric={OPTIMIZATION_METRIC}"
        )

        selected_params, optimization_report = (
            run_hyperparameter_optimization(
                X=X_internal_train_bal.reset_index(drop=True),
                y=y_internal_train_bal.reset_index(drop=True),
                output_dir=output_dir,
            )
        )
    else:
        print("[HYPEROPT] Disabled: using DIGIGARD default profile.")

        if MODEL_TYPE == "CATBOOST":
            selected_params = default_catboost_params(
                RANDOM_STATE
            )
        else:
            selected_params = default_lightgbm_params(
                RANDOM_STATE
            )

        optimization_report = {
            "enabled": False,
            "backend": None,
            "metric": None,
            "n_trials_requested": 0,
            "cv_folds": 0,
            "selected_profile": "DIGIGARD default",
            "selected_params_before_early_stopping":
                selected_params,
        }

    write_json(
        output_dir / "selected_params_before_early_stopping.json",
        selected_params,
    )

    # -------------------------------------------------------------------------
    # Provisional model:
    # train on internal training only, use internal validation for early stopping.
    # -------------------------------------------------------------------------
    print("[FIT] Provisional model with internal validation / early stopping...")

    provisional_model = build_model(
        selected_params
    )

    provisional_model = fit_with_early_stopping(
        model=provisional_model,
        X_train=X_internal_train_bal,
        y_train=y_internal_train_bal,
        X_val=X_internal_val_bal,
        y_val=y_internal_val_bal,
    )

    selected_rounds = best_boosting_iteration(
        provisional_model
    )

    print(
        f"[FIT] Selected boosting iterations: {selected_rounds}"
    )

    # -------------------------------------------------------------------------
    # Validation reports.
    # -------------------------------------------------------------------------
    p_val_full = predict_probability_class1(
        provisional_model,
        X_internal_val,
    )

    val_full_metrics, val_full_cm = binary_metrics(
        y_internal_val,
        p_val_full,
        threshold=PROBABILITY_THRESHOLD,
    )

    val_balanced = repeated_balanced_evaluation(
        model=provisional_model,
        X=X_internal_val,
        y=y_internal_val,
        seed0=RANDOM_STATE + 30_000,
    )

    validation_report = {
        "target": TARGET,
        "model_type": MODEL_TYPE,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "internal_validation": {
            "raw_full_validation": {
                "rows": int(len(X_internal_val)),
                "class_counts": class_counts(y_internal_val),
                "metrics": val_full_metrics,
                "confusion_matrix":
                    val_full_cm.astype(int).tolist(),
            },
            "balanced_repeated": val_balanced,
        },
        "selected_boosting_iterations": selected_rounds,
    }

    write_json(
        output_dir / "validation_report.json",
        validation_report,
    )

    # -------------------------------------------------------------------------
    # Final refit on the complete Stage-02 training data.
    # -------------------------------------------------------------------------
    (
        X_final_train,
        y_final_train,
        final_balance_report,
    ) = maybe_balance_training(
        X=X_stage02_train,
        y=y_stage02_train,
        seed=RANDOM_STATE + 40_000,
    )

    final_params = params_for_final_refit(
        selected_params=selected_params,
        selected_boosting_rounds=selected_rounds,
        seed=RANDOM_STATE,
    )

    write_json(
        output_dir / "final_model_params.json",
        final_params,
    )

    if REFIT_ON_FULL_STAGE02_TRAIN:
        print(
            "[FIT] Refitting fresh final model on complete Stage-02 "
            "training data with frozen settings..."
        )

        final_model = build_model(
            final_params
        )

        final_model = fit_without_validation(
            model=final_model,
            X_train=X_final_train,
            y_train=y_final_train,
        )

        final_training_scope = (
            "complete Stage-02 train.csv after configured balancing"
        )
    else:
        print(
            "[FIT] REFIT_ON_FULL_STAGE02_TRAIN=False; "
            "saving provisional model."
        )

        final_model = provisional_model
        final_training_scope = (
            "internal training subset only; validation used for early stopping"
        )

    # -------------------------------------------------------------------------
    # Save model.
    # -------------------------------------------------------------------------
    joblib_path: Optional[Path] = None

    if SAVE_JOBLIB_MODEL:
        joblib_path = (
            output_dir
            / "best_model.joblib"
        )
        joblib.dump(
            final_model,
            joblib_path,
        )

    native_model_path = save_native_model(
        final_model,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Final untouched test evaluation.
    # -------------------------------------------------------------------------
    print("[TEST] Evaluating final model on untouched Stage-02 test.csv...")

    p_test = predict_probability_class1(
        final_model,
        X_test,
    )

    test_pred = probability_to_class(
        p_test,
        PROBABILITY_THRESHOLD,
    )

    test_full_metrics, test_full_cm = binary_metrics(
        y_test,
        p_test,
        threshold=PROBABILITY_THRESHOLD,
    )

    test_balanced = repeated_balanced_evaluation(
        model=final_model,
        X=X_test,
        y=y_test,
        seed0=RANDOM_STATE + 50_000,
    )

    test_report = {
        "target": TARGET,
        "model_type": MODEL_TYPE,
        "model_run": run_name,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "test_set_role": (
            "untouched held-out Stage-02 test set; not used for tuning, "
            "early stopping or model selection"
        ),
        "full_test": {
            "rows": int(len(X_test)),
            "class_counts": class_counts(y_test),
            "metrics": test_full_metrics,
            "confusion_matrix":
                test_full_cm.astype(int).tolist(),
        },
        "balanced_repeated": test_balanced,
    }

    write_json(
        output_dir / "test_report.json",
        test_report,
    )

    # -------------------------------------------------------------------------
    # Test probability output.
    # -------------------------------------------------------------------------
    test_predictions_path: Optional[Path] = None

    if SAVE_TEST_PREDICTIONS:
        probability_output = test_df.copy()

        probability_output[
            f"{TARGET}_probability_class0"
        ] = 1.0 - p_test

        probability_output[
            f"{TARGET}_probability_class1"
        ] = p_test

        probability_output[
            f"{TARGET}_probability_threshold"
        ] = PROBABILITY_THRESHOLD

        probability_output[
            f"{TARGET}_predicted"
        ] = test_pred

        test_predictions_path = (
            output_dir
            / "test_predictions.csv"
        )

        probability_output.to_csv(
            test_predictions_path,
            index=False,
        )

    # -------------------------------------------------------------------------
    # Feature importance.
    # -------------------------------------------------------------------------
    feature_importance_path = save_feature_importance(
        model=final_model,
        features=features,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # Save exact feature/model contract for downstream DIGIGARD stages.
    # -------------------------------------------------------------------------
    feature_contract = {
        "pipeline": "DIGIGARD",
        "stage": 3,
        "target": TARGET,
        "features": features,
        "feature_count": len(features),
        "model_type": MODEL_TYPE,
        "probability_method": "predict_proba",
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "positive_probability_column": 1,
        "default_probability_threshold":
            PROBABILITY_THRESHOLD,
        "preprocessing": {
            "numeric_only": True,
            "scaling": False,
            "learned_imputation": False,
            "missing_values": (
                "retained as NaN; handled natively by selected tree model"
            ),
            "infinity": "converted to NaN",
        },
    }

    write_json(
        output_dir / "features.json",
        feature_contract,
    )

    # -------------------------------------------------------------------------
    # Training report.
    # -------------------------------------------------------------------------
    elapsed_seconds = float(
        time.time() - start_time
    )

    training_report = {
        "pipeline": "DIGIGARD",
        "stage": 3,
        "stage_name": "train_binary_classifier",
        "target": TARGET,
        "model_type": MODEL_TYPE,
        "run_name": run_name,
        "random_state": RANDOM_STATE,
        "feature_count": len(features),
        "features": features,
        "source": {
            "train_csv": str(TRAIN_CSV.resolve()),
            "test_csv": str(TEST_CSV.resolve()),
            "features_json": str(FEATURES_JSON.resolve()),
        },
        "internal_split": {
            "validation_size": VAL_SIZE,
            "stratified": STRATIFY_INTERNAL_VALIDATION,
            "internal_train_rows":
                int(len(X_internal_train)),
            "internal_validation_rows":
                int(len(X_internal_val)),
            "internal_train_class_counts":
                class_counts(y_internal_train),
            "internal_validation_class_counts":
                class_counts(y_internal_val),
        },
        "preprocessing": {
            "train": train_preprocess_report,
            "test": test_preprocess_report,
        },
        "balancing": {
            "internal_training":
                internal_balance_report,
            "final_training":
                final_balance_report,
            "balanced_evaluation_excerpts":
                N_BALANCED_EVAL_EXCERPTS,
            "balanced_evaluation_minority_fraction":
                EVAL_MINORITY_FRACTION,
        },
        "hyperparameter_optimization":
            optimization_report,
        "selected_boosting_iterations":
            selected_rounds,
        "selected_params_before_early_stopping":
            selected_params,
        "final_model_params":
            final_params,
        "final_refit": {
            "enabled":
                REFIT_ON_FULL_STAGE02_TRAIN,
            "training_scope":
                final_training_scope,
            "rows":
                int(len(X_final_train)),
            "class_counts":
                class_counts(y_final_train),
        },
        "validation_summary": {
            "full_metrics":
                val_full_metrics,
            "balanced_metrics_mean":
                val_balanced["metrics_mean"],
        },
        "test_summary": {
            "full_metrics":
                test_full_metrics,
            "balanced_metrics_mean":
                test_balanced["metrics_mean"],
        },
        "artifacts": {
            "joblib_model":
                str(joblib_path.resolve())
                if joblib_path is not None
                else None,
            "native_model":
                str(native_model_path.resolve())
                if native_model_path is not None
                else None,
            "test_predictions":
                str(test_predictions_path.resolve())
                if test_predictions_path is not None
                else None,
            "feature_importance":
                str(feature_importance_path.resolve())
                if feature_importance_path is not None
                else None,
        },
        "probability_interpretation": {
            "positive_probability":
                "predict_proba(X)[:, 1]",
            "default_threshold":
                PROBABILITY_THRESHOLD,
            "calibration_warning": (
                "Because training uses class balancing by undersampling, "
                "raw predicted probabilities should not automatically be "
                "interpreted as calibrated real-world prevalence probabilities. "
                "Use a dedicated calibration stage if calibrated probabilities "
                "are required."
            ),
        },
        "environment": {
            "python":
                sys.version,
            "platform":
                platform.platform(),
            "numpy":
                safe_package_version("numpy"),
            "pandas":
                safe_package_version("pandas"),
            "scikit_learn":
                safe_package_version("scikit-learn"),
            "joblib":
                safe_package_version("joblib"),
            "catboost":
                safe_package_version("catboost"),
            "lightgbm":
                safe_package_version("lightgbm"),
            "optuna":
                safe_package_version("optuna"),
        },
        "elapsed_seconds":
            elapsed_seconds,
    }

    write_json(
        output_dir / "training_report.json",
        training_report,
    )

    # -------------------------------------------------------------------------
    # Model manifest for downstream loading.
    # -------------------------------------------------------------------------
    model_manifest = {
        "pipeline": "DIGIGARD",
        "stage": 3,
        "target": TARGET,
        "model_type": MODEL_TYPE,
        "run_name": run_name,
        "hyperparameter_optimization":
            USE_HYPERPARAMETER_OPTIMIZATION,
        "joblib_model":
            str(joblib_path)
            if joblib_path is not None
            else None,
        "native_model":
            str(native_model_path)
            if native_model_path is not None
            else None,
        "features_json":
            str(output_dir / "features.json"),
        "test_predictions":
            str(test_predictions_path)
            if test_predictions_path is not None
            else None,
        "probability_method":
            "predict_proba",
        "positive_class":
            POSITIVE_CLASS,
        "positive_probability_column":
            1,
        "default_probability_threshold":
            PROBABILITY_THRESHOLD,
    }

    write_json(
        output_dir / "model_manifest.json",
        model_manifest,
    )

    # -------------------------------------------------------------------------
    # Human-readable report.
    # -------------------------------------------------------------------------
    report_lines: List[str] = []

    report_lines.append("DIGIGARD STAGE 03")
    report_lines.append("=" * 72)
    report_lines.append(f"TARGET: {TARGET}")
    report_lines.append(f"MODEL_TYPE: {MODEL_TYPE}")
    report_lines.append(
        "HYPERPARAMETER_OPTIMIZATION: "
        f"{USE_HYPERPARAMETER_OPTIMIZATION}"
    )
    report_lines.append(f"RUN_NAME: {run_name}")
    report_lines.append(f"RANDOM_STATE: {RANDOM_STATE}")
    report_lines.append(f"N_FEATURES: {len(features)}")
    report_lines.append("")

    report_lines.append("=== TRAINING METHODOLOGY ===")
    report_lines.append(
        "  - Stage-02 train.csv was internally split into train/validation."
    )
    report_lines.append(
        "  - Stage-02 test.csv was not used for tuning or early stopping."
    )
    report_lines.append(
        "  - Training uses balanced undersampling when BALANCE_TRAINING=True."
    )
    report_lines.append(
        "  - Hyperparameter optimization, when enabled, runs only on the "
        "internal training data."
    )
    report_lines.append(
        "  - Internal validation determines the best boosting iteration."
    )
    report_lines.append(
        "  - A fresh final model is then fitted on the complete Stage-02 "
        "training set with frozen settings."
    )
    report_lines.append(
        "  - The final model is evaluated on the untouched Stage-02 test set."
    )
    report_lines.append("")

    report_lines.append("=== INTERNAL VALIDATION: FULL ===")
    report_lines.append(
        format_metrics(
            val_full_metrics
        )
    )
    report_lines.append("")

    report_lines.append("=== INTERNAL VALIDATION: BALANCED MEAN ===")
    report_lines.append(
        format_metrics(
            val_balanced["metrics_mean"]
        )
    )
    report_lines.append("")

    report_lines.append("=== FINAL TEST: FULL HELD-OUT SET ===")
    report_lines.append(
        format_metrics(
            test_full_metrics
        )
    )
    report_lines.append("")

    report_lines.append("=== FINAL TEST: BALANCED REPEATED MEAN ===")
    report_lines.append(
        format_metrics(
            test_balanced["metrics_mean"]
        )
    )
    report_lines.append("")

    report_lines.append("=== SELECTED MODEL ===")
    report_lines.append(
        f"  boosting_iterations: {selected_rounds}"
    )
    report_lines.append(
        f"  native_model: "
        f"{native_model_path if native_model_path else 'not saved'}"
    )
    report_lines.append(
        f"  joblib_model: "
        f"{joblib_path if joblib_path else 'not saved'}"
    )
    report_lines.append("")

    report_lines.append("=== PROBABILITY NOTE ===")
    report_lines.append(
        "  Positive-class probability is predict_proba(X)[:, 1]."
    )
    report_lines.append(
        f"  Default hard-class threshold is {PROBABILITY_THRESHOLD:.2f}."
    )
    report_lines.append(
        "  Because training is balanced by undersampling, these raw probabilities "
        "are not automatically calibrated to the original class prevalence."
    )

    report_path = (
        output_dir
        / "REPORT.txt"
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    print()
    print("=" * 88)
    print("DIGIGARD STAGE 03 COMPLETE")
    print("=" * 88)
    print(f"TARGET             : {TARGET}")
    print(f"MODEL              : {MODEL_TYPE}")
    print(f"RUN                 : {run_name}")
    print(
        f"SELECTED ITERATIONS : {selected_rounds}"
    )
    print()
    print("[FINAL TEST - FULL]")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "f1",
        "mcc",
        "roc_auc",
        "average_precision",
    ]:
        print(
            f"  {key:22s}: "
            f"{test_full_metrics[key]:.6f}"
        )
    print()
    print(f"[OK] Output directory -> {output_dir}")
    print(f"[OK] Training report  -> {output_dir / 'training_report.json'}")
    print(f"[OK] Test report      -> {output_dir / 'test_report.json'}")
    print(f"[OK] Manifest         -> {output_dir / 'model_manifest.json'}")
    print(f"[OK] Human report     -> {report_path}")


if __name__ == "__main__":
    main()
