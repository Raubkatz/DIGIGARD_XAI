#!/usr/bin/env python3
"""
07_DIGIGARD_catboost_predicted_class_feature_distributions.py

DIGIGARD Stage 07:
CatBoost predicted-class feature-distribution analysis.

This stage runs AFTER:
    01_DIGIGARD_merge_and_clean_project_histories.py
    02_DIGIGARD_create_train_test_splits.py
    03_DIGIGARD_train_classifier.py

Recommended preceding analyses:
    04_DIGIGARD_evaluate_catboost_classifier.py
    05_DIGIGARD_catboost_shap_analysis.py
    06_DIGIGARD_catboost_lime_ice_analysis.py

This script performs NO training and NO hyperparameter optimization.

It loads ONE saved Stage-03 CatBoostClassifier for ONE target and analyzes:

    1) full_dataset
       = Stage-02 train.csv + Stage-02 test.csv

    2) training_dataset
       = Stage-02 train.csv

    3) testing_dataset
       = Stage-02 test.csv

For every model feature, observations are grouped by the class PREDICTED by the
saved CatBoost model:

    predicted class 0
    predicted class 1

The script then saves feature-distribution statistics and boxplots.

===============================================================================
PRIMARY DISTRIBUTION VIEW
===============================================================================

For EVERY numeric model feature:

    feature values | predicted class 0
    feature values | predicted class 1

A conventional boxplot is created from ALL available rows.

This is the primary descriptive view.

===============================================================================
SECONDARY REPEATED BALANCED VIEW
===============================================================================

To preserve the earlier OBJ post-hoc boxplot methodology, an optional repeated
balanced analysis is also performed.

Default:
    100 repetitions
    50% of each predicted class per repetition

For each repetition:
    - sample the same number from predicted class 0 and predicted class 1
    - sampling is without replacement within the iteration
    - class counts are therefore exactly balanced

The repeated sampled observations are pooled only for the secondary balanced
boxplot.

IMPORTANT:
The repeated-balanced pooled boxplot contains observations that can appear in
more than one iteration. It is therefore a robustness / visualization view, not
an independent-sample statistical dataset. The ordinary all-row boxplot remains
the main distribution plot.

===============================================================================
SAVED STATISTICS FOR EACH FEATURE AND PREDICTED CLASS
===============================================================================

    n total
    n non-missing
    n missing
    missing fraction

    mean
    standard deviation
    minimum
    q05
    q25
    median
    q75
    q95
    maximum
    IQR

Between predicted classes:
    difference in means: class1 - class0
    difference in medians: class1 - class0
    ratio of medians when defined
    pooled-standard-deviation standardized mean difference
    absolute standardized mean difference

The analysis is descriptive with respect to MODEL PREDICTIONS.
It does not claim that the feature distributions are causal.

===============================================================================
OUTPUT STRUCTURE
===============================================================================

DIGIGARD_07_catboost_predicted_class_distributions/
    <TARGET>/
        <MODEL_RUN>/
            distribution_analysis_config.json
            distribution_analysis_report.json
            distribution_analysis_summary.csv
            DISTRIBUTION_ANALYSIS_REPORT.txt
            catboost_native_feature_importance.csv

            full_dataset/
                predictions.csv
                predicted_class_counts.csv
                feature_distribution_statistics.csv
                feature_class_comparison.csv
                feature_plot_manifest.csv
                DATASET_REPORT.txt

                boxplots_all_rows/
                    001_<feature>.png/.eps
                    ...

                boxplots_balanced_repeated/
                    001_<feature>.png/.eps
                    ...

            training_dataset/
                same structure

            testing_dataset/
                same structure

            cross_dataset_comparison/
                feature_class_comparison_full_train_test.csv
                distribution_cross_dataset_report.json

Dependencies
------------
    numpy
    pandas
    matplotlib
    catboost
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

from catboost import CatBoostClassifier


# =============================================================================
# CONFIG: ONE TARGET + ONE CATBOOST MODEL PER EXECUTION
# =============================================================================

TARGET = "isBugPresent"

# Alternatives:
# TARGET = "isBugfix"
# TARGET = "isSZZBugIntroducer"


MODEL_RUN = "catboost_default_rs42"

# Alternative:
# MODEL_RUN = "catboost_optuna_rs42"


# =============================================================================
# PIPELINE PATHS
# =============================================================================

RANDOM_STATE = 42

STAGE02_ROOT = Path("DIGIGARD_02_train_test_splits")
STAGE03_ROOT = Path("DIGIGARD_03_models")
STAGE07_ROOT = Path(
    "DIGIGARD_07_catboost_predicted_class_distributions"
)

STAGE02_TARGET_ROOT = STAGE02_ROOT / TARGET
SPLIT_DIR = STAGE02_TARGET_ROOT / f"splits_{RANDOM_STATE}"

TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

MODEL_DIR = STAGE03_ROOT / TARGET / MODEL_RUN
MODEL_MANIFEST = MODEL_DIR / "model_manifest.json"
MODEL_FEATURES_JSON = MODEL_DIR / "features.json"
MODEL_PATH = MODEL_DIR / "best_model.cbm"

OUTPUT_DIR = STAGE07_ROOT / TARGET / MODEL_RUN


# =============================================================================
# DISTRIBUTION SETTINGS
# =============================================================================

# Primary all-row boxplots for every model feature.
SAVE_ALL_ROW_BOXPLOTS = True

# Reproduce the earlier repeated-balanced predicted-class boxplot logic.
SAVE_REPEATED_BALANCED_BOXPLOTS = True

N_BOX_ITERATIONS = 100
BOX_SAMPLING_RATIO = 0.50

# If one predicted class is very small, always keep at least this many rows from
# each class where possible.
MIN_ROWS_PER_CLASS_PER_BALANCED_ITERATION = 1

# Conventional boxplots become unreadable with extreme outliers in some metrics.
# The statistics CSV still retains the complete data; this controls only markers.
SHOW_BOXPLOT_OUTLIERS = False

# Add individual sample jitter points to the PRIMARY all-row boxplot.
# Disabled by default because project histories can be large.
SHOW_RAW_POINTS = False
RAW_POINTS_MAX_PER_CLASS = 1000
RAW_POINT_ALPHA = 0.20
RAW_POINT_SIZE = 7.0

# Rank plot filenames by CatBoost native feature importance.
# Every feature is still plotted.
RANK_FEATURES_BY_NATIVE_IMPORTANCE = True


# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

SAVE_PREDICTIONS = True
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
    "full_dataset":
        "Full target dataset (train + test)",
    "training_dataset":
        "Training dataset",
    "testing_dataset":
        "Testing dataset",
}

DATASET_INTERPRETATION = {
    "full_dataset": (
        "Descriptive feature distributions grouped by the saved CatBoost "
        "model's predicted class on the union of Stage-02 training and testing "
        "rows. Because training rows are included, this is not an independent "
        "generalization analysis."
    ),
    "training_dataset": (
        "In-sample feature distributions grouped by the saved CatBoost model's "
        "predicted class on Stage-02 training rows."
    ),
    "testing_dataset": (
        "Held-out feature distributions grouped by the saved CatBoost model's "
        "predicted class on Stage-02 test rows. This is the most important split "
        "for examining whether class-associated feature distributions persist "
        "on held-out observations."
    ),
}


# =============================================================================
# COLUMN NAMES
# =============================================================================

PRED_CLASS_COL = f"{TARGET}__predicted_class"
PRED_PROBABILITY_COL = f"{TARGET}__predicted_probability_class1"


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

if N_BOX_ITERATIONS <= 0:
    raise ValueError(
        "N_BOX_ITERATIONS must be >= 1."
    )

if not 0.0 < BOX_SAMPLING_RATIO <= 1.0:
    raise ValueError(
        "BOX_SAMPLING_RATIO must be in (0,1]."
    )

if MIN_ROWS_PER_CLASS_PER_BALANCED_ITERATION <= 0:
    raise ValueError(
        "MIN_ROWS_PER_CLASS_PER_BALANCED_ITERATION must be >= 1."
    )

if RAW_POINTS_MAX_PER_CLASS <= 0:
    raise ValueError(
        "RAW_POINTS_MAX_PER_CLASS must be >= 1."
    )


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def ensure_dir(
    path: Path,
) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    ensure_dir(
        path.parent
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def safe_package_version(
    package_name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(
            package_name
        )
    except Exception:
        return None


def safe_file_name(
    value: str,
) -> str:
    return "".join(
        character
        if character.isalnum()
        or character in {
            "_",
            "-",
            ".",
        }
        else "_"
        for character in str(
            value
        )
    )


def save_figure(
    fig: plt.Figure,
    base_path: Path,
) -> None:
    ensure_dir(
        base_path.parent
    )

    if SAVE_PNG:
        fig.savefig(
            base_path.with_suffix(
                ".png"
            ),
            dpi=PLOT_DPI,
            bbox_inches="tight",
        )

    if SAVE_EPS:
        fig.savefig(
            base_path.with_suffix(
                ".eps"
            ),
            bbox_inches="tight",
        )

    plt.close(
        fig
    )


def class_counts(
    values: np.ndarray,
) -> Dict[str, int]:
    series = pd.Series(
        np.asarray(
            values,
            dtype=int,
        )
    )

    counts = (
        series
        .value_counts()
        .sort_index()
    )

    return {
        str(
            int(
                key
            )
        ):
            int(
                value
            )
        for key, value
        in counts.items()
    }


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
        float(
            value
        )
        for value in pd.unique(
            numeric
        )
    )

    unexpected = [
        value
        for value in observed
        if value not in {
            0.0,
            1.0,
        }
    ]

    if unexpected:
        raise ValueError(
            f"{dataset_name}: target '{TARGET}' is not binary 0/1. "
            f"Unexpected values: {unexpected}"
        )

    return numeric.astype(
        int
    )


def preprocess_numeric_features(
    df: pd.DataFrame,
    features: List[str],
    dataset_name: str,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name}: required model features are missing:\n"
            + "\n".join(
                missing
            )
        )

    X = df[
        features
    ].copy()

    coercion_losses: Dict[
        str,
        int,
    ] = {}

    for feature in features:
        original = X[
            feature
        ]

        before = int(
            original.notna().sum()
        )

        if pd.api.types.is_numeric_dtype(
            original
        ):
            converted = pd.to_numeric(
                original,
                errors="coerce",
            )

        else:
            text = (
                original
                .astype(
                    "string"
                )
                .str.strip()
                .str.replace(
                    ",",
                    ".",
                    regex=False,
                )
            )

            converted = pd.to_numeric(
                text,
                errors="coerce",
            )

        after = int(
            converted.notna().sum()
        )

        loss = max(
            0,
            before - after,
        )

        if loss > 0:
            coercion_losses[
                feature
            ] = loss

        X[
            feature
        ] = converted

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    report = {
        "dataset":
            dataset_name,
        "rows":
            int(
                len(
                    X
                )
            ),
        "feature_count":
            len(
                features
            ),
        "numeric_coercions_to_nan":
            coercion_losses,
        "total_missing_cells":
            int(
                X.isna().sum().sum()
            ),
        "rows_with_any_missing_feature":
            int(
                X.isna().any(
                    axis=1
                ).sum()
            ),
        "rows_with_all_features_missing":
            int(
                X.isna().all(
                    axis=1
                ).sum()
            ),
        "all_missing_features": [
            feature
            for feature in features
            if X[
                feature
            ].isna().all()
        ],
    }

    return (
        X,
        report,
    )


# =============================================================================
# LOAD MODEL / CONTRACT
# =============================================================================

def load_model_manifest() -> Dict[
    str,
    Any,
]:
    if not MODEL_MANIFEST.exists():
        raise SystemExit(
            "Stage-03 model_manifest.json not found:\n"
            f"  {MODEL_MANIFEST.resolve()}"
        )

    with MODEL_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(
            handle
        )

    if manifest.get(
        "target"
    ) != TARGET:
        raise ValueError(
            "Target mismatch between Stage-07 configuration and "
            "Stage-03 model manifest."
        )

    model_type = str(
        manifest.get(
            "model_type",
            "",
        )
    ).strip().upper()

    if model_type != "CATBOOST":
        raise ValueError(
            "Stage 07 is CatBoost-only. "
            f"Loaded model_type={model_type!r}."
        )

    if manifest.get(
        "run_name"
    ) != MODEL_RUN:
        raise ValueError(
            "MODEL_RUN mismatch between Stage-07 configuration "
            "and Stage-03 model manifest."
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
        payload = json.load(
            handle
        )

    if isinstance(
        payload,
        list,
    ):
        features = list(
            payload
        )

    elif isinstance(
        payload,
        dict,
    ):
        features = list(
            payload.get(
                "features",
                [],
            )
        )

    else:
        raise ValueError(
            "Unsupported Stage-03 features.json format."
        )

    if not features:
        raise ValueError(
            "No model features found."
        )

    if len(
        features
    ) != len(
        set(
            features
        )
    ):
        raise ValueError(
            "Duplicate model features found."
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
        str(
            MODEL_PATH
        )
    )

    return model


def resolve_probability_threshold(
    manifest: Dict[
        str,
        Any,
    ],
) -> float:
    threshold = float(
        manifest.get(
            "default_probability_threshold",
            0.50,
        )
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"Invalid probability threshold: {threshold}"
        )

    return threshold


# =============================================================================
# CATBOOST OUTPUTS / FEATURE IMPORTANCE
# =============================================================================

def predict_model_outputs(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    threshold: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    probabilities = np.asarray(
        model.predict_proba(
            X
        ),
        dtype=float,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[
            1
        ] != 2
    ):
        raise ValueError(
            "Expected binary predict_proba output of shape (n,2); "
            f"got {probabilities.shape}."
        )

    probability_class1 = probabilities[
        :,
        1,
    ]

    predicted_class = (
        probability_class1
        >= threshold
    ).astype(
        int
    )

    return (
        probability_class1,
        predicted_class,
    )


def catboost_native_importance(
    model: CatBoostClassifier,
    features: List[str],
) -> pd.DataFrame:
    importance = np.asarray(
        model.get_feature_importance(),
        dtype=float,
    )

    if len(
        importance
    ) != len(
        features
    ):
        raise ValueError(
            "CatBoost feature-importance length does not match "
            "the saved model feature contract."
        )

    result = pd.DataFrame({
        "feature":
            features,
        "catboost_native_importance":
            importance,
    })

    total = float(
        result[
            "catboost_native_importance"
        ].sum()
    )

    if total > 0:
        result[
            "catboost_native_importance_normalized"
        ] = (
            result[
                "catboost_native_importance"
            ]
            / total
        )
    else:
        result[
            "catboost_native_importance_normalized"
        ] = 0.0

    result = result.sort_values(
        [
            "catboost_native_importance",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    result.insert(
        0,
        "catboost_native_rank",
        np.arange(
            1,
            len(
                result
            )
            + 1,
            dtype=int,
        ),
    )

    return result


# =============================================================================
# DISTRIBUTION STATISTICS
# =============================================================================

def finite_numeric_values(
    series: pd.Series,
) -> np.ndarray:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return numeric.dropna().to_numpy(
        dtype=float
    )


def one_class_distribution_statistics(
    values: pd.Series,
    total_rows_in_class: int,
) -> Dict[
    str,
    float | int,
]:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    finite = numeric.dropna().to_numpy(
        dtype=float
    )

    n_nonmissing = int(
        len(
            finite
        )
    )

    n_missing = int(
        total_rows_in_class
        - n_nonmissing
    )

    if n_nonmissing == 0:
        return {
            "n_total":
                int(
                    total_rows_in_class
                ),
            "n_nonmissing":
                0,
            "n_missing":
                n_missing,
            "missing_fraction":
                float(
                    n_missing
                    / max(
                        1,
                        total_rows_in_class,
                    )
                ),
            "mean":
                float(
                    "nan"
                ),
            "std":
                float(
                    "nan"
                ),
            "min":
                float(
                    "nan"
                ),
            "q05":
                float(
                    "nan"
                ),
            "q25":
                float(
                    "nan"
                ),
            "median":
                float(
                    "nan"
                ),
            "q75":
                float(
                    "nan"
                ),
            "q95":
                float(
                    "nan"
                ),
            "max":
                float(
                    "nan"
                ),
            "iqr":
                float(
                    "nan"
                ),
        }

    q05, q25, median, q75, q95 = (
        np.quantile(
            finite,
            [
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ],
        )
    )

    return {
        "n_total":
            int(
                total_rows_in_class
            ),
        "n_nonmissing":
            n_nonmissing,
        "n_missing":
            n_missing,
        "missing_fraction":
            float(
                n_missing
                / max(
                    1,
                    total_rows_in_class,
                )
            ),
        "mean":
            float(
                np.mean(
                    finite
                )
            ),
        "std":
            float(
                np.std(
                    finite,
                    ddof=1,
                )
            )
            if n_nonmissing > 1
            else 0.0,
        "min":
            float(
                np.min(
                    finite
                )
            ),
        "q05":
            float(
                q05
            ),
        "q25":
            float(
                q25
            ),
        "median":
            float(
                median
            ),
        "q75":
            float(
                q75
            ),
        "q95":
            float(
                q95
            ),
        "max":
            float(
                np.max(
                    finite
                )
            ),
        "iqr":
            float(
                q75
                - q25
            ),
    }


def standardized_mean_difference(
    class0_values: np.ndarray,
    class1_values: np.ndarray,
) -> float:
    x0 = np.asarray(
        class0_values,
        dtype=float,
    )

    x1 = np.asarray(
        class1_values,
        dtype=float,
    )

    if (
        len(
            x0
        ) < 2
        or len(
            x1
        ) < 2
    ):
        return float(
            "nan"
        )

    var0 = float(
        np.var(
            x0,
            ddof=1,
        )
    )

    var1 = float(
        np.var(
            x1,
            ddof=1,
        )
    )

    denominator_df = (
        len(
            x0
        )
        + len(
            x1
        )
        - 2
    )

    if denominator_df <= 0:
        return float(
            "nan"
        )

    pooled_variance = (
        (
            (
                len(
                    x0
                )
                - 1
            )
            * var0
        )
        + (
            (
                len(
                    x1
                )
                - 1
            )
            * var1
        )
    ) / denominator_df

    if (
        not np.isfinite(
            pooled_variance
        )
        or pooled_variance
        <= 0.0
    ):
        return float(
            "nan"
        )

    pooled_std = math.sqrt(
        pooled_variance
    )

    return float(
        (
            np.mean(
                x1
            )
            - np.mean(
                x0
            )
        )
        / pooled_std
    )


def feature_statistics_and_comparison(
    X: pd.DataFrame,
    predicted_class: np.ndarray,
    features: List[str],
    native_importance_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pred = np.asarray(
        predicted_class,
        dtype=int,
    )

    class0_mask = (
        pred == 0
    )

    class1_mask = (
        pred == 1
    )

    n_class0 = int(
        class0_mask.sum()
    )

    n_class1 = int(
        class1_mask.sum()
    )

    if (
        n_class0 == 0
        or n_class1 == 0
    ):
        raise ValueError(
            "Predicted-class distribution analysis requires both "
            "predicted classes 0 and 1.\n"
            f"Predicted counts: {class_counts(pred)}"
        )

    stats_rows: List[
        Dict[str, Any]
    ] = []

    comparison_rows: List[
        Dict[str, Any]
    ] = []

    native_map = (
        native_importance_df
        .set_index(
            "feature"
        )
    )

    for feature in features:
        class0_series = X.loc[
            class0_mask,
            feature,
        ]

        class1_series = X.loc[
            class1_mask,
            feature,
        ]

        stats0 = (
            one_class_distribution_statistics(
                class0_series,
                n_class0,
            )
        )

        stats1 = (
            one_class_distribution_statistics(
                class1_series,
                n_class1,
            )
        )

        stats_rows.append({
            "feature":
                feature,
            "predicted_class":
                0,
            **stats0,
        })

        stats_rows.append({
            "feature":
                feature,
            "predicted_class":
                1,
            **stats1,
        })

        x0 = finite_numeric_values(
            class0_series
        )

        x1 = finite_numeric_values(
            class1_series
        )

        mean_difference = (
            float(
                stats1[
                    "mean"
                ]
                - stats0[
                    "mean"
                ]
            )
            if np.isfinite(
                stats0[
                    "mean"
                ]
            )
            and np.isfinite(
                stats1[
                    "mean"
                ]
            )
            else float(
                "nan"
            )
        )

        median_difference = (
            float(
                stats1[
                    "median"
                ]
                - stats0[
                    "median"
                ]
            )
            if np.isfinite(
                stats0[
                    "median"
                ]
            )
            and np.isfinite(
                stats1[
                    "median"
                ]
            )
            else float(
                "nan"
            )
        )

        median_ratio = float(
            "nan"
        )

        if (
            np.isfinite(
                stats0[
                    "median"
                ]
            )
            and np.isfinite(
                stats1[
                    "median"
                ]
            )
            and float(
                stats0[
                    "median"
                ]
            )
            != 0.0
        ):
            median_ratio = float(
                stats1[
                    "median"
                ]
                / stats0[
                    "median"
                ]
            )

        smd = standardized_mean_difference(
            x0,
            x1,
        )

        native_rank = float(
            native_map.loc[
                feature,
                "catboost_native_rank",
            ]
        )

        native_importance = float(
            native_map.loc[
                feature,
                "catboost_native_importance",
            ]
        )

        native_relative = float(
            native_map.loc[
                feature,
                "catboost_native_importance_normalized",
            ]
        )

        comparison_rows.append({
            "feature":
                feature,
            "catboost_native_rank":
                int(
                    native_rank
                ),
            "catboost_native_importance":
                native_importance,
            "catboost_native_importance_normalized":
                native_relative,
            "predicted_class_0_n_total":
                n_class0,
            "predicted_class_1_n_total":
                n_class1,
            "predicted_class_0_n_nonmissing":
                int(
                    stats0[
                        "n_nonmissing"
                    ]
                ),
            "predicted_class_1_n_nonmissing":
                int(
                    stats1[
                        "n_nonmissing"
                    ]
                ),
            "class0_mean":
                stats0[
                    "mean"
                ],
            "class1_mean":
                stats1[
                    "mean"
                ],
            "mean_difference_class1_minus_class0":
                mean_difference,
            "class0_median":
                stats0[
                    "median"
                ],
            "class1_median":
                stats1[
                    "median"
                ],
            "median_difference_class1_minus_class0":
                median_difference,
            "median_ratio_class1_over_class0":
                median_ratio,
            "class0_iqr":
                stats0[
                    "iqr"
                ],
            "class1_iqr":
                stats1[
                    "iqr"
                ],
            "class0_missing_fraction":
                stats0[
                    "missing_fraction"
                ],
            "class1_missing_fraction":
                stats1[
                    "missing_fraction"
                ],
            "standardized_mean_difference_class1_minus_class0":
                smd,
            "absolute_standardized_mean_difference":
                float(
                    abs(
                        smd
                    )
                )
                if np.isfinite(
                    smd
                )
                else float(
                    "nan"
                ),
        })

    stats_df = pd.DataFrame(
        stats_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if RANK_FEATURES_BY_NATIVE_IMPORTANCE:
        comparison_df = comparison_df.sort_values(
            [
                "catboost_native_rank",
                "feature",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )
    else:
        comparison_df = comparison_df.sort_values(
            [
                "absolute_standardized_mean_difference",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )

    comparison_df = comparison_df.reset_index(
        drop=True
    )

    comparison_df.insert(
        0,
        "distribution_plot_rank",
        np.arange(
            1,
            len(
                comparison_df
            )
            + 1,
            dtype=int,
        ),
    )

    return (
        stats_df,
        comparison_df,
    )


# =============================================================================
# REPEATED BALANCED SAMPLING BY PREDICTED CLASS
# =============================================================================

def balanced_predicted_class_indices(
    predicted_class: np.ndarray,
    sampling_ratio: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    pred = np.asarray(
        predicted_class,
        dtype=int,
    )

    class0_indices = np.flatnonzero(
        pred == 0
    )

    class1_indices = np.flatnonzero(
        pred == 1
    )

    if (
        len(
            class0_indices
        ) == 0
        or len(
            class1_indices
        ) == 0
    ):
        raise ValueError(
            "Balanced predicted-class sampling requires both classes."
        )

    minority_available = min(
        len(
            class0_indices
        ),
        len(
            class1_indices
        ),
    )

    n_pick = int(
        round(
            minority_available
            * sampling_ratio
        )
    )

    n_pick = max(
        MIN_ROWS_PER_CLASS_PER_BALANCED_ITERATION,
        n_pick,
    )

    n_pick = min(
        n_pick,
        len(
            class0_indices
        ),
        len(
            class1_indices
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    chosen0 = rng.choice(
        class0_indices,
        size=n_pick,
        replace=False,
    )

    chosen1 = rng.choice(
        class1_indices,
        size=n_pick,
        replace=False,
    )

    return (
        np.asarray(
            chosen0,
            dtype=int,
        ),
        np.asarray(
            chosen1,
            dtype=int,
        ),
    )


def pooled_repeated_balanced_values(
    feature_values: pd.Series,
    predicted_class: np.ndarray,
    seed0: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    numeric_values = pd.to_numeric(
        feature_values,
        errors="coerce",
    ).replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).to_numpy(
        dtype=float
    )

    pooled0: List[
        np.ndarray
    ] = []

    pooled1: List[
        np.ndarray
    ] = []

    rows_per_class: List[
        int
    ] = []

    nonmissing_rows_per_class0: List[
        int
    ] = []

    nonmissing_rows_per_class1: List[
        int
    ] = []

    for iteration in range(
        N_BOX_ITERATIONS
    ):
        chosen0, chosen1 = (
            balanced_predicted_class_indices(
                predicted_class=
                    predicted_class,
                sampling_ratio=
                    BOX_SAMPLING_RATIO,
                seed=
                    seed0 + iteration,
            )
        )

        values0 = numeric_values[
            chosen0
        ]

        values1 = numeric_values[
            chosen1
        ]

        finite0 = values0[
            np.isfinite(
                values0
            )
        ]

        finite1 = values1[
            np.isfinite(
                values1
            )
        ]

        pooled0.append(
            finite0
        )

        pooled1.append(
            finite1
        )

        rows_per_class.append(
            int(
                len(
                    chosen0
                )
            )
        )

        nonmissing_rows_per_class0.append(
            int(
                len(
                    finite0
                )
            )
        )

        nonmissing_rows_per_class1.append(
            int(
                len(
                    finite1
                )
            )
        )

    concatenated0 = (
        np.concatenate(
            pooled0
        )
        if pooled0
        else np.asarray(
            [],
            dtype=float,
        )
    )

    concatenated1 = (
        np.concatenate(
            pooled1
        )
        if pooled1
        else np.asarray(
            [],
            dtype=float,
        )
    )

    diagnostics = {
        "iterations":
            N_BOX_ITERATIONS,
        "sampling_ratio":
            BOX_SAMPLING_RATIO,
        "rows_sampled_per_class_per_iteration":
            int(
                rows_per_class[
                    0
                ]
            )
            if rows_per_class
            else 0,
        "pooled_nonmissing_values_class0":
            int(
                len(
                    concatenated0
                )
            ),
        "pooled_nonmissing_values_class1":
            int(
                len(
                    concatenated1
                )
            ),
        "mean_nonmissing_values_per_iteration_class0":
            float(
                np.mean(
                    nonmissing_rows_per_class0
                )
            )
            if nonmissing_rows_per_class0
            else 0.0,
        "mean_nonmissing_values_per_iteration_class1":
            float(
                np.mean(
                    nonmissing_rows_per_class1
                )
            )
            if nonmissing_rows_per_class1
            else 0.0,
    }

    return (
        concatenated0,
        concatenated1,
        diagnostics,
    )


# =============================================================================
# BOXPLOTS
# =============================================================================

def add_optional_raw_points(
    ax: plt.Axes,
    class0_values: np.ndarray,
    class1_values: np.ndarray,
    seed: int,
) -> None:
    if not SHOW_RAW_POINTS:
        return

    rng = np.random.default_rng(
        seed
    )

    for x_position, values in [
        (
            1.0,
            class0_values,
        ),
        (
            2.0,
            class1_values,
        ),
    ]:
        values = np.asarray(
            values,
            dtype=float,
        )

        if len(
            values
        ) > RAW_POINTS_MAX_PER_CLASS:
            selected = rng.choice(
                np.arange(
                    len(
                        values
                    )
                ),
                size=
                    RAW_POINTS_MAX_PER_CLASS,
                replace=False,
            )

            plot_values = values[
                selected
            ]

        else:
            plot_values = values

        jitter = rng.uniform(
            -0.08,
            0.08,
            size=len(
                plot_values
            ),
        )

        ax.scatter(
            np.full(
                len(
                    plot_values
                ),
                x_position,
            )
            + jitter,
            plot_values,
            s=RAW_POINT_SIZE,
            alpha=RAW_POINT_ALPHA,
        )


def create_boxplot(
    class0_values: np.ndarray,
    class1_values: np.ndarray,
    feature: str,
    title_prefix: str,
    subtitle: str,
    output_base: Path,
    seed: int,
) -> bool:
    class0_values = np.asarray(
        class0_values,
        dtype=float,
    )

    class1_values = np.asarray(
        class1_values,
        dtype=float,
    )

    class0_values = class0_values[
        np.isfinite(
            class0_values
        )
    ]

    class1_values = class1_values[
        np.isfinite(
            class1_values
        )
    ]

    if (
        len(
            class0_values
        ) == 0
        or len(
            class1_values
        ) == 0
    ):
        return False

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            6.5,
        )
    )

    ax.boxplot(
        [
            class0_values,
            class1_values,
        ],
        labels=[
            "Predicted class 0",
            "Predicted class 1",
        ],
        showfliers=
            SHOW_BOXPLOT_OUTLIERS,
    )

    add_optional_raw_points(
        ax=
            ax,
        class0_values=
            class0_values,
        class1_values=
            class1_values,
        seed=
            seed,
    )

    ax.set_ylabel(
        feature
    )

    ax.set_xlabel(
        "CatBoost predicted class"
    )

    ax.set_title(
        f"{title_prefix}\n"
        f"{feature}\n"
        f"{subtitle}"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    save_figure(
        fig,
        output_base,
    )

    return True


# =============================================================================
# ONE DATASET
# =============================================================================

def analyze_one_dataset(
    dataset_key: str,
    source_df: pd.DataFrame,
    model: CatBoostClassifier,
    features: List[str],
    native_importance_df: pd.DataFrame,
    threshold: float,
    seed: int,
) -> Dict[str, Any]:
    display_name = (
        DATASET_DISPLAY_NAMES[
            dataset_key
        ]
    )

    interpretation = (
        DATASET_INTERPRETATION[
            dataset_key
        ]
    )

    dataset_dir = (
        OUTPUT_DIR
        / dataset_key
    )

    all_boxplot_dir = (
        dataset_dir
        / "boxplots_all_rows"
    )

    balanced_boxplot_dir = (
        dataset_dir
        / "boxplots_balanced_repeated"
    )

    ensure_dir(
        dataset_dir
    )

    if SAVE_ALL_ROW_BOXPLOTS:
        ensure_dir(
            all_boxplot_dir
        )

    if SAVE_REPEATED_BALANCED_BOXPLOTS:
        ensure_dir(
            balanced_boxplot_dir
        )

    print()
    print("=" * 88)
    print(
        f"DIGIGARD STAGE 07: {display_name}"
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
    )

    X, preprocessing_report = (
        preprocess_numeric_features(
            df=
                source_df,
            features=
                features,
            dataset_name=
                dataset_key,
        )
    )

    (
        probability_class1,
        predicted_class,
    ) = predict_model_outputs(
        model=
            model,
        X=
            X,
        threshold=
            threshold,
    )

    predicted_counts = (
        class_counts(
            predicted_class
        )
    )

    print(
        f"[PREDICT] predicted class counts="
        f"{predicted_counts}"
    )

    if (
        int(
            predicted_counts.get(
                "0",
                0,
            )
        ) == 0
        or int(
            predicted_counts.get(
                "1",
                0,
            )
        ) == 0
    ):
        raise ValueError(
            f"{dataset_key}: model predicted only one class; "
            "class-wise feature distribution boxplots require both classes."
        )

    # ---------------------------------------------------------------------
    # Save model predictions alongside metadata and target.
    # ---------------------------------------------------------------------
    predictions_df = (
        source_df.copy()
        .reset_index(
            drop=True
        )
    )

    predictions_df[
        PRED_CLASS_COL
    ] = predicted_class

    predictions_df[
        PRED_PROBABILITY_COL
    ] = probability_class1

    predictions_path = (
        dataset_dir
        / "predictions.csv"
    )

    if SAVE_PREDICTIONS:
        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

    counts_df = pd.DataFrame({
        "predicted_class": [
            0,
            1,
        ],
        "count": [
            int(
                np.sum(
                    predicted_class
                    == 0
                )
            ),
            int(
                np.sum(
                    predicted_class
                    == 1
                )
            ),
        ],
    })

    counts_df[
        "fraction"
    ] = (
        counts_df[
            "count"
        ]
        / len(
            predicted_class
        )
    )

    predicted_class_counts_path = (
        dataset_dir
        / "predicted_class_counts.csv"
    )

    counts_df.to_csv(
        predicted_class_counts_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Distribution statistics.
    # ---------------------------------------------------------------------
    (
        statistics_df,
        comparison_df,
    ) = feature_statistics_and_comparison(
        X=
            X,
        predicted_class=
            predicted_class,
        features=
            features,
        native_importance_df=
            native_importance_df,
    )

    statistics_path = (
        dataset_dir
        / "feature_distribution_statistics.csv"
    )

    comparison_path = (
        dataset_dir
        / "feature_class_comparison.csv"
    )

    statistics_df.to_csv(
        statistics_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Boxplots for EVERY model feature.
    # ---------------------------------------------------------------------
    class0_mask = (
        predicted_class
        == 0
    )

    class1_mask = (
        predicted_class
        == 1
    )

    plot_manifest_rows: List[
        Dict[str, Any]
    ] = []

    for _, feature_row in comparison_df.iterrows():
        rank = int(
            feature_row[
                "distribution_plot_rank"
            ]
        )

        feature = str(
            feature_row[
                "feature"
            ]
        )

        class0_values = (
            finite_numeric_values(
                X.loc[
                    class0_mask,
                    feature,
                ]
            )
        )

        class1_values = (
            finite_numeric_values(
                X.loc[
                    class1_mask,
                    feature,
                ]
            )
        )

        file_stem = (
            f"{rank:03d}_"
            f"{safe_file_name(feature)}"
        )

        all_plot_base = (
            all_boxplot_dir
            / file_stem
        )

        all_plot_saved = False

        if SAVE_ALL_ROW_BOXPLOTS:
            all_plot_saved = create_boxplot(
                class0_values=
                    class0_values,
                class1_values=
                    class1_values,
                feature=
                    feature,
                title_prefix=
                    display_name,
                subtitle=(
                    "All available rows grouped by model-predicted class"
                ),
                output_base=
                    all_plot_base,
                seed=
                    seed
                    + rank,
            )

        balanced_plot_base = (
            balanced_boxplot_dir
            / file_stem
        )

        balanced_plot_saved = False
        balanced_diagnostics: Dict[
            str,
            Any,
        ] = {}

        if SAVE_REPEATED_BALANCED_BOXPLOTS:
            (
                balanced_class0,
                balanced_class1,
                balanced_diagnostics,
            ) = pooled_repeated_balanced_values(
                feature_values=
                    X[
                        feature
                    ],
                predicted_class=
                    predicted_class,
                seed0=
                    seed
                    + 100_000
                    + rank
                    * 1_000,
            )

            balanced_plot_saved = create_boxplot(
                class0_values=
                    balanced_class0,
                class1_values=
                    balanced_class1,
                feature=
                    feature,
                title_prefix=
                    display_name,
                subtitle=(
                    f"Repeated balanced predicted-class sampling: "
                    f"{N_BOX_ITERATIONS} iterations, "
                    f"ratio={BOX_SAMPLING_RATIO:.2f}"
                ),
                output_base=
                    balanced_plot_base,
                seed=
                    seed
                    + 200_000
                    + rank,
            )

        plot_manifest_rows.append({
            "distribution_plot_rank":
                rank,
            "feature":
                feature,
            "catboost_native_rank":
                int(
                    feature_row[
                        "catboost_native_rank"
                    ]
                ),
            "class0_nonmissing_rows":
                int(
                    len(
                        class0_values
                    )
                ),
            "class1_nonmissing_rows":
                int(
                    len(
                        class1_values
                    )
                ),
            "all_rows_boxplot_saved":
                bool(
                    all_plot_saved
                ),
            "all_rows_boxplot_base":
                str(
                    all_plot_base
                )
                if all_plot_saved
                else None,
            "balanced_boxplot_saved":
                bool(
                    balanced_plot_saved
                ),
            "balanced_boxplot_base":
                str(
                    balanced_plot_base
                )
                if balanced_plot_saved
                else None,
            "balanced_rows_per_class_per_iteration":
                balanced_diagnostics.get(
                    "rows_sampled_per_class_per_iteration"
                ),
            "balanced_pooled_nonmissing_class0":
                balanced_diagnostics.get(
                    "pooled_nonmissing_values_class0"
                ),
            "balanced_pooled_nonmissing_class1":
                balanced_diagnostics.get(
                    "pooled_nonmissing_values_class1"
                ),
        })

        print(
            f"[BOXPLOT] {rank:03d}/{len(comparison_df):03d} "
            f"{feature}"
        )

    plot_manifest_df = pd.DataFrame(
        plot_manifest_rows
    )

    plot_manifest_path = (
        dataset_dir
        / "feature_plot_manifest.csv"
    )

    plot_manifest_df.to_csv(
        plot_manifest_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Summary / report.
    # ---------------------------------------------------------------------
    top_differences = (
        comparison_df
        .sort_values(
            [
                "absolute_standardized_mean_difference",
                "feature",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )
        .head(
            min(
                25,
                len(
                    comparison_df
                ),
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
            7,
        "stage_name":
            "catboost_predicted_class_feature_distributions",
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
            len(
                features
            ),
        "threshold":
            threshold,
        "predicted_class_counts":
            predicted_counts,
        "predicted_positive_fraction":
            float(
                np.mean(
                    predicted_class
                )
            ),
        "preprocessing":
            preprocessing_report,
        "primary_distribution_analysis": {
            "grouping_variable":
                "CatBoost predicted class",
            "classes": [
                0,
                1,
            ],
            "uses_all_rows":
                True,
            "show_boxplot_outliers":
                SHOW_BOXPLOT_OUTLIERS,
        },
        "balanced_distribution_analysis": {
            "enabled":
                SAVE_REPEATED_BALANCED_BOXPLOTS,
            "iterations":
                N_BOX_ITERATIONS,
            "sampling_ratio":
                BOX_SAMPLING_RATIO,
            "sampling_without_replacement_within_iteration":
                True,
            "same_number_from_each_predicted_class":
                True,
            "important_note": (
                "Rows can appear in multiple repetitions. The pooled repeated "
                "balanced boxplot is a robustness visualization, not an "
                "independent-sample inferential dataset."
            ),
        },
        "top_features_by_absolute_standardized_mean_difference":
            top_differences,
        "artifacts": {
            "predictions_csv":
                str(
                    predictions_path
                )
                if SAVE_PREDICTIONS
                else None,
            "predicted_class_counts_csv":
                str(
                    predicted_class_counts_path
                ),
            "feature_distribution_statistics_csv":
                str(
                    statistics_path
                ),
            "feature_class_comparison_csv":
                str(
                    comparison_path
                ),
            "feature_plot_manifest_csv":
                str(
                    plot_manifest_path
                ),
            "all_rows_boxplot_dir":
                str(
                    all_boxplot_dir
                )
                if SAVE_ALL_ROW_BOXPLOTS
                else None,
            "balanced_boxplot_dir":
                str(
                    balanced_boxplot_dir
                )
                if SAVE_REPEATED_BALANCED_BOXPLOTS
                else None,
        },
    }

    write_json(
        dataset_dir
        / "distribution_analysis_report.json",
        dataset_report,
    )

    # ---------------------------------------------------------------------
    # Human-readable report.
    # ---------------------------------------------------------------------
    report_lines: List[
        str
    ] = []

    report_lines.append(
        f"DIGIGARD STAGE 07 - {display_name}"
    )

    report_lines.append(
        "=" * 80
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

    report_lines.append(
        f"PROBABILITY THRESHOLD: {threshold}"
    )

    report_lines.append(
        f"PREDICTED CLASS COUNTS: {predicted_counts}"
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
        "PRIMARY BOXPLOTS"
    )

    report_lines.append(
        "Every model feature is plotted using all finite feature values, "
        "grouped by the class predicted by the saved CatBoost model."
    )

    report_lines.append(
        "These are descriptive feature distributions conditioned on model "
        "predictions; they do not imply causality."
    )

    report_lines.append("")

    report_lines.append(
        "REPEATED BALANCED BOXPLOTS"
    )

    report_lines.append(
        f"Iterations: {N_BOX_ITERATIONS}"
    )

    report_lines.append(
        f"Sampling ratio: {BOX_SAMPLING_RATIO}"
    )

    report_lines.append(
        "Each iteration samples equal numbers from predicted class 0 and "
        "predicted class 1 without replacement within the iteration."
    )

    report_lines.append(
        "Rows may recur across iterations; the pooled repeated boxplot is a "
        "robustness visualization."
    )

    report_lines.append("")

    report_lines.append(
        "LARGEST CLASS-ASSOCIATED DISTRIBUTION DIFFERENCES"
    )

    for rank, item in enumerate(
        top_differences[
            :20
        ],
        start=1,
    ):
        report_lines.append(
            f"  {rank:>2}. "
            f"{item['feature']}: "
            f"SMD={item['standardized_mean_difference_class1_minus_class0']:.6g}, "
            f"|SMD|={item['absolute_standardized_mean_difference']:.6g}, "
            f"median0={item['class0_median']:.8g}, "
            f"median1={item['class1_median']:.8g}"
        )

    (
        dataset_dir
        / "DATASET_REPORT.txt"
    ).write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    return dataset_report


# =============================================================================
# CROSS-DATASET COMPARISON
# =============================================================================

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

    merged: Optional[
        pd.DataFrame
    ] = None

    for dataset_key in DATASET_ORDER:
        comparison_csv = (
            results[
                dataset_key
            ][
                "artifacts"
            ][
                "feature_class_comparison_csv"
            ]
        )

        df = pd.read_csv(
            comparison_csv
        )

        keep_columns = [
            "feature",
            "catboost_native_rank",
            "catboost_native_importance_normalized",
            "class0_mean",
            "class1_mean",
            "mean_difference_class1_minus_class0",
            "class0_median",
            "class1_median",
            "median_difference_class1_minus_class0",
            "standardized_mean_difference_class1_minus_class0",
            "absolute_standardized_mean_difference",
        ]

        df = df[
            [
                column
                for column in keep_columns
                if column in df.columns
            ]
        ].copy()

        rename = {
            column:
                f"{dataset_key}__{column}"
            for column in df.columns
            if column != "feature"
        }

        df = df.rename(
            columns=rename
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

    # Stability of standardized mean differences across full/train/test.
    smd_columns = [
        f"{dataset_key}__standardized_mean_difference_class1_minus_class0"
        for dataset_key in DATASET_ORDER
        if (
            f"{dataset_key}__standardized_mean_difference_class1_minus_class0"
            in merged.columns
        )
    ]

    if smd_columns:
        merged[
            "smd_mean_across_datasets"
        ] = merged[
            smd_columns
        ].mean(
            axis=1
        )

        merged[
            "smd_std_across_datasets"
        ] = merged[
            smd_columns
        ].std(
            axis=1,
            ddof=0,
        )

        merged[
            "absolute_smd_mean_across_datasets"
        ] = (
            merged[
                smd_columns
            ]
            .abs()
            .mean(
                axis=1
            )
        )

    cross_path = (
        comparison_dir
        / "feature_class_comparison_full_train_test.csv"
    )

    merged.to_csv(
        cross_path,
        index=False,
    )

    report = {
        "comparison_csv":
            str(
                cross_path
            ),
        "datasets":
            DATASET_ORDER,
        "meaning": (
            "Compares predicted-class-conditioned feature-distribution "
            "differences across full, training, and held-out testing data."
        ),
    }

    write_json(
        comparison_dir
        / "distribution_cross_dataset_report.json",
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
            "Stage-07 output directory already exists and "
            "OVERWRITE_EXISTING_ANALYSIS=False:\n"
            f"  {OUTPUT_DIR.resolve()}"
        )

    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 88)
    print(
        "DIGIGARD STAGE 07 - CATBOOST PREDICTED-CLASS FEATURE DISTRIBUTIONS"
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

    threshold = (
        resolve_probability_threshold(
            manifest
        )
    )

    native_importance_df = (
        catboost_native_importance(
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
            RANDOM_STATE
            + 10_000,
        "training_dataset":
            RANDOM_STATE
            + 20_000,
        "testing_dataset":
            RANDOM_STATE
            + 30_000,
    }

    results: Dict[
        str,
        Dict[str, Any],
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
            threshold=
                threshold,
            seed=
                seeds[
                    dataset_key
                ],
        )

    cross_report = (
        build_cross_dataset_comparison(
            results
        )
    )

    elapsed_seconds = float(
        time.time()
        - start_time
    )

    # ---------------------------------------------------------------------
    # Compact root summary.
    # ---------------------------------------------------------------------
    summary_rows: List[
        Dict[str, Any],
    ] = []

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        comparison_df = pd.read_csv(
            result[
                "artifacts"
            ][
                "feature_class_comparison_csv"
            ]
        )

        largest = (
            comparison_df
            .sort_values(
                "absolute_standardized_mean_difference",
                ascending=False,
                kind="stable",
            )
            .iloc[
                0
            ]
        )

        summary_rows.append({
            "dataset":
                dataset_key,
            "dataset_display_name":
                DATASET_DISPLAY_NAMES[
                    dataset_key
                ],
            "rows":
                result[
                    "rows"
                ],
            "predicted_class_0_count":
                int(
                    result[
                        "predicted_class_counts"
                    ].get(
                        "0",
                        0,
                    )
                ),
            "predicted_class_1_count":
                int(
                    result[
                        "predicted_class_counts"
                    ].get(
                        "1",
                        0,
                    )
                ),
            "predicted_positive_fraction":
                result[
                    "predicted_positive_fraction"
                ],
            "largest_distribution_difference_feature":
                largest[
                    "feature"
                ],
            "largest_absolute_standardized_mean_difference":
                largest[
                    "absolute_standardized_mean_difference"
                ],
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "distribution_analysis_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Master report.
    # ---------------------------------------------------------------------
    master_report = {
        "pipeline":
            "DIGIGARD",
        "stage":
            7,
        "stage_name":
            "catboost_predicted_class_feature_distribution_analysis",
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
        "probability_threshold":
            threshold,
        "feature_count":
            len(
                features
            ),
        "features":
            features,
        "grouping_variable":
            "CatBoost predicted class",
        "primary_analysis":
            "all-row boxplot and class-wise descriptive statistics for every feature",
        "repeated_balanced_analysis": {
            "enabled":
                SAVE_REPEATED_BALANCED_BOXPLOTS,
            "iterations":
                N_BOX_ITERATIONS,
            "sampling_ratio":
                BOX_SAMPLING_RATIO,
            "same_number_from_each_predicted_class":
                True,
        },
        "datasets":
            results,
        "cross_dataset_comparison":
            cross_report,
        "artifacts": {
            "native_feature_importance_csv":
                str(
                    native_importance_path.resolve()
                ),
            "summary_csv":
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
        },
        "elapsed_seconds":
            elapsed_seconds,
    }

    master_report_path = (
        OUTPUT_DIR
        / "distribution_analysis_report.json"
    )

    write_json(
        master_report_path,
        master_report,
    )

    # ---------------------------------------------------------------------
    # Human-readable root report.
    # ---------------------------------------------------------------------
    report_lines: List[
        str
    ] = []

    report_lines.append(
        "DIGIGARD STAGE 07 - CATBOOST PREDICTED-CLASS FEATURE DISTRIBUTIONS"
    )

    report_lines.append(
        "=" * 84
    )

    report_lines.append(
        f"TARGET: {TARGET}"
    )

    report_lines.append(
        f"MODEL_RUN: {MODEL_RUN}"
    )

    report_lines.append(
        f"MODEL: {MODEL_PATH.resolve()}"
    )

    report_lines.append(
        f"N_FEATURES: {len(features)}"
    )

    report_lines.append(
        f"PROBABILITY_THRESHOLD: {threshold}"
    )

    report_lines.append("")

    report_lines.append(
        "ANALYSIS DEFINITION"
    )

    report_lines.append(
        "For every model feature, feature values are grouped by the class "
        "predicted by the saved CatBoost model."
    )

    report_lines.append(
        "Primary plots use all available rows. A secondary repeated-balanced "
        "plot reproduces the earlier predicted-class sampling approach."
    )

    report_lines.append(
        "These distributions are descriptive associations with model-predicted "
        "class and are not causal effects."
    )

    report_lines.append("")

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        report_lines.append(
            "=" * 84
        )

        report_lines.append(
            DATASET_DISPLAY_NAMES[
                dataset_key
            ]
        )

        report_lines.append(
            "=" * 84
        )

        report_lines.append(
            DATASET_INTERPRETATION[
                dataset_key
            ]
        )

        report_lines.append(
            f"Rows: {result['rows']}"
        )

        report_lines.append(
            f"Predicted class counts: "
            f"{result['predicted_class_counts']}"
        )

        comparison_df = pd.read_csv(
            result[
                "artifacts"
            ][
                "feature_class_comparison_csv"
            ]
        )

        top = (
            comparison_df
            .sort_values(
                "absolute_standardized_mean_difference",
                ascending=False,
                kind="stable",
            )
            .head(
                min(
                    20,
                    len(
                        comparison_df
                    ),
                )
            )
        )

        report_lines.append("")

        report_lines.append(
            "Largest predicted-class feature distribution differences:"
        )

        for rank, (
            _,
            row,
        ) in enumerate(
            top.iterrows(),
            start=1,
        ):
            report_lines.append(
                f"  {rank:>2}. "
                f"{row['feature']}: "
                f"SMD={row['standardized_mean_difference_class1_minus_class0']:.6g}, "
                f"|SMD|={row['absolute_standardized_mean_difference']:.6g}, "
                f"median0={row['class0_median']:.8g}, "
                f"median1={row['class1_median']:.8g}"
            )

        report_lines.append("")

    human_report_path = (
        OUTPUT_DIR
        / "DISTRIBUTION_ANALYSIS_REPORT.txt"
    )

    human_report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Config snapshot.
    # ---------------------------------------------------------------------
    config = {
        "pipeline":
            "DIGIGARD",
        "stage":
            7,
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
        "save_all_row_boxplots":
            SAVE_ALL_ROW_BOXPLOTS,
        "save_repeated_balanced_boxplots":
            SAVE_REPEATED_BALANCED_BOXPLOTS,
        "n_box_iterations":
            N_BOX_ITERATIONS,
        "box_sampling_ratio":
            BOX_SAMPLING_RATIO,
        "show_boxplot_outliers":
            SHOW_BOXPLOT_OUTLIERS,
        "show_raw_points":
            SHOW_RAW_POINTS,
        "rank_features_by_native_importance":
            RANK_FEATURES_BY_NATIVE_IMPORTANCE,
        "save_png":
            SAVE_PNG,
        "save_eps":
            SAVE_EPS,
    }

    write_json(
        OUTPUT_DIR
        / "distribution_analysis_config.json",
        config,
    )

    print()
    print("=" * 88)
    print(
        "DIGIGARD STAGE 07 COMPLETE"
    )
    print("=" * 88)

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        print(
            f"{DATASET_DISPLAY_NAMES[dataset_key]}:"
        )

        print(
            f"  predicted counts="
            f"{result['predicted_class_counts']}"
        )

        print(
            f"  statistics="
            f"{result['artifacts']['feature_distribution_statistics_csv']}"
        )

        print(
            f"  all-row boxplots="
            f"{result['artifacts']['all_rows_boxplot_dir']}"
        )

        print(
            f"  balanced boxplots="
            f"{result['artifacts']['balanced_boxplot_dir']}"
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
