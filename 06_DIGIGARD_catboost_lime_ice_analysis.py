#!/usr/bin/env python3
"""
06_DIGIGARD_catboost_lime_ice_analysis.py

DIGIGARD Stage 06: CatBoost-only LIME + ICE explainability analysis.

This stage runs AFTER:
    01_DIGIGARD_merge_and_clean_project_histories.py
    02_DIGIGARD_create_train_test_splits.py
    03_DIGIGARD_train_classifier.py

Recommended preceding analyses:
    04_DIGIGARD_evaluate_catboost_classifier.py
    05_DIGIGARD_catboost_shap_analysis.py

This script performs NO model training and NO hyperparameter optimization.

It loads ONE saved Stage-03 CatBoostClassifier for ONE target and analyzes:

    1) full_dataset
       = Stage-02 train.csv + Stage-02 test.csv

    2) training_dataset
       = Stage-02 train.csv

    3) testing_dataset
       = Stage-02 test.csv

===============================================================================
LIME
===============================================================================

LIME is used for LOCAL explanations of selected representative rows.

For each explained row this script saves:
    - true target
    - CatBoost predicted class
    - CatBoost P(class=1)
    - selected LIME feature weights
    - feature values
    - absolute LIME weights
    - LIME local surrogate fidelity score
    - LIME local surrogate prediction
    - LIME intercept
    - compact JSON explanation
    - CSV
    - TXT
    - HTML
    - local bar plot

Representative rows include, when available:
    - highest P(class=1)
    - lowest P(class=1)
    - predictions nearest the decision threshold
    - strongest false positives
    - strongest false negatives
    - random true-class-0 rows
    - random true-class-1 rows

A descriptive aggregate of the selected LIME explanations is also saved:
    - explanation frequency
    - mean signed LIME weight
    - mean absolute LIME weight
    - median absolute LIME weight
    - positive / negative weight fractions

IMPORTANT LIME NOTE:
The DIGIGARD model itself accepts numeric NaNs natively. LIME's tabular
perturbation space is constructed from a finite numeric matrix, so this script
creates a LIME-only median-filled representation using medians learned ONLY
from the Stage-02 training split. If a row contains missing values, its LIME
explanation therefore describes the CatBoost model around that median-filled
representation. The original model and all ICE analyses remain unchanged.

===============================================================================
ICE
===============================================================================

ICE = Individual Conditional Expectation.

For each selected important feature and each selected observation:
    - keep all other feature values fixed
    - replace the selected feature with values over a grid
    - call the saved CatBoost model
    - record P(class=1)

This yields one response curve per observation.

For each analyzed feature this script saves:
    - complete ICE long-form CSV
    - PDP summary (mean ICE curve)
    - median response
    - standard deviation
    - q10 / q25 / q75 / q90 response bands
    - centered ICE (cICE)
    - raw ICE + PDP plot
    - centered ICE plot
    - PDP + distribution-band plot
    - response/heterogeneity statistics

ICE feature selection:
    1) if Stage-05 global SHAP ranking exists for this dataset, use its ranking;
    2) otherwise fall back to CatBoost native feature importance.

The Stage-05 SHAP stage is therefore useful but NOT required.

===============================================================================
OUTPUT STRUCTURE
===============================================================================

DIGIGARD_06_catboost_lime_ice/
    <TARGET>/
        <MODEL_RUN>/
            lime_ice_analysis_config.json
            lime_ice_analysis_report.json
            lime_ice_analysis_summary.csv
            LIME_ICE_ANALYSIS_REPORT.txt
            catboost_native_feature_importance.csv

            full_dataset/
                lime/
                    selected_rows.csv
                    lime_explanations_long.csv
                    lime_feature_aggregate.csv
                    lime_report.json
                    rows/
                        row_XXXXXXXX/
                            explanation.csv
                            explanation.json
                            explanation.txt
                            explanation.html
                            explanation.png/.eps
                ice/
                    ice_feature_summary.csv
                    ice_report.json
                    <rank>_<feature>/
                        ice_curves_long.csv
                        pdp_summary.csv
                        ice_and_pdp.png/.eps
                        centered_ice.png/.eps
                        pdp_distribution.png/.eps
                DATASET_REPORT.txt

            training_dataset/
                same structure

            testing_dataset/
                same structure

            cross_dataset_comparison/
                lime_feature_aggregate_full_train_test.csv
                ice_feature_summary_full_train_test.csv
                explanation_feature_source_report.json

Dependencies
------------
Required:
    numpy
    pandas
    matplotlib
    catboost
    lime

Install LIME, if needed:
    pip install lime
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
# CONFIG: ONE TARGET + ONE CATBOOST MODEL PER RUN
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
STAGE05_ROOT = Path("DIGIGARD_05_catboost_shap")
STAGE06_ROOT = Path("DIGIGARD_06_catboost_lime_ice")

STAGE02_TARGET_ROOT = STAGE02_ROOT / TARGET
SPLIT_DIR = STAGE02_TARGET_ROOT / f"splits_{RANDOM_STATE}"

TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

MODEL_DIR = STAGE03_ROOT / TARGET / MODEL_RUN
MODEL_MANIFEST = MODEL_DIR / "model_manifest.json"
MODEL_FEATURES_JSON = MODEL_DIR / "features.json"
MODEL_PATH = MODEL_DIR / "best_model.cbm"

STAGE05_MODEL_DIR = STAGE05_ROOT / TARGET / MODEL_RUN

OUTPUT_DIR = STAGE06_ROOT / TARGET / MODEL_RUN


# =============================================================================
# LIME SETTINGS
# =============================================================================

ENABLE_LIME = True

# LIME local explanation size.
LIME_NUM_FEATURES = 15

# Number of perturbed neighborhood samples per explanation.
LIME_NUM_SAMPLES = 5000

# False:
#     continuous feature names remain direct model feature names.
#
# True:
#     LIME explains discretized intervals such as "feature <= value".
LIME_DISCRETIZE_CONTINUOUS = False

LIME_KERNEL_WIDTH: Optional[float] = None
LIME_FEATURE_SELECTION = "auto"
LIME_DISTANCE_METRIC = "euclidean"

# Number of representative rows selected for each category.
LIME_ROWS_PER_CATEGORY = 5

# Random examples from each true class.
LIME_RANDOM_ROWS_PER_CLASS = 5

# Optional brute-force mode.
# False is recommended for large datasets.
LIME_ANALYZE_ALL_ROWS = False

# Optional hard cap if LIME_ANALYZE_ALL_ROWS=True.
LIME_ALL_ROWS_MAX: Optional[int] = None

SAVE_LIME_HTML = True
SAVE_LIME_PLOTS = True


# =============================================================================
# ICE SETTINGS
# =============================================================================

ENABLE_ICE = True

# Analyze the strongest features according to Stage-05 SHAP if available,
# otherwise CatBoost native importance.
ICE_TOP_N_FEATURES = 12

# Number of individual observations represented by ICE curves.
ICE_SAMPLE_ROWS = 200

# Number of feature-grid values for continuous features.
ICE_GRID_POINTS = 25

# Use the central empirical range to reduce domination by extreme outliers.
ICE_GRID_LOWER_QUANTILE = 0.05
ICE_GRID_UPPER_QUANTILE = 0.95

# If a feature has this many or fewer unique finite values, use the actual
# observed values rather than a quantile grid.
ICE_USE_UNIQUE_VALUES_IF_AT_MOST = 20

# Save raw and centered ICE.
SAVE_ICE_LONG_CSV = True
SAVE_ICE_PLOTS = True


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
        "Descriptive explanation analysis on the union of Stage-02 training "
        "and testing observations. It contains training rows and is therefore "
        "not an independent generalization set."
    ),
    "training_dataset": (
        "In-sample explanation analysis on Stage-02 training observations."
    ),
    "testing_dataset": (
        "Held-out explanation analysis on Stage-02 test observations. "
        "This is the most important split for assessing explanation behavior "
        "on unseen held-out observations."
    ),
}


# =============================================================================
# VALIDATE CONFIG
# =============================================================================

if LIME_NUM_FEATURES <= 0:
    raise ValueError("LIME_NUM_FEATURES must be >= 1.")

if LIME_NUM_SAMPLES <= 0:
    raise ValueError("LIME_NUM_SAMPLES must be >= 1.")

if LIME_ROWS_PER_CATEGORY <= 0:
    raise ValueError("LIME_ROWS_PER_CATEGORY must be >= 1.")

if LIME_RANDOM_ROWS_PER_CLASS < 0:
    raise ValueError("LIME_RANDOM_ROWS_PER_CLASS must be >= 0.")

if ICE_TOP_N_FEATURES <= 0:
    raise ValueError("ICE_TOP_N_FEATURES must be >= 1.")

if ICE_SAMPLE_ROWS <= 0:
    raise ValueError("ICE_SAMPLE_ROWS must be >= 1.")

if ICE_GRID_POINTS <= 1:
    raise ValueError("ICE_GRID_POINTS must be >= 2.")

if not 0.0 <= ICE_GRID_LOWER_QUANTILE < ICE_GRID_UPPER_QUANTILE <= 1.0:
    raise ValueError(
        "ICE grid quantiles must satisfy "
        "0 <= lower < upper <= 1."
    )


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


def safe_package_version(
    package_name: str,
) -> Optional[str]:
    try:
        return importlib_metadata.version(
            package_name
        )
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, np.ndarray):
        return [
            json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(value)


def safe_file_name(text: str) -> str:
    return "".join(
        ch
        if ch.isalnum()
        or ch in {
            "_",
            "-",
            ".",
        }
        else "_"
        for ch in str(text)
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

    plt.close(fig)


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
        float(value)
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
        int
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
                .astype("string")
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

        converted = converted.astype(float)

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
# LOAD MODEL + CONTRACT
# =============================================================================

def load_model_manifest() -> Dict[
    str,
    Any,
]:
    if not MODEL_MANIFEST.exists():
        raise SystemExit(
            "Stage-03 model manifest not found:\n"
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
            "Target mismatch between Stage-06 configuration and "
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
            "Stage 06 is CatBoost-only. "
            f"Loaded model_type={model_type!r}."
        )

    if manifest.get(
        "run_name"
    ) != MODEL_RUN:
        raise ValueError(
            "MODEL_RUN mismatch between Stage-06 configuration "
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
            "Duplicate feature names in Stage-03 contract."
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
# CATBOOST OUTPUTS
# =============================================================================

def predict_probability_matrix(
    model: CatBoostClassifier,
    X: pd.DataFrame,
) -> np.ndarray:
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
            "Expected binary predict_proba output with shape (n,2); "
            f"got {probabilities.shape}."
        )

    return probabilities


def model_outputs(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    threshold: float,
) -> Dict[
    str,
    np.ndarray,
]:
    probabilities = (
        predict_probability_matrix(
            model,
            X,
        )
    )

    p1 = probabilities[
        :,
        1,
    ]

    return {
        "probability_class0":
            probabilities[
                :,
                0,
            ],
        "probability_class1":
            p1,
        "predicted_class":
            (
                p1
                >= threshold
            ).astype(
                int
            ),
    }


# =============================================================================
# CATBOOST NATIVE IMPORTANCE + OPTIONAL STAGE-05 RANKING
# =============================================================================

def catboost_native_importance(
    model: CatBoostClassifier,
    features: List[str],
) -> pd.DataFrame:
    values = np.asarray(
        model.get_feature_importance(),
        dtype=float,
    )

    if len(
        values
    ) != len(
        features
    ):
        raise ValueError(
            "CatBoost feature importance does not match feature contract."
        )

    out = pd.DataFrame({
        "feature":
            features,
        "catboost_native_importance":
            values,
    })

    total = float(
        out[
            "catboost_native_importance"
        ].sum()
    )

    out[
        "catboost_native_importance_normalized"
    ] = (
        out[
            "catboost_native_importance"
        ]
        / total
        if total > 0
        else 0.0
    )

    out = out.sort_values(
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

    out.insert(
        0,
        "catboost_native_rank",
        np.arange(
            1,
            len(
                out
            )
            + 1,
            dtype=int,
        ),
    )

    return out


def resolve_ice_feature_ranking(
    dataset_key: str,
    native_importance_df: pd.DataFrame,
    features: List[str],
) -> Tuple[
    List[str],
    Dict[str, Any],
]:
    stage05_csv = (
        STAGE05_MODEL_DIR
        / dataset_key
        / "global_shap_importance.csv"
    )

    if stage05_csv.exists():
        try:
            stage05_df = pd.read_csv(
                stage05_csv
            )

            if (
                "feature"
                in stage05_df.columns
            ):
                ranked = [
                    str(
                        feature
                    )
                    for feature
                    in stage05_df[
                        "feature"
                    ].tolist()
                    if str(
                        feature
                    )
                    in features
                ]

                if ranked:
                    return (
                        ranked,
                        {
                            "source":
                                "Stage-05 global SHAP ranking",
                            "path":
                                str(
                                    stage05_csv
                                ),
                        },
                    )

        except Exception as exc:
            print(
                "[WARN] Could not use Stage-05 SHAP ranking "
                f"for {dataset_key}: {exc}"
            )

    ranked = [
        str(
            feature
        )
        for feature
        in native_importance_df[
            "feature"
        ].tolist()
        if str(
            feature
        )
        in features
    ]

    return (
        ranked,
        {
            "source":
                "CatBoost native feature importance",
            "path":
                None,
        },
    )


# =============================================================================
# LIME PREPARATION
# =============================================================================

def import_lime():
    try:
        from lime.lime_tabular import (
            LimeTabularExplainer,
        )

        return LimeTabularExplainer

    except Exception as exc:
        raise ImportError(
            "Stage 06 LIME analysis requires package 'lime'. "
            "Install it with: pip install lime"
        ) from exc


def build_lime_training_representation(
    X_stage02_train: pd.DataFrame,
    features: List[str],
) -> Tuple[
    pd.DataFrame,
    Dict[str, float],
    Dict[str, Any],
]:
    """
    Build finite LIME-only training data using medians learned ONLY from the
    Stage-02 training split.

    This does not change the CatBoost model or ICE inputs.
    """
    medians: Dict[
        str,
        float
    ] = {}

    all_missing_features: List[
        str
    ] = []

    X_lime = X_stage02_train[
        features
    ].copy()

    for feature in features:
        values = pd.to_numeric(
            X_lime[
                feature
            ],
            errors="coerce",
        ).astype(float)

        finite = values[
            np.isfinite(
                values
            )
        ]

        if len(
            finite
        ) == 0:
            median = 0.0

            all_missing_features.append(
                feature
            )

        else:
            median = float(
                finite.median()
            )

        medians[
            feature
        ] = median

        X_lime[
            feature
        ] = (
            values
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                median
            )
        )

    matrix = X_lime.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        matrix
    ).all():
        raise RuntimeError(
            "LIME training representation still contains non-finite values."
        )

    report = {
        "rows":
            int(
                len(
                    X_lime
                )
            ),
        "features":
            len(
                features
            ),
        "imputation_method":
            "feature median learned only from Stage-02 train.csv",
        "all_missing_features_filled_with_zero":
            all_missing_features,
        "medians":
            medians,
    }

    return (
        X_lime,
        medians,
        report,
    )


def lime_impute_dataset(
    X: pd.DataFrame,
    medians: Dict[
        str,
        float,
    ],
    features: List[str],
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    X_lime = X[
        features
    ].copy()

    missing_before: Dict[
        str,
        int
    ] = {}

    for feature in features:
        values = pd.to_numeric(
            X_lime[
                feature
            ],
            errors="coerce",
        ).astype(float).replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        missing_before[
            feature
        ] = int(
            values.isna().sum()
        )

        X_lime[
            feature
        ] = values.fillna(
            float(
                medians[
                    feature
                ]
            )
        )

    return (
        X_lime,
        {
            "rows":
                int(
                    len(
                        X_lime
                    )
                ),
            "missing_values_replaced":
                int(
                    sum(
                        missing_before.values()
                    )
                ),
            "missing_by_feature":
                {
                    feature:
                        count
                    for feature, count
                    in missing_before.items()
                    if count > 0
                },
        },
    )


def build_lime_explainer(
    X_lime_training: pd.DataFrame,
    features: List[str],
):
    LimeTabularExplainer = (
        import_lime()
    )

    kwargs: Dict[
        str,
        Any
    ] = {
        "training_data":
            X_lime_training.to_numpy(
                dtype=float
            ),
        "feature_names":
            features,
        "class_names": [
            "class_0",
            "class_1",
        ],
        "mode":
            "classification",
        "discretize_continuous":
            LIME_DISCRETIZE_CONTINUOUS,
        "feature_selection":
            LIME_FEATURE_SELECTION,
        "random_state":
            RANDOM_STATE,
    }

    if LIME_KERNEL_WIDTH is not None:
        kwargs[
            "kernel_width"
        ] = float(
            LIME_KERNEL_WIDTH
        )

    return LimeTabularExplainer(
        **kwargs
    )


def make_lime_predict_fn(
    model: CatBoostClassifier,
    features: List[str],
):
    def predict_fn(
        array_like: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(
            array_like,
            dtype=float,
        )

        if array.ndim == 1:
            array = array.reshape(
                1,
                -1,
            )

        frame = pd.DataFrame(
            array,
            columns=features,
        )

        return predict_probability_matrix(
            model,
            frame,
        )

    return predict_fn


# =============================================================================
# LIME ROW SELECTION
# =============================================================================

def take_ranked(
    indices: np.ndarray,
    scores: np.ndarray,
    k: int,
    descending: bool,
) -> List[int]:
    if len(
        indices
    ) == 0:
        return []

    values = scores[
        indices
    ]

    if descending:
        order = np.argsort(
            -values,
            kind="stable",
        )
    else:
        order = np.argsort(
            values,
            kind="stable",
        )

    return (
        indices[
            order
        ][
            :min(
                k,
                len(
                    order
                ),
            )
        ]
        .astype(
            int
        )
        .tolist()
    )


def select_lime_rows(
    y_true: np.ndarray,
    probability_class1: np.ndarray,
    predicted_class: np.ndarray,
    threshold: float,
    seed: int,
) -> Tuple[
    List[int],
    pd.DataFrame,
]:
    n_rows = len(
        y_true
    )

    if LIME_ANALYZE_ALL_ROWS:
        selected = np.arange(
            n_rows,
            dtype=int,
        )

        if (
            LIME_ALL_ROWS_MAX
            is not None
            and len(
                selected
            )
            > LIME_ALL_ROWS_MAX
        ):
            rng = np.random.default_rng(
                seed
            )

            selected = np.sort(
                rng.choice(
                    selected,
                    size=LIME_ALL_ROWS_MAX,
                    replace=False,
                )
            )

        selection_df = pd.DataFrame({
            "row_index":
                selected,
            "category":
                "all_rows",
        })

        return (
            selected.astype(
                int
            ).tolist(),
            selection_df,
        )

    y = np.asarray(
        y_true,
        dtype=int,
    )

    p = np.asarray(
        probability_class1,
        dtype=float,
    )

    pred = np.asarray(
        predicted_class,
        dtype=int,
    )

    categories: Dict[
        str,
        List[int]
    ] = {}

    all_idx = np.arange(
        n_rows,
        dtype=int,
    )

    categories[
        "highest_probability"
    ] = take_ranked(
        all_idx,
        p,
        LIME_ROWS_PER_CATEGORY,
        descending=True,
    )

    categories[
        "lowest_probability"
    ] = take_ranked(
        all_idx,
        p,
        LIME_ROWS_PER_CATEGORY,
        descending=False,
    )

    nearest = np.argsort(
        np.abs(
            p
            - threshold
        ),
        kind="stable",
    )[
        :min(
            LIME_ROWS_PER_CATEGORY,
            n_rows,
        )
    ]

    categories[
        "nearest_threshold"
    ] = nearest.astype(
        int
    ).tolist()

    false_positive_idx = np.flatnonzero(
        (y == 0)
        & (pred == 1)
    )

    categories[
        "strongest_false_positives"
    ] = take_ranked(
        false_positive_idx,
        p,
        LIME_ROWS_PER_CATEGORY,
        descending=True,
    )

    false_negative_idx = np.flatnonzero(
        (y == 1)
        & (pred == 0)
    )

    categories[
        "strongest_false_negatives"
    ] = take_ranked(
        false_negative_idx,
        p,
        LIME_ROWS_PER_CATEGORY,
        descending=False,
    )

    rng = np.random.default_rng(
        seed
    )

    for class_value in [
        0,
        1,
    ]:
        class_idx = np.flatnonzero(
            y == class_value
        )

        n_pick = min(
            LIME_RANDOM_ROWS_PER_CLASS,
            len(
                class_idx
            ),
        )

        if n_pick > 0:
            chosen = np.sort(
                rng.choice(
                    class_idx,
                    size=n_pick,
                    replace=False,
                )
            ).astype(
                int
            ).tolist()

        else:
            chosen = []

        categories[
            f"random_true_class_{class_value}"
        ] = chosen

    rows: List[
        Dict[str, Any]
    ] = []

    unique_ordered: List[
        int
    ] = []

    seen = set()

    for category, indices in categories.items():
        for category_rank, row_index in enumerate(
            indices,
            start=1,
        ):
            rows.append({
                "row_index":
                    int(
                        row_index
                    ),
                "category":
                    category,
                "category_rank":
                    category_rank,
                "true_class":
                    int(
                        y[
                            row_index
                        ]
                    ),
                "predicted_class":
                    int(
                        pred[
                            row_index
                        ]
                    ),
                "probability_class1":
                    float(
                        p[
                            row_index
                        ]
                    ),
                "distance_to_threshold":
                    float(
                        abs(
                            p[
                                row_index
                            ]
                            - threshold
                        )
                    ),
            })

            if row_index not in seen:
                seen.add(
                    row_index
                )

                unique_ordered.append(
                    int(
                        row_index
                    )
                )

    selection_df = pd.DataFrame(
        rows
    )

    return (
        unique_ordered,
        selection_df,
    )


# =============================================================================
# LIME EXPLANATION
# =============================================================================

def scalarize_lime_local_prediction(
    local_pred: Any,
) -> Optional[float]:
    try:
        arr = np.asarray(
            local_pred,
            dtype=float,
        ).reshape(
            -1
        )

        if len(
            arr
        ) > 0:
            return float(
                arr[
                    0
                ]
            )

    except Exception:
        pass

    return None


def explain_lime_row(
    explainer: Any,
    predict_fn: Any,
    X_lime: pd.DataFrame,
    X_original: pd.DataFrame,
    source_df: pd.DataFrame,
    features: List[str],
    row_index: int,
    model_probability: float,
    predicted_class: int,
    dataset_lime_rows_dir: Path,
) -> Tuple[
    List[
        Dict[str, Any]
    ],
    Dict[str, Any],
]:
    row_array = (
        X_lime.iloc[
            row_index
        ].to_numpy(
            dtype=float
        )
    )

    explanation = (
        explainer.explain_instance(
            data_row=
                row_array,
            predict_fn=
                predict_fn,
            labels=(
                1,
            ),
            num_features=
                min(
                    LIME_NUM_FEATURES,
                    len(
                        features
                    ),
                ),
            num_samples=
                LIME_NUM_SAMPLES,
            distance_metric=
                LIME_DISTANCE_METRIC,
        )
    )

    map_payload = (
        explanation.as_map()
    )

    if 1 not in map_payload:
        raise RuntimeError(
            "LIME explanation does not contain positive-class label 1."
        )

    local_weights = map_payload[
        1
    ]

    # LIME returns feature indices in its internal representation.
    rows: List[
        Dict[str, Any]
    ] = []

    for rank, (
        feature_index,
        weight,
    ) in enumerate(
        sorted(
            local_weights,
            key=lambda item:
                abs(
                    float(
                        item[
                            1
                        ]
                    )
                ),
            reverse=True,
        ),
        start=1,
    ):
        feature_index = int(
            feature_index
        )

        if (
            feature_index < 0
            or feature_index
            >= len(
                features
            )
        ):
            continue

        feature = features[
            feature_index
        ]

        rows.append({
            "row_index":
                int(
                    row_index
                ),
            "rank":
                rank,
            "feature":
                feature,
            "original_feature_value":
                json_safe(
                    X_original.iloc[
                        row_index
                    ][
                        feature
                    ]
                ),
            "lime_feature_value":
                float(
                    X_lime.iloc[
                        row_index
                    ][
                        feature
                    ]
                ),
            "lime_weight":
                float(
                    weight
                ),
            "abs_lime_weight":
                float(
                    abs(
                        float(
                            weight
                        )
                    )
                ),
        })

    intercept: Optional[
        float
    ] = None

    try:
        intercept = float(
            explanation.intercept[
                1
            ]
        )
    except Exception:
        pass

    local_pred = (
        scalarize_lime_local_prediction(
            getattr(
                explanation,
                "local_pred",
                None,
            )
        )
    )

    fidelity_score: Optional[
        float
    ] = None

    try:
        fidelity_score = float(
            explanation.score
        )
    except Exception:
        pass

    true_class = int(
        source_df.iloc[
            row_index
        ][
            TARGET
        ]
    )

    row_dir = (
        dataset_lime_rows_dir
        / (
            f"row_{row_index:08d}"
        )
    )

    ensure_dir(
        row_dir
    )

    explanation_df = pd.DataFrame(
        rows
    )

    explanation_df.to_csv(
        row_dir
        / "explanation.csv",
        index=False,
    )

    explanation_summary = {
        "row_index":
            int(
                row_index
            ),
        "true_class":
            true_class,
        "predicted_class":
            int(
                predicted_class
            ),
        "catboost_probability_class1":
            float(
                model_probability
            ),
        "lime_local_surrogate_prediction":
            local_pred,
        "lime_local_fidelity_score":
            fidelity_score,
        "lime_intercept_class1":
            intercept,
        "lime_num_samples":
            LIME_NUM_SAMPLES,
        "lime_num_features_requested":
            LIME_NUM_FEATURES,
        "lime_discretize_continuous":
            LIME_DISCRETIZE_CONTINUOUS,
        "feature_explanations":
            rows,
    }

    write_json(
        row_dir
        / "explanation.json",
        explanation_summary,
    )

    text_lines: List[
        str
    ] = []

    text_lines.append(
        f"DIGIGARD LIME EXPLANATION - ROW {row_index}"
    )

    text_lines.append(
        "=" * 72
    )

    text_lines.append(
        f"TARGET: {TARGET}"
    )

    text_lines.append(
        f"TRUE CLASS: {true_class}"
    )

    text_lines.append(
        f"PREDICTED CLASS: {predicted_class}"
    )

    text_lines.append(
        f"CATBOOST P(CLASS=1): {model_probability:.8g}"
    )

    text_lines.append(
        f"LIME LOCAL PREDICTION: {local_pred}"
    )

    text_lines.append(
        f"LIME FIDELITY SCORE: {fidelity_score}"
    )

    text_lines.append(
        f"LIME INTERCEPT CLASS 1: {intercept}"
    )

    text_lines.append("")

    text_lines.append(
        "LOCAL LIME WEIGHTS"
    )

    for item in rows:
        text_lines.append(
            f"  {int(item['rank']):>2}. "
            f"{item['feature']}: "
            f"weight={item['lime_weight']:.8g}, "
            f"abs={item['abs_lime_weight']:.8g}, "
            f"original_value={item['original_feature_value']}, "
            f"lime_value={item['lime_feature_value']:.8g}"
        )

    (
        row_dir
        / "explanation.txt"
    ).write_text(
        "\n".join(
            text_lines
        ),
        encoding="utf-8",
    )

    if SAVE_LIME_HTML:
        try:
            explanation.save_to_file(
                str(
                    row_dir
                    / "explanation.html"
                ),
                labels=(
                    1,
                ),
            )

        except TypeError:
            # Compatibility fallback for LIME releases whose save_to_file
            # signature does not expose labels.
            explanation.save_to_file(
                str(
                    row_dir
                    / "explanation.html"
                )
            )

        except Exception as exc:
            print(
                f"[WARN] LIME HTML save failed for row {row_index}: {exc}"
            )

    if SAVE_LIME_PLOTS:
        plot_df = (
            explanation_df
            .sort_values(
                "abs_lime_weight",
                ascending=True,
            )
        )

        fig_height = max(
            5.0,
            0.50
            * len(
                plot_df
            )
            + 2.5,
        )

        fig, ax = plt.subplots(
            figsize=(
                10.0,
                fig_height,
            )
        )

        ax.barh(
            plot_df[
                "feature"
            ],
            plot_df[
                "lime_weight"
            ],
        )

        ax.axvline(
            0.0,
            linewidth=0.9,
        )

        ax.set_xlabel(
            "LIME local surrogate weight for class 1"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            f"LIME local explanation — row {row_index}\n"
            f"true={true_class} | "
            f"pred={predicted_class} | "
            f"P(1)={model_probability:.5f}"
        )

        ax.grid(
            axis="x",
            alpha=0.20,
        )

        fig.tight_layout()

        save_figure(
            fig,
            row_dir
            / "explanation"
        )

    return (
        rows,
        explanation_summary,
    )


def build_lime_aggregate(
    lime_long_df: pd.DataFrame,
    n_explained_rows: int,
) -> pd.DataFrame:
    if lime_long_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "explanation_count",
                "explanation_fraction",
                "mean_lime_weight",
                "mean_abs_lime_weight",
                "median_abs_lime_weight",
                "positive_weight_fraction",
                "negative_weight_fraction",
                "mean_rank",
            ]
        )

    grouped = (
        lime_long_df
        .groupby(
            "feature",
            sort=False,
        )
    )

    rows: List[
        Dict[str, Any]
    ] = []

    for feature, group in grouped:
        weights = pd.to_numeric(
            group[
                "lime_weight"
            ],
            errors="coerce",
        )

        abs_weights = np.abs(
            weights.to_numpy(
                dtype=float
            )
        )

        rows.append({
            "feature":
                feature,
            "explanation_count":
                int(
                    group[
                        "row_index"
                    ].nunique()
                ),
            "explanation_fraction":
                float(
                    group[
                        "row_index"
                    ].nunique()
                    / max(
                        1,
                        n_explained_rows,
                    )
                ),
            "mean_lime_weight":
                float(
                    weights.mean()
                ),
            "mean_abs_lime_weight":
                float(
                    np.mean(
                        abs_weights
                    )
                ),
            "median_abs_lime_weight":
                float(
                    np.median(
                        abs_weights
                    )
                ),
            "positive_weight_fraction":
                float(
                    np.mean(
                        weights.to_numpy(
                            dtype=float
                        )
                        > 0
                    )
                ),
            "negative_weight_fraction":
                float(
                    np.mean(
                        weights.to_numpy(
                            dtype=float
                        )
                        < 0
                    )
                ),
            "mean_rank":
                float(
                    pd.to_numeric(
                        group[
                            "rank"
                        ],
                        errors="coerce",
                    ).mean()
                ),
        })

    out = pd.DataFrame(
        rows
    )

    out = out.sort_values(
        [
            "mean_abs_lime_weight",
            "explanation_count",
            "feature",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    out.insert(
        0,
        "lime_aggregate_rank",
        np.arange(
            1,
            len(
                out
            )
            + 1,
            dtype=int,
        ),
    )

    return out


def run_lime_analysis(
    dataset_key: str,
    source_df: pd.DataFrame,
    X_original: pd.DataFrame,
    X_lime: pd.DataFrame,
    explainer: Any,
    predict_fn: Any,
    model_outputs_payload: Dict[
        str,
        np.ndarray,
    ],
    features: List[str],
    threshold: float,
    dataset_dir: Path,
    seed: int,
    lime_imputation_report: Dict[
        str,
        Any,
    ],
) -> Dict[str, Any]:
    lime_dir = (
        dataset_dir
        / "lime"
    )

    rows_dir = (
        lime_dir
        / "rows"
    )

    ensure_dir(
        rows_dir
    )

    if not ENABLE_LIME:
        return {
            "enabled":
                False,
        }

    y_true = validate_binary_target(
        source_df[
            TARGET
        ],
        dataset_key,
    ).to_numpy(
        dtype=int
    )

    probability = np.asarray(
        model_outputs_payload[
            "probability_class1"
        ],
        dtype=float,
    )

    predicted_class = np.asarray(
        model_outputs_payload[
            "predicted_class"
        ],
        dtype=int,
    )

    (
        selected_rows,
        selection_df,
    ) = select_lime_rows(
        y_true=
            y_true,
        probability_class1=
            probability,
        predicted_class=
            predicted_class,
        threshold=
            threshold,
        seed=
            seed,
    )

    selection_path = (
        lime_dir
        / "selected_rows.csv"
    )

    selection_df.to_csv(
        selection_path,
        index=False,
    )

    print(
        f"[LIME] {dataset_key}: "
        f"{len(selected_rows)} unique rows selected."
    )

    all_long_rows: List[
        Dict[str, Any]
    ] = []

    explanation_summaries: List[
        Dict[str, Any]
    ] = []

    failures: List[
        Dict[str, Any]
    ] = []

    for counter, row_index in enumerate(
        selected_rows,
        start=1,
    ):
        print(
            f"[LIME] {dataset_key}: "
            f"{counter}/{len(selected_rows)} "
            f"row={row_index}"
        )

        try:
            (
                long_rows,
                summary,
            ) = explain_lime_row(
                explainer=
                    explainer,
                predict_fn=
                    predict_fn,
                X_lime=
                    X_lime,
                X_original=
                    X_original,
                source_df=
                    source_df,
                features=
                    features,
                row_index=
                    row_index,
                model_probability=
                    float(
                        probability[
                            row_index
                        ]
                    ),
                predicted_class=
                    int(
                        predicted_class[
                            row_index
                        ]
                    ),
                dataset_lime_rows_dir=
                    rows_dir,
            )

            true_class = int(
                y_true[
                    row_index
                ]
            )

            for item in long_rows:
                item[
                    "true_class"
                ] = true_class

                item[
                    "predicted_class"
                ] = int(
                    predicted_class[
                        row_index
                    ]
                )

                item[
                    "probability_class1"
                ] = float(
                    probability[
                        row_index
                    ]
                )

                item[
                    "lime_local_fidelity_score"
                ] = summary.get(
                    "lime_local_fidelity_score"
                )

                item[
                    "lime_local_surrogate_prediction"
                ] = summary.get(
                    "lime_local_surrogate_prediction"
                )

            all_long_rows.extend(
                long_rows
            )

            explanation_summaries.append(
                summary
            )

        except Exception as exc:
            failures.append({
                "row_index":
                    int(
                        row_index
                    ),
                "error":
                    f"{type(exc).__name__}: {exc}",
            })

            print(
                f"[WARN] LIME failed for row {row_index}: "
                f"{type(exc).__name__}: {exc}"
            )

    lime_long_df = pd.DataFrame(
        all_long_rows
    )

    lime_long_path = (
        lime_dir
        / "lime_explanations_long.csv"
    )

    lime_long_df.to_csv(
        lime_long_path,
        index=False,
    )

    aggregate_df = build_lime_aggregate(
        lime_long_df=
            lime_long_df,
        n_explained_rows=
            len(
                explanation_summaries
            ),
    )

    aggregate_path = (
        lime_dir
        / "lime_feature_aggregate.csv"
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    failures_path = (
        lime_dir
        / "lime_failures.csv"
    )

    pd.DataFrame(
        failures,
        columns=[
            "row_index",
            "error",
        ],
    ).to_csv(
        failures_path,
        index=False,
    )

    fidelity_values = [
        summary[
            "lime_local_fidelity_score"
        ]
        for summary in explanation_summaries
        if summary.get(
            "lime_local_fidelity_score"
        ) is not None
    ]

    local_prediction_errors = []

    for summary in explanation_summaries:
        local_pred = summary.get(
            "lime_local_surrogate_prediction"
        )

        model_p = summary.get(
            "catboost_probability_class1"
        )

        if (
            local_pred
            is not None
            and model_p
            is not None
        ):
            local_prediction_errors.append(
                float(
                    local_pred
                    - model_p
                )
            )

    report = {
        "enabled":
            True,
        "n_rows_requested":
            int(
                len(
                    selected_rows
                )
            ),
        "n_rows_successfully_explained":
            int(
                len(
                    explanation_summaries
                )
            ),
        "n_failures":
            int(
                len(
                    failures
                )
            ),
        "num_features_per_explanation":
            min(
                LIME_NUM_FEATURES,
                len(
                    features
                ),
            ),
        "num_perturbation_samples":
            LIME_NUM_SAMPLES,
        "discretize_continuous":
            LIME_DISCRETIZE_CONTINUOUS,
        "distance_metric":
            LIME_DISTANCE_METRIC,
        "feature_selection":
            LIME_FEATURE_SELECTION,
        "lime_only_imputation":
            lime_imputation_report,
        "interpretation_warning": (
            "LIME is a local surrogate explanation. The aggregate table "
            "summarizes only the selected explained rows and must not be "
            "interpreted as a replacement for global SHAP or global model "
            "importance."
        ),
        "fidelity_score_mean":
            float(
                np.mean(
                    fidelity_values
                )
            )
            if fidelity_values
            else None,
        "fidelity_score_std":
            float(
                np.std(
                    fidelity_values,
                    ddof=1,
                )
            )
            if len(
                fidelity_values
            ) > 1
            else None,
        "local_surrogate_probability_error_mean":
            float(
                np.mean(
                    local_prediction_errors
                )
            )
            if local_prediction_errors
            else None,
        "local_surrogate_probability_error_mae":
            float(
                np.mean(
                    np.abs(
                        local_prediction_errors
                    )
                )
            )
            if local_prediction_errors
            else None,
        "top_aggregate_features":
            aggregate_df.head(
                min(
                    25,
                    len(
                        aggregate_df
                    ),
                )
            ).to_dict(
                orient="records"
            ),
        "artifacts": {
            "selected_rows_csv":
                str(
                    selection_path
                ),
            "lime_explanations_long_csv":
                str(
                    lime_long_path
                ),
            "lime_feature_aggregate_csv":
                str(
                    aggregate_path
                ),
            "lime_failures_csv":
                str(
                    failures_path
                ),
            "rows_directory":
                str(
                    rows_dir
                ),
        },
    }

    write_json(
        lime_dir
        / "lime_report.json",
        report,
    )

    return report


# =============================================================================
# ICE ROW SAMPLE + GRID
# =============================================================================

def stratified_sample_indices(
    y_true: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(
        y_true,
        dtype=int,
    )

    n_rows = len(
        y
    )

    if n_rows <= max_rows:
        return np.arange(
            n_rows,
            dtype=int,
        )

    rng = np.random.default_rng(
        seed
    )

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    selected_parts: List[
        np.ndarray
    ] = []

    remaining = max_rows

    for class_index, (
        class_value,
        class_count,
    ) in enumerate(
        zip(
            classes,
            counts,
        )
    ):
        class_rows = np.flatnonzero(
            y == class_value
        )

        if class_index == len(
            classes
        ) - 1:
            n_pick = min(
                remaining,
                len(
                    class_rows
                ),
            )

        else:
            proportional = int(
                round(
                    max_rows
                    * (
                        class_count
                        / n_rows
                    )
                )
            )

            n_pick = min(
                max(
                    1,
                    proportional,
                ),
                len(
                    class_rows
                ),
                remaining,
            )

        if n_pick > 0:
            chosen = rng.choice(
                class_rows,
                size=n_pick,
                replace=False,
            )

            selected_parts.append(
                np.asarray(
                    chosen,
                    dtype=int,
                )
            )

            remaining -= n_pick

    selected = np.unique(
        np.concatenate(
            selected_parts
        )
    )

    if len(
        selected
    ) < max_rows:
        remaining_pool = np.setdiff1d(
            np.arange(
                n_rows,
                dtype=int,
            ),
            selected,
            assume_unique=False,
        )

        n_extra = min(
            max_rows
            - len(
                selected
            ),
            len(
                remaining_pool
            ),
        )

        if n_extra > 0:
            extra = rng.choice(
                remaining_pool,
                size=n_extra,
                replace=False,
            )

            selected = np.concatenate(
                [
                    selected,
                    extra,
                ]
            )

    return np.sort(
        selected.astype(
            int
        )
    )


def feature_grid(
    values: pd.Series,
) -> np.ndarray:
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

    if len(
        finite
    ) == 0:
        return np.asarray(
            [],
            dtype=float,
        )

    unique = np.unique(
        finite
    )

    if len(
        unique
    ) <= ICE_USE_UNIQUE_VALUES_IF_AT_MOST:
        return np.sort(
            unique
        )

    lower = float(
        np.quantile(
            finite,
            ICE_GRID_LOWER_QUANTILE,
        )
    )

    upper = float(
        np.quantile(
            finite,
            ICE_GRID_UPPER_QUANTILE,
        )
    )

    if not np.isfinite(
        lower
    ) or not np.isfinite(
        upper
    ):
        return np.asarray(
            [],
            dtype=float,
        )

    if upper <= lower:
        return np.asarray(
            [
                lower
            ],
            dtype=float,
        )

    # Quantile-spaced grid places more resolution where observations exist.
    quantiles = np.linspace(
        ICE_GRID_LOWER_QUANTILE,
        ICE_GRID_UPPER_QUANTILE,
        ICE_GRID_POINTS,
    )

    grid = np.quantile(
        finite,
        quantiles,
    )

    grid = np.unique(
        np.asarray(
            grid,
            dtype=float,
        )
    )

    return grid


# =============================================================================
# ICE ANALYSIS FOR ONE FEATURE
# =============================================================================

def ice_feature_analysis(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    source_df: pd.DataFrame,
    feature: str,
    feature_rank: int,
    selected_rows: np.ndarray,
    feature_dir: Path,
    dataset_display_name: str,
) -> Dict[str, Any]:
    grid = feature_grid(
        X[
            feature
        ]
    )

    if len(
        grid
    ) == 0:
        return {
            "feature":
                feature,
            "feature_rank":
                feature_rank,
            "status":
                "skipped",
            "reason":
                "no finite values available",
        }

    X_sample = (
        X.iloc[
            selected_rows
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    source_sample = (
        source_df.iloc[
            selected_rows
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    y_sample = validate_binary_target(
        source_sample[
            TARGET
        ],
        f"ICE/{feature}",
    ).to_numpy(
        dtype=int
    )

    original_feature_values = pd.to_numeric(
        X_sample[
            feature
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    original_probabilities = (
        predict_probability_matrix(
            model,
            X_sample,
        )[
            :,
            1,
        ]
    )

    curves = np.empty(
        (
            len(
                X_sample
            ),
            len(
                grid
            ),
        ),
        dtype=float,
    )

    for grid_index, grid_value in enumerate(
        grid
    ):
        modified = X_sample.copy()

        modified[
            feature
        ] = float(
            grid_value
        )

        curves[
            :,
            grid_index
        ] = (
            predict_probability_matrix(
                model,
                modified,
            )[
                :,
                1,
            ]
        )

    centered_curves = (
        curves
        - curves[
            :,
            [
                0
            ],
        ]
    )

    pdp_mean = np.mean(
        curves,
        axis=0,
    )

    pdp_median = np.median(
        curves,
        axis=0,
    )

    pdp_std = np.std(
        curves,
        axis=0,
        ddof=0,
    )

    q10 = np.quantile(
        curves,
        0.10,
        axis=0,
    )

    q25 = np.quantile(
        curves,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        curves,
        0.75,
        axis=0,
    )

    q90 = np.quantile(
        curves,
        0.90,
        axis=0,
    )

    centered_mean = np.mean(
        centered_curves,
        axis=0,
    )

    individual_ranges = (
        np.max(
            curves,
            axis=1,
        )
        - np.min(
            curves,
            axis=1,
        )
    )

    pdp_range = float(
        np.max(
            pdp_mean
        )
        - np.min(
            pdp_mean
        )
    )

    feature_grid_series = pd.Series(
        grid
    )

    pdp_series = pd.Series(
        pdp_mean
    )

    spearman_grid_pdp = (
        float(
            feature_grid_series.corr(
                pdp_series,
                method="spearman",
            )
        )
        if len(
            grid
        ) >= 3
        and pdp_series.nunique() >= 2
        else float(
            "nan"
        )
    )

    # ---------------------------------------------------------------------
    # Long-form ICE table.
    # ---------------------------------------------------------------------
    long_rows: List[
        Dict[str, Any]
    ] = []

    for sample_index in range(
        len(
            X_sample
        )
    ):
        original_global_row = int(
            selected_rows[
                sample_index
            ]
        )

        for grid_index, grid_value in enumerate(
            grid
        ):
            long_rows.append({
                "sample_local_index":
                    sample_index,
                "row_index":
                    original_global_row,
                "true_class":
                    int(
                        y_sample[
                            sample_index
                        ]
                    ),
                "original_feature_value":
                    json_safe(
                        original_feature_values[
                            sample_index
                        ]
                    ),
                "original_probability_class1":
                    float(
                        original_probabilities[
                            sample_index
                        ]
                    ),
                "grid_index":
                    grid_index,
                "grid_value":
                    float(
                        grid_value
                    ),
                "ice_probability_class1":
                    float(
                        curves[
                            sample_index,
                            grid_index,
                        ]
                    ),
                "centered_ice":
                    float(
                        centered_curves[
                            sample_index,
                            grid_index,
                        ]
                    ),
            })

    ice_long_df = pd.DataFrame(
        long_rows
    )

    ice_long_path = (
        feature_dir
        / "ice_curves_long.csv"
    )

    if SAVE_ICE_LONG_CSV:
        ice_long_df.to_csv(
            ice_long_path,
            index=False,
        )

    pdp_summary_df = pd.DataFrame({
        "grid_index":
            np.arange(
                len(
                    grid
                ),
                dtype=int,
            ),
        "grid_value":
            grid,
        "pdp_mean_probability":
            pdp_mean,
        "median_probability":
            pdp_median,
        "std_probability":
            pdp_std,
        "q10_probability":
            q10,
        "q25_probability":
            q25,
        "q75_probability":
            q75,
        "q90_probability":
            q90,
        "mean_centered_ice":
            centered_mean,
    })

    pdp_summary_path = (
        feature_dir
        / "pdp_summary.csv"
    )

    pdp_summary_df.to_csv(
        pdp_summary_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Plots.
    # ---------------------------------------------------------------------
    if SAVE_ICE_PLOTS:
        fig, ax = plt.subplots(
            figsize=(
                9.5,
                6.5,
            )
        )

        ax.plot(
            grid,
            curves.T,
            alpha=0.15,
            linewidth=0.7,
        )

        ax.plot(
            grid,
            pdp_mean,
            linewidth=2.5,
            label="PDP / mean ICE",
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "CatBoost P(class=1)"
        )

        ax.set_title(
            f"{dataset_display_name}\n"
            f"ICE + PDP — rank {feature_rank}: {feature}"
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        save_figure(
            fig,
            feature_dir
            / "ice_and_pdp"
        )

        fig, ax = plt.subplots(
            figsize=(
                9.5,
                6.5,
            )
        )

        ax.plot(
            grid,
            centered_curves.T,
            alpha=0.15,
            linewidth=0.7,
        )

        ax.plot(
            grid,
            centered_mean,
            linewidth=2.5,
            label="Mean centered ICE",
        )

        ax.axhline(
            0.0,
            linewidth=0.9,
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Change in P(class=1) from first grid value"
        )

        ax.set_title(
            f"{dataset_display_name}\n"
            f"Centered ICE — rank {feature_rank}: {feature}"
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        save_figure(
            fig,
            feature_dir
            / "centered_ice"
        )

        fig, ax = plt.subplots(
            figsize=(
                9.5,
                6.5,
            )
        )

        ax.plot(
            grid,
            pdp_mean,
            linewidth=2.5,
            label="Mean / PDP",
        )

        ax.plot(
            grid,
            pdp_median,
            linewidth=1.5,
            linestyle="--",
            label="Median",
        )

        ax.fill_between(
            grid,
            q10,
            q90,
            alpha=0.15,
            label="10%-90% ICE band",
        )

        ax.fill_between(
            grid,
            q25,
            q75,
            alpha=0.20,
            label="25%-75% ICE band",
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "CatBoost P(class=1)"
        )

        ax.set_title(
            f"{dataset_display_name}\n"
            f"ICE response distribution — rank {feature_rank}: {feature}"
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.grid(
            alpha=0.20
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()

        save_figure(
            fig,
            feature_dir
            / "pdp_distribution"
        )

    report = {
        "feature":
            feature,
        "feature_rank":
            int(
                feature_rank
            ),
        "status":
            "ok",
        "n_ice_rows":
            int(
                len(
                    X_sample
                )
            ),
        "n_grid_points":
            int(
                len(
                    grid
                )
            ),
        "grid_min":
            float(
                np.min(
                    grid
                )
            ),
        "grid_max":
            float(
                np.max(
                    grid
                )
            ),
        "pdp_probability_min":
            float(
                np.min(
                    pdp_mean
                )
            ),
        "pdp_probability_max":
            float(
                np.max(
                    pdp_mean
                )
            ),
        "pdp_probability_range":
            pdp_range,
        "mean_individual_probability_range":
            float(
                np.mean(
                    individual_ranges
                )
            ),
        "median_individual_probability_range":
            float(
                np.median(
                    individual_ranges
                )
            ),
        "max_individual_probability_range":
            float(
                np.max(
                    individual_ranges
                )
            ),
        "mean_heterogeneity_std_across_grid":
            float(
                np.mean(
                    pdp_std
                )
            ),
        "max_heterogeneity_std_across_grid":
            float(
                np.max(
                    pdp_std
                )
            ),
        "mean_absolute_end_to_start_probability_change":
            float(
                np.mean(
                    np.abs(
                        centered_curves[
                            :,
                            -1
                        ]
                    )
                )
            ),
        "spearman_grid_value_vs_pdp_probability":
            spearman_grid_pdp,
        "artifacts": {
            "ice_curves_long_csv":
                str(
                    ice_long_path
                )
                if SAVE_ICE_LONG_CSV
                else None,
            "pdp_summary_csv":
                str(
                    pdp_summary_path
                ),
            "plot_directory":
                str(
                    feature_dir
                ),
        },
    }

    write_json(
        feature_dir
        / "ice_feature_report.json",
        report,
    )

    return report


# =============================================================================
# ICE DATASET ANALYSIS
# =============================================================================

def run_ice_analysis(
    dataset_key: str,
    source_df: pd.DataFrame,
    X: pd.DataFrame,
    model: CatBoostClassifier,
    ranked_features: List[str],
    ranking_source: Dict[
        str,
        Any,
    ],
    dataset_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    ice_dir = (
        dataset_dir
        / "ice"
    )

    ensure_dir(
        ice_dir
    )

    if not ENABLE_ICE:
        return {
            "enabled":
                False,
        }

    y_true = validate_binary_target(
        source_df[
            TARGET
        ],
        dataset_key,
    ).to_numpy(
        dtype=int
    )

    selected_rows = stratified_sample_indices(
        y_true=
            y_true,
        max_rows=
            ICE_SAMPLE_ROWS,
        seed=
            seed,
    )

    selected_rows_path = (
        ice_dir
        / "ice_selected_rows.csv"
    )

    pd.DataFrame({
        "row_index":
            selected_rows,
        "true_class":
            y_true[
                selected_rows
            ],
    }).to_csv(
        selected_rows_path,
        index=False,
    )

    top_features = [
        feature
        for feature
        in ranked_features
        if feature
        in X.columns
    ][
        :min(
            ICE_TOP_N_FEATURES,
            len(
                ranked_features
            ),
        )
    ]

    print(
        f"[ICE] {dataset_key}: "
        f"{len(selected_rows)} rows, "
        f"{len(top_features)} features."
    )

    feature_reports: List[
        Dict[str, Any]
    ] = []

    for rank, feature in enumerate(
        top_features,
        start=1,
    ):
        print(
            f"[ICE] {dataset_key}: "
            f"{rank}/{len(top_features)} "
            f"{feature}"
        )

        feature_dir = (
            ice_dir
            / (
                f"{rank:02d}_"
                f"{safe_file_name(feature)}"
            )
        )

        ensure_dir(
            feature_dir
        )

        try:
            feature_report = (
                ice_feature_analysis(
                    model=
                        model,
                    X=
                        X,
                    source_df=
                        source_df,
                    feature=
                        feature,
                    feature_rank=
                        rank,
                    selected_rows=
                        selected_rows,
                    feature_dir=
                        feature_dir,
                    dataset_display_name=
                        DATASET_DISPLAY_NAMES[
                            dataset_key
                        ],
                )
            )

        except Exception as exc:
            feature_report = {
                "feature":
                    feature,
                "feature_rank":
                    rank,
                "status":
                    "failed",
                "error":
                    f"{type(exc).__name__}: {exc}",
            }

            print(
                f"[WARN] ICE failed for feature {feature}: "
                f"{type(exc).__name__}: {exc}"
            )

        feature_reports.append(
            feature_report
        )

    summary_df = pd.DataFrame(
        feature_reports
    )

    summary_path = (
        ice_dir
        / "ice_feature_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    report = {
        "enabled":
            True,
        "feature_ranking_source":
            ranking_source,
        "n_selected_rows":
            int(
                len(
                    selected_rows
                )
            ),
        "n_requested_features":
            int(
                len(
                    top_features
                )
            ),
        "grid_points_target":
            ICE_GRID_POINTS,
        "grid_quantile_range": [
            ICE_GRID_LOWER_QUANTILE,
            ICE_GRID_UPPER_QUANTILE,
        ],
        "feature_reports":
            feature_reports,
        "interpretation_note": (
            "ICE varies one selected feature while holding each sampled "
            "observation's remaining features fixed. Curves can include feature "
            "combinations that are rare or unrealistic when features are strongly "
            "dependent, so ICE should be interpreted together with feature "
            "distributions and domain knowledge."
        ),
        "artifacts": {
            "selected_rows_csv":
                str(
                    selected_rows_path
                ),
            "ice_feature_summary_csv":
                str(
                    summary_path
                ),
        },
    }

    write_json(
        ice_dir
        / "ice_report.json",
        report,
    )

    return report


# =============================================================================
# DATASET REPORT
# =============================================================================

def write_dataset_report(
    dataset_key: str,
    dataset_dir: Path,
    preprocessing_report: Dict[
        str,
        Any,
    ],
    lime_report: Dict[
        str,
        Any,
    ],
    ice_report: Dict[
        str,
        Any,
    ],
) -> None:
    lines: List[
        str
    ] = []

    lines.append(
        f"DIGIGARD STAGE 06 - {DATASET_DISPLAY_NAMES[dataset_key]}"
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
        DATASET_INTERPRETATION[
            dataset_key
        ]
    )

    lines.append("")

    lines.append(
        "PREPROCESSING"
    )

    lines.append(
        f"  rows: {preprocessing_report['rows']}"
    )

    lines.append(
        f"  feature_count: {preprocessing_report['feature_count']}"
    )

    lines.append(
        f"  total_missing_cells: "
        f"{preprocessing_report['total_missing_cells']}"
    )

    lines.append("")

    lines.append(
        "LIME"
    )

    if lime_report.get(
        "enabled"
    ):
        lines.append(
            f"  successfully explained rows: "
            f"{lime_report['n_rows_successfully_explained']}"
        )

        lines.append(
            f"  failed rows: "
            f"{lime_report['n_failures']}"
        )

        lines.append(
            f"  perturbation samples per row: "
            f"{lime_report['num_perturbation_samples']}"
        )

        lines.append(
            f"  mean local fidelity: "
            f"{lime_report.get('fidelity_score_mean')}"
        )

        lines.append(
            f"  local surrogate probability MAE: "
            f"{lime_report.get('local_surrogate_probability_error_mae')}"
        )

        lines.append(
            "  NOTE: LIME uses a training-median-filled finite representation "
            "only for its local perturbation space."
        )

        lines.append(
            "  LIME feature aggregation is descriptive over selected rows, "
            "not a global model importance."
        )

    else:
        lines.append(
            "  disabled"
        )

    lines.append("")

    lines.append(
        "ICE"
    )

    if ice_report.get(
        "enabled"
    ):
        lines.append(
            f"  sampled rows: "
            f"{ice_report['n_selected_rows']}"
        )

        lines.append(
            f"  analyzed features: "
            f"{ice_report['n_requested_features']}"
        )

        lines.append(
            f"  feature ranking source: "
            f"{ice_report['feature_ranking_source']['source']}"
        )

        for feature_report in ice_report[
            "feature_reports"
        ]:
            if feature_report.get(
                "status"
            ) != "ok":
                continue

            lines.append(
                f"  {int(feature_report['feature_rank']):>2}. "
                f"{feature_report['feature']}: "
                f"PDP range="
                f"{feature_report['pdp_probability_range']:.6g}, "
                f"mean individual range="
                f"{feature_report['mean_individual_probability_range']:.6g}, "
                f"mean heterogeneity std="
                f"{feature_report['mean_heterogeneity_std_across_grid']:.6g}"
            )

    else:
        lines.append(
            "  disabled"
        )

    (
        dataset_dir
        / "DATASET_REPORT.txt"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# =============================================================================
# ONE DATASET
# =============================================================================

def analyze_one_dataset(
    dataset_key: str,
    source_df: pd.DataFrame,
    model: CatBoostClassifier,
    features: List[str],
    probability_threshold: float,
    lime_explainer: Any,
    lime_predict_fn: Any,
    lime_medians: Dict[
        str,
        float,
    ],
    native_importance_df: pd.DataFrame,
    seed: int,
) -> Dict[str, Any]:
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
        f"DIGIGARD STAGE 06: "
        f"{DATASET_DISPLAY_NAMES[dataset_key]}"
    )
    print("=" * 88)

    X, preprocessing_report = (
        preprocess_numeric_features(
            source_df,
            features,
            dataset_key,
        )
    )

    outputs = model_outputs(
        model=
            model,
        X=
            X,
        threshold=
            probability_threshold,
    )

    X_lime, lime_imputation_report = (
        lime_impute_dataset(
            X=
                X,
            medians=
                lime_medians,
            features=
                features,
        )
    )

    lime_report = run_lime_analysis(
        dataset_key=
            dataset_key,
        source_df=
            source_df,
        X_original=
            X,
        X_lime=
            X_lime,
        explainer=
            lime_explainer,
        predict_fn=
            lime_predict_fn,
        model_outputs_payload=
            outputs,
        features=
            features,
        threshold=
            probability_threshold,
        dataset_dir=
            dataset_dir,
        seed=
            seed,
        lime_imputation_report=
            lime_imputation_report,
    )

    (
        ranked_features,
        ranking_source,
    ) = resolve_ice_feature_ranking(
        dataset_key=
            dataset_key,
        native_importance_df=
            native_importance_df,
        features=
            features,
    )

    ice_report = run_ice_analysis(
        dataset_key=
            dataset_key,
        source_df=
            source_df,
        X=
            X,
        model=
            model,
        ranked_features=
            ranked_features,
        ranking_source=
            ranking_source,
        dataset_dir=
            dataset_dir,
        seed=
            seed + 10_000,
    )

    write_dataset_report(
        dataset_key=
            dataset_key,
        dataset_dir=
            dataset_dir,
        preprocessing_report=
            preprocessing_report,
        lime_report=
            lime_report,
        ice_report=
            ice_report,
    )

    report = {
        "dataset":
            dataset_key,
        "dataset_display_name":
            DATASET_DISPLAY_NAMES[
                dataset_key
            ],
        "interpretation":
            DATASET_INTERPRETATION[
                dataset_key
            ],
        "rows":
            int(
                len(
                    source_df
                )
            ),
        "preprocessing":
            preprocessing_report,
        "lime":
            lime_report,
        "ice":
            ice_report,
    }

    write_json(
        dataset_dir
        / "lime_ice_dataset_report.json",
        report,
    )

    return report


# =============================================================================
# CROSS-DATASET COMPARISON
# =============================================================================

def load_csv_if_exists(
    path: Optional[
        str
    ],
) -> Optional[
    pd.DataFrame
]:
    if not path:
        return None

    file_path = Path(
        path
    )

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(
            file_path
        )
    except Exception:
        return None


def cross_dataset_comparison(
    results: Dict[
        str,
        Dict[str, Any],
    ],
) -> Dict[str, Any]:
    out_dir = (
        OUTPUT_DIR
        / "cross_dataset_comparison"
    )

    ensure_dir(
        out_dir
    )

    # ---------------------------------------------------------------------
    # LIME aggregate comparison.
    # ---------------------------------------------------------------------
    lime_merged: Optional[
        pd.DataFrame
    ] = None

    for dataset_key in DATASET_ORDER:
        lime_artifacts = (
            results[
                dataset_key
            ][
                "lime"
            ].get(
                "artifacts",
                {},
            )
        )

        lime_path = lime_artifacts.get(
            "lime_feature_aggregate_csv"
        )

        lime_df = load_csv_if_exists(
            lime_path
        )

        if (
            lime_df is None
            or lime_df.empty
        ):
            continue

        keep = [
            column
            for column
            in [
                "feature",
                "lime_aggregate_rank",
                "explanation_count",
                "explanation_fraction",
                "mean_lime_weight",
                "mean_abs_lime_weight",
                "median_abs_lime_weight",
            ]
            if column
            in lime_df.columns
        ]

        lime_df = lime_df[
            keep
        ].copy()

        rename = {
            column:
                f"{dataset_key}__{column}"
            for column in keep
            if column != "feature"
        }

        lime_df = lime_df.rename(
            columns=rename
        )

        if lime_merged is None:
            lime_merged = lime_df
        else:
            lime_merged = lime_merged.merge(
                lime_df,
                on="feature",
                how="outer",
            )

    lime_comparison_path: Optional[
        Path
    ] = None

    if lime_merged is not None:
        lime_comparison_path = (
            out_dir
            / "lime_feature_aggregate_full_train_test.csv"
        )

        lime_merged.to_csv(
            lime_comparison_path,
            index=False,
        )

    # ---------------------------------------------------------------------
    # ICE summary comparison.
    # ---------------------------------------------------------------------
    ice_merged: Optional[
        pd.DataFrame
    ] = None

    for dataset_key in DATASET_ORDER:
        ice_artifacts = (
            results[
                dataset_key
            ][
                "ice"
            ].get(
                "artifacts",
                {},
            )
        )

        ice_path = ice_artifacts.get(
            "ice_feature_summary_csv"
        )

        ice_df = load_csv_if_exists(
            ice_path
        )

        if (
            ice_df is None
            or ice_df.empty
            or "feature"
            not in ice_df.columns
        ):
            continue

        keep = [
            column
            for column
            in [
                "feature",
                "feature_rank",
                "status",
                "pdp_probability_range",
                "mean_individual_probability_range",
                "mean_heterogeneity_std_across_grid",
                "max_heterogeneity_std_across_grid",
                "mean_absolute_end_to_start_probability_change",
                "spearman_grid_value_vs_pdp_probability",
            ]
            if column
            in ice_df.columns
        ]

        ice_df = ice_df[
            keep
        ].copy()

        rename = {
            column:
                f"{dataset_key}__{column}"
            for column in keep
            if column != "feature"
        }

        ice_df = ice_df.rename(
            columns=rename
        )

        if ice_merged is None:
            ice_merged = ice_df
        else:
            ice_merged = ice_merged.merge(
                ice_df,
                on="feature",
                how="outer",
            )

    ice_comparison_path: Optional[
        Path
    ] = None

    if ice_merged is not None:
        ice_comparison_path = (
            out_dir
            / "ice_feature_summary_full_train_test.csv"
        )

        ice_merged.to_csv(
            ice_comparison_path,
            index=False,
        )

    ranking_sources = {
        dataset_key:
            results[
                dataset_key
            ][
                "ice"
            ].get(
                "feature_ranking_source"
            )
        for dataset_key
        in DATASET_ORDER
    }

    ranking_source_path = (
        out_dir
        / "explanation_feature_source_report.json"
    )

    write_json(
        ranking_source_path,
        ranking_sources,
    )

    report = {
        "lime_comparison_csv":
            str(
                lime_comparison_path
            )
            if lime_comparison_path
            is not None
            else None,
        "ice_comparison_csv":
            str(
                ice_comparison_path
            )
            if ice_comparison_path
            is not None
            else None,
        "feature_ranking_sources":
            ranking_sources,
    }

    write_json(
        out_dir
        / "cross_dataset_lime_ice_report.json",
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
            "Stage-06 output directory already exists and "
            "OVERWRITE_EXISTING_ANALYSIS=False:\n"
            f"  {OUTPUT_DIR.resolve()}"
        )

    ensure_dir(
        OUTPUT_DIR
    )

    print("=" * 88)
    print(
        "DIGIGARD STAGE 06 - CATBOOST LIME + ICE EXPLAINABILITY"
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

    # ---------------------------------------------------------------------
    # Stage-02 data.
    # ---------------------------------------------------------------------
    train_df = pd.read_csv(
        TRAIN_CSV,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_CSV,
        low_memory=False,
    )

    X_train, train_preprocess = (
        preprocess_numeric_features(
            train_df,
            features,
            "Stage-02 train.csv / LIME reference",
        )
    )

    # LIME's finite perturbation distribution MUST be derived from training only.
    (
        X_lime_training,
        lime_medians,
        lime_training_representation_report,
    ) = build_lime_training_representation(
        X_stage02_train=
            X_train,
        features=
            features,
    )

    lime_explainer = (
        build_lime_explainer(
            X_lime_training=
                X_lime_training,
            features=
                features,
        )
        if ENABLE_LIME
        else None
    )

    lime_predict_fn = (
        make_lime_predict_fn(
            model=
                model,
            features=
                features,
        )
        if ENABLE_LIME
        else None
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
            probability_threshold=
                probability_threshold,
            lime_explainer=
                lime_explainer,
            lime_predict_fn=
                lime_predict_fn,
            lime_medians=
                lime_medians,
            native_importance_df=
                native_importance_df,
            seed=
                seeds[
                    dataset_key
                ],
        )

    cross_report = cross_dataset_comparison(
        results
    )

    elapsed_seconds = float(
        time.time()
        - start_time
    )

    # ---------------------------------------------------------------------
    # Compact summary CSV.
    # ---------------------------------------------------------------------
    summary_rows: List[
        Dict[str, Any]
    ] = []

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        lime_report = result[
            "lime"
        ]

        ice_report = result[
            "ice"
        ]

        ok_ice_features = [
            item
            for item
            in ice_report.get(
                "feature_reports",
                [],
            )
            if item.get(
                "status"
            )
            == "ok"
        ]

        top_ice = (
            ok_ice_features[
                0
            ]
            if ok_ice_features
            else {}
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
            "lime_rows_explained":
                lime_report.get(
                    "n_rows_successfully_explained"
                ),
            "lime_failures":
                lime_report.get(
                    "n_failures"
                ),
            "lime_mean_fidelity":
                lime_report.get(
                    "fidelity_score_mean"
                ),
            "lime_local_probability_mae":
                lime_report.get(
                    "local_surrogate_probability_error_mae"
                ),
            "ice_features_analyzed":
                len(
                    ok_ice_features
                ),
            "top_ice_feature":
                top_ice.get(
                    "feature"
                ),
            "top_ice_pdp_probability_range":
                top_ice.get(
                    "pdp_probability_range"
                ),
            "top_ice_mean_individual_probability_range":
                top_ice.get(
                    "mean_individual_probability_range"
                ),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "lime_ice_analysis_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------------------
    # Master machine report.
    # ---------------------------------------------------------------------
    master_report = {
        "pipeline":
            "DIGIGARD",
        "stage":
            6,
        "stage_name":
            "catboost_lime_ice_explainability",
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
            probability_threshold,
        "feature_count":
            len(
                features
            ),
        "features":
            features,
        "lime": {
            "enabled":
                ENABLE_LIME,
            "num_features":
                LIME_NUM_FEATURES,
            "num_samples":
                LIME_NUM_SAMPLES,
            "discretize_continuous":
                LIME_DISCRETIZE_CONTINUOUS,
            "training_reference":
                "Stage-02 train.csv only",
            "training_reference_preprocessing":
                train_preprocess,
            "finite_training_representation":
                lime_training_representation_report,
        },
        "ice": {
            "enabled":
                ENABLE_ICE,
            "top_n_features":
                ICE_TOP_N_FEATURES,
            "sample_rows":
                ICE_SAMPLE_ROWS,
            "grid_points":
                ICE_GRID_POINTS,
            "grid_quantile_range": [
                ICE_GRID_LOWER_QUANTILE,
                ICE_GRID_UPPER_QUANTILE,
            ],
        },
        "datasets":
            results,
        "cross_dataset_comparison":
            cross_report,
        "artifacts": {
            "catboost_native_feature_importance_csv":
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
            "lime":
                safe_package_version(
                    "lime"
                ),
        },
        "elapsed_seconds":
            elapsed_seconds,
    }

    master_report_path = (
        OUTPUT_DIR
        / "lime_ice_analysis_report.json"
    )

    write_json(
        master_report_path,
        master_report,
    )

    # ---------------------------------------------------------------------
    # Human-readable master report.
    # ---------------------------------------------------------------------
    lines: List[
        str
    ] = []

    lines.append(
        "DIGIGARD STAGE 06 - CATBOOST LIME + ICE EXPLAINABILITY"
    )

    lines.append(
        "=" * 82
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

    lines.append(
        f"PROBABILITY_THRESHOLD: {probability_threshold}"
    )

    lines.append("")

    lines.append(
        "LIME METHODOLOGY"
    )

    lines.append(
        "LIME explains selected individual predictions using a locally "
        "weighted surrogate model around each observation."
    )

    lines.append(
        "The finite LIME perturbation distribution is constructed only from "
        "Stage-02 training data."
    )

    lines.append(
        "Numeric missing values are replaced by Stage-02 training medians only "
        "inside the LIME representation. CatBoost itself is not retrained or "
        "modified."
    )

    lines.append(
        "Aggregated LIME weights summarize selected local explanations and "
        "must not be treated as a global feature-importance replacement."
    )

    lines.append("")

    lines.append(
        "ICE METHODOLOGY"
    )

    lines.append(
        "ICE changes one selected feature over an empirical grid while holding "
        "all other feature values of each sampled observation fixed."
    )

    lines.append(
        "Each line is one observation's P(class=1) response. "
        "The mean line is the corresponding PDP."
    )

    lines.append(
        "Centered ICE subtracts each observation's response at the first grid "
        "value to emphasize heterogeneous response shapes."
    )

    lines.append("")

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        lines.append(
            "=" * 82
        )

        lines.append(
            DATASET_DISPLAY_NAMES[
                dataset_key
            ]
        )

        lines.append(
            "=" * 82
        )

        lines.append(
            DATASET_INTERPRETATION[
                dataset_key
            ]
        )

        lines.append("")

        lime_report = result[
            "lime"
        ]

        if lime_report.get(
            "enabled"
        ):
            lines.append(
                f"LIME explained rows: "
                f"{lime_report['n_rows_successfully_explained']}"
            )

            lines.append(
                f"LIME failures: "
                f"{lime_report['n_failures']}"
            )

            lines.append(
                f"LIME mean fidelity: "
                f"{lime_report.get('fidelity_score_mean')}"
            )

            lines.append(
                f"LIME local probability MAE: "
                f"{lime_report.get('local_surrogate_probability_error_mae')}"
            )

            lines.append(
                "Top aggregate LIME features:"
            )

            for item in lime_report.get(
                "top_aggregate_features",
                [],
            )[
                :10
            ]:
                lines.append(
                    f"  {int(item['lime_aggregate_rank']):>2}. "
                    f"{item['feature']}: "
                    f"mean|weight|={item['mean_abs_lime_weight']:.8g}, "
                    f"frequency={item['explanation_fraction']:.4f}"
                )

        lines.append("")

        ice_report = result[
            "ice"
        ]

        if ice_report.get(
            "enabled"
        ):
            lines.append(
                "ICE feature ranking source: "
                f"{ice_report['feature_ranking_source']['source']}"
            )

            lines.append(
                "ICE feature summaries:"
            )

            for item in ice_report.get(
                "feature_reports",
                [],
            ):
                if item.get(
                    "status"
                ) != "ok":
                    continue

                lines.append(
                    f"  {int(item['feature_rank']):>2}. "
                    f"{item['feature']}: "
                    f"PDP range={item['pdp_probability_range']:.6g}, "
                    f"mean individual range="
                    f"{item['mean_individual_probability_range']:.6g}, "
                    f"mean heterogeneity std="
                    f"{item['mean_heterogeneity_std_across_grid']:.6g}"
                )

        lines.append("")

    human_report_path = (
        OUTPUT_DIR
        / "LIME_ICE_ANALYSIS_REPORT.txt"
    )

    human_report_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Configuration snapshot.
    # ---------------------------------------------------------------------
    config = {
        "pipeline":
            "DIGIGARD",
        "stage":
            6,
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
        "lime": {
            "enabled":
                ENABLE_LIME,
            "num_features":
                LIME_NUM_FEATURES,
            "num_samples":
                LIME_NUM_SAMPLES,
            "discretize_continuous":
                LIME_DISCRETIZE_CONTINUOUS,
            "kernel_width":
                LIME_KERNEL_WIDTH,
            "feature_selection":
                LIME_FEATURE_SELECTION,
            "distance_metric":
                LIME_DISTANCE_METRIC,
            "rows_per_category":
                LIME_ROWS_PER_CATEGORY,
            "random_rows_per_class":
                LIME_RANDOM_ROWS_PER_CLASS,
            "analyze_all_rows":
                LIME_ANALYZE_ALL_ROWS,
            "all_rows_max":
                LIME_ALL_ROWS_MAX,
        },
        "ice": {
            "enabled":
                ENABLE_ICE,
            "top_n_features":
                ICE_TOP_N_FEATURES,
            "sample_rows":
                ICE_SAMPLE_ROWS,
            "grid_points":
                ICE_GRID_POINTS,
            "grid_lower_quantile":
                ICE_GRID_LOWER_QUANTILE,
            "grid_upper_quantile":
                ICE_GRID_UPPER_QUANTILE,
            "use_unique_values_if_at_most":
                ICE_USE_UNIQUE_VALUES_IF_AT_MOST,
        },
        "save_png":
            SAVE_PNG,
        "save_eps":
            SAVE_EPS,
    }

    write_json(
        OUTPUT_DIR
        / "lime_ice_analysis_config.json",
        config,
    )

    print()
    print("=" * 88)
    print(
        "DIGIGARD STAGE 06 COMPLETE"
    )
    print("=" * 88)

    for dataset_key in DATASET_ORDER:
        result = results[
            dataset_key
        ]

        print(
            DATASET_DISPLAY_NAMES[
                dataset_key
            ]
        )

        print(
            "  LIME rows explained: "
            f"{result['lime'].get('n_rows_successfully_explained')}"
        )

        print(
            "  ICE features analyzed: "
            f"{len([x for x in result['ice'].get('feature_reports', []) if x.get('status') == 'ok'])}"
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
