#!/usr/bin/env python3
"""
02_DIGIGARD_create_train_test_splits.py

DIGIGARD Stage 02: create train/test splits for EVERY configured target.

This stage runs AFTER:
    01_DIGIGARD_merge_and_clean_project_histories.py

and BEFORE:
    CatBoost / model-training stages.

Input
-----
DIGIGARD_01_merged_data/
    DIGIGARD_merged_dataset.csv
    DIGIGARD_feature_target_contract.json

The feature list and possible targets are NOT redefined here.
They are loaded from the Stage-01 contract so every downstream DIGIGARD stage
uses exactly the same model-feature definition.

Default targets loaded from Stage 01:
    isBugPresent
    isBugfix
    isSZZBugIntroducer

Default split:
    80% train
    20% test
    random_state = 42
    stratified by the current target

Output
------
DIGIGARD_02_train_test_splits/
    pipeline_config.json
    split_summary.csv

    isBugPresent/
        features.json
        target.json
        splits_42/
            train.csv
            test.csv
            split_report.json

    isBugfix/
        features.json
        target.json
        splits_42/
            train.csv
            test.csv
            split_report.json

    isSZZBugIntroducer/
        features.json
        target.json
        splits_42/
            train.csv
            test.csv
            split_report.json

Important
---------
- Each target gets its OWN train/test split.
- Only the configured FEATURES may later be used as model inputs.
- Other ground-truth columns are deliberately NOT written into a target's split
  files, preventing accidental target leakage.
- Project/file/history metadata are retained in train.csv/test.csv for tracing and
  later analyses, but are NOT included in features.json.
- Missing target rows are dropped separately for each target.
- Feature NaNs are retained. CatBoost can handle numeric missing values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# =============================================================================
# CONFIG
# =============================================================================

STAGE01_DIR = Path("DIGIGARD_01_merged_data")

MERGED_CSV = STAGE01_DIR / "DIGIGARD_merged_dataset.csv"
CONTRACT_JSON = STAGE01_DIR / "DIGIGARD_feature_target_contract.json"

OUTPUT_ROOT = Path("DIGIGARD_02_train_test_splits")

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Standard DIGIGARD default:
# stratify every target independently using its own 0/1 ground truth.
STRATIFY = True

# Optional group-aware mode retained from the previous OBJ pipeline.
#
# None:
#     ordinary row-level train_test_split.
#
# "source_project":
#     entire projects are assigned either to train or test.
#
# IMPORTANT:
# GroupShuffleSplit cannot simultaneously guarantee exact class stratification.
GROUP_COL: Optional[str] = None
REQUIRE_GROUP_HOLDOUT = False

# Ground truths are expected to be binary 0/1.
STRICT_BINARY_TARGET = True

# Rows without a target cannot be used for supervised training.
DROP_MISSING_TARGET = True

# A row with no usable model features is not informative for model training.
DROP_ROWS_WITH_ALL_FEATURES_MISSING = True

# Fail if an expected model feature is absent from the merged Stage-01 dataset.
STRICT_FEATURE_SCHEMA = True

# For a normal stratified split, fail rather than silently changing methodology
# if stratification is impossible.
ALLOW_UNSTRATIFIED_FALLBACK = False


# =============================================================================
# HELPERS
# =============================================================================

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_contract() -> Dict[str, Any]:
    if not CONTRACT_JSON.exists():
        raise SystemExit(
            "Stage-01 feature/target contract not found:\n"
            f"  {CONTRACT_JSON.resolve()}\n\n"
            "Run 01_DIGIGARD_merge_and_clean_project_histories.py first."
        )

    with CONTRACT_JSON.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)

    required_keys = {
        "features",
        "targets",
        "metadata_columns",
    }

    missing = sorted(required_keys - set(contract))
    if missing:
        raise SystemExit(
            "Invalid Stage-01 contract. Missing keys:\n"
            + "\n".join(missing)
        )

    return contract


def validate_contract(
    features: List[str],
    targets: List[str],
    metadata_columns: List[str],
) -> None:
    if not features:
        raise ValueError("Stage-01 contract contains no features.")

    if not targets:
        raise ValueError("Stage-01 contract contains no targets.")

    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature names detected in Stage-01 contract.")

    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate target names detected in Stage-01 contract.")

    overlap_ft = sorted(set(features) & set(targets))
    overlap_fm = sorted(set(features) & set(metadata_columns))
    overlap_tm = sorted(set(targets) & set(metadata_columns))

    if overlap_ft or overlap_fm or overlap_tm:
        raise ValueError(
            "Invalid Stage-01 contract: feature/target/metadata overlap detected.\n"
            f"FEATURES ∩ TARGETS: {overlap_ft}\n"
            f"FEATURES ∩ METADATA: {overlap_fm}\n"
            f"TARGETS ∩ METADATA: {overlap_tm}"
        )


def to_numeric_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip().str.lower()

    replacements = {
        "true": "1",
        "false": "0",
        "yes": "1",
        "no": "0",
        "null": pd.NA,
        "none": pd.NA,
        "nan": pd.NA,
        "na": pd.NA,
        "n/a": pd.NA,
    }

    text = text.replace(replacements)
    text = text.str.replace(",", ".", regex=False)

    return pd.to_numeric(text, errors="coerce")


def class_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()

    out: Dict[str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "NaN"
        else:
            numeric = float(key)
            if numeric.is_integer():
                label = str(int(numeric))
            else:
                label = str(numeric)

        out[label] = int(value)

    return out


def class_fractions(series: pd.Series) -> Dict[str, float]:
    counts = series.value_counts(normalize=True, dropna=False).sort_index()

    out: Dict[str, float] = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "NaN"
        else:
            numeric = float(key)
            if numeric.is_integer():
                label = str(int(numeric))
            else:
                label = str(numeric)

        out[label] = float(value)

    return out


def prepare_target_dataset(
    merged: pd.DataFrame,
    target: str,
    features: List[str],
    metadata_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build one target-specific supervised dataset.

    Other targets are intentionally excluded to prevent label leakage.
    """
    if target not in merged.columns:
        raise ValueError(
            f"Target '{target}' is missing from the Stage-01 merged dataset."
        )

    missing_features = [
        feature
        for feature in features
        if feature not in merged.columns
    ]

    if missing_features and STRICT_FEATURE_SCHEMA:
        raise ValueError(
            f"Target '{target}': expected model features are missing:\n"
            + "\n".join(missing_features)
        )

    available_metadata = [
        column
        for column in metadata_columns
        if column in merged.columns
    ]

    # Add missing features only when explicitly allowed.
    working = merged.copy()

    for feature in missing_features:
        working[feature] = np.nan

    # IMPORTANT:
    # Current target + metadata + FEATURES only.
    # Other possible targets are NOT copied into this target experiment.
    selected_columns = available_metadata + features + [target]
    df = working[selected_columns].copy()

    rows_initial = int(len(df))

    # Re-normalize the current target defensively.
    df[target] = to_numeric_target(df[target])

    missing_target_before_drop = int(df[target].isna().sum())

    if DROP_MISSING_TARGET:
        df = df.loc[df[target].notna()].copy()

    rows_after_target_drop = int(len(df))

    # Check binary ground truth.
    observed_values = sorted(
        float(value)
        for value in pd.unique(df[target].dropna())
    )

    unexpected_values = [
        value
        for value in observed_values
        if value not in {0.0, 1.0}
    ]

    if unexpected_values and STRICT_BINARY_TARGET:
        raise ValueError(
            f"Target '{target}' contains non-binary values: "
            f"{unexpected_values}"
        )

    if unexpected_values:
        df = df.loc[df[target].isin([0, 1])].copy()

    df[target] = df[target].astype(int)

    all_features_missing_count = int(
        df[features].isna().all(axis=1).sum()
    )

    if DROP_ROWS_WITH_ALL_FEATURES_MISSING:
        df = df.loc[
            ~df[features].isna().all(axis=1)
        ].copy()

    rows_final = int(len(df))

    counts = df[target].value_counts().to_dict()

    if len(counts) != 2:
        raise ValueError(
            f"Target '{target}' does not contain both binary classes "
            f"after filtering. Counts: {counts}"
        )

    preparation_report = {
        "target": target,
        "rows_initial": rows_initial,
        "missing_target_rows": missing_target_before_drop,
        "rows_after_missing_target_drop": rows_after_target_drop,
        "unexpected_non_binary_values": unexpected_values,
        "rows_all_features_missing": all_features_missing_count,
        "rows_final_for_splitting": rows_final,
        "class_counts_before_split": class_counts(df[target]),
        "class_fractions_before_split": class_fractions(df[target]),
        "features_missing_from_merged_dataset": missing_features,
        "metadata_columns_retained": available_metadata,
    }

    return df, preparation_report


def ordinary_split(
    df: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    stratify_values = df[target] if STRATIFY else None

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=stratify_values,
        )

        split_mode = (
            "train_test_split_stratified"
            if STRATIFY
            else "train_test_split"
        )

        return train_df, test_df, split_mode

    except ValueError as exc:
        if STRATIFY and ALLOW_UNSTRATIFIED_FALLBACK:
            print(
                f"[WARN] Stratified split failed for {target}: {exc}\n"
                "[WARN] Falling back to ordinary shuffled split."
            )

            train_df, test_df = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                shuffle=True,
                stratify=None,
            )

            return (
                train_df,
                test_df,
                "train_test_split_unstratified_fallback",
            )

        raise


def group_split(
    df: pd.DataFrame,
    target: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if group_col not in df.columns:
        if REQUIRE_GROUP_HOLDOUT:
            raise ValueError(
                f"GROUP_COL='{group_col}' requested but missing "
                f"for target '{target}'."
            )

        print(
            f"[WARN] GROUP_COL='{group_col}' missing for {target}. "
            "Falling back to ordinary row-level splitting."
        )

        return ordinary_split(df, target)

    groups = df[group_col].astype("string").fillna("<MISSING_GROUP>")

    if groups.nunique(dropna=False) < 2:
        raise ValueError(
            f"Target '{target}': group holdout requires at least "
            f"two unique groups in '{group_col}'."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df[target],
            groups=groups,
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return (
        train_df,
        test_df,
        f"group_shuffle_split__{group_col}",
    )


def check_split_integrity(
    source_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    group_col: Optional[str],
) -> Dict[str, Any]:
    train_indices = set(train_df.index.tolist())
    test_indices = set(test_df.index.tolist())

    overlapping_indices = train_indices & test_indices
    union_indices = train_indices | test_indices
    source_indices = set(source_df.index.tolist())

    report: Dict[str, Any] = {
        "row_index_overlap_count": len(overlapping_indices),
        "all_source_rows_assigned_exactly_once": (
            union_indices == source_indices
            and len(overlapping_indices) == 0
        ),
    }

    if overlapping_indices:
        raise RuntimeError(
            f"Target '{target}': train/test row overlap detected."
        )

    if union_indices != source_indices:
        raise RuntimeError(
            f"Target '{target}': not all prepared rows were assigned "
            "to exactly one split."
        )

    if group_col is not None and group_col in source_df.columns:
        train_groups = set(
            train_df[group_col]
            .astype("string")
            .fillna("<MISSING_GROUP>")
            .tolist()
        )
        test_groups = set(
            test_df[group_col]
            .astype("string")
            .fillna("<MISSING_GROUP>")
            .tolist()
        )

        overlap_groups = sorted(train_groups & test_groups)

        report.update({
            "group_column": group_col,
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "group_overlap_count": len(overlap_groups),
            "overlapping_groups": overlap_groups,
        })

        if GROUP_COL is not None and overlap_groups:
            raise RuntimeError(
                f"Target '{target}': group leakage detected in "
                f"'{group_col}': {overlap_groups}"
            )

    return report


# =============================================================================
# ONE TARGET
# =============================================================================

def process_target(
    merged: pd.DataFrame,
    target: str,
    features: List[str],
    metadata_columns: List[str],
) -> Dict[str, Any]:
    print()
    print("=" * 80)
    print(f"TARGET: {target}")
    print("=" * 80)

    target_df, preparation_report = prepare_target_dataset(
        merged=merged,
        target=target,
        features=features,
        metadata_columns=metadata_columns,
    )

    print(
        f"[DATA] rows available for split: {len(target_df):,}"
    )
    print(
        f"[DATA] class counts: "
        f"{class_counts(target_df[target])}"
    )

    if GROUP_COL is None:
        train_df, test_df, split_mode = ordinary_split(
            target_df,
            target,
        )
    else:
        train_df, test_df, split_mode = group_split(
            target_df,
            target,
            GROUP_COL,
        )

    integrity = check_split_integrity(
        source_df=target_df,
        train_df=train_df,
        test_df=test_df,
        target=target,
        group_col=GROUP_COL,
    )

    # Reset index only AFTER integrity checks.
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    target_root = OUTPUT_ROOT / target
    split_dir = target_root / f"splits_{RANDOM_STATE}"

    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save the exact feature contract inside every target namespace so future model
    # stages can operate without guessing or inferring feature columns.
    write_json(
        target_root / "features.json",
        features,
    )

    write_json(
        target_root / "target.json",
        {
            "target": target,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
    )

    split_report = {
        "pipeline": "DIGIGARD",
        "stage": 2,
        "stage_name": "create_train_test_splits",
        "target": target,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "split_mode": split_mode,
        "stratify_requested": STRATIFY,
        "group_col": GROUP_COL,
        "feature_count": len(features),
        "features": features,
        "preparation": preparation_report,
        "train": {
            "rows": int(len(train_df)),
            "fraction_of_prepared_dataset": float(
                len(train_df) / len(target_df)
            ),
            "class_counts": class_counts(train_df[target]),
            "class_fractions": class_fractions(train_df[target]),
            "csv": str(train_path),
        },
        "test": {
            "rows": int(len(test_df)),
            "fraction_of_prepared_dataset": float(
                len(test_df) / len(target_df)
            ),
            "class_counts": class_counts(test_df[target]),
            "class_fractions": class_fractions(test_df[target]),
            "csv": str(test_path),
        },
        "integrity_checks": integrity,
        "important": {
            "other_targets_in_split_files": False,
            "metadata_retained_but_not_features": True,
            "feature_nans_retained": True,
            "test_set_not_used_for_training_at_this_stage": True,
        },
    }

    report_path = split_dir / "split_report.json"
    write_json(report_path, split_report)

    print(
        f"[TRAIN] {len(train_df):,} rows | "
        f"{class_counts(train_df[target])}"
    )
    print(
        f"[TEST ] {len(test_df):,} rows | "
        f"{class_counts(test_df[target])}"
    )
    print(f"[OK] {train_path}")
    print(f"[OK] {test_path}")
    print(f"[OK] {report_path}")

    return {
        "target": target,
        "rows_before_target_filtering": preparation_report["rows_initial"],
        "rows_used": int(len(target_df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "train_class_0": int((train_df[target] == 0).sum()),
        "train_class_1": int((train_df[target] == 1).sum()),
        "test_class_0": int((test_df[target] == 0).sum()),
        "test_class_1": int((test_df[target] == 1).sum()),
        "split_mode": split_mode,
        "train_csv": str(train_path),
        "test_csv": str(test_path),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if not MERGED_CSV.exists():
        raise SystemExit(
            "Stage-01 merged CSV not found:\n"
            f"  {MERGED_CSV.resolve()}\n\n"
            "Run 01_DIGIGARD_merge_and_clean_project_histories.py first."
        )

    contract = load_contract()

    features = list(contract["features"])
    targets = list(contract["targets"])
    metadata_columns = list(contract["metadata_columns"])

    validate_contract(
        features=features,
        targets=targets,
        metadata_columns=metadata_columns,
    )

    print("=" * 80)
    print("DIGIGARD STAGE 02 - TRAIN/TEST SPLIT CREATION")
    print("=" * 80)
    print(f"Merged CSV      : {MERGED_CSV.resolve()}")
    print(f"Contract        : {CONTRACT_JSON.resolve()}")
    print(f"Output root     : {OUTPUT_ROOT.resolve()}")
    print(f"Targets         : {targets}")
    print(f"Features        : {len(features)}")
    print(f"Test size       : {TEST_SIZE}")
    print(f"Random state    : {RANDOM_STATE}")
    print(f"Stratify        : {STRATIFY}")
    print(f"Group column    : {GROUP_COL}")

    merged = pd.read_csv(
        MERGED_CSV,
        low_memory=False,
    )

    missing_targets = [
        target
        for target in targets
        if target not in merged.columns
    ]

    if missing_targets:
        raise SystemExit(
            "The Stage-01 merged dataset is missing configured targets:\n"
            + "\n".join(missing_targets)
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for target in targets:
        try:
            summary = process_target(
                merged=merged,
                target=target,
                features=features,
                metadata_columns=metadata_columns,
            )
            summary_rows.append(summary)

        except Exception as exc:
            failures.append({
                "target": target,
                "error": f"{type(exc).__name__}: {exc}",
            })

            print()
            print(f"[ERROR] Target '{target}' failed:")
            print(f"        {type(exc).__name__}: {exc}")

    # Global configuration copied to Stage 02.
    pipeline_config = {
        "pipeline": "DIGIGARD",
        "stage": 2,
        "stage_name": "create_train_test_splits",
        "input_merged_csv": str(MERGED_CSV),
        "input_contract": str(CONTRACT_JSON),
        "output_root": str(OUTPUT_ROOT),
        "targets": targets,
        "features": features,
        "feature_count": len(features),
        "metadata_columns": metadata_columns,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": STRATIFY,
        "group_col": GROUP_COL,
        "strict_binary_target": STRICT_BINARY_TARGET,
        "drop_missing_target": DROP_MISSING_TARGET,
        "drop_rows_with_all_features_missing":
            DROP_ROWS_WITH_ALL_FEATURES_MISSING,
        "allow_unstratified_fallback":
            ALLOW_UNSTRATIFIED_FALLBACK,
    }

    write_json(
        OUTPUT_ROOT / "pipeline_config.json",
        pipeline_config,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    failures_path = OUTPUT_ROOT / "split_failures.csv"
    pd.DataFrame(
        failures,
        columns=["target", "error"],
    ).to_csv(
        failures_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("DIGIGARD STAGE 02 COMPLETE")
    print("=" * 80)
    print(f"Successful targets: {len(summary_rows)}/{len(targets)}")
    print(f"Failed targets    : {len(failures)}")
    print(f"[OK] Summary      -> {summary_path}")
    print(f"[OK] Failures     -> {failures_path}")
    print(
        f"[OK] Config       -> "
        f"{OUTPUT_ROOT / 'pipeline_config.json'}"
    )

    if failures:
        raise SystemExit(
            "\nOne or more target split jobs failed. "
            "Inspect split_failures.csv before model training."
        )


if __name__ == "__main__":
    main()
