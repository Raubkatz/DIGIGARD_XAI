#!/usr/bin/env python3
"""
01_DIGIGARD_merge_and_clean_project_histories.py

DIGIGARD Stage 01: merge + light-clean all project-history CSV files.

Input
-----
DATA_ESEIW_final/*.csv

Each CSV is treated as one project/repository history.

This stage deliberately does NOT choose one prediction target yet. Instead it keeps
all three allowed ground-truth columns so later stages can run the same pipeline for
any of them.

Outputs
-------
DIGIGARD_01_merged_data/
    DIGIGARD_merged_dataset.csv
    DIGIGARD_feature_target_contract.json
    DIGIGARD_features.json
    DIGIGARD_targets.json
    DIGIGARD_merge_report.json
    DIGIGARD_file_summary.csv

Important
---------
- FEATURES below contains only the active/non-commented features requested for the
  new DIGIGARD pipeline.
- TARGETS contains all three possible ground truths.
- Project/file identifiers are retained as METADATA only. They are NOT model features.
- Missing requested columns in an individual project are added as NaN and reported.
- No rows are removed because of negative feature values at this stage.
- No duplicate rows are removed at this stage.
- Targets are not filtered to 0/1 here; unexpected values are reported for inspection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR = Path("DATA_ESEIW")
OUTPUT_DIR = Path("DIGIGARD_01_merged_data")
OUTPUT_CSV_NAME = "DIGIGARD_merged_dataset.csv"

PATTERN = "*.csv"
RECURSIVE = False

# Light-cleaning controls.
DROP_COMPLETELY_EMPTY_ROWS = True
REMOVE_UNNAMED_COLUMNS = True
NORMALIZE_BLANK_STRINGS = True
COERCE_FEATURES_AND_TARGETS_TO_NUMERIC = True
REPLACE_POS_NEG_INFINITY_WITH_NAN = True

# If True, stop immediately when any requested feature/target is missing from a
# project CSV. Recommended default is False because the merge report should expose
# schema differences without preventing inspection of the complete dataset.
STRICT_REQUIRED_SCHEMA = False

# Optional Parquet copy. CSV remains the canonical Stage-01 output.
WRITE_PARQUET = False


# =============================================================================
# TARGET CONTRACT
# =============================================================================

TARGETS: List[str] = [
    "isBugPresent",
    "isBugfix",
    "isSZZBugIntroducer",
]


# =============================================================================
# FEATURE CONTRACT
# Only the ACTIVE / NON-COMMENTED features from the requested selection.
# =============================================================================

FEATURES: List[str] = [
    # Bug and refactor counts
    "isRefactor",
    "totalRefactors",

    # DOK core values
    "dok0.004AuthorPreCommit",
    "dok0.004AuthorPostCommit",
    "dok0.004Sum",
    "dok0.004Avg",
    "dok0.004Median",
    "dok0.004Std",
    "dok0.004Min",
    "dok0.004Max",

    # DOK pre-commit statistics
    "dok0.004PreCommitSum",
    "dok0.004PreCommitAvg",
    "dok0.004PreCommitMedian",
    "dok0.004PreCommitMin",
    "dok0.004PreCommitMax",

    # DOK relative statistics
    "dok0.004RelativeAvg",
    "dok0.004RelativeMedian",
    "dok0.004RelativeMin",
    "dok0.004RelativeMax",

    # File Bus Factor
    "dok0.004FileBusFactor0.2",

    # Project Bus Factor
    "dok0.004ProjectBusFactor0.2",

    # Revision metadata
    "revision",
    "age",
    "authors",
    "totalAuthors",

    # LOC metrics
    "totalLoc",
    "locFirst",
    "locLast",
    "locLast-First",

    # Delta totals
    "deltaTotalAddedToLast",
    "deltaTotalRemovedToLast",
    "deltaTotalReplacedToLast",

    # Current change
    "added",
    "removed",
    "replaced",
    "modified",

    # Relative change
    "relativeAdded",
    "relativeRemoved",
    "relativeReplaced",
    "relativeModified",

    # LOC deltas (temporal)
    "deltaLocToLast",
    "deltaLocToPrev",
    "deltaLocToNext",

    "deltaLocAddedToPrev",
    "deltaLocRemovedToPrev",
    "deltaLocReplacedToPrev",
    "deltaLocModifiedToPrev",

    "deltaLocAddedToNext",
    "deltaLocRemovedToNext",
    "deltaLocReplacedToNext",
    "deltaLocModifiedToNext",

    # 180-day history metrics
    "180hcpf1",
    "180hcpf2",
    "180fileHcpf",
    "180commitHcpf",
    "180hcm1",
    "180hcm2",
    "180edhcm",
    "180ldhcm",
    "180lgdhcm",

    # 90-day history metrics
    "90hcpf1",
    "90hcpf2",
    "90fileHcpf",
    "90commitHcpf",
    "90hcm1",
    "90hcm2",
    "90edhcm",
    "90ldhcm",
    "90lgdhcm",

    # Risk & complexity
    "nrOfFunctions",
    "ccsSum",
    "ccsAvg",
    "ccsMed",
    "ccsMin",
    "ccsMax",
    "ccsStd",

    # Dependency
    "dependencyChanged",
    "dependencyChangedPrev5Commits",
]


# =============================================================================
# METADATA CONTRACT
# These columns are retained for project/file/history reconstruction and later
# analyses, but they are explicitly NOT part of FEATURES.
# =============================================================================

# Always generated from the input filename.
GENERATED_METADATA_COLUMNS: List[str] = [
    "source_project",
    "source_file",
]

# Retained from a project CSV when present. Missing values are filled with NaN so
# the merged schema remains stable across repositories.
PRESERVED_METADATA_COLUMNS: List[str] = [
    "project",
    "hash",
    "date",
    "fileId",
    "fileName",
]

METADATA_COLUMNS: List[str] = (
    GENERATED_METADATA_COLUMNS + PRESERVED_METADATA_COLUMNS
)

REQUIRED_SELECTED_COLUMNS: List[str] = FEATURES + TARGETS
FINAL_COLUMN_ORDER: List[str] = METADATA_COLUMNS + FEATURES + TARGETS


# =============================================================================
# BASIC VALIDATION OF THIS SCRIPT'S CONTRACT
# =============================================================================

def validate_contract() -> None:
    """Fail early if the configured feature/target contract itself is invalid."""

    def duplicates(values: List[str]) -> List[str]:
        seen = set()
        dupes = []
        for value in values:
            if value in seen and value not in dupes:
                dupes.append(value)
            seen.add(value)
        return dupes

    feature_dupes = duplicates(FEATURES)
    target_dupes = duplicates(TARGETS)
    metadata_dupes = duplicates(METADATA_COLUMNS)

    if feature_dupes:
        raise ValueError(f"Duplicate feature names in FEATURES: {feature_dupes}")
    if target_dupes:
        raise ValueError(f"Duplicate target names in TARGETS: {target_dupes}")
    if metadata_dupes:
        raise ValueError(f"Duplicate metadata names: {metadata_dupes}")

    overlap_ft = sorted(set(FEATURES) & set(TARGETS))
    overlap_fm = sorted(set(FEATURES) & set(METADATA_COLUMNS))
    overlap_tm = sorted(set(TARGETS) & set(METADATA_COLUMNS))

    if overlap_ft or overlap_fm or overlap_tm:
        raise ValueError(
            "Feature/target/metadata contracts overlap:\n"
            f"FEATURES ∩ TARGETS = {overlap_ft}\n"
            f"FEATURES ∩ METADATA = {overlap_fm}\n"
            f"TARGETS ∩ METADATA = {overlap_tm}"
        )


# =============================================================================
# CSV READING
# =============================================================================

def sniff_delimiter(path: Path, sample_bytes: int = 64_000) -> str:
    """Try to detect common CSV delimiters; fall back to comma."""
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            sample = handle.read(sample_bytes)

        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=[",", ";", "\t", "|"],
        )
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_flex(path: Path) -> Tuple[pd.DataFrame, str, str]:
    """
    Read one project CSV with delimiter detection and a small encoding fallback.

    Returns
    -------
    dataframe, delimiter, encoding
    """
    delimiter = sniff_delimiter(path)
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                low_memory=False,
                encoding=encoding,
            )
            return df, delimiter, encoding
        except Exception as exc:
            last_error = exc

    # Final fallback: Python parser can tolerate some malformed CSV layouts better.
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            low_memory=False,
            encoding="utf-8",
            engine="python",
            on_bad_lines="warn",
        )
        return df, delimiter, "utf-8/python-engine"
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Could not read CSV {path}: {last_error}")


# =============================================================================
# LIGHT CLEANING
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding whitespace from headers and reject duplicate names."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            "Duplicate column names after header cleanup: "
            + ", ".join(map(str, duplicated))
        )

    return df


def drop_unnamed_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop common accidental pandas index columns such as 'Unnamed: 0'."""
    if not REMOVE_UNNAMED_COLUMNS:
        return df, []

    unnamed = [
        column
        for column in df.columns
        if str(column).strip().lower().startswith("unnamed:")
    ]

    if unnamed:
        df = df.drop(columns=unnamed)

    return df, unnamed


def to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Robust conversion for selected features/targets.

    Handles:
    - ordinary numeric dtypes
    - surrounding whitespace
    - decimal commas
    - common boolean strings
    - common textual missing-value markers
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()
    lower = text.str.lower()

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

    normalized = lower.replace(replacements)
    normalized = normalized.str.replace(",", ".", regex=False)

    return pd.to_numeric(normalized, errors="coerce")


def target_value_summary(series: pd.Series) -> Dict[str, Any]:
    """Return target distribution plus non-binary values without changing them."""
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()

    counts = non_missing.value_counts(dropna=False).sort_index()
    distribution = {
        str(key): int(value)
        for key, value in counts.items()
    }

    unexpected = sorted(
        float(value)
        for value in pd.unique(non_missing)
        if float(value) not in {0.0, 1.0}
    )

    return {
        "missing": int(numeric.isna().sum()),
        "non_missing": int(non_missing.shape[0]),
        "distribution": distribution,
        "unexpected_non_binary_values": unexpected,
    }


def clean_and_align_project(
    df: pd.DataFrame,
    source_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Light-clean one project and align it to the DIGIGARD Stage-01 schema."""
    rows_original = int(len(df))
    cols_original = list(map(str, df.columns))

    df = clean_column_names(df)
    df, removed_unnamed = drop_unnamed_columns(df)

    if NORMALIZE_BLANK_STRINGS:
        # Replace strings containing only whitespace with NaN.
        df = df.replace(r"^\s*$", np.nan, regex=True)

    if DROP_COMPLETELY_EMPTY_ROWS:
        df = df.dropna(how="all").copy()

    # Capture schema before adding missing requested columns.
    current_columns = set(df.columns)

    missing_features = [c for c in FEATURES if c not in current_columns]
    missing_targets = [c for c in TARGETS if c not in current_columns]
    missing_metadata = [
        c for c in PRESERVED_METADATA_COLUMNS if c not in current_columns
    ]

    if STRICT_REQUIRED_SCHEMA and (missing_features or missing_targets):
        raise ValueError(
            f"{source_path.name}: missing required selected columns.\n"
            f"Missing features: {missing_features}\n"
            f"Missing targets: {missing_targets}"
        )

    # Add absent selected columns as NaN so every project receives the same schema.
    for column in FEATURES + TARGETS + PRESERVED_METADATA_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    # Source identifiers are generated after schema alignment so they cannot be
    # discarded accidentally.
    df["source_project"] = source_path.stem
    df["source_file"] = source_path.name

    # Convert only model features and possible targets to numeric. Metadata remains
    # untouched and therefore cannot accidentally enter the model contract.
    coercion_to_nan: Dict[str, int] = {}

    if COERCE_FEATURES_AND_TARGETS_TO_NUMERIC:
        for column in FEATURES + TARGETS:
            before_non_missing = int(df[column].notna().sum())
            converted = to_numeric_series(df[column])
            after_non_missing = int(converted.notna().sum())
            coercion_to_nan[column] = max(
                0,
                before_non_missing - after_non_missing,
            )
            df[column] = converted

    if REPLACE_POS_NEG_INFINITY_WITH_NAN:
        df[FEATURES + TARGETS] = df[FEATURES + TARGETS].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Extra columns are reported but deliberately excluded from the merged output.
    allowed_input_columns = set(
        FEATURES + TARGETS + PRESERVED_METADATA_COLUMNS
    )
    extra_columns = sorted(
        column
        for column in df.columns
        if column not in allowed_input_columns
        and column not in GENERATED_METADATA_COLUMNS
    )

    # Final, explicit column contract.
    df = df[FINAL_COLUMN_ORDER].copy()

    target_summaries = {
        target: target_value_summary(df[target])
        for target in TARGETS
    }

    report: Dict[str, Any] = {
        "source_file": source_path.name,
        "source_project": source_path.stem,
        "rows_original": rows_original,
        "rows_after_light_cleaning": int(len(df)),
        "original_column_count": int(len(cols_original)),
        "final_column_count": int(len(df.columns)),
        "removed_unnamed_columns": removed_unnamed,
        "missing_features": missing_features,
        "missing_targets": missing_targets,
        "missing_preserved_metadata": missing_metadata,
        "extra_columns_not_kept": extra_columns,
        "numeric_coercions_to_nan": {
            key: value
            for key, value in coercion_to_nan.items()
            if value > 0
        },
        "target_summary": target_summaries,
    }

    return df, report


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def discover_input_files() -> List[Path]:
    if RECURSIVE:
        return sorted(INPUT_DIR.rglob(PATTERN))
    return sorted(INPUT_DIR.glob(PATTERN))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    validate_contract()

    if not INPUT_DIR.exists():
        raise SystemExit(
            f"Input directory does not exist: {INPUT_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files()
    if not input_files:
        raise SystemExit(
            f"No CSV files matched {PATTERN!r} in {INPUT_DIR.resolve()}"
        )

    print("=" * 80)
    print("DIGIGARD STAGE 01 - MERGE + LIGHT CLEAN PROJECT HISTORIES")
    print("=" * 80)
    print(f"Input directory : {INPUT_DIR.resolve()}")
    print(f"Project CSVs    : {len(input_files)}")
    print(f"Active features : {len(FEATURES)}")
    print(f"Possible targets: {TARGETS}")
    print()

    merged_chunks: List[pd.DataFrame] = []
    per_file_reports: List[Dict[str, Any]] = []

    for index, path in enumerate(input_files, start=1):
        print(f"[{index:>3}/{len(input_files)}] {path.name}")

        raw_df, delimiter, encoding = read_csv_flex(path)
        cleaned_df, report = clean_and_align_project(raw_df, path)

        report["detected_delimiter"] = delimiter
        report["encoding_used"] = encoding

        merged_chunks.append(cleaned_df)
        per_file_reports.append(report)

        print(
            f"      rows={len(cleaned_df):,} | "
            f"missing_features={len(report['missing_features'])} | "
            f"missing_targets={len(report['missing_targets'])} | "
            f"extras_dropped={len(report['extra_columns_not_kept'])}"
        )

    merged = pd.concat(
        merged_chunks,
        ignore_index=True,
        sort=False,
    )

    # Final defensive enforcement of the exact Stage-01 schema.
    merged = merged[FINAL_COLUMN_ORDER].copy()

    output_csv = OUTPUT_DIR / OUTPUT_CSV_NAME
    merged.to_csv(output_csv, index=False)

    # Reusable contracts for every subsequent DIGIGARD stage.
    features_path = OUTPUT_DIR / "DIGIGARD_features.json"
    targets_path = OUTPUT_DIR / "DIGIGARD_targets.json"
    contract_path = OUTPUT_DIR / "DIGIGARD_feature_target_contract.json"

    write_json(features_path, FEATURES)
    write_json(targets_path, TARGETS)

    contract = {
        "pipeline": "DIGIGARD",
        "stage": 1,
        "stage_name": "merge_and_light_clean_project_histories",
        "input_directory": str(INPUT_DIR),
        "merged_csv": str(output_csv),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "target_count": len(TARGETS),
        "targets": TARGETS,
        "generated_metadata_columns": GENERATED_METADATA_COLUMNS,
        "preserved_metadata_columns": PRESERVED_METADATA_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "final_column_order": FINAL_COLUMN_ORDER,
        "notes": {
            "metadata_are_model_features": False,
            "negative_values_removed": False,
            "duplicate_rows_removed": False,
            "targets_filtered_to_binary": False,
            "missing_selected_columns_filled_with_nan": True,
        },
    }
    write_json(contract_path, contract)

    # Compact per-project CSV summary.
    file_summary_rows: List[Dict[str, Any]] = []
    for report in per_file_reports:
        row: Dict[str, Any] = {
            "source_project": report["source_project"],
            "source_file": report["source_file"],
            "rows_original": report["rows_original"],
            "rows_after_light_cleaning": report["rows_after_light_cleaning"],
            "missing_feature_count": len(report["missing_features"]),
            "missing_target_count": len(report["missing_targets"]),
            "extra_column_count": len(report["extra_columns_not_kept"]),
            "numeric_coercion_problem_count": sum(
                report["numeric_coercions_to_nan"].values()
            ),
            "detected_delimiter": report["detected_delimiter"],
            "encoding_used": report["encoding_used"],
        }

        for target in TARGETS:
            target_info = report["target_summary"][target]
            row[f"{target}__missing"] = target_info["missing"]
            row[f"{target}__count_0"] = target_info["distribution"].get(
                "0.0",
                target_info["distribution"].get("0", 0),
            )
            row[f"{target}__count_1"] = target_info["distribution"].get(
                "1.0",
                target_info["distribution"].get("1", 0),
            )
            row[f"{target}__unexpected_values"] = json.dumps(
                target_info["unexpected_non_binary_values"]
            )

        file_summary_rows.append(row)

    file_summary_df = pd.DataFrame(file_summary_rows)
    file_summary_path = OUTPUT_DIR / "DIGIGARD_file_summary.csv"
    file_summary_df.to_csv(file_summary_path, index=False)

    # Full machine-readable merge report.
    global_target_summary = {
        target: target_value_summary(merged[target])
        for target in TARGETS
    }

    report_path = OUTPUT_DIR / "DIGIGARD_merge_report.json"
    merge_report = {
        "pipeline": "DIGIGARD",
        "stage": 1,
        "input_directory": str(INPUT_DIR.resolve()),
        "output_directory": str(OUTPUT_DIR.resolve()),
        "merged_csv": str(output_csv.resolve()),
        "file_count": len(input_files),
        "total_rows": int(len(merged)),
        "final_column_count": int(len(merged.columns)),
        "feature_count": len(FEATURES),
        "target_count": len(TARGETS),
        "metadata_column_count": len(METADATA_COLUMNS),
        "final_columns": list(merged.columns),
        "global_target_summary": global_target_summary,
        "per_file": per_file_reports,
    }
    write_json(report_path, merge_report)

    if WRITE_PARQUET:
        parquet_path = output_csv.with_suffix(".parquet")
        try:
            merged.to_parquet(parquet_path, index=False)
            print(f"[OK] Parquet copy       -> {parquet_path}")
        except Exception as exc:
            print(f"[WARN] Could not write Parquet: {exc}")

    print()
    print("=" * 80)
    print("STAGE 01 COMPLETE")
    print("=" * 80)
    print(f"Merged projects : {len(input_files)}")
    print(f"Merged rows     : {len(merged):,}")
    print(f"Features        : {len(FEATURES)}")
    print(f"Targets         : {len(TARGETS)}")
    print(f"Final columns   : {len(merged.columns)}")
    print()
    print(f"[OK] Merged CSV        -> {output_csv}")
    print(f"[OK] Feature contract  -> {features_path}")
    print(f"[OK] Target contract   -> {targets_path}")
    print(f"[OK] Full contract     -> {contract_path}")
    print(f"[OK] File summary      -> {file_summary_path}")
    print(f"[OK] Merge report      -> {report_path}")


if __name__ == "__main__":
    main()
