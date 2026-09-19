"""
validator.py

Dataset validation logic — ported from Kaggle prototyping (Steps 5-7).

validate_structure() checks that don't need a target column.
validate_target() checks that need a target column.
validate_dataset() combines both into one rolled-up result.
"""

import pandas as pd

# Tunable thresholds — move to app/core/config.py once that exists (Phase 3+)
HIGH_CARDINALITY_RATIO = 0.5
CLASS_IMBALANCE_THRESHOLD = 0.8


def validate_structure(df: pd.DataFrame) -> list:
    """Step 5: duplicate rows, empty columns, constant columns, high-cardinality columns."""
    results = []

    dup_count = int(df.duplicated().sum())
    results.append({
        "check": "duplicate_rows",
        "status": "ok" if dup_count == 0 else "warning",
        "detail": f"{dup_count} duplicate row(s) found"
    })

    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    results.append({
        "check": "empty_columns",
        "status": "ok" if not empty_cols else "warning",
        "detail": f"Empty columns: {empty_cols}" if empty_cols else "No empty columns"
    })

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) == 1]
    results.append({
        "check": "constant_columns",
        "status": "ok" if not constant_cols else "warning",
        "detail": f"Constant columns: {constant_cols}" if constant_cols else "No constant columns"
    })

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    high_cardinality = [
        col for col in categorical_cols
        if df[col].nunique() > HIGH_CARDINALITY_RATIO * len(df)
    ]
    results.append({
        "check": "high_cardinality_categorical",
        "status": "ok" if not high_cardinality else "warning",
        "detail": f"High-cardinality columns: {high_cardinality}" if high_cardinality else "None found"
    })

    return results


def validate_target(df: pd.DataFrame, target_column: str) -> list:
    """Step 6: missing target values, class imbalance. Requires a chosen target column."""
    results = []

    missing_target = int(df[target_column].isnull().sum())
    results.append({
        "check": "target_missing_values",
        "status": "ok" if missing_target == 0 else "error",
        "detail": f"{missing_target} missing value(s) in target column"
    })

    value_counts = df[target_column].value_counts(normalize=True)
    max_class_ratio = round(float(value_counts.max()), 3)
    results.append({
        "check": "class_imbalance",
        "status": "ok" if max_class_ratio < CLASS_IMBALANCE_THRESHOLD else "warning",
        "detail": f"Largest class makes up {max_class_ratio * 100:.1f}% of the data"
    })

    return results


def validate_dataset(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Step 7: the single entry point.

    target_column=None -> structural checks only (used right after upload).
    target_column set  -> structural + target checks (used after target selection).
    """
    structure_checks = validate_structure(df)

    if target_column is not None:
        target_checks = validate_target(df, target_column)
    else:
        target_checks = []

    all_checks = structure_checks + target_checks
    has_errors = any(check["status"] == "error" for check in all_checks)
    has_warnings = any(check["status"] == "warning" for check in all_checks)

    if has_errors:
        overall_status = "error"
    elif has_warnings:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "overall_status": overall_status,
        "checks": all_checks,
    }
