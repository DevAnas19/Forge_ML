"""
Tests for validator.py -- same approach as test_profiler.py: small,
hand-built data where we know the "correct" answer, not loan.csv.
"""

import pandas as pd
from app.services.validator import validate_structure, validate_target, validate_dataset


def test_validate_structure_detects_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    checks = validate_structure(df)

    dup_check = next(c for c in checks if c["check"] == "duplicate_rows")
    assert dup_check["status"] == "warning"
    assert "1 duplicate" in dup_check["detail"]


def test_validate_structure_detects_empty_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    checks = validate_structure(df)

    empty_check = next(c for c in checks if c["check"] == "empty_columns")
    assert empty_check["status"] == "warning"
    assert "b" in empty_check["detail"]


def test_validate_structure_detects_constant_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
    checks = validate_structure(df)

    constant_check = next(c for c in checks if c["check"] == "constant_columns")
    assert constant_check["status"] == "warning"
    assert "b" in constant_check["detail"]


def test_validate_structure_flags_high_cardinality_id_column():
    # 10 rows, 10 unique string values -- classic ID-column pattern,
    # the exact kind of thing that flagged Loan_ID in your real dataset
    df = pd.DataFrame({
        "id": [f"ID{i}" for i in range(10)],
        "category": ["A"] * 10,
    })
    checks = validate_structure(df)

    cardinality_check = next(c for c in checks if c["check"] == "high_cardinality_categorical")
    assert cardinality_check["status"] == "warning"
    assert "id" in cardinality_check["detail"]


def test_validate_structure_passes_clean_data():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "b": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],  # repeated values, not all-unique
    })
    checks = validate_structure(df)

    assert all(c["status"] == "ok" for c in checks)


def test_validate_target_detects_missing_values():
    df = pd.DataFrame({"target": ["Y", "N", None, "Y"]})
    checks = validate_target(df, "target")

    missing_check = next(c for c in checks if c["check"] == "target_missing_values")
    assert missing_check["status"] == "error"
    assert "1 missing" in missing_check["detail"]


def test_validate_target_detects_class_imbalance():
    # 9 "Y" out of 10 -- 90%, above the 0.8 threshold, should warn
    df = pd.DataFrame({"target": ["Y"] * 9 + ["N"]})
    checks = validate_target(df, "target")

    imbalance_check = next(c for c in checks if c["check"] == "class_imbalance")
    assert imbalance_check["status"] == "warning"


def test_validate_target_accepts_balanced_classes():
    df = pd.DataFrame({"target": ["Y"] * 5 + ["N"] * 5})
    checks = validate_target(df, "target")

    imbalance_check = next(c for c in checks if c["check"] == "class_imbalance")
    assert imbalance_check["status"] == "ok"


def test_validate_dataset_without_target_only_runs_structural_checks():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = validate_dataset(df, target_column=None)

    check_names = [c["check"] for c in result["checks"]]
    assert "target_missing_values" not in check_names
    assert "duplicate_rows" in check_names


def test_validate_dataset_overall_status_is_error_when_target_has_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3], "target": ["Y", None, "N"]})
    result = validate_dataset(df, target_column="target")

    assert result["overall_status"] == "error"