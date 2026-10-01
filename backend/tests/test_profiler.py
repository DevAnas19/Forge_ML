"""
Tests for profiler.py -- pure functions, no database or network needed,
so these run fast and don't depend on Docker/Postgres being up at all.
"""

import pandas as pd
from app.services.profiler import (
    get_basic_info,
    get_column_types,
    get_numerical_stats,
    get_categorical_stats,
    profile_dataset,
)


def make_sample_df():
    """
    A small, hand-built DataFrame with KNOWN values -- this is the key idea
    behind unit testing: we're not testing on loan.csv where we don't know
    the exact right answer by heart, we're testing on data small enough
    that we can compute the expected result ourselves and check the
    function gets it right.
    """
    return pd.DataFrame({
        "age": [25, 30, 35, None],
        "income": [50000, 60000, 70000, 80000],
        "city": ["Surat", "Mumbai", "Surat", None],
    })


def test_get_basic_info_counts_rows_and_columns():
    df = make_sample_df()
    info = get_basic_info(df)

    assert info["rows"] == 4
    assert info["columns"] == 3
    assert info["missing_values"] == 2  # one None in age, one None in city
    assert info["duplicate_rows"] == 0


def test_get_column_types_splits_correctly():
    df = make_sample_df()
    types = get_column_types(df)

    assert "age" in types["numerical_columns"]
    assert "income" in types["numerical_columns"]
    assert "city" in types["categorical_columns"]
    assert "city" not in types["numerical_columns"]


def test_get_numerical_stats_computes_correct_mean():
    df = make_sample_df()
    stats = get_numerical_stats(df, ["income"])

    # 50000 + 60000 + 70000 + 80000 = 260000, / 4 = 65000 -- known by hand
    assert stats["income"]["mean"] == 65000.0
    assert stats["income"]["missing"] == 0


def test_get_numerical_stats_handles_missing_values():
    df = make_sample_df()
    stats = get_numerical_stats(df, ["age"])

    assert stats["age"]["missing"] == 1
    # mean of 25, 30, 35 (ignoring the None) = 30
    assert stats["age"]["mean"] == 30.0


def test_get_categorical_stats_counts_frequencies():
    df = make_sample_df()
    stats = get_categorical_stats(df, ["city"])

    assert stats["city"]["unique_values"] == 2  # Surat, Mumbai
    assert stats["city"]["frequency"]["Surat"] == 2
    assert stats["city"]["missing"] == 1


def test_profile_dataset_returns_all_expected_keys():
    df = make_sample_df()
    profile = profile_dataset(df, dataset_name="test.csv")

    assert profile["dataset_name"] == "test.csv"
    assert "basic_info" in profile
    assert "column_types" in profile
    assert "numerical_stats" in profile
    assert "categorical_stats" in profile


def test_profile_dataset_output_is_json_serializable():
    """
    This test exists BECAUSE of the real bug you hit in Phase 1 -- numpy
    types silently breaking JSON serialization. This test would have
    caught that bug automatically, before it ever reached a live request.
    """
    import json
    df = make_sample_df()
    profile = profile_dataset(df, dataset_name="test.csv")

    # If anything in here is a numpy type instead of a native Python type,
    # this line throws TypeError -- exactly the error you saw in production.
    json.dumps(profile)