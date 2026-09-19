"""
profiler.py

Dataset profiling logic — ported from Kaggle prototyping (Steps 1-4).

Given a pandas DataFrame, produces a JSON-serializable profile:
basic info, column types, and per-column statistics.
"""

import pandas as pd
import numpy as np


def get_basic_info(df: pd.DataFrame) -> dict:
    """Step 1: row/column counts, missing values, duplicates."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_column_types(df: pd.DataFrame) -> dict:
    """Step 2: split columns into numerical vs categorical."""
    numerical = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include="object").columns.tolist()
    return {
        "numerical_columns": numerical,
        "categorical_columns": categorical,
    }


def get_numerical_stats(df: pd.DataFrame, numerical_cols: list) -> dict:
    """Step 3: mean/median/min/max/std/missing for each numerical column."""
    stats = {}
    for col in numerical_cols:
        # float(...) around every value: pandas/numpy return numpy.float64 or
        # numpy.int64 depending on the column's dtype, and neither is a plain
        # Python type. round() alone doesn't fix this -- it just rounds
        # whatever numpy type it was given. FastAPI's JSON encoder only
        # recognizes native Python int/float/str/bool/None/list/dict, so an
        # un-cast numpy type causes a 500 error at response time, even though
        # the exact same code looked fine printed in a notebook.
        mean_val = df[col].mean()
        median_val = df[col].median()
        min_val = df[col].min()
        max_val = df[col].max()
        std_val = df[col].std()

        stats[col] = {
            "mean": round(float(mean_val), 2) if pd.notnull(mean_val) else None,
            "median": round(float(median_val), 2) if pd.notnull(median_val) else None,
            "min": round(float(min_val), 2) if pd.notnull(min_val) else None,
            "max": round(float(max_val), 2) if pd.notnull(max_val) else None,
            "std": round(float(std_val), 2) if pd.notnull(std_val) else None,
            "missing": int(df[col].isnull().sum()),
        }
    return stats


def get_categorical_stats(df: pd.DataFrame, categorical_cols: list) -> dict:
    """Step 3: unique value count + frequency distribution for each categorical column."""
    stats = {}
    for col in categorical_cols:
        # value_counts().to_dict() converts the index (column values) to
        # plain Python via the dict keys, but the *counts* stay numpy.int64.
        # Cast each count explicitly.
        frequency = {
            str(k): int(v) for k, v in df[col].value_counts().to_dict().items()
        }
        stats[col] = {
            "unique_values": int(df[col].nunique()),
            "frequency": frequency,
            "missing": int(df[col].isnull().sum()),
        }
    return stats


def profile_dataset(df: pd.DataFrame, dataset_name: str = "dataset") -> dict:
    """
    Step 4: the single entry point.

    This is what /api/datasets/upload calls directly.
    """
    basic_info = get_basic_info(df)
    col_types = get_column_types(df)
    numerical_stats = get_numerical_stats(df, col_types["numerical_columns"])
    categorical_stats = get_categorical_stats(df, col_types["categorical_columns"])

    return {
        "dataset_name": dataset_name,
        "basic_info": basic_info,
        "column_types": col_types,
        "numerical_stats": numerical_stats,
        "categorical_stats": categorical_stats,
    }