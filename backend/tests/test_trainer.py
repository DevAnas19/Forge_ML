"""
Tests for trainer.py -- training is slower than profiling/validation, so
we keep the dataset tiny and focus on STRUCTURE (did it produce the right
shape of output, the right keys, sensible types) rather than exact
predicted values, since those can vary slightly across sklearn/xgboost
versions even with a fixed random_seed.
"""

import pandas as pd
from app.services.trainer import run_experiment, encode_target


def make_sample_df():
    """
    A small but genuinely learnable synthetic dataset -- feature values
    correlate clearly with the target, so models can actually learn
    something real rather than just guessing on noise. 20 rows is enough
    for a 80/20 split to produce non-trivial train/test sets.
    """
    return pd.DataFrame({
        "income": [20000, 25000, 30000, 80000, 90000, 100000, 22000, 85000,
                   28000, 95000, 21000, 88000, 31000, 99000, 24000, 92000,
                   29000, 83000, 26000, 97000],
        "category": ["low", "low", "low", "high", "high", "high", "low", "high",
                     "low", "high", "low", "high", "low", "high", "low", "high",
                     "low", "high", "low", "high"],
        "approved": ["N", "N", "N", "Y", "Y", "Y", "N", "Y",
                     "N", "Y", "N", "Y", "N", "Y", "N", "Y",
                     "N", "Y", "N", "Y"],
    })


def test_encode_target_converts_to_binary():
    y_train = pd.Series(["Y", "N", "Y", "N"])
    y_test = pd.Series(["N", "Y"])

    y_train_enc, y_test_enc, encoder = encode_target(y_train, y_test)

    assert set(y_train_enc) <= {0, 1}
    assert set(y_test_enc) <= {0, 1}
    assert list(encoder.classes_) == sorted(encoder.classes_)  # alphabetical: N=0, Y=1


def test_run_experiment_returns_expected_keys():
    df = make_sample_df()
    result = run_experiment(
        df,
        model_name="logistic_regression",
        target_column="approved",
        numerical_cols=["income"],
        categorical_cols=["category"],
    )

    expected_keys = {
        "experiment_id", "model_name", "hyperparameters", "metrics",
        "training_time", "timestamp", "random_seed", "feature_count",
        "status", "label_classes", "pipeline",
    }
    assert expected_keys <= set(result.keys())


def test_run_experiment_status_is_completed():
    df = make_sample_df()
    result = run_experiment(
        df, model_name="random_forest", target_column="approved",
        numerical_cols=["income"], categorical_cols=["category"],
    )
    assert result["status"] == "completed"


def test_run_experiment_metrics_are_in_valid_range():
    """
    Doesn't check for a SPECIFIC accuracy number -- that would make this
    test fragile and fail on tiny, irrelevant changes. Instead checks the
    metrics are at least valid probabilities/scores (between 0 and 1),
    which should always hold regardless of exact model behavior.
    """
    df = make_sample_df()
    result = run_experiment(
        df, model_name="xgboost", target_column="approved",
        numerical_cols=["income"], categorical_cols=["category"],
    )

    metrics = result["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_run_experiment_same_seed_gives_same_result():
    """
    This is the actual test of spec section 29 rule 5: reproducibility.
    Same data, same seed, same everything -- should produce identical
    metrics both times, proving random_seed is actually being respected.
    """
    df = make_sample_df()

    result1 = run_experiment(
        df, model_name="logistic_regression", target_column="approved",
        numerical_cols=["income"], categorical_cols=["category"], random_seed=42,
    )
    result2 = run_experiment(
        df, model_name="logistic_regression", target_column="approved",
        numerical_cols=["income"], categorical_cols=["category"], random_seed=42,
    )

    assert result1["metrics"]["accuracy"] == result2["metrics"]["accuracy"]
    assert result1["metrics"]["f1"] == result2["metrics"]["f1"]


def test_run_experiment_rejects_unknown_model_name():
    df = make_sample_df()
    try:
        run_experiment(
            df, model_name="not_a_real_model", target_column="approved",
            numerical_cols=["income"], categorical_cols=["category"],
        )
        assert False, "Expected a ValueError for an unknown model name"
    except ValueError:
        pass  # this is the expected, correct behavior