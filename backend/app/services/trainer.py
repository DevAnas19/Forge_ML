"""
trainer.py

The orchestration layer for Phase 2: ties together preprocessing (ml/preprocessing.py),
splitting + model selection (ml/classification.py), and evaluation (evaluator.py)
into one run_experiment() call, plus artifact saving.

This mirrors profiler.py/validator.py's role in Phase 1 -- one clean entry
point that main.py's endpoint calls.
"""

import os
import time
import uuid
from datetime import datetime, timezone

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.ml.preprocessing import build_preprocessing_pipeline
from app.ml.classification import get_classification_model, split_data
from app.services.evaluator import evaluate_classification

ARTIFACTS_DIR = "artifacts"


def encode_target(y_train, y_test):
    """
    Encodes a text target ("Y"/"N", "Approved"/"Rejected", etc.) to 0/1.

    Using LabelEncoder instead of a hardcoded {"N": 0, "Y": 1} map (like the
    Kaggle version had) makes this work for ANY two-class target, not just
    this one dataset -- important since ForgeML is meant to handle datasets
    the user uploads, not just loan.csv.

    Returns the encoded labels plus the encoder itself, so predictions can
    later be decoded back to the original text labels.
    """
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, encoder


def train_model(preprocessor, model, X_train, y_train):
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    full_pipeline.fit(X_train, y_train)
    return full_pipeline


def run_experiment(
    df,
    model_name: str,
    target_column: str,
    numerical_cols: list,
    categorical_cols: list,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    feature_cols = numerical_cols + categorical_cols

    X_train, X_test, y_train, y_test = split_data(
        df, target_column, feature_cols, test_size, random_seed
    )

    y_train_encoded, y_test_encoded, label_encoder = encode_target(y_train, y_test)

    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    model = get_classification_model(model_name)

    start_time = time.time()
    trained_pipeline = train_model(preprocessor, model, X_train, y_train_encoded)
    training_time = round(time.time() - start_time, 3)

    metrics = evaluate_classification(trained_pipeline, X_test, y_test_encoded)

    return {
        "experiment_id": str(uuid.uuid4()),
        "model_name": model_name,
        "hyperparameters": {k: str(v) for k, v in model.get_params().items()},
        "metrics": metrics,
        "training_time": training_time,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "feature_count": len(feature_cols),
        "status": "completed",
        "label_classes": label_encoder.classes_.tolist(),  # e.g. ["N", "Y"] -- 0 maps to classes_[0]
        "pipeline": trained_pipeline,  # kept in memory, stripped before the API response
    }


def save_model_artifact(pipeline, model_name: str, experiment_id: str) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    filename = f"{safe_name}_{experiment_id}.pkl"
    artifact_path = os.path.join(ARTIFACTS_DIR, filename)
    joblib.dump(pipeline, artifact_path)
    return artifact_path