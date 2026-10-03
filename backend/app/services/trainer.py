"""
trainer.py

The orchestration layer for Phase 2: ties together preprocessing (ml/preprocessing.py),
splitting + model selection (ml/classification.py), and evaluation (evaluator.py)
into one run_experiment() call.

Artifact "saving" now means serializing to bytes (for storage in Postgres),
not writing to the local filesystem -- see Experiment.artifact_data.
"""

import io
import time
import uuid
from datetime import datetime, timezone

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.ml.preprocessing import build_preprocessing_pipeline
from app.ml.classification import get_classification_model, split_data
from app.services.evaluator import evaluate_classification


def encode_target(y_train, y_test):
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


def serialize_pipeline(pipeline) -> bytes:
    """
    Serializes a trained pipeline to raw bytes using joblib, entirely in
    memory (no file written to disk) -- these bytes are what gets stored
    directly in the experiments.artifact_data column.
    """
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return buffer.getvalue()


def deserialize_pipeline(data: bytes):
    """The inverse of serialize_pipeline() -- used at predict/explain time."""
    buffer = io.BytesIO(data)
    return joblib.load(buffer)


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
        "label_classes": label_encoder.classes_.tolist(),
        "pipeline": trained_pipeline,
    }