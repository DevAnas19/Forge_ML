"""
main.py

ForgeML backend entry point.

Phase 1: dataset upload -> profile + structural validation.
Phase 2 (current): run one or more classification experiments on an
uploaded dataset -- preprocessing, training, evaluation.

No database yet (Phase 3) -- results are returned directly, not persisted.
Because there's no persisted dataset to reference by ID yet, the experiment
endpoint re-accepts the CSV file directly, alongside the experiment settings.
This will get cleaner once Phase 3 adds Postgres.
"""

from typing import List

from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd

from app.services.profiler import profile_dataset
from app.services.validator import validate_dataset
from app.ml.classification import get_classification_model  # noqa: F401 (keeps model registry importable from here too)
from app.services.trainer import run_experiment

app = FastAPI(title="ForgeML API")


@app.get("/")
def root():
    return {"status": "ForgeML backend running"}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accepts a CSV file, profiles it, and runs structural validation
    (no target column yet -- that comes in a later phase).
    """
    df = pd.read_csv(file.file)

    profile = profile_dataset(df, dataset_name=file.filename)
    validation = validate_dataset(df)  # target_column=None -> structural checks only

    return {
        "profile": profile,
        "validation": validation,
    }


@app.post("/api/experiments")
async def create_experiments(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    numerical_columns: str = Form(...),   # comma-separated, e.g. "ApplicantIncome,LoanAmount"
    categorical_columns: str = Form(...), # comma-separated, e.g. "Gender,Education"
    model_names: str = Form(...),         # comma-separated, e.g. "logistic_regression,random_forest,xgboost"
    test_size: float = Form(0.2),
    random_seed: int = Form(42),
):
    """
    Runs one experiment per requested model on the uploaded dataset and
    returns the comparison list (spec section 10). Trained pipelines stay
    in server memory only for now -- there's no artifact-saving endpoint
    yet, that's the next thing to wire up once this works end to end.
    """
    df = pd.read_csv(file.file)

    numerical_cols = [c.strip() for c in numerical_columns.split(",") if c.strip()]
    categorical_cols = [c.strip() for c in categorical_columns.split(",") if c.strip()]
    requested_models = [m.strip() for m in model_names.split(",") if m.strip()]

    results = []
    for model_name in requested_models:
        result = run_experiment(
            df,
            model_name=model_name,
            target_column=target_column,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            test_size=test_size,
            random_seed=random_seed,
        )
        # Strip the in-memory pipeline object before returning -- it's not
        # JSON-serializable and isn't meant to leave the server anyway.
        result_for_response = {k: v for k, v in result.items() if k != "pipeline"}
        results.append(result_for_response)

    return {"experiments": results}