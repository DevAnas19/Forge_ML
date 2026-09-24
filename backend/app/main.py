"""
main.py

ForgeML backend entry point.

Phase 1: dataset upload -> profile + structural validation.
Phase 2: run classification experiments -- preprocessing, training, evaluation.
Phase 3: persistence -- datasets and experiments saved to Postgres.
Phase 4 (current): model registry + prediction API. Every trained pipeline
is now saved to disk automatically; registering a model just points a
database row at an artifact that already exists.
"""

import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body
from sqlalchemy import text
from sqlalchemy.orm import Session
import pandas as pd
import joblib

from app.core.database import engine, Base, get_db
import app.models
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.model_registry import RegisteredModel
from app.core.schemas import ExperimentRequest, RegisterModelRequest

from app.services.profiler import profile_dataset
from app.services.validator import validate_dataset
from app.services.trainer import run_experiment, save_model_artifact, build_artifact_path
from app.services.explainer import compute_global_importance, explain_prediction

from app.services.llm import analyze_dataset_profile
from app.core.schemas import DatasetAnalysisResponse
from pydantic import ValidationError

from app.services.llm import analyze_dataset_profile, create_experiment_plan, VALID_MODEL_NAMES
from app.core.schemas import ExperimentPlanRequest, ExperimentPlanResponse

from app.services.llm import analyze_experiment_results

DATASET_STORAGE_DIR = "storage/datasets"

app = FastAPI(title="ForgeML API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # the Next.js dev server -- tighten this to your real domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        columns = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = 'experiments'")
        ).scalars().all()
        if "label_classes" not in columns:
            conn.execute(text("ALTER TABLE experiments ADD COLUMN label_classes JSONB"))


@app.get("/")
def root():
    return {"status": "ForgeML backend running"}


def serialize_experiment(exp: Experiment) -> dict:
    return {
        "experiment_id": str(exp.id),
        "dataset_id": str(exp.dataset_id),
        "model_name": exp.model_name,
        "hyperparameters": exp.parameters,
        "metrics": exp.metrics,
        "label_classes": getattr(exp, "label_classes", None),
        "status": exp.status,
        "training_time": exp.training_time,
        "timestamp": exp.created_at.isoformat() if exp.created_at else None,
    }


def serialize_dataset(ds: Dataset) -> dict:
    return {
        "dataset_id": str(ds.id),
        "name": ds.name,
        "rows": ds.rows,
        "columns": ds.columns,
        "target_column": ds.target_column,
        "task_type": ds.task_type,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }


def serialize_model(m: RegisteredModel) -> dict:
    return {
        "model_id": str(m.id),
        "experiment_id": str(m.experiment_id),
        "name": m.name,
        "version": m.version,
        "artifact_path": m.artifact_path,
        "status": m.status,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_csv(file.file)

    profile = profile_dataset(df, dataset_name=file.filename)
    validation = validate_dataset(df)

    os.makedirs(DATASET_STORAGE_DIR, exist_ok=True)
    dataset_id = uuid.uuid4()
    file_path = os.path.join(DATASET_STORAGE_DIR, f"{dataset_id}.csv")
    df.to_csv(file_path, index=False)

    dataset_row = Dataset(
        id=dataset_id,
        name=file.filename,
        file_path=file_path,
        rows=profile["basic_info"]["rows"],
        columns=profile["basic_info"]["columns"],
    )
    db.add(dataset_row)
    db.commit()
    db.refresh(dataset_row)

    return {
        "dataset": serialize_dataset(dataset_row),
        "profile": profile,
        "validation": validation,
    }


@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [serialize_dataset(d) for d in datasets]


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.file_path)
    profile = profile_dataset(df, dataset_name=dataset.name)

    return {
        "dataset": serialize_dataset(dataset),
        "profile": profile,
    }


@app.post("/api/experiments")
def create_experiments(request: ExperimentRequest, db: Session = Depends(get_db)):
    try:
        dataset_uuid = uuid.UUID(request.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.file_path)

    dataset.target_column = request.target_column
    dataset.task_type = "classification"
    db.commit()

    results = []
    for model_name in request.model_names:
        result = run_experiment(
            df,
            model_name=model_name,
            target_column=request.target_column,
            numerical_cols=request.numerical_columns,
            categorical_cols=request.categorical_columns,
            test_size=request.test_size,
            random_seed=request.random_seed,
        )

        experiment_row = Experiment(
            dataset_id=dataset.id,
            model_name=result["model_name"],
            parameters=result["hyperparameters"],
            metrics=result["metrics"],
            label_classes=result["label_classes"],
            status=result["status"],
            training_time=result["training_time"],
        )
        db.add(experiment_row)
        db.commit()
        db.refresh(experiment_row)

        # Save the trained pipeline to disk NOW, using the real database
        # experiment_id -- so registering this model later doesn't require
        # retraining, it just points at a file that already exists.
        save_model_artifact(result["pipeline"], result["model_name"], str(experiment_row.id))

        results.append(serialize_experiment(experiment_row))

    return {"experiments": results}


@app.get("/api/experiments")
def list_experiments(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    return [serialize_experiment(e) for e in experiments]


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    try:
        experiment_uuid = uuid.UUID(experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment_id format")

    experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return serialize_experiment(experiment)


@app.post("/api/models/register")
def register_model(request: RegisterModelRequest, db: Session = Depends(get_db)):
    try:
        experiment_uuid = uuid.UUID(request.experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment_id format")

    experiment = db.query(Experiment).filter(Experiment.id == experiment_uuid).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    artifact_path = build_artifact_path(experiment.model_name, str(experiment.id))
    if not os.path.exists(artifact_path):
        raise HTTPException(
            status_code=404,
            detail="Trained artifact not found on disk for this experiment -- it may need to be retrained."
        )

    # Simple auto-versioning: count how many times this model name has
    # already been registered, and call this one the next version.
    existing_count = db.query(RegisteredModel).filter(RegisteredModel.name == experiment.model_name).count()
    version = f"v{existing_count + 1}"

    model_row = RegisteredModel(
        experiment_id=experiment.id,
        name=experiment.model_name,
        version=version,
        artifact_path=artifact_path,
        status="registered",
    )
    db.add(model_row)
    db.commit()
    db.refresh(model_row)

    return serialize_model(model_row)


@app.get("/api/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(RegisteredModel).order_by(RegisteredModel.created_at.desc()).all()
    return [serialize_model(m) for m in models]


@app.get("/api/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return serialize_model(model)


@app.post("/api/models/{model_id}/predict")
def predict(model_id: str, payload: dict = Body(...), db: Session = Depends(get_db)):
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    experiment = db.query(Experiment).filter(Experiment.id == model.experiment_id).first()
    label_classes = getattr(experiment, "label_classes", None) if experiment else None
    if not experiment or not label_classes:
        raise HTTPException(status_code=500, detail="Label classes missing for this model's experiment")

    if not os.path.exists(model.artifact_path):
        raise HTTPException(status_code=500, detail="Model artifact file is missing from disk")

    pipeline = joblib.load(model.artifact_path)

    input_df = pd.DataFrame([payload])

    try:
        predicted_class_index = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature in input: {e}")

    predicted_label = label_classes[predicted_class_index]
    predicted_probability = round(float(probabilities[predicted_class_index]), 4)

    return {
        "prediction": predicted_label,
        "probability": predicted_probability,
    }

@app.get("/api/models/{model_id}/explain")
def explain_model_global(model_id: str, db: Session = Depends(get_db)):
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    experiment = db.query(Experiment).filter(Experiment.id == model.experiment_id).first()
    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()

    if not os.path.exists(model.artifact_path):
        raise HTTPException(status_code=500, detail="Model artifact file is missing from disk")

    pipeline = joblib.load(model.artifact_path)
    raw_df = pd.read_csv(dataset.file_path)

    return compute_global_importance(pipeline, model.name, raw_df)


@app.post("/api/models/{model_id}/explain")
def explain_model_prediction(model_id: str, payload: dict = Body(...), db: Session = Depends(get_db)):
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    experiment = db.query(Experiment).filter(Experiment.id == model.experiment_id).first()
    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()

    if not os.path.exists(model.artifact_path):
        raise HTTPException(status_code=500, detail="Model artifact file is missing from disk")

    pipeline = joblib.load(model.artifact_path)
    raw_df = pd.read_csv(dataset.file_path)
    input_row_df = pd.DataFrame([payload])

    try:
        result = explain_prediction(pipeline, model.name, input_row_df, raw_df)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature in input: {e}")

    return result

@app.post("/api/assistant/analyze-dataset")
def analyze_dataset_endpoint(payload: dict = Body(...), db: Session = Depends(get_db)):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.file_path)
    profile = profile_dataset(df, dataset_name=dataset.name)

    try:
        raw_result = analyze_dataset_profile(profile)
        validated = DatasetAnalysisResponse(**raw_result)
    except Exception as e:
        # Covers both a malformed JSON response from the LLM AND a
        # response that IS valid JSON but doesn't match our expected
        # shape (spec section 17: never trust LLM output blindly --
        # validate it before it reaches the user).
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned an unusable response: {str(e)}"
        )

    return validated.model_dump()

@app.post("/api/assistant/create-plan")
def create_plan_endpoint(request: ExperimentPlanRequest, db: Session = Depends(get_db)):
    try:
        dataset_uuid = uuid.UUID(request.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")

    dataset = db.query(Dataset).filter(Dataset.id == dataset_uuid).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.file_path)
    profile = profile_dataset(df, dataset_name=dataset.name)

    try:
        raw_plan = create_experiment_plan(request.goal, profile)
        plan = ExperimentPlanResponse(**raw_plan)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned an unusable plan: {str(e)}")

    # Structural validation (right shape) is NOT the same as semantic
    # validation (right VALUES). A plan can be perfectly valid JSON matching
    # our schema and still name a target column that doesn't exist in this
    # dataset, or a model we don't actually support -- so check both,
    # separately, before this plan is trusted enough to return.
    all_columns = (
        profile["column_types"]["numerical_columns"]
        + profile["column_types"]["categorical_columns"]
    )
    if plan.target not in all_columns:
        raise HTTPException(
            status_code=502,
            detail=f"LLM proposed an invalid target column: '{plan.target}'"
        )

    invalid_models = [m for m in plan.models if m not in VALID_MODEL_NAMES]
    if invalid_models:
        raise HTTPException(
            status_code=502,
            detail=f"LLM proposed unsupported model(s): {invalid_models}"
        )

    return plan.model_dump()

@app.post("/api/assistant/analyze-experiments")
def analyze_experiments_endpoint(payload: dict = Body(...), db: Session = Depends(get_db)):
    dataset_id = payload.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")

    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format")

    experiments = (
        db.query(Experiment)
        .filter(Experiment.dataset_id == dataset_uuid, Experiment.status == "completed")
        .order_by(Experiment.created_at.desc())
        .all()
    )

    if not experiments:
        raise HTTPException(status_code=404, detail="No completed experiments found for this dataset")

    # Same principle as everywhere else -- pull real data OUT of the
    # database and hand it to the LLM as read-only context. The LLM never
    # touches the database directly, and never sees anything except what
    # we've already computed and stored ourselves.
    experiment_dicts = [serialize_experiment(e) for e in experiments]

    analysis_text = analyze_experiment_results(experiment_dicts)

    return {"analysis": analysis_text}