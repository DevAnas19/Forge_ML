"""
main.py

ForgeML backend entry point.

Phase 1-7: dataset upload/profiling/validation, preprocessing/training,
PostgreSQL persistence, model registry + prediction, SHAP explainability,
LLM assistant endpoints.

IMPORTANT: dataset CSV content and trained model artifacts are stored
directly in Postgres (Dataset.file_content, Experiment.artifact_data),
not on local disk. This is required for deployment on platforms like
Render's free tier, where the web service filesystem is ephemeral --
local files disappear on every restart, redeploy, or free-tier spin-down.
Postgres is the only storage that actually persists in that environment.
"""

import io
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
import pandas as pd

from app.core.database import engine, Base, get_db
import app.models
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.model_registry import RegisteredModel
from app.core.schemas import ExperimentRequest, RegisterModelRequest, ExperimentPlanRequest, ExperimentPlanResponse, DatasetAnalysisResponse

from app.services.profiler import profile_dataset
from app.services.validator import validate_dataset
from app.services.trainer import run_experiment, serialize_pipeline, deserialize_pipeline
from app.services.explainer import compute_global_importance, explain_prediction
from app.services.llm import analyze_dataset_profile, create_experiment_plan, analyze_experiment_results, VALID_MODEL_NAMES


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            columns = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'experiments'")
            ).scalars().all()
            if "label_classes" not in columns:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN label_classes JSONB"))
            if "artifact_data" not in columns:
                conn.execute(text("ALTER TABLE experiments ADD COLUMN artifact_data BYTEA"))

            dataset_columns = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = 'datasets'")
            ).scalars().all()
            if "file_content" not in dataset_columns:
                conn.execute(text("ALTER TABLE datasets ADD COLUMN file_content TEXT"))

    yield


app = FastAPI(title="ForgeML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://forge-ml-gamma.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "label_classes": exp.label_classes,
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
        "status": m.status,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def load_dataset_df(dataset: Dataset) -> pd.DataFrame:
    """Reads a dataset's dataframe from its stored content in Postgres."""
    return pd.read_csv(io.StringIO(dataset.file_content))


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    df = pd.read_csv(file.file)

    profile = profile_dataset(df, dataset_name=file.filename)
    validation = validate_dataset(df)

    csv_text = df.to_csv(index=False)

    dataset_row = Dataset(
        name=file.filename,
        file_path=None,
        file_content=csv_text,
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

    df = load_dataset_df(dataset)
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

    df = load_dataset_df(dataset)

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

        artifact_bytes = serialize_pipeline(result["pipeline"])

        experiment_row = Experiment(
            dataset_id=dataset.id,
            model_name=result["model_name"],
            parameters=result["hyperparameters"],
            metrics=result["metrics"],
            label_classes=result["label_classes"],
            artifact_data=artifact_bytes,
            status=result["status"],
            training_time=result["training_time"],
        )
        db.add(experiment_row)
        db.commit()
        db.refresh(experiment_row)

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

    if not experiment.artifact_data:
        raise HTTPException(
            status_code=404,
            detail="No trained artifact stored for this experiment -- it may need to be retrained."
        )

    existing_count = db.query(RegisteredModel).filter(RegisteredModel.name == experiment.model_name).count()
    version = f"v{existing_count + 1}"

    model_row = RegisteredModel(
        experiment_id=experiment.id,
        name=experiment.model_name,
        version=version,
        artifact_path="stored-in-database",
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


def load_model_pipeline_and_experiment(model_id: str, db: Session):
    try:
        model_uuid = uuid.UUID(model_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_id format")

    model = db.query(RegisteredModel).filter(RegisteredModel.id == model_uuid).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    experiment = db.query(Experiment).filter(Experiment.id == model.experiment_id).first()
    if not experiment or not experiment.artifact_data:
        raise HTTPException(status_code=500, detail="Model artifact data is missing")

    pipeline = deserialize_pipeline(experiment.artifact_data)
    return model, experiment, pipeline


@app.post("/api/models/{model_id}/predict")
def predict(model_id: str, payload: dict = Body(...), db: Session = Depends(get_db)):
    model, experiment, pipeline = load_model_pipeline_and_experiment(model_id, db)

    if not experiment.label_classes:
        raise HTTPException(status_code=500, detail="Label classes missing for this model's experiment")

    input_df = pd.DataFrame([payload])

    try:
        predicted_class_index = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature in input: {e}")

    predicted_label = experiment.label_classes[predicted_class_index]
    predicted_probability = round(float(probabilities[predicted_class_index]), 4)

    return {
        "prediction": predicted_label,
        "probability": predicted_probability,
    }


@app.get("/api/models/{model_id}/explain")
def explain_model_global(model_id: str, db: Session = Depends(get_db)):
    model, experiment, pipeline = load_model_pipeline_and_experiment(model_id, db)

    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    raw_df = load_dataset_df(dataset)

    return compute_global_importance(pipeline, model.name, raw_df)


@app.post("/api/models/{model_id}/explain")
def explain_model_prediction(model_id: str, payload: dict = Body(...), db: Session = Depends(get_db)):
    model, experiment, pipeline = load_model_pipeline_and_experiment(model_id, db)

    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    raw_df = load_dataset_df(dataset)
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

    df = load_dataset_df(dataset)
    profile = profile_dataset(df, dataset_name=dataset.name)

    try:
        raw_result = analyze_dataset_profile(profile)
        validated = DatasetAnalysisResponse(**raw_result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned an unusable response: {str(e)}")

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

    df = load_dataset_df(dataset)
    profile = profile_dataset(df, dataset_name=dataset.name)

    try:
        raw_plan = create_experiment_plan(request.goal, profile)
        plan = ExperimentPlanResponse(**raw_plan)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM returned an unusable plan: {str(e)}")

    all_columns = (
        profile["column_types"]["numerical_columns"]
        + profile["column_types"]["categorical_columns"]
    )
    if plan.target not in all_columns:
        raise HTTPException(status_code=502, detail=f"LLM proposed an invalid target column: '{plan.target}'")

    invalid_models = [m for m in plan.models if m not in VALID_MODEL_NAMES]
    if invalid_models:
        raise HTTPException(status_code=502, detail=f"LLM proposed unsupported model(s): {invalid_models}")

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

    experiment_dicts = [serialize_experiment(e) for e in experiments]
    analysis_text = analyze_experiment_results(experiment_dicts)

    return {"analysis": analysis_text}