"""
main.py

ForgeML backend entry point.

Phase 1: dataset upload -> profile + structural validation.
Phase 2: run classification experiments -- preprocessing, training, evaluation.
Phase 3 (current): persistence. Datasets are now saved to disk + a database
row; experiments reference a dataset_id instead of re-uploading the file,
and results are saved to Postgres so experiment history survives a restart.
"""

import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd

from app.core.database import engine, Base, get_db
import app.models
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.core.schemas import ExperimentRequest

from app.services.profiler import profile_dataset
from app.services.validator import validate_dataset
from app.services.trainer import run_experiment

DATASET_STORAGE_DIR = "storage/datasets"

app = FastAPI(title="ForgeML API")


@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)


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