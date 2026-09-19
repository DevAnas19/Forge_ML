"""
main.py

ForgeML backend entry point.

Phase 1 (current): dataset upload -> profile + structural validation.
No database yet -- results are returned directly, not persisted.
"""

from fastapi import FastAPI, UploadFile, File
import pandas as pd

from app.services.profiler import profile_dataset
from app.services.validator import validate_dataset

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
