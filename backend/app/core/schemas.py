"""
schemas.py

Pydantic models define the *shape* of API requests/responses.
"""

from typing import List
from pydantic import BaseModel


class ExperimentRequest(BaseModel):
    dataset_id: str
    target_column: str
    numerical_columns: List[str]
    categorical_columns: List[str]
    model_names: List[str]
    test_size: float = 0.2
    random_seed: int = 42


class RegisterModelRequest(BaseModel):
    experiment_id: str

class DatasetAnalysisResponse(BaseModel):
    recommended_preprocessing: dict
    potential_risks: List[str]

class ExperimentPlanRequest(BaseModel):
    dataset_id: str
    goal: str


class ExperimentPlanResponse(BaseModel):
    task: str
    target: str
    models: List[str]
    preprocessing: dict
    metrics: List[str]