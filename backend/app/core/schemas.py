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