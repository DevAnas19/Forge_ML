"""
schemas.py

Pydantic models define the *shape* of API requests/responses -- FastAPI
uses them to validate incoming JSON automatically and reject bad requests
with a clear 422 error before your route code even runs.
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