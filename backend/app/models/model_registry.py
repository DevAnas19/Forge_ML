"""
model_registry.py

SQLAlchemy model for the `models` table (spec section 22) -- this is the
Model Registry (spec section 12), tracking which trained artifact is
registered/deployed. Named model_registry.py (not models.py) to avoid a
name clash with the ML sense of "model" used everywhere else in the app.
"""

import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class RegisteredModel(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False, default="v1")
    artifact_path = Column(String, nullable=False)
    status = Column(String, nullable=False, default="trained")  # trained / validated / registered / deployed
    created_at = Column(DateTime(timezone=True), server_default=func.now())