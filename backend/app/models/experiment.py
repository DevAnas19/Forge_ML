"""
experiment.py

SQLAlchemy model for the `experiments` table (spec section 22).

artifact_data stores the trained pipeline's serialized bytes directly in
Postgres -- same reasoning as Dataset.file_content: local disk doesn't
survive a restart on Render's free tier, Postgres does.
"""

import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.core.database import Base

JSONType = JSON().with_variant(JSONB, "postgresql")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String, nullable=False)
    parameters = Column(JSONType, nullable=True)
    metrics = Column(JSONType, nullable=True)
    label_classes = Column(JSONType, nullable=True)
    artifact_data = Column(LargeBinary, nullable=True)  # serialized (joblib) trained pipeline
    status = Column(String, nullable=False, default="pending")
    training_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())