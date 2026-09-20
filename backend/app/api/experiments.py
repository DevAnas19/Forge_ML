"""
experiment.py

SQLAlchemy model for the `experiments` table (spec section 22).
"""

import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.core.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True)
    metrics = Column(JSONB, nullable=True)
    label_classes = Column(JSONB, nullable=True)  # e.g. ["N", "Y"] -- needed to decode predictions later
    status = Column(String, nullable=False, default="pending")
    training_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())