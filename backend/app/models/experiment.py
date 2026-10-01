"""
experiment.py

SQLAlchemy model for the `experiments` table (spec section 22).
"""

import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from app.core.database import Base

# JSON().with_variant(JSONB, "postgresql") means: use real JSONB when
# running against Postgres (production, and the docker-compose setup),
# but fall back to plain JSON everywhere else -- specifically so SQLite
# (used only in tests) can also create this table.
JSONType = JSON().with_variant(JSONB, "postgresql")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String, nullable=False)
    parameters = Column(JSONType, nullable=True)
    metrics = Column(JSONType, nullable=True)
    label_classes = Column(JSONType, nullable=True)
    status = Column(String, nullable=False, default="pending")
    training_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())