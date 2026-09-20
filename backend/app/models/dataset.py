"""
dataset.py

SQLAlchemy model for the `datasets` table (spec section 22).
"""

import uuid
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    target_column = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())