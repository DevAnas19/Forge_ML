"""
dataset.py

SQLAlchemy model for the `datasets` table (spec section 22).

file_content stores the raw CSV text directly in Postgres -- necessary
because Render's (and most PaaS free tiers') web service filesystem is
ephemeral: local files disappear on every restart/redeploy/spin-down.
Postgres is the only genuinely persistent storage available here, so the
actual file content lives there now, not on local disk.
"""

import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=True)   # kept for display/debugging only, not read from anymore
    file_content = Column(Text, nullable=True)  # the actual CSV data, as text
    rows = Column(Integer, nullable=False)
    columns = Column(Integer, nullable=False)
    target_column = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())