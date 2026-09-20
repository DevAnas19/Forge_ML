"""
database.py

SQLAlchemy engine + session setup, and the get_db() dependency that
FastAPI route handlers will use to talk to Postgres.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Every SQLAlchemy model (in app/models/) inherits from this Base.
# Base.metadata.create_all(engine) uses this to know which tables to create.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency: yields a database session for the duration of one
    request, and always closes it afterward -- even if the request raises
    an error. Used in route handlers as: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()