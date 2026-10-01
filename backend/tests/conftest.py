"""
conftest.py

Pytest auto-discovers this file and makes its fixtures available to every
test in this folder, without needing to import anything. This is where we
set up an isolated, disposable test database -- so running tests never
touches your real Postgres data.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db

# In-memory SQLite -- created fresh, lives only for the test run, then
# disappears. StaticPool is needed specifically because in-memory SQLite
# databases normally don't persist across different connections, and
# FastAPI's dependency injection opens a new connection per request.
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """
    Swaps out the real get_db() dependency with one pointing at the test
    database -- this is FastAPI's built-in mechanism for exactly this
    situation, called dependency overriding.
    """
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture()
def client():
    """
    Creates all tables fresh before each test function, and drops them
    afterward -- so every single test starts from a completely clean,
    empty database, with zero leftover data from a previous test affecting
    this one.
    """
    Base.metadata.create_all(bind=engine)
    yield TestClient(app)
    Base.metadata.drop_all(bind=engine)