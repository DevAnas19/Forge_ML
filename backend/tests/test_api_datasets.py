"""
Tests for the /api/datasets endpoints -- using the isolated test database
from conftest.py. Each test uploads a small real CSV (built with pandas,
written to a temp file) to exercise the actual upload code path end to end.
"""

import io


def make_csv_bytes():
    """
    Builds a tiny CSV in memory -- this is what gets sent as the uploaded
    'file', simulating exactly what a real browser file upload looks like
    to FastAPI's UploadFile.
    """
    csv_content = "income,category,approved\n20000,low,N\n80000,high,Y\n30000,low,N\n"
    return io.BytesIO(csv_content.encode())


def test_upload_dataset_returns_profile_and_dataset_id(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", make_csv_bytes(), "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "dataset" in data
    assert "profile" in data
    assert "validation" in data
    assert data["dataset"]["rows"] == 3
    assert data["dataset"]["columns"] == 3


def test_uploaded_dataset_appears_in_list(client):
    client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", make_csv_bytes(), "text/csv")},
    )

    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert len(datasets) == 1
    assert datasets[0]["name"] == "test.csv"


def test_get_single_dataset_by_id(client):
    upload_response = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", make_csv_bytes(), "text/csv")},
    )
    dataset_id = upload_response.json()["dataset"]["dataset_id"]

    response = client.get(f"/api/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.json()["dataset"]["dataset_id"] == dataset_id


def test_get_dataset_with_invalid_id_format_returns_400(client):
    response = client.get("/api/datasets/not-a-real-uuid")
    assert response.status_code == 400


def test_get_dataset_with_nonexistent_id_returns_404(client):
    import uuid
    fake_id = str(uuid.uuid4())
    response = client.get(f"/api/datasets/{fake_id}")
    assert response.status_code == 404