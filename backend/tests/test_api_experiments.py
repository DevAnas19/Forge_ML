"""
Tests for /api/experiments and the model registry/predict flow --
end-to-end through the real HTTP endpoints, against the isolated test
database and real (but tiny) training.
"""

import io


def make_learnable_csv_bytes():
    """
    Needs more rows than the Phase 1 dataset tests, since training an
    actual model (even a tiny one) needs enough data for a meaningful
    80/20 split. Same clearly-learnable pattern as test_trainer.py.
    """
    rows = ["income,category,approved"]
    for i in range(20):
        income = 20000 + i * 1000 if i < 10 else 80000 + i * 1000
        category = "low" if i < 10 else "high"
        approved = "N" if i < 10 else "Y"
        rows.append(f"{income},{category},{approved}")
    return io.BytesIO("\n".join(rows).encode())


def upload_test_dataset(client):
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("loans.csv", make_learnable_csv_bytes(), "text/csv")},
    )
    return response.json()["dataset"]["dataset_id"]


def test_create_experiment_trains_and_persists(client):
    dataset_id = upload_test_dataset(client)

    response = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": "approved",
        "numerical_columns": ["income"],
        "categorical_columns": ["category"],
        "model_names": ["logistic_regression"],
    })

    assert response.status_code == 200
    experiments = response.json()["experiments"]
    assert len(experiments) == 1
    assert experiments[0]["model_name"] == "logistic_regression"
    assert experiments[0]["status"] == "completed"
    assert "f1" in experiments[0]["metrics"]


def test_experiment_appears_in_list_after_creation(client):
    dataset_id = upload_test_dataset(client)
    client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": "approved",
        "numerical_columns": ["income"],
        "categorical_columns": ["category"],
        "model_names": ["logistic_regression"],
    })

    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_create_experiment_with_invalid_dataset_id_returns_404(client):
    import uuid
    response = client.post("/api/experiments", json={
        "dataset_id": str(uuid.uuid4()),
        "target_column": "approved",
        "numerical_columns": ["income"],
        "categorical_columns": ["category"],
        "model_names": ["logistic_regression"],
    })
    assert response.status_code == 404


def test_register_model_and_predict_end_to_end(client):
    """
    The big one: the FULL lifecycle in one test -- upload, train, register,
    predict -- proving every piece connects correctly, the same chain of
    actions a real user does through the UI.
    """
    dataset_id = upload_test_dataset(client)

    exp_response = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": "approved",
        "numerical_columns": ["income"],
        "categorical_columns": ["category"],
        "model_names": ["logistic_regression"],
    })
    experiment_id = exp_response.json()["experiments"][0]["experiment_id"]

    register_response = client.post("/api/models/register", json={
        "experiment_id": experiment_id,
    })
    assert register_response.status_code == 200
    model_id = register_response.json()["model_id"]

    predict_response = client.post(
        f"/api/models/{model_id}/predict",
        json={"income": 90000, "category": "high"},
    )
    assert predict_response.status_code == 200
    result = predict_response.json()
    assert result["prediction"] in ("Y", "N")
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_with_missing_feature_returns_400(client):
    dataset_id = upload_test_dataset(client)
    exp_response = client.post("/api/experiments", json={
        "dataset_id": dataset_id,
        "target_column": "approved",
        "numerical_columns": ["income"],
        "categorical_columns": ["category"],
        "model_names": ["logistic_regression"],
    })
    experiment_id = exp_response.json()["experiments"][0]["experiment_id"]
    model_id = client.post("/api/models/register", json={"experiment_id": experiment_id}).json()["model_id"]

    # Deliberately leave out "category" -- the model trained on both
    # income AND category, so this should fail cleanly, not crash.
    response = client.post(f"/api/models/{model_id}/predict", json={"income": 50000})
    assert response.status_code == 400