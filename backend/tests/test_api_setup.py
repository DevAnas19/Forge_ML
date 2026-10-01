def test_root_endpoint_works(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ForgeML backend running"