from fastapi.testclient import TestClient
from viet_qa.api.main import app

client = TestClient(app)

def test_health_check_smoke():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data

def test_predict_extractive_validation_error():
    # Attempting to query predict without valid Pydantic format should 422
    response = client.post("/predict/extractive", json={"context": "Hello, this is context. missing question."})
    assert response.status_code == 422 # Unprocessable Entity
