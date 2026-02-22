from fastapi.testclient import TestClient
from api.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_returns_json():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "I love this product"})
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "score" in data