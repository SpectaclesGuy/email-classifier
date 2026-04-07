from fastapi.testclient import TestClient

from api.main import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classify_email_shape(monkeypatch):
    def _fake_predict(*args, **kwargs):
        return {
            "predicted_category": "informational",
            "confidence_score": 0.5,
            "priority_score": 20,
            "priority_band": "low",
            "explanation": ["stub"],
            "extracted_signals": {"urgency_keyword_count": 0},
        }

    monkeypatch.setattr("api.main.predict_email", _fake_predict)
    client = TestClient(app)
    payload = {"subject": "Hello", "body": "World"}
    response = client.post("/classify-email", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "priority_score" in data



