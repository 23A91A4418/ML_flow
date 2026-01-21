import pytest
import numpy as np
from src import inference_api


@pytest.fixture
def client():
    inference_api.app.testing = True
    return inference_api.app.test_client()


class DummyScaler:
    def transform(self, X):
        return np.array(X)


class DummyModel:
    def predict(self, X):
        # always return 1 for testing
        return np.ones(len(X), dtype=int)


@pytest.fixture(autouse=True)
def mock_model_and_scaler():
    inference_api.model = DummyModel()
    inference_api.scaler = DummyScaler()


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json["status"] == "ok"


def test_predict_valid_input(client):
    payload = {
        "features": [[1] * 30]  # breast cancer dataset has 30 features
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "predictions" in res.json
    assert res.json["predictions"] == [1]


def test_predict_missing_features(client):
    payload = {"wrong_key": [[1] * 30]}
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    assert "error" in res.json
