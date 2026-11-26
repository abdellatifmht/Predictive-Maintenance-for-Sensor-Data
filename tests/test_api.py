from fastapi.testclient import TestClient
import numpy as np
import pytest

from predictive_maintenance.api import main
from predictive_maintenance.api.inference import ModelService, rul_to_class


@pytest.fixture
def api(artifact, monkeypatch):
    path, _, _, _ = artifact
    service = ModelService(path)
    monkeypatch.setattr(main, "model_service", service)
    with TestClient(main.app) as client:
        yield client, service


def sensor_readings(raw_data):
    return raw_data.query("unit == 1").drop(columns=["unit", "cycle"]).to_dict("records")


def test_api_prediction_matches_offline_features(api, artifact, raw_data):
    client, _ = api
    _, model, features, _ = artifact
    response = client.post(
        "/predict", json={"unit": "engine_1", "readings": sensor_readings(raw_data)}
    )
    assert response.status_code == 200
    result = response.json()
    X = features.query("unit == 1").tail(1).drop(columns=["unit", "cycle"])
    assert result["predicted_rul"] == pytest.approx(max(float(model.predict(X)[0]), 0), abs=0.005)
    assert result["unit"] == "engine_1"
    assert result["n_readings_used"] == 10
    assert result["n_readings_received"] == 15
    assert client.get("/health").json()["min_readings_required"] == 10


@pytest.mark.parametrize("problem", ["short", "empty", "missing_sensor", "unknown_sensor"])
def test_api_rejects_invalid_requests(api, raw_data, problem):
    client, _ = api
    readings = sensor_readings(raw_data)
    if problem == "short":
        readings = readings[:1]
    elif problem == "empty":
        readings = []
    elif problem == "missing_sensor":
        del readings[-1]["sensor_2"]
    else:
        readings[-1]["sensor_99"] = 1
    response = client.post("/predict", json={"readings": readings})
    assert response.status_code == 422


def test_api_accepts_legacy_unit_id(api, raw_data):
    client, _ = api
    result = client.post(
        "/predict", json={"unit_id": "legacy", "readings": sensor_readings(raw_data)}
    )
    assert result.status_code == 200
    assert result.json()["unit"] == "legacy"


def test_api_rejects_nonfinite_sensor(raw_data):
    from predictive_maintenance.api.schemas import SensorReading

    reading = sensor_readings(raw_data)[0]
    reading["os_1"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SensorReading.model_validate(reading)


def test_untrained_api_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "model_service", ModelService(tmp_path))
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize(
    "rul,label", [(10, "imminent_failure"), (10.1, "anomaly"), (30, "anomaly"), (30.1, "healthy")]
)
def test_health_class_boundaries(rul, label):
    assert rul_to_class(rul) == label
