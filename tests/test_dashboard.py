import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from predictive_maintenance import config


@pytest.mark.parametrize("successful", [True, False])
def test_dashboard_handles_one_request_and_incomplete_log(tmp_path, monkeypatch, successful):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monitoring = tmp_path / "monitoring"
    logs = monitoring / "logs"
    logs.mkdir(parents=True)
    reference = {"features": {"sensor_2": {"values": [1.0, 2.0, 3.0]}}}
    (monitoring / "reference_profile.json").write_text(json.dumps(reference), encoding="utf-8")
    entry = {
        "timestamp": "2026-11-26T12:00:00+00:00",
        "unit_id": "engine_1",
        "status": "ok" if successful else "error: unavailable",
        "drift_injected": False,
        "last_reading": {"sensor_2": 2.0},
    }
    if successful:
        entry.update(predicted_rul=100.0, predicted_class="healthy", latency_ms=10.0)
    (logs / "predictions.jsonl").write_text(json.dumps(entry) + '\n{"partial":', encoding="utf-8")
    dashboard = Path(__file__).parents[1] / "src/predictive_maintenance/monitoring/dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=30)
    assert not app.exception
