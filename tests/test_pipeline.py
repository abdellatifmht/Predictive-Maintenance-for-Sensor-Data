import json

import pandas as pd

from predictive_maintenance import pipeline
from predictive_maintenance.api.inference import ModelService
from predictive_maintenance.config import RAW_COLUMNS
from predictive_maintenance.models import train
from predictive_maintenance.models.classical.lightgbm_model import LightGBMModel
from predictive_maintenance.models.classical.linear_regression import LinearRegressionModel
from predictive_maintenance.models.classical.xgboost_model import XGBoostModel


def test_full_classical_pipeline_without_torch(tmp_path, monkeypatch, raw_data):
    external = tmp_path / "external"
    external.mkdir()
    for name in ("train", "test"):
        raw_data[RAW_COLUMNS].to_csv(
            external / f"{name}_FD001.txt", sep=" ", header=False, index=False
        )
    pd.DataFrame([10, 20, 30]).to_csv(external / "RUL_FD001.txt", header=False, index=False)
    for constant, folder in [
        ("EXTERNAL_DATA_DIR", "external"),
        ("RAW_DATA_DIR", "raw"),
        ("INTERIM_DATA_DIR", "interim"),
        ("PROCESSED_DATA_DIR", "processed"),
        ("MODELS_DIR", "models"),
        ("RESULTS_DIR", "results"),
    ]:
        monkeypatch.setattr(pipeline, constant, tmp_path / folder)
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        train,
        "get_models",
        lambda: [
            LinearRegressionModel(),
            XGBoostModel(n_estimators=2, n_jobs=1),
            LightGBMModel(n_estimators=2, n_jobs=1),
        ],
    )

    def unwanted_lstm(*args, **kwargs):
        raise AssertionError("--no-train-lstm must skip LSTM training")

    monkeypatch.setattr(train, "train_final_lstm", unwanted_lstm)
    pipeline.main(n_splits=3, train_lstm=False)
    info = json.loads((tmp_path / "models/best_model.json").read_text())
    assert info["selection_metric"] == "groupkfold_RMSE"
    report = pd.read_csv(tmp_path / "results/evaluation.csv")
    assert report.n_engines.iloc[0] == 3
    service = ModelService(tmp_path / "models")
    service.load()
    readings = raw_data.query("unit == 1").drop(columns=["unit", "cycle"]).to_dict("records")
    assert service.predict(readings)["predicted_rul"] >= 0
