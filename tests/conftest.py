import json

import pandas as pd
import pytest

from predictive_maintenance.config import RAW_COLUMNS
from predictive_maintenance.data.preprocessing import preprocess_data
from predictive_maintenance.features.engineering import create_features
from predictive_maintenance.models.classical.linear_regression import LinearRegressionModel


@pytest.fixture
def raw_data():
    rows = []
    for unit in (1, 2, 3):
        for cycle in range(1, 16):
            row = {col: float(unit + cycle * (i + 1)) for i, col in enumerate(RAW_COLUMNS)}
            row.update(unit=unit, cycle=cycle, sensor_1=518.67, os_3=100.0)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def artifact(tmp_path, raw_data):
    clean, constants = preprocess_data(raw_data, is_train=True)
    features = create_features(
        clean.drop(columns="RUL"), [col for col in clean if col.startswith("sensor_")]
    )
    X = features.drop(columns=["unit", "cycle"])
    model = LinearRegressionModel()
    model.fit(X, clean["RUL"])
    model.save(tmp_path / "linear_regression.pkl")
    (tmp_path / "features.json").write_text(json.dumps(X.columns.tolist()), encoding="utf-8")
    (tmp_path / "constant_columns.json").write_text(json.dumps(constants), encoding="utf-8")
    info = {
        "model_name": "LinearRegression",
        "test_RMSE": 1.0,
        "test_MAE": 0.5,
        "preprocessing": {"trend_window": 10, "rul_cap": 125},
        "files": {"model": "linear_regression.pkl", "features": "features.json"},
    }
    (tmp_path / "best_model.json").write_text(json.dumps(info), encoding="utf-8")
    return tmp_path, model, features, clean
