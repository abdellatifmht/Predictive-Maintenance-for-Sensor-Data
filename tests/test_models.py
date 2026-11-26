import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.models import train
from predictive_maintenance.models.base import BaseModel
from predictive_maintenance.models.evaluate import align_predictions, phm08_score
from predictive_maintenance.models.predict import predict


def test_saved_linear_model_includes_scaler(artifact):
    path, model, features, _ = artifact
    X = features.drop(columns=["unit", "cycle"])
    loaded = BaseModel.load(path / "linear_regression.pkl")
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), atol=1e-10)


def test_batch_predictions_follow_saved_feature_order(artifact):
    path, model, features, _ = artifact
    features[features.columns[::-1]].to_csv(path / "features.csv", index=False)
    predict(path / "best_model.json", path / "features.csv", path / "predictions.csv")
    result = pd.read_csv(path / "predictions.csv")
    expected = model.predict(features.drop(columns=["unit", "cycle"])).clip(min=0)
    np.testing.assert_allclose(result.prediction, expected, atol=1e-8)


def test_model_selection_ignores_test_set_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train, "RESULTS_DIR", tmp_path)
    test_results = [
        {"model": "LinearRegression", "MAE": 10.0, "RMSE": 12.0},
        {"model": "XGBoost", "MAE": 1.0, "RMSE": 2.0},
    ]
    cv_results = [
        {"model": "LinearRegression", "MAE": 2.0, "RMSE": 3.0},
        {"model": "XGBoost", "MAE": 5.0, "RMSE": 6.0},
    ]
    info = train.save_best_model(test_results, cv_results)
    assert info["model_name"] == "LinearRegression"
    assert info["validation_RMSE"] == 3.0
    assert info["test_RMSE"] == 12.0


def test_group_cross_validation_keeps_engines_separate(raw_data, monkeypatch):
    observed = []

    def check_split(X_train, y_train, X_val, y_val, fold):
        assert set(y_train) & set(y_val) == set()
        assert "unit" not in X_train
        observed.append(fold)
        return []

    monkeypatch.setattr(train, "train_models", check_split)
    train.cross_validate(raw_data, raw_data.unit, n_splits=3)
    assert observed == [1, 2, 3]


def test_load_data_rejects_misaligned_labels(tmp_path, raw_data):
    raw_data.to_csv(tmp_path / "features.csv", index=False)
    raw_data[["unit", "cycle"]].assign(RUL=1).iloc[::-1].to_csv(
        tmp_path / "labels.csv", index=False
    )
    with pytest.raises(ValueError, match="aligned"):
        train.load_data(tmp_path / "features.csv", tmp_path / "labels.csv")


def test_evaluation_rejects_silently_dropped_rows():
    labels = pd.DataFrame({"unit": [1, 2], "cycle": [1, 2], "RUL": [10, 20]})
    bad = pd.DataFrame({"unit": [1, 3], "cycle": [1, 2], "prediction": [10, 20]})
    with pytest.raises(ValueError, match="matching labels"):
        align_predictions(bad, labels)
    with pytest.raises(ValueError, match="every engine"):
        align_predictions(bad.iloc[:1], labels)


def test_phm08_penalizes_late_predictions_more():
    assert phm08_score([10], [20]) > phm08_score([10], [0])
    assert phm08_score([10], [10]) == 0
