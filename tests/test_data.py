import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.data.ingestion import load_cmapss
from predictive_maintenance.data.preprocessing import (
    add_rul_from_ground_truth,
    create_rul,
    handle_missing_values,
    preprocess_data,
)
from predictive_maintenance.data.validation import validate_dataset


def test_training_rul_is_capped_and_ends_at_zero(raw_data):
    result = create_rul(raw_data, cap=5)
    assert result.RUL.max() == 5
    assert (result.groupby("unit").tail(1).RUL == 0).all()


def test_test_rul_uses_ground_truth_with_nondefault_index(raw_data):
    data = raw_data.iloc[::-1].copy()
    data.index = np.arange(100, 100 + len(data))
    result = add_rul_from_ground_truth(data, pd.DataFrame({"RUL": [10, 20, 30]}), cap=None)
    expected = result.unit.map({1: 10, 2: 20, 3: 30}) + 15 - result.cycle
    pd.testing.assert_series_equal(result.RUL, expected, check_names=False)


def test_test_preprocessing_reuses_training_schema(raw_data):
    train, constants = preprocess_data(raw_data, is_train=True)
    raw_data["sensor_2"] = 5  # Constant in test only; it must remain a feature.
    test, _ = preprocess_data(
        raw_data,
        is_train=False,
        constant_columns=constants,
        rul_df=pd.DataFrame({"RUL": [10, 20, 30]}),
    )
    assert train.columns.tolist() == test.columns.tolist()
    assert "sensor_2" in test


def test_identifiers_are_preserved_for_one_engine(raw_data):
    result, constants = preprocess_data(raw_data.query("unit == 1"), is_train=True)
    assert "unit" not in constants
    validate_dataset(result)


def test_missing_values_use_only_prior_cycle_in_same_engine(raw_data):
    raw_data.loc[1, "sensor_2"] = np.nan
    filled = handle_missing_values(raw_data.iloc[::-1])
    assert filled.loc[1, "sensor_2"] == filled.loc[0, "sensor_2"]
    raw_data.loc[15, "sensor_2"] = np.nan
    with pytest.raises(ValueError, match="initial"):
        handle_missing_values(raw_data)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "fractional", "infinite", "negative_rul", "empty"]
)
def test_invalid_datasets_are_rejected(raw_data, invalid):
    data, _ = preprocess_data(raw_data, is_train=True)
    if invalid == "duplicate":
        duplicate = data.iloc[[0]].copy()
        duplicate["sensor_2"] += 1
        data = pd.concat([data, duplicate], ignore_index=True)
    elif invalid == "fractional":
        data["unit"] = data.unit.astype(float)
        data.loc[0, "unit"] = 1.5
    elif invalid == "infinite":
        data.loc[0, "sensor_2"] = np.inf
    elif invalid == "negative_rul":
        data.loc[0, "RUL"] = -1
    else:
        data = data.iloc[:0]
    with pytest.raises(ValueError):
        validate_dataset(data)


def test_raw_loader_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cmapss(tmp_path / "missing.txt")


def test_raw_loader_rejects_wrong_column_count(tmp_path):
    source = tmp_path / "malformed.txt"
    source.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="26 columns"):
        load_cmapss(source)
