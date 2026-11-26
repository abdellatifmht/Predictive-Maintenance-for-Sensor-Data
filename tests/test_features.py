import pandas as pd
import pytest

from predictive_maintenance.features.engineering import add_trend_features, engineer_dataset


def test_slopes_are_causal_and_do_not_cross_engine_boundaries():
    data = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_2": [2.0, 4.0, 6.0, 100.0, 97.0, 94.0],
        }
    )
    slopes = add_trend_features(data, sensor_columns=["sensor_2"]).sensor_2_slope
    assert slopes.tolist() == pytest.approx([0, 2, 2, 0, -3, -3])
    data.loc[2, "sensor_2"] = 1000
    changed = add_trend_features(data, sensor_columns=["sensor_2"])
    assert changed.loc[1, "sensor_2_slope"] == slopes.iloc[1]


def test_feature_and_label_files_stay_aligned_when_input_is_unsorted(tmp_path, raw_data):
    raw_data["RUL"] = 15 - raw_data.cycle
    source = tmp_path / "source.csv"
    raw_data.sample(frac=1, random_state=42).to_csv(source, index=False)
    engineer_dataset(source, tmp_path / "features.csv", tmp_path / "labels.csv")
    features = pd.read_csv(tmp_path / "features.csv")
    labels = pd.read_csv(tmp_path / "labels.csv")
    pd.testing.assert_frame_equal(features[["unit", "cycle"]], labels[["unit", "cycle"]])
    assert "RUL" not in features
