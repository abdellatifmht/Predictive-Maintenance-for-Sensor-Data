import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.monitoring.drift import (
    compare_all_features,
    compare_distributions,
    population_stability_index,
)


def test_identical_distributions_are_stable():
    reference = np.linspace(0, 1, 100)
    result = compare_distributions("sensor_2", reference, reference)
    assert result.psi == pytest.approx(0.0)
    assert not result.is_drifted


def test_shifted_constant_reference_is_detected():
    reference = np.ones(100)
    result = compare_distributions("sensor_2", reference, reference + 10)
    assert result.psi_level == "alert"
    assert result.ks_flagged


@pytest.mark.parametrize("sample", [[], [np.nan], [np.inf]])
def test_invalid_drift_samples_are_rejected(sample):
    with pytest.raises(ValueError):
        population_stability_index(np.ones(10), sample)


def test_missing_or_empty_monitoring_sensors_are_skipped():
    reference = pd.DataFrame({"sensor_2": [1, 2], "sensor_3": [np.nan, np.nan]})
    current = pd.DataFrame({"sensor_2": [1, 2], "sensor_3": [1, 2]})
    results = compare_all_features(reference, current, ["sensor_2", "sensor_3", "sensor_4"])
    assert [result.feature for result in results] == ["sensor_2"]
