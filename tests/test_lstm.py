import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from predictive_maintenance.models.deep_learning.lstm import LSTMModel


def test_lstm_short_history_padding_and_checkpoint_round_trip(tmp_path):
    torch.set_num_threads(1)
    X = pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_2": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "constant": [1.0] * 6,
        }
    )
    y = pd.Series([2.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    model = LSTMModel(sequence_length=5, hidden_size=4, epochs=1, batch_size=3, dropout=0.0)
    model.fit(X, y, X.iloc[:3], y.iloc[:3])
    sequences, units = model._build_inference_sequences(X)
    assert units == [1, 2]
    assert sequences.shape == (2, 5, 2)
    np.testing.assert_allclose(sequences[0, 0], sequences[0, 1])
    before = model.predict(X)
    model.save(tmp_path / "lstm")
    loaded = LSTMModel.load(tmp_path / "lstm.pt")
    np.testing.assert_allclose(loaded.predict(X), before, atol=1e-6)
    assert np.isfinite(before).all()
