"""Models, with PyTorch loaded only when LSTM is requested."""

from .base import BaseModel
from .classical.lightgbm_model import LightGBMModel
from .classical.linear_regression import LinearRegressionModel
from .classical.xgboost_model import XGBoostModel

__all__ = ["BaseModel", "LSTMModel", "LightGBMModel", "LinearRegressionModel", "XGBoostModel"]


def __getattr__(name):
    if name == "LSTMModel":
        from .deep_learning.lstm import LSTMModel

        return LSTMModel
    raise AttributeError(name)
