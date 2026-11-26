import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.models.base import BaseModel


class LinearRegressionModel(BaseModel):
    """Persist the scaler with the estimator for consistent batch and API inference."""

    def __init__(self):
        super().__init__(name="LinearRegression")
        self.model = make_pipeline(StandardScaler(), LinearRegression())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
