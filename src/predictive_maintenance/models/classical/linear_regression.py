import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model for RUL prediction."""

    def __init__(self):
        super().__init__(name="LinearRegression")
        self.model = LinearRegression()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Train the Linear Regression model on the given data."""
        self.model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions using the trained Linear Regression model."""
        return self.model.predict(X)
