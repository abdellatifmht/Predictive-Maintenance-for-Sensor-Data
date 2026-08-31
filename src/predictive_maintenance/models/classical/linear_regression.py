from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from predictive_maintenance.models import BaseModel


class LinearRegressionModel(BaseModel):
    """Linear Regression model for RUL prediction."""

    def __init__(self):
        super().__init__(name="LinearRegression")
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Train the Linear Regression model on the given data."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions using the trained Linear Regression model."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: Path) -> None:
        """Save the trained model and scaler to the specified path."""
        import joblib
        joblib.dump(self.model, path)
        joblib.dump(self.scaler, path.with_suffix('.scaler'))
