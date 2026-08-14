import pandas as pd
from lightgbm import LGBMRegressor

from .base import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM model for RUL prediction."""

    def __init__(
            self,
            n_estimators: int = 500,
            learning_rate: float = 0.05,
            num_leaves: int = 31,
            max_depth: int = -1,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            random_state: int = 42,
            ):
        super().__init__(name="LightGBM")
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Train the LightGBM model on the given data."""
        self.model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions using the trained LightGBM model."""
        return self.model.predict(X)

    def feature_importances(self) -> pd.Series:
        """Get feature importances from the trained LightGBM model."""
        return self.model.feature_importances_
