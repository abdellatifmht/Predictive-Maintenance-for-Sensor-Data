import pandas as pd
from xgboost import XGBRegressor

from predictive_maintenance.models import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost model for RUL prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 4,
    ):
        super().__init__(name="XGBoost")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model on the given data."""
        self.model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions using the trained XGBoost model."""
        return self.model.predict(X)

    def feature_importances(self) -> pd.Series:
        """Get feature importances from the trained XGBoost model."""
        return self.model.feature_importances_
