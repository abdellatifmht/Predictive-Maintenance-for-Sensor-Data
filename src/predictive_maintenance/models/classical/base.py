from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import joblib


class BaseModel(ABC):
    """Base class for all models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions on the given data."""
        pass

    def save(self, path: Path) -> None:
        """Save the model to the given path."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: Path):
        """Load the model from the given path."""
        return joblib.load(path)
