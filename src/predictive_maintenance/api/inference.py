import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from predictive_maintenance.config import MODELS_DIR
from predictive_maintenance.features.engineering import add_trend_features
from predictive_maintenance.models.base import BaseModel
from predictive_maintenance.models.deep_learning.lstm import LSTMModel


logger.info("predictive_maintenance_api")

ROLLING_WINDOW = 6

RUL_THRESHOLD_ANOMALY = 30
RUL_THRESHOLD_IMMINENT = 10


def rul_to_class(rul: float) -> str:
    """Convert a RUL value to a class label based on predefined thresholds."""
    if rul <= RUL_THRESHOLD_IMMINENT:
        return "imminent_failure"
    elif rul <= RUL_THRESHOLD_ANOMALY:
        return "anomaly"
    else:
        return "healthy"


class ModelService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.model_name: str | None = None
        self.model_type: str | None = None
        self.best_model_info: dict | None = None
        self.constant_columns: list[str] = []
        self.feature_columns: list[str] | None = None

    def load(self):
        t0 = time.time()

        best_model_path = self.models_dir / "best_model.json"
        if not best_model_path.exists():
            raise FileNotFoundError(
                f"{best_model_path} introuvable — lance `make train` avant de démarrer l'API."
            )
        with open(best_model_path) as f:
            self.best_model_info = json.load(f)

        self.model_name = self.best_model_info["model_name"]

        constant_columns_path = self.models_dir / "constant_columns.json"
        if constant_columns_path.exists():
            with open(constant_columns_path) as f:
                self.constant_columns = json.load(f)
        else:
            logger.warning(f"{constant_columns_path} introuvable — aucune colonne constante ne sera filtrée.")

        if self.model_name == "LSTM":
            self.model_type = "lstm"
            self.model = LSTMModel.load(self.models_dir / "lstm")
        else:
            self.model_type = "classical"
            filename = self.best_model_info["files"]["model"]
            self.model = BaseModel.load(self.models_dir / filename)

            features_filename = self.best_model_info["files"]["features"]
            with open(self.models_dir / features_filename) as f:
                self.feature_columns = json.load(f)

        logger.info(
            "Modèle chargé: %s (%s) en %.2fs — test RMSE=%.2f",
            self.model_name, self.model_type, time.time() - t0,
            self.best_model_info["test_RMSE"],
        )

    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def min_readings_required(self) -> int:
        if self.model_type == "lstm":
            return self.model.sequence_length
        return 1

    def _build_engineering_features(self, readings: list[dict]) -> pd.DataFrame:
        """Build a DataFrame with engineered features from raw sensor readings."""
        df = pd.DataFrame(readings)
        columns_to_drop = [col for col in self.constant_columns if col in df.columns]
        df = df.drop(columns=columns_to_drop, errors="ignore")

        df.insert(0, "cycle", range(1, len(df) + 1))
        df.insert(0, "unit", 1)

        sensor_columns = [col for col in df.columns if col.startswith("sensor_")]
        df = add_trend_features(df, sensor_columns=sensor_columns)

        return df

    def _predict(self, df: pd.DataFrame) -> int:
        """Make a prediction using the loaded model."""

        if self.model_type == "lstm":
            # LSTM needs the complete temporal sequence.
            predicted_rul = self.model.predict(df)[0]

        else:
            # Classical models only need the last observation.
            last_row = df.iloc[[-1]].drop(
                columns=["unit", "cycle"],
                errors="ignore",
            )

            last_row = last_row.reindex(columns=self.feature_columns)

            if last_row.isna().any(axis=None):
                missing_cols = last_row.columns[
                    last_row.isna().any()
                ].tolist()

                raise ValueError(
                    f"Missing values in the input features for prediction: "
                    f"{missing_cols}"
                )

            predicted_rul = self.model.predict(last_row)[0]

        return int(predicted_rul)

    def predict(self, readings: list[dict], unit: str | None = None) -> dict:
        """Predict RUL and class based on a list of sensor readings."""

        if not self.is_ready():
            raise RuntimeError(
                "Model is not loaded. Call load() before predicting."
            )

        t0 = time.time()
        n_readings_received = len(readings)

        engineered_df = self._build_engineering_features(readings)

        if len(engineered_df) < self.min_readings_required:
            raise ValueError(
                f"Not enough readings for {self.model_name}. "
                f"Received {len(engineered_df)}, "
                f"but {self.min_readings_required} are required."
            )

        if self.model_type == "lstm":
            n_used = self.model.sequence_length
        else:
            n_used = 1

        rul = self._predict(engineered_df)
        rul = max(rul, 0.0)

        latency_ms = (time.time() - t0) * 1000

        return {
            "unit": unit,
            "predicted_rul": round(rul, 2),
            "predicted_class": rul_to_class(rul),
            "model_name": self.model_name,
            "n_readings_received": n_readings_received,
            "n_readings_used": n_used,
            "latency_ms": round(latency_ms, 2),
        }


model_service = ModelService()
