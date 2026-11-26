"""Load the trained artifact and apply the same causal features used offline."""

import json
from pathlib import Path
import time

from loguru import logger
import numpy as np
import pandas as pd

from predictive_maintenance.config import MODELS_DIR, TREND_WINDOW
from predictive_maintenance.features.engineering import create_features
from predictive_maintenance.models.predict import load_best_model

RUL_THRESHOLD_ANOMALY = 30
RUL_THRESHOLD_IMMINENT = 10


def rul_to_class(rul: float) -> str:
    if rul <= RUL_THRESHOLD_IMMINENT:
        return "imminent_failure"
    if rul <= RUL_THRESHOLD_ANOMALY:
        return "anomaly"
    return "healthy"


class ModelService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.model_name = None
        self.model_type = None
        self.best_model_info = None
        self.constant_columns = []
        self.feature_columns = None
        self.trend_window = TREND_WINDOW

    def load(self):
        self.model = None
        manifest = self.models_dir / "best_model.json"
        with manifest.open(encoding="utf-8") as handle:
            info = json.load(handle)
        # This is required: silently guessing dropped sensors changes the feature schema.
        with (self.models_dir / "constant_columns.json").open(encoding="utf-8") as handle:
            self.constant_columns = json.load(handle)
        model, name = load_best_model(manifest)
        self.model_name = name
        self.model_type = "lstm" if name == "LSTM" else "classical"
        self.trend_window = info.get("preprocessing", {}).get("trend_window", TREND_WINDOW)
        if self.model_type == "classical":
            with (self.models_dir / info["files"]["features"]).open(encoding="utf-8") as handle:
                self.feature_columns = json.load(handle)
        else:
            self.feature_columns = model.feature_columns
        self.best_model_info = info
        self.model = model
        logger.info("Loaded {} ({})", name, self.model_type)

    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def min_readings_required(self) -> int:
        if self.model_type == "lstm":
            # The earliest feature in the sequence also needs its trend history.
            return self.model.sequence_length + self.trend_window - 1
        return self.trend_window

    def _build_engineering_features(self, readings: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(readings).drop(columns=self.constant_columns, errors="ignore")
        df.insert(0, "cycle", range(1, len(df) + 1))
        df.insert(0, "unit", 1)
        return create_features(
            df,
            sensor_columns=[col for col in df if col.startswith("sensor_")],
            trend_window=self.trend_window,
        )

    def predict(self, readings: list[dict], unit: str | None = None) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model is not loaded. Call load() before predicting.")
        if len(readings) < self.min_readings_required:
            raise ValueError(
                f"Not enough readings: {self.min_readings_required} consecutive readings "
                f"are required for {self.model_name}; received {len(readings)}."
            )
        started = time.perf_counter()
        n_used = self.min_readings_required
        df = self._build_engineering_features(readings[-n_used:])
        if self.model_type == "lstm":
            predicted = self.model.predict(df)[0]
        else:
            last_row = df.iloc[[-1]].reindex(columns=self.feature_columns)
            if not np.isfinite(last_row.to_numpy(dtype=float)).all():
                raise ValueError("Input is missing required finite sensor values")
            predicted = self.model.predict(last_row)[0]
        if not np.isfinite(predicted):
            raise RuntimeError("Model returned a non-finite prediction")
        rul = max(float(predicted), 0.0)
        return {
            "unit": unit,
            "predicted_rul": round(rul, 2),
            "predicted_class": rul_to_class(rul),
            "model_name": self.model_name,
            "n_readings_received": len(readings),
            "n_readings_used": n_used,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }


model_service = ModelService()
