"""Batch predictions from the artifact selected by training cross-validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from predictive_maintenance.config import MODELS_DIR, PROCESSED_DATA_DIR
from predictive_maintenance.models.base import BaseModel

app = typer.Typer()


def load_features(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    if df.empty:
        raise ValueError(f"Features file is empty: {features_path}")
    if not {"unit", "cycle"}.issubset(df.columns):
        raise ValueError("Features must contain unit and cycle")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["unit", "cycle"], errors="ignore").copy()


def load_best_model(best_model_path: Path):
    with best_model_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    name = info["model_name"]
    if name == "LSTM":
        from predictive_maintenance.models.deep_learning.lstm import LSTMModel

        return LSTMModel.load(best_model_path.parent / info["files"]["weights"]), name
    return BaseModel.load(best_model_path.parent / info["files"]["model"]), name


def prepare_lstm_output(df: pd.DataFrame, predictions) -> pd.DataFrame:
    identifiers = (
        df.loc[df.groupby("unit")["cycle"].idxmax(), ["unit", "cycle"]]
        .sort_values("unit")
        .reset_index(drop=True)
    )
    if len(identifiers) != len(predictions):
        raise ValueError("LSTM prediction count does not match number of engines")
    return identifiers.assign(prediction=np.asarray(predictions).reshape(-1))


def prepare_classical_output(df: pd.DataFrame, predictions) -> pd.DataFrame:
    if len(df) != len(predictions):
        raise ValueError("Prediction count does not match input rows")
    return df[["unit", "cycle"]].assign(prediction=np.asarray(predictions).reshape(-1))


def predict(model_path: Path, features_path: Path, output_path: Path) -> None:
    model, name = load_best_model(model_path)
    df = load_features(features_path)
    if name == "LSTM":
        output = prepare_lstm_output(df, np.clip(model.predict(df), 0, None))
    else:
        with model_path.open(encoding="utf-8") as handle:
            info = json.load(handle)
        with (model_path.parent / info["files"]["features"]).open(encoding="utf-8") as handle:
            columns = json.load(handle)
        X = prepare_features(df).reindex(columns=columns)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Features are missing required finite model inputs")
        output = prepare_classical_output(df, np.clip(model.predict(X), 0, None))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


@app.command()
def main(
    model_path: Path = MODELS_DIR / "best_model.json",
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    output_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
) -> None:
    predict(model_path, features_path, output_path)


if __name__ == "__main__":
    app()
