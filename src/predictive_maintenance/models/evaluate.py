"""Strictly align predictions and labels, then report the last cycle per engine."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import typer

from predictive_maintenance.config import PROCESSED_DATA_DIR, RESULTS_DIR

app = typer.Typer()
TARGET_COLUMN = "RUL"


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    return pd.read_csv(predictions_path)


def load_labels(labels_path: Path) -> pd.DataFrame:
    return pd.read_csv(labels_path)


def phm08_score(y_true, y_pred) -> float:
    errors = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.expm1(np.where(errors < 0, -errors / 13, errors / 10)).sum())


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "PHM08": phm08_score(y_true, y_pred),
    }


def evaluate_by_unit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit, group in predictions.groupby("unit"):
        metrics = calculate_metrics(group["RUL"], group["prediction"])
        rows.append({"unit": unit, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"]})
    return pd.DataFrame(rows)


def align_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["unit", "cycle"]
    if not set(keys + ["prediction"]).issubset(predictions):
        raise ValueError("Predictions must contain unit, cycle and prediction")
    if not set(keys + [TARGET_COLUMN]).issubset(labels):
        raise ValueError("Labels must contain unit, cycle and RUL")
    if predictions.empty or labels.empty:
        raise ValueError("Predictions and labels must be non-empty")
    joined = predictions.merge(
        labels[keys + [TARGET_COLUMN]], on=keys, how="left", validate="one_to_one"
    )
    if joined[TARGET_COLUMN].isna().any():
        raise ValueError("Predictions contain unit/cycle keys without matching labels")
    if not np.isfinite(joined[["RUL", "prediction"]].to_numpy()).all():
        raise ValueError("Labels and predictions must be finite")
    # Require the real last cycle of every labelled engine, even for LSTM output.
    expected = labels.loc[labels.groupby("unit")["cycle"].idxmax(), keys]
    actual = joined.loc[joined.groupby("unit")["cycle"].idxmax(), keys]
    if set(map(tuple, expected.to_numpy())) != set(map(tuple, actual.to_numpy())):
        raise ValueError("Predictions must include the last labelled cycle of every engine")
    return joined


def evaluate(
    predictions_path: Path,
    labels_path: Path,
    output_path: Path,
    evaluation_by_unit_path: Path,
) -> None:
    joined = align_predictions(load_predictions(predictions_path), load_labels(labels_path))
    last = joined.loc[joined.groupby("unit")["cycle"].idxmax()].sort_values("unit")
    metrics = calculate_metrics(last["RUL"], last["prediction"])
    metrics["n_engines"] = len(last)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_by_unit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)
    evaluate_by_unit(last).to_csv(evaluation_by_unit_path, index=False)


@app.command()
def main(
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    output_path: Path = RESULTS_DIR / "evaluation.csv",
    evaluation_by_unit_path: Path = RESULTS_DIR / "evaluation_by_unit.csv",
) -> None:
    evaluate(predictions_path, labels_path, output_path, evaluation_by_unit_path)


if __name__ == "__main__":
    app()
