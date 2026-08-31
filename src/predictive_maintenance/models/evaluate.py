from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

TARGET_COLUMN = "RUL"


def load_predictions(
    predictions_path: Path,
) -> pd.DataFrame:
    """
    Load model predictions.
    """

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: "
            f"{predictions_path}"
        )

    return pd.read_csv(
        predictions_path
    )


def load_labels(
    labels_path: Path,
) -> pd.DataFrame:
    """
    Load ground-truth labels.
    """

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}"
        )

    return pd.read_csv(
        labels_path
    )


def phm08_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate the PHM08 score.
    """
    errors = y_pred - y_true

    score = 0.0

    for error in errors:
        if error < 0:
            score += np.exp(-error / 13) - 1
        else:
            score += np.exp(error / 10) - 1

    return score


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Calculate regression metrics and the PHM08 score.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "PHM08": phm08_score(y_true, y_pred),
    }


def evaluate_by_unit(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    results = []

    for unit, group in predictions.groupby("unit"):

        mae = mean_absolute_error(
            group["RUL"],
            group["prediction"],
        )

        rmse = mean_squared_error(
            group["RUL"],
            group["prediction"],
        ) ** 0.5

        results.append(
            {
                "unit": unit,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return pd.DataFrame(results)


def evaluate(
    predictions_path: Path,
    labels_path: Path,
    output_path: Path,
    evaluation_by_unit_path: Path
) -> None:
    """
    Evaluate model predictions.
    """

    logger.info(
        "Starting model evaluation..."
    )

    predictions = load_predictions(
        predictions_path
    )

    labels = load_labels(
        labels_path
    )

    if TARGET_COLUMN not in labels.columns:
        raise ValueError(
            f"Missing target column: "
            f"{TARGET_COLUMN}"
        )

    if "unit" in predictions.columns and "cycle" in predictions.columns:

        predictions = predictions.merge(
            labels[
                ["unit", "cycle", TARGET_COLUMN]
            ],
            on=["unit", "cycle"],
            how="inner",
        )

        y_true = predictions[
            TARGET_COLUMN
        ]

        y_pred = predictions[
            "prediction"
        ]

    else:

        if len(predictions) != len(labels):
            raise ValueError(
                "Predictions and labels have "
                "different numbers of rows."
            )

        y_true = labels[
            TARGET_COLUMN
        ]

        y_pred = predictions[
            "prediction"
        ]

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    evaluation_by_unit = evaluate_by_unit(
        predictions
    )

    logger.info(
        f"MAE:  {metrics['MAE']:.4f}"
    )

    logger.info(
        f"RMSE: {metrics['RMSE']:.4f}"
    )

    logger.info(
        f"R²:   {metrics['R2']:.4f}"
    )
    logger.info(
        f"PHM08: {metrics['PHM08']:.4f}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_path,
        index=False,
    )

    logger.success(
        f"Metrics saved to {output_path}"
    )

    evaluation_by_unit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_by_unit.to_csv(
        evaluation_by_unit_path,
        index=False,
    )

    logger.success(
        f"Evaluation by unit saved to {evaluation_by_unit_path}"
    )
