from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from predictive_maintenance.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from predictive_maintenance.utils.helpers import (
    load_dataframe,
    save_dataframe,
)


def add_statistical_features(
    df: pd.DataFrame,
    sensor_columns: list[str] = None,
    window: int = 6
) -> pd.DataFrame:
    """
    Add rolling statistical features for sensor measurements.

    Features:
        - rolling mean
        - rolling standard deviation
    """
    logger.info("Adding statistical features...")

    df = df.copy()
    sensor_columns = sensor_columns if sensor_columns is not None else [
        col for col in df.columns if col.startswith("sensor")
    ]
    for sensor in sensor_columns:
        grouped = df.groupby("unit")[sensor]

        df[f"{sensor}_rolling_mean_{window}"] = (
            grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).mean()
            )
        )

        df[f"{sensor}_rolling_std_{window}"] = (
            grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).std().fillna(0)
            )
        )
    nan_counts = df.isna().sum()

    nan_counts = nan_counts[nan_counts > 0]

    if not nan_counts.empty:
        logger.warning(
            f"NaN detected in features:\n{nan_counts}"
        )
    return df


def add_trend_features(
    df: pd.DataFrame,
    window: int = 10,
    sensor_columns: list[str] = None,
) -> pd.DataFrame:
    """
    Add local trend features using rolling linear regression.

    The slope represents the local degradation trend of each sensor.
    """
    logger.info(
        f"Adding trend features with window={window}..."
    )

    df = df.copy()

    def calculate_slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))

        return np.polyfit(x, values, 1)[0]

    sensor_columns = sensor_columns if sensor_columns is not None else [
        col for col in df.columns if col.startswith("sensor")
    ]
    for sensor in sensor_columns:
        df[f"{sensor}_slope"] = (
            df.groupby("unit")[sensor]
            .transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=1,
                ).apply(
                    calculate_slope,
                    raw=True,
                )
            )
        )
    nan_counts = df.isna().sum()

    nan_counts = nan_counts[nan_counts > 0]

    if not nan_counts.empty:
        logger.warning(
            f"NaN detected in features:\n{nan_counts}"
        )
    return df


def create_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    """
    logger.info("Starting feature engineering...")

    df = df.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    # df = add_statistical_features(df, sensor_columns)
    df = add_trend_features(df, window=10, sensor_columns=sensor_columns)

    logger.success(
        f"Feature engineering completed: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


def engineer_dataset(
    input_path: Path,
    output_path: Path,
    labels_path: Path,
) -> None:
    """
    Load preprocessed data, create features and save the result.
    """
    logger.info(
        f"Loading preprocessed data from {input_path}"
    )

    df = load_dataframe(input_path)

    labels = df[["unit", "cycle", "RUL"]].copy()

    features_input = df.drop(columns=["RUL"])

    sensor_columns = [
        col for col in df.columns if col.startswith("sensor")
    ]

    features = create_features(features_input, sensor_columns)
    save_dataframe(
        features,
        output_path,
        file_format="csv",
        index=False,
    )

    save_dataframe(
        labels,
        labels_path,
        file_format="csv",
        index=False,
    )

    logger.success(
        f"Features saved to {output_path} and labels saved to {labels_path}"
    )
