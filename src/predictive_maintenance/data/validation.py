from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from predictive_maintenance.config import INTERIM_DATA_DIR, REQUIRED_COLUMNS
from predictive_maintenance.utils.helpers import load_dataframe

app = typer.Typer()


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate the schema of the DataFrame against the expected columns.
    """
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        logger.error(f"Missing columns in the DataFrame: {missing_columns}")
        raise ValueError(f"Missing columns in the DataFrame: {missing_columns}")

    logger.success("DataFrame schema validation passed. All required columns are present.")


def validate_missing_values(df: pd.DataFrame) -> None:
    """
    Validate that there are no missing values in the DataFrame.
    """
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]

    if not missing_columns.empty:
        logger.error(f"DataFrame contains missing values in columns: {missing_columns.index.tolist()}")
        raise ValueError(f"DataFrame contains missing values in columns: {missing_columns.index.tolist()}")

    logger.success("DataFrame missing values validation passed. No missing values found.")


def validate_duplicates(df: pd.DataFrame) -> None:
    """
    Check for duplicate rows in the DataFrame and raise an error if any are found.
    """
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        logger.error(f"DataFrame contains {duplicates} duplicate rows.")
        raise ValueError(f"DataFrame contains {duplicates} duplicate rows.")
    logger.success("DataFrame duplicate validation passed. No duplicate rows found.")


def validate_unit_cycle(df: pd.DataFrame) -> None:
    """
    Validate that the 'unit' and 'cycle' columns contain valid values.
    """
    if (df["unit"] <= 0).any():
        logger.error("Invalid values found in 'unit' column. All values must be positive integers.")
        raise ValueError("Invalid values found in 'unit' column. All values must be positive integers.")

    if (df["cycle"] < 0).any():
        logger.error("Invalid values found in 'cycle' column. All values must be non-negative integers.")
        raise ValueError("Invalid values found in 'cycle' column. All values must be non-negative integers.")

    for unit, group in df.groupby("unit"):
        if not group["cycle"].is_monotonic_increasing:
            logger.error(f"'cycle' values for unit {unit} are not monotonically increasing.")
            raise ValueError(f"'cycle' values for unit {unit} are not monotonically increasing.")

    logger.success("DataFrame 'unit' and 'cycle' validation passed. All values are valid.")


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """
    Validate that all sensor columns contain numeric values.
    """
    sensor_columns = [col for col in df.columns if col.startswith("sensor_") or col.startswith("os_")]
    non_numeric_columns = [col for col in sensor_columns if not pd.api.types.is_numeric_dtype(df[col])]

    if non_numeric_columns:
        logger.error(f"Columns {non_numeric_columns} contain non-numeric values.")
        raise ValueError(f"Columns {non_numeric_columns} contain non-numeric values.")

    logger.success("DataFrame numeric columns validation passed. All sensor columns contain numeric values.")


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Run all validation checks on the DataFrame.
    """
    validate_schema(df)
    validate_missing_values(df)
    validate_duplicates(df)
    validate_unit_cycle(df)
    validate_numeric_columns(df)

    logger.success("All dataset validation checks passed successfully.")


@app.command()
def main(
    input_file: Path = INTERIM_DATA_DIR / "train_preprocessed.csv",
) -> None:
    """
    Validate the preprocessed dataset.

    Parameters
    ----------
    input_file : Path
        Path to the preprocessed dataset file.
    """
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    df = load_dataframe(input_file)

    validate_dataset(df)


if __name__ == "__main__":
    app()
