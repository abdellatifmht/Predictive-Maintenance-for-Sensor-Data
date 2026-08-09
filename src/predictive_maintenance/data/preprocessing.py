from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from predictive_maintenance.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COLUMNS
from predictive_maintenance.utils.helpers import (
    save_dataframe,
    validate_required_columns,
)

app = typer.Typer()


def load_data(path: Path) -> pd.DataFrame:
    """
    Load a raw dataset from a specified path.

    Parameters
    ----------
    path : Path
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with standardized column names.
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading dataset from {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    logger.success(
        f"Dataset loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that contain only NaN values from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.

    Returns
    -------
    pd.DataFrame
        DataFrame with empty columns removed.
    """
    initial_columns = df.shape[1]
    df_cleaned = df.dropna(axis=1, how='all')
    removed_columns = initial_columns - df_cleaned.shape[1]

    if removed_columns > 0:
        logger.info(f"Removed {removed_columns} empty columns.")
    else:
        logger.info("No empty columns to remove.")

    return df_cleaned


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with zero variance (constant columns) from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.

    Returns
    -------
    pd.DataFrame
        DataFrame with constant columns removed.
    """
    constant_columns = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        df = df.drop(columns=constant_columns)
        logger.info(f"Removed {len(constant_columns)} constant columns: {constant_columns}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    