from pathlib import Path

import pandas as pd
from loguru import logger

from predictive_maintenance.config import RAW_DATA_DIR, EXTERNAL_DATA_DIR, RAW_COLUMNS
from predictive_maintenance.utils.helpers import (
    save_dataframe,
    validate_required_columns,
)


def load_cmapss(path: Path) -> pd.DataFrame:
    """
    Load a raw NASA CMAPSS dataset.

    Parameters
    ----------
    path : Path
        Path to the CMAPSS .txt file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with standardized column names.
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading CMAPSS dataset from {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    logger.success(
        f"CMAPSS dataset loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


def load_rul_data(path: Path) -> pd.DataFrame:
    """
    Load RUL ground-truth data.

    Parameters
    ----------
    path : Path
        Path to the RUL .txt file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing RUL values.
    """
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading RUL data from {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    logger.success(
        f"RUL data loaded: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    return df


def ingest_dataset(
    train_path: Path,
    test_path: Path,
    rul_path: Path,
    output_dir: Path,
) -> None:
    """
    Execute the complete CMAPSS data ingestion process.
    """
    logger.info("Starting data ingestion process...")

    # Load datasets
    train_df = load_cmapss(train_path)
    test_df = load_cmapss(test_path)
    rul_df = load_rul_data(rul_path)

    # Validate schemas
    logger.info("Validating dataset schemas...")

    validate_required_columns(train_df, RAW_COLUMNS)
    validate_required_columns(test_df, RAW_COLUMNS)
    validate_required_columns(rul_df, ["RUL"])

    # Save datasets
    logger.info(f"Saving ingested datasets to {output_dir}")

    train_output_path = output_dir / "train_data.csv"
    test_output_path = output_dir / "test_data.csv"
    rul_output_path = output_dir / "rul_data.csv"

    save_dataframe(
        train_df,
        train_output_path,
        file_format="csv",
        index=False,
    )

    save_dataframe(
        test_df,
        test_output_path,
        file_format="csv",
        index=False,
    )

    save_dataframe(
        rul_df,
        rul_output_path,
        file_format="csv",
        index=False,
    )

    logger.success(
        "Data ingestion process completed successfully."
    )
