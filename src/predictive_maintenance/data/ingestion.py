<<<<<<< HEAD
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from predictive_maintenance.config import RAW_DATA_DIR, COLUMNS
from predictive_maintenance.utils.helpers import (
    save_dataframe,
    validate_required_columns,
)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

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
        names=COLUMNS,
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


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

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

    validate_required_columns(train_df, COLUMNS)
    validate_required_columns(test_df, COLUMNS)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    train_path: Path = RAW_DATA_DIR / "train_FD001.txt",
    test_path: Path = RAW_DATA_DIR / "test_FD001.txt",
    rul_path: Path = RAW_DATA_DIR / "RUL_FD001.txt",
    output_dir: Path = RAW_DATA_DIR,
) -> None:
    """
    Ingest NASA CMAPSS datasets.
    """
    ingest_dataset(
        train_path=train_path,
        test_path=test_path,
        rul_path=rul_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    app()
=======
def 
>>>>>>> 6b6f4126abbe7c97def6b5a89eb094e0a7f041ba
