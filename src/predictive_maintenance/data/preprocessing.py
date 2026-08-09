from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from predictive_maintenance.config import RAW_DATA_DIR, INTERIM_DATA_DIR, COLUMNS
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
    """ Handle missing values using forward-fill followed by backward-fill. """

    missing_values_count = df.isnull().sum().sum()
    if missing_values_count == 0:
        logger.info("No missing values found.")
        return df

    logger.info(f"Handling {missing_values_count} missing values using forward-fill and backward-fill.")

    # Fill values within each engine independently
    df = df.groupby('unit').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

    missing_values_count_after = df.isnull().sum().sum()

    logger.success(f"Missing values handled. Remaining missing values: {missing_values_count_after}")

    return df


def sort_by_unit_and_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the DataFrame by 'unit' and 'cycle' columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to sort.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame.
    """
    return df.sort_values(by=['unit', 'cycle']).reset_index(drop=True)


def create_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a Remaining Useful Life (RUL) column for each engine unit.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing engine data.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'RUL' column.
    """
    logger.info("Creating Remaining Useful Life (RUL) column.")

    # Calculate RUL for each unit
    max_cycles = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = max_cycles - df['cycle']

    logger.success("RUL column created successfully.")

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataset by performing the following steps:
    1. Validate required columns.
    2. Remove empty columns.
    3. Remove constant columns.
    4. Handle missing values.
    5. Sort by 'unit' and 'cycle'.
    6. Create RUL column.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset.
    """
    validate_required_columns(df, COLUMNS)
    df = remove_empty_columns(df)
    df = remove_constant_columns(df)
    df = handle_missing_values(df)
    df = sort_by_unit_and_cycle(df)
    df = create_rul(df)

    logger.success(f"Data preprocessing completed: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


def preprocess_dataset(input_path: Path, output_path: Path) -> None:
    """
    Execute the data preprocessing pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the raw dataset file.
    output_path : Path
        Path to save the preprocessed dataset.
    """
    logger.info(f"Starting preprocessing for dataset: {input_path}")

    df = load_data(input_path)
    df_preprocessed = preprocess_data(df)

    save_dataframe(df_preprocessed, output_path, file_format="csv", index=False)

    logger.success(f"Preprocessed dataset saved to: {output_path}")


@app.command()
def main(
    input_file: Path = RAW_DATA_DIR / "train_data.csv",
    output_file: Path = INTERIM_DATA_DIR / "train_preprocessed.csv",
) -> None:
    """
    Preprocess the raw dataset and save the preprocessed dataset.

    Parameters
    ----------
    input_file : Path
        Path to the raw dataset file.
    output_file : Path
        Path to save the preprocessed dataset.
    """
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    preprocess_dataset(input_file, output_file)


if __name__ == "__main__":
    app()
