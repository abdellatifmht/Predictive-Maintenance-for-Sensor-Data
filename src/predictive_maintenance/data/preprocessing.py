import json
from pathlib import Path

from loguru import logger
import pandas as pd

from predictive_maintenance.config import RAW_COLUMNS
from predictive_maintenance.utils.helpers import (
    save_dataframe,
    validate_required_columns,
)


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

    df = pd.read_csv(path)

    logger.success(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

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
    df_cleaned = df.dropna(axis=1, how="all")
    removed_columns = initial_columns - df_cleaned.shape[1]

    if removed_columns > 0:
        logger.info(f"Removed {removed_columns} empty columns.")
    else:
        logger.info("No empty columns to remove.")

    return df_cleaned


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify columns with zero variance (constant columns).
    """
    return [
        col
        for col in df.columns
        if col not in ("unit", "cycle", "RUL") and df[col].nunique(dropna=False) <= 1
    ]


def remove_constant_columns(
    df: pd.DataFrame,
    constant_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove constant columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    constant_columns : list[str] | None
        Columns to drop. If None, they are computed from `df` itself (original
        behaviour — use this ONLY for the train set). If provided, those exact
        columns are dropped (ignoring any that are missing) — use this for the
        test set, passing the list computed on train, so both datasets end up
        with an identical, consistent feature schema.
    """
    if constant_columns is None:
        constant_columns = get_constant_columns(df)

    columns_to_drop = [col for col in constant_columns if col in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        logger.info(f"Removed {len(columns_to_drop)} constant columns: {columns_to_drop}")
    else:
        logger.info("No constant columns to remove.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using forward-fill followed by backward-fill."""

    missing_values_count = df.isnull().sum().sum()
    if missing_values_count == 0:
        logger.info("No missing values found.")
        return df

    logger.info(
        f"Handling {missing_values_count} missing values using forward-fill and backward-fill."
    )

    # Fill values within each engine independently
    # Use only earlier observations; future readings must not leak into a cycle's features.
    df = sort_by_unit_and_cycle(df).copy()
    value_columns = [col for col in df.columns if col not in ("unit", "cycle", "RUL")]
    df[value_columns] = df.groupby("unit")[value_columns].ffill()
    if df[value_columns].isna().any().any():
        raise ValueError("Missing initial sensor readings cannot be filled causally.")

    missing_values_count_after = df.isnull().sum().sum()

    logger.success(
        f"Missing values handled. Remaining missing values: {missing_values_count_after}"
    )

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
    return df.sort_values(by=["unit", "cycle"]).reset_index(drop=True)


def create_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """
    Create a Remaining Useful Life (RUL) column for each engine unit.

    ONLY valid for a TRAIN-style dataset where each unit's trajectory runs
    all the way to failure (its last recorded cycle IS the failure). For a
    TEST-style dataset with truncated trajectories, use
    `add_rul_from_ground_truth` instead — applying this function to test data
    would silently assign RUL=0 to every unit's last cycle, which is wrong.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing engine data (must run to failure).
    cap : int | None
        Piecewise-linear RUL cap. In early life, an engine shows almost no
        sign of degradation, so distinguishing "500 cycles left" from
        "300 cycles left" adds noise rather than signal — the standard
        practice on C-MAPSS (Heimes, 2008) is to cap RUL at a fixed ceiling
        (125 cycles is the commonly used value in the literature). Pass
        `cap=None` to disable and keep the raw linear RUL.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'RUL' column.
    """
    logger.info("Creating Remaining Useful Life (RUL) column.")

    # Calculate RUL for each unit
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]

    if cap is not None:
        n_capped = (df["RUL"] > cap).sum()
        df["RUL"] = df["RUL"].clip(upper=cap)
        logger.info(f"RUL capped at {cap} cycles ({n_capped:,} rows affected).")

    logger.success("RUL column created successfully.")

    return df


def add_rul_from_ground_truth(
    df: pd.DataFrame,
    rul_df: pd.DataFrame,
    cap: int | None = 125,
) -> pd.DataFrame:
    """
    Create a per-cycle RUL column for a TRUNCATED (test-style) dataset, using
    ground-truth RUL values known only at each unit's LAST recorded cycle.

    Each test trajectory stops before failure. `rul_df` (e.g. RUL_FD001.txt)
    gives the true remaining life at that last recorded cycle. We extend that
    known value backwards through the trajectory:

        RUL(cycle) = RUL_true_at_last_cycle + (max_cycle_in_trajectory - cycle)

    Parameters
    ----------
    df : pd.DataFrame
        Test-style DataFrame (truncated trajectories), must contain 'unit' and 'cycle'.
    rul_df : pd.DataFrame
        DataFrame with one row per unit, in unit order, containing a 'RUL' column
        (this matches the raw format of RUL_FD001.txt loaded via `load_rul_data`).
    cap : int | None
        Same piecewise-linear cap as `create_rul` — MUST be the same value used
        for the training set, otherwise train and test targets follow different
        scales.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'RUL' column.
    """
    logger.info("Creating Remaining Useful Life (RUL) column from ground truth (test set).")

    rul_lookup = rul_df.reset_index(drop=True).copy()
    rul_lookup["unit"] = rul_lookup.index + 1  # RUL_FD001.txt is ordered by unit_number, 1-indexed
    rul_lookup = rul_lookup.rename(columns={"RUL": "_rul_at_last_cycle"})

    df = df.merge(
        rul_lookup[["unit", "_rul_at_last_cycle"]], on="unit", how="left", validate="many_to_one"
    )
    max_cycle = df.groupby("unit")["cycle"].transform("max")

    if df["_rul_at_last_cycle"].isna().any():
        missing_units = df.loc[df["_rul_at_last_cycle"].isna(), "unit"].unique().tolist()
        raise ValueError(f"No ground-truth RUL found for units: {missing_units}")

    df["RUL"] = df["_rul_at_last_cycle"] + (max_cycle - df["cycle"])
    df = df.drop(columns=["_rul_at_last_cycle"])

    if cap is not None:
        n_capped = (df["RUL"] > cap).sum()
        df["RUL"] = df["RUL"].clip(upper=cap)
        logger.info(f"RUL capped at {cap} cycles ({n_capped:,} rows affected).")

    logger.success("RUL column (ground truth) created successfully.")

    return df


def preprocess_data(
    df: pd.DataFrame,
    is_train: bool,
    constant_columns: list[str] | None = None,
    rul_df: pd.DataFrame | None = None,
    rul_cap: int | None = 125,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Preprocess the raw dataset by performing the following steps:
    1. Validate required columns.
    2. Remove empty columns.
    3. Remove constant columns.
    4. Handle missing values.
    5. Sort by 'unit' and 'cycle'.
    6. Create RUL column (from trajectory end for train, from ground truth for test).

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset to preprocess.
    is_train : bool
        True for a train-style dataset (trajectories run to failure) — RUL is
        computed directly. False for a test-style (truncated) dataset — RUL is
        reconstructed from `rul_df` (required in that case).
    constant_columns : list[str] | None
        Explicit list of constant columns to drop. Pass None for train (they
        will be detected from the train data itself and returned so the SAME
        list can be reused for the test set — see `get_constant_columns`).
    rul_df : pd.DataFrame | None
        Ground-truth RUL data (e.g. rul_data.csv), REQUIRED when is_train=False.
    rul_cap : int | None
        Piecewise-linear RUL cap, forwarded to `create_rul` /
        `add_rul_from_ground_truth`. Must be identical for train and test.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The preprocessed dataset, and the list of constant columns that were
        removed (pass this back in for the test set so schemas match).
    """
    if not is_train and rul_df is None:
        raise ValueError("rul_df is required when is_train=False (test-style dataset).")
    if not is_train and constant_columns is None:
        raise ValueError("Training constant_columns are required for test preprocessing.")

    validate_required_columns(df, RAW_COLUMNS)
    if constant_columns is None:
        constant_columns = get_constant_columns(df)
    df = remove_constant_columns(df, constant_columns=constant_columns)

    df = sort_by_unit_and_cycle(df)
    df = handle_missing_values(df)
    df = sort_by_unit_and_cycle(df)

    if is_train:
        df = create_rul(df, cap=rul_cap)
    else:
        df = add_rul_from_ground_truth(df, rul_df, cap=rul_cap)

    logger.success(f"Data preprocessing completed: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df, constant_columns


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    is_train: bool,
    constant_columns_path: Path,
    rul_path: Path | None = None,
    rul_cap: int | None = 125,
) -> None:
    """
    Execute the data preprocessing pipeline.

    Parameters
    ----------
    input_path : Path
        Path to the raw dataset file.
    output_path : Path
        Path to save the preprocessed dataset.
    is_train : bool
        Whether `input_path` is the train (trajectories run to failure) or
        test (truncated) dataset.
    constant_columns_path : Path
        Where the list of constant columns is saved (train) or loaded from
        (test) — a small JSON file, so the test set drops EXACTLY the same
        columns as the train set, even if run as a separate CLI invocation.
    rul_path : Path | None
        Path to the ground-truth RUL file — REQUIRED when is_train=False.
    rul_cap : int | None
        Piecewise-linear RUL cap (see `create_rul`).
    """
    logger.info(f"Starting preprocessing for dataset: {input_path}")

    df = load_data(input_path)

    if is_train:
        constant_columns = None  # detected from this (train) data
        rul_df = None
    else:
        if rul_path is None:
            raise ValueError("--rul-path is required when preprocessing the test set.")
        if not constant_columns_path.exists():
            raise FileNotFoundError(
                f"{constant_columns_path} not found — preprocess the TRAIN set first, "
                "it saves this file for the test set to reuse."
            )
        with open(constant_columns_path) as f:
            constant_columns = json.load(f)
        rul_df = (
            load_data(rul_path)
            if rul_path.suffix == ".csv"
            else pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
        )

    df_preprocessed, constant_columns = preprocess_data(
        df,
        is_train=is_train,
        constant_columns=constant_columns,
        rul_df=rul_df,
        rul_cap=rul_cap,
    )

    save_dataframe(df_preprocessed, output_path, file_format="csv", index=False)
    logger.success(f"Preprocessed dataset saved to: {output_path}")

    if is_train:
        constant_columns_path.parent.mkdir(parents=True, exist_ok=True)
        with open(constant_columns_path, "w") as f:
            json.dump(constant_columns, f)
        logger.success(f"Constant columns saved to: {constant_columns_path}")
