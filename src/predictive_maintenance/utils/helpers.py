from pathlib import Path
import pandas as pd


def save_dataframe(
    df: pd.DataFrame,
    file_path: Path,
    file_format: str = "csv",
    index: bool = False
) -> None:
    """
    Save a DataFrame to a specified file format.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (Path): The path where the file will be saved.
    file_format (str): The format to save the file in ('csv' or 'parquet').
    index (bool): Whether to write row names (index).

    Raises:
    ValueError: If an unsupported file format is provided.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "csv":
        df.to_csv(file_path, index=index)
    elif file_format == "parquet":
        df.to_parquet(file_path, index=index)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Use 'csv' or 'parquet'.")


# def load_dataframe(
#     file_path: Path, 
#     file_format: str = "csv"
# ) -> pd.DataFrame:
#     """
#     Load a DataFrame from a specified file format.

#     Parameters:
#     file_path (Path): The path of the file to load.
#     file_format (str): The format of the file to load ('csv' or 'parquet').

#     Returns:
#     pd.DataFrame: The loaded DataFrame.

#     Raises:
#     ValueError: If an unsupported file format is provided.
#     """

#     if not file_path.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")

#     if file_format == "csv":
#         return pd.read_csv(file_path)
#     elif file_format == "parquet":
#         return pd.read_parquet(file_path)
#     else:
#         raise ValueError(f"Unsupported file format: {file_format}. Use 'csv' or 'parquet'.")


def validate_required_colums(
        df: pd.DataFrame, 
        required_columns: list[str]
):
    """
    Validate that the required columns are present in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate.
    required_columns (list[str]): A list of required column names.

    Raises:
    ValueError: If any required columns are missing from the DataFrame.
    """

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
