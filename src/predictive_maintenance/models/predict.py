from pathlib import Path
import json

import pandas as pd
from loguru import logger

from predictive_maintenance.models.base import BaseModel
from predictive_maintenance.models import LSTMModel


# ============================================================
# Load features
# ============================================================

def load_features(
    features_path: Path,
) -> pd.DataFrame:
    """
    Load processed features.
    """

    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}"
        )

    logger.info(
        f"Loading features from {features_path}"
    )

    df = pd.read_csv(features_path)

    if df.empty:
        raise ValueError(
            f"Features file is empty: {features_path}"
        )

    return df


# ============================================================
# Prepare classical ML features
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare features for classical ML models.

    Classical models don't need engine identifiers.
    """

    return df.drop(
        columns=["unit", "cycle"],
        errors="ignore",
    ).copy()


# ============================================================
# Load best model
# ============================================================

def load_best_model(
    best_model_path: Path,
):
    """
    Load the model specified by best_model.json.

    Classical models:
        .pkl

    LSTM:
        .pt
    """

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Best model file not found: {best_model_path}"
        )

    logger.info(
        f"Loading model information from {best_model_path}"
    )

    # --------------------------------------------------------
    # Read best_model.json
    # --------------------------------------------------------

    with open(best_model_path, "r", encoding="utf-8") as f:
        best_model_info = json.load(f)

    if "model_name" not in best_model_info:
        raise ValueError(
            "best_model.json does not contain 'model_name'."
        )

    if "files" not in best_model_info:
        raise ValueError(
            "best_model.json does not contain 'files'."
        )

    model_name = best_model_info["model_name"]
    files = best_model_info["files"]

    logger.info(
        f"Best model selected: {model_name}"
    )

    # --------------------------------------------------------
    # Classical models
    # --------------------------------------------------------

    if model_name != "LSTM":

        if "model" not in files:
            raise ValueError(
                f"Missing 'model' entry in best_model.json "
                f"for {model_name}."
            )

        model_filename = files["model"]

        model_path = (
            best_model_path.parent / model_filename
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        logger.info(
            f"Loading {model_name} from {model_path}"
        )

        model = BaseModel.load(model_path)

        return model, model_name

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    if "weights" not in files:
        raise ValueError(
            "Missing 'weights' entry in best_model.json "
            "for LSTM."
        )

    weights_filename = files["weights"]

    weights_path = (
        best_model_path.parent / weights_filename
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"LSTM weights not found: {weights_path}"
        )

    logger.info(
        f"Loading LSTM from {weights_path}"
    )

    model = LSTMModel.load(weights_path)

    return model, model_name


# ============================================================
# Prepare LSTM output
# ============================================================

def prepare_lstm_output(
    df: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    """
    Create the output DataFrame for LSTM predictions.

    The LSTM predicts one RUL value per engine.

    Therefore we keep only the last recorded cycle
    of each engine.
    """

    # --------------------------------------------------------
    # Get last cycle of every engine
    # --------------------------------------------------------

    identifiers = (
        df.loc[
            df.groupby("unit")["cycle"].idxmax(),
            ["unit", "cycle"],
        ]
        .sort_values("unit")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Convert predictions to 1D
    # --------------------------------------------------------

    predictions = predictions.reshape(-1)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(identifiers) != len(predictions):
        raise ValueError(
            "LSTM prediction count does not match "
            "the number of engines.\n"
            f"Number of engines: {len(identifiers)}\n"
            f"Number of predictions: {len(predictions)}"
        )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    predictions_df = identifiers.copy()

    predictions_df["prediction"] = predictions

    return predictions_df


# ============================================================
# Prepare classical output
# ============================================================

def prepare_classical_output(
    df: pd.DataFrame,
    predictions,
) -> pd.DataFrame:
    """
    Create the output DataFrame for classical models.

    Classical models produce one prediction per row.
    """

    identifiers = df[
        ["unit", "cycle"]
    ].copy()

    predictions = predictions.reshape(-1)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(identifiers) != len(predictions):
        raise ValueError(
            "Classical model prediction count does not match "
            "the number of input rows.\n"
            f"Number of rows: {len(identifiers)}\n"
            f"Number of predictions: {len(predictions)}"
        )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    predictions_df = identifiers.copy()

    predictions_df["prediction"] = predictions

    return predictions_df


# ============================================================
# Main prediction function
# ============================================================

def predict(
    model_path: Path,
    features_path: Path,
    output_path: Path,
) -> None:
    """
    Generate predictions using the model specified
    by best_model.json.

    Classical models:
        one prediction per row.

    LSTM:
        one prediction per engine using the last
        available temporal sequence.
    """

    logger.info("========================================")
    logger.info("Starting prediction")
    logger.info("========================================")

    # ========================================================
    # Load model
    # ========================================================

    model, model_name = load_best_model(
        model_path
    )

    # ========================================================
    # Load features
    # ========================================================

    df = load_features(
        features_path
    )

    logger.info(
        f"Input dataset shape: {df.shape}"
    )

    # ========================================================
    # Validate required columns
    # ========================================================

    required_columns = [
        "unit",
        "cycle",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ========================================================
    # LSTM
    # ========================================================

    if model_name == "LSTM":

        logger.info(
            "Using LSTM prediction mode."
        )

        # IMPORTANT:
        # Keep unit and cycle.
        # LSTM needs them to build temporal sequences.

        X = df.copy()

        logger.info(
            f"LSTM prediction input shape: {X.shape}"
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(X)

        predictions = predictions.reshape(-1)

        # ----------------------------------------------------
        # Remove negative RUL predictions
        # ----------------------------------------------------

        predictions = predictions.clip(min=0)

        logger.info(
            f"LSTM generated {len(predictions)} predictions "
            f"for {df['unit'].nunique()} engines."
        )

        # ----------------------------------------------------
        # Create output
        # ----------------------------------------------------

        predictions_df = prepare_lstm_output(
            df=df,
            predictions=predictions,
        )

    # ========================================================
    # Classical models
    # ========================================================

    else:

        logger.info(
            f"Using classical model prediction mode: "
            f"{model_name}"
        )

        # ----------------------------------------------------
        # Remove unit and cycle
        # ----------------------------------------------------

        X = prepare_features(df)

        logger.info(
            f"Classical prediction input shape: {X.shape}"
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(X)

        predictions = predictions.reshape(-1)

        # ----------------------------------------------------
        # Remove negative RUL predictions
        # ----------------------------------------------------

        predictions = predictions.clip(min=0)

        logger.info(
            f"{model_name} generated "
            f"{len(predictions)} predictions."
        )

        # ----------------------------------------------------
        # Create output
        # ----------------------------------------------------

        predictions_df = prepare_classical_output(
            df=df,
            predictions=predictions,
        )

    # ========================================================
    # Save predictions
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # Final logs
    # ========================================================

    logger.success(
        f"Predictions saved to {output_path}"
    )

    logger.info(
        f"Output shape: {predictions_df.shape}"
    )

    logger.info(
        f"Model used: {model_name}"
    )

    logger.info("========================================")
    logger.success("Prediction completed successfully.")
    logger.info("========================================")
