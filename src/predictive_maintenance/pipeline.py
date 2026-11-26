"""
Reproducible ingestion, preprocessing, feature engineering, training and evaluation.

Usage:
    python -m predictive_maintenance.pipeline
    python -m predictive_maintenance.pipeline --train-lstm   # optional experiment
"""

import shutil

from loguru import logger
import typer

from predictive_maintenance.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RUL_CAP,
    SETTINGS,
)
from predictive_maintenance.data.ingestion import ingest_dataset
from predictive_maintenance.data.preprocessing import preprocess_dataset
from predictive_maintenance.data.validation import validate_dataset
from predictive_maintenance.features.engineering import engineer_dataset
from predictive_maintenance.models.evaluate import evaluate
from predictive_maintenance.models.predict import predict
from predictive_maintenance.models.train import train_pipeline
from predictive_maintenance.utils.helpers import load_dataframe

app = typer.Typer()


@app.command()
def main(
    n_splits: int = SETTINGS.get("n_splits", 5),
    train_lstm: bool = False,
) -> None:
    """Run the complete pipeline end to end, from raw .txt files to saved, evaluated models."""

    logger.info("################################################")
    logger.info("STEP 1/4 — Data ingestion")
    logger.info("################################################")

    ingest_dataset(
        train_path=EXTERNAL_DATA_DIR / "train_FD001.txt",
        test_path=EXTERNAL_DATA_DIR / "test_FD001.txt",
        rul_path=EXTERNAL_DATA_DIR / "RUL_FD001.txt",
        output_dir=RAW_DATA_DIR,
    )

    logger.info("################################################")
    logger.info("STEP 2/4 — Preprocessing (train, then test)")
    logger.info("################################################")

    constant_columns_path = INTERIM_DATA_DIR / "constant_columns.json"

    preprocess_dataset(
        input_path=RAW_DATA_DIR / "train_data.csv",
        output_path=INTERIM_DATA_DIR / "train_preprocessed.csv",
        is_train=True,
        constant_columns_path=constant_columns_path,
        rul_cap=RUL_CAP,
    )

    preprocess_dataset(
        input_path=RAW_DATA_DIR / "test_data.csv",
        output_path=INTERIM_DATA_DIR / "test_preprocessed.csv",
        is_train=False,
        constant_columns_path=constant_columns_path,
        rul_path=RAW_DATA_DIR / "rul_data.csv",
        rul_cap=RUL_CAP,
    )

    logger.info("################################################")
    logger.info("STEP 2b/4 — Validation")
    logger.info("################################################")

    validate_dataset(load_dataframe(INTERIM_DATA_DIR / "train_preprocessed.csv"))
    validate_dataset(load_dataframe(INTERIM_DATA_DIR / "test_preprocessed.csv"))

    logger.info("################################################")
    logger.info("STEP 3/4 — Feature engineering")
    logger.info("################################################")

    engineer_dataset(
        input_path=INTERIM_DATA_DIR / "train_preprocessed.csv",
        output_path=PROCESSED_DATA_DIR / "train_features.csv",
        labels_path=PROCESSED_DATA_DIR / "train_labels.csv",
    )

    engineer_dataset(
        input_path=INTERIM_DATA_DIR / "test_preprocessed.csv",
        output_path=PROCESSED_DATA_DIR / "test_features.csv",
        labels_path=PROCESSED_DATA_DIR / "test_labels.csv",
    )

    logger.info("################################################")
    logger.info("STEP 4/4 — Training (CV diagnostic + final models + test evaluation)")
    logger.info("################################################")

    train_pipeline(
        features_path=PROCESSED_DATA_DIR / "train_features.csv",
        labels_path=PROCESSED_DATA_DIR / "train_labels.csv",
        test_features_path=PROCESSED_DATA_DIR / "test_features.csv",
        test_labels_path=PROCESSED_DATA_DIR / "test_labels.csv",
        n_splits=n_splits,
        train_lstm=train_lstm,
    )

    logger.info("################################################")
    logger.info("STEP 4/4 — Prediction on test set")
    logger.info("################################################")
    predict(
        model_path=MODELS_DIR / "best_model.json",
        features_path=PROCESSED_DATA_DIR / "test_features.csv",
        output_path=PROCESSED_DATA_DIR / "test_predictions.csv",
    )

    logger.info("################################################")
    logger.info("STEP 4/4 — Evaluation")
    logger.info("################################################")
    evaluate(
        predictions_path=PROCESSED_DATA_DIR / "test_predictions.csv",
        labels_path=PROCESSED_DATA_DIR / "test_labels.csv",
        output_path=RESULTS_DIR / "evaluation.csv",
        evaluation_by_unit_path=RESULTS_DIR / "evaluation_by_unit.csv",
    )

    logger.success("################################################")
    logger.success("Pipeline completed — see models/best_model.json for the selected model.")
    logger.success("################################################")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(constant_columns_path, MODELS_DIR / "constant_columns.json")
    logger.info(
        f"Copied {constant_columns_path} -> {MODELS_DIR / 'constant_columns.json'} (for serving)."
    )


if __name__ == "__main__":
    app()
