from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from loguru import logger
from sklearn.model_selection import GroupKFold

from predictive_maintenance.config import (
    MODELS_DIR,
    RESULTS_DIR,
)
from predictive_maintenance.models import (
    LinearRegressionModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
)


N_SPLITS = 5
TARGET = "RUL"
ID_COLUMNS = ["unit", "cycle"]


def load_data(features_path: Path, labels_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load features and labels from CSV files, and perform basic validation checks.
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    if "RUL" not in labels.columns:
        raise ValueError("Missing RUL column")

    for col in ID_COLUMNS:
        if col not in features.columns:
            raise ValueError(f"Missing column: {col}")

    if len(features) != len(labels):
        raise ValueError("Features and labels have different sizes")

    if features.isna().sum().sum() > 0:
        raise ValueError("Features contain NaN values")

    if labels[TARGET].isna().sum() > 0:
        raise ValueError("Labels contain NaN values")

    return features, labels[TARGET]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data for model training by dropping ID columns.

    NOTE: le LSTM ne doit JAMAIS passer par cette fonction -- il a besoin de
    'unit'/'cycle' pour reconstruire ses séquences par moteur. Partout ci-dessous
    où un DataFrame est destiné au LSTM, c'est le DataFrame complet (avec
    identifiants) qui est utilisé, pas le résultat de prepare_data().
    """
    return df.drop(columns=ID_COLUMNS, errors="ignore")


def evaluate(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate the model on the given data and return MAE and RMSE metrics.
    """
    pred = model.predict(X)
    pred = pred.clip(min=0)

    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
    }


def get_models() -> list[object]:
    """
    Return a list of model instances to be trained.

    Le LSTM n'est PAS inclus ici volontairement : cette liste alimente
    `cross_validate`, qui appelle `prepare_data()` (drop unit/cycle) avant fit,
    ce qui casserait le LSTM. Il a son propre chemin d'entraînement
    (`train_final_lstm`) et sa propre évaluation dans `evaluate_test`.
    """
    return [
        LinearRegressionModel(),
        XGBoostModel(),
        LightGBMModel(),
    ]


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, fold: int) -> list[dict]:
    """
    Train multiple models on the training data and evaluate them on the validation data.
    """

    results = []

    for model in get_models():
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_val, y_val)

        logger.success(
            f"Fold {fold} | {model.name} | "
            f"MAE={metrics['MAE']:.2f} | "
            f"RMSE={metrics['RMSE']:.2f}"
        )

        results.append({
            "fold": fold,
            "model": model.name,
            **metrics,
        })

    return results


def cross_validate(features: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[dict]:
    """
    Perform GroupKFold cross-validation on the given features and labels.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = features["unit"]

    results = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(features, y, groups),
        start=1
    ):
        train = features.iloc[train_idx]
        val = features.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Check for engine leakage
        if set(train["unit"]) & set(val["unit"]):
            raise ValueError("Engine leakage detected")

        X_train = prepare_data(train)
        X_val = prepare_data(val)

        results.extend(
            train_models(
                X_train,
                y_train,
                X_val,
                y_val,
                fold,
            )
        )

    return results


def save_cv_results(results: list[dict]) -> None:
    """
    Save cross-validation results to CSV files.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_DIR / "groupkfold_results.csv",
        index=False,
    )

    summary = (
        df.groupby("model")
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
    )

    logger.success("\nCross-validation results:")
    logger.info(summary)


MODEL_FILES = {
    "LinearRegression": "linear_regression.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
}


def train_final_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train final models on the entire dataset and save them to disk.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models = {}

    for model in get_models():
        logger.success(f"Training {model.name}...")

        model.fit(X, y)

        filename = MODEL_FILES[model.name]
        model.save(MODELS_DIR / filename)

        # Save feature names for later use
        feature_file = MODELS_DIR / filename.replace(
            ".pkl",
            "_features.json",
        )
        if model.name in ["XGBoost", "LightGBM"]:
            feature_importance_file = MODELS_DIR / filename.replace(
                ".pkl",
                "_feature_importance.csv",
            )
            pd.DataFrame({
                "feature": X.columns.tolist(),
                "importance": model.model.feature_importances_,
            }).to_csv(feature_importance_file, index=False)

        with open(feature_file, "w") as f:
            json.dump(X.columns.tolist(), f)

        models[model.name] = model

    return models


def train_final_lstm(features: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2) -> LSTMModel:
    """
    Train the final LSTM on the (train) dataset given, with a held-out slice
    of ENGINES (never rows) used for validation / early stopping.

    IMPORTANT : `features` doit être le jeu de TRAIN, jamais le jeu de test --
    entraîner sur le test annulerait l'évaluation qui suit (le modèle aurait
    déjà vu les données sur lesquelles on prétend le juger).
    """
    units = sorted(features["unit"].unique())
    val_units = set(units[-max(1, int(len(units) * val_fraction)):])
    train_mask = ~features["unit"].isin(val_units)
    val_mask = features["unit"].isin(val_units)

    logger.success("Training LSTM...")

    model = LSTMModel()
    # attention : features complet (avec unit/cycle), PAS prepare_data(features)
    model.fit(features[train_mask], y[train_mask], features[val_mask], y[val_mask])
    model.save(MODELS_DIR / "lstm")

    return model


def get_last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the last cycle for each unit in the DataFrame.
    """
    return (
        df.loc[df.groupby("unit")["cycle"].idxmax()]
        .sort_values("unit")
        .reset_index(drop=True)
    )


def evaluate_test(models: dict, test_features: pd.DataFrame, test_y: pd.Series) -> list[dict]:
    """
    Evaluate the trained models on the test dataset and return the results.
    """
    # Create a DataFrame with unit, cycle, and RUL for evaluation
    labels = pd.DataFrame({
        "unit": test_features["unit"],
        "cycle": test_features["cycle"],
        "RUL": test_y,
    })
    labels = get_last_cycle(labels)
    y_true = labels["RUL"].values

    last_cycle_test = get_last_cycle(test_features)
    X_test = prepare_data(last_cycle_test)

    results = []

    for name, model in models.items():
        if name == "LSTM":
            pred = model.predict(test_features)
        else:
            pred = model.predict(X_test)

        pred = pred.clip(min=0)

        mae = mean_absolute_error(y_true, pred)
        rmse = mean_squared_error(y_true, pred) ** 0.5

        logger.success(
            f"{name} | "
            f"MAE={mae:.2f} | "
            f"RMSE={rmse:.2f}"
        )

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "n_engines": len(y_true),
        })

    return results


def save_best_model(results: list[dict]) -> dict:
    """
    Save the best model based on RMSE to a JSON file.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("RMSE")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        RESULTS_DIR / "test_evaluation.csv",
        index=False,
    )

    best = df.iloc[0]
    name = best["model"]

    # Le LSTM n'a pas d'entrée dans MODEL_FILES (format de sauvegarde
    # différent : un seul .pt contenant poids + scaler + colonnes de
    # features, pas de .pkl + _features.json séparés) -- cas traité à part
    # pour ne pas planter avec un KeyError si le LSTM s'avère être le
    # meilleur modèle.
    if name == "LSTM":
        files = {"weights": "lstm.pt"}
    else:
        filename = MODEL_FILES[name]
        files = {
            "model": filename,
            "features": filename.replace(
                ".pkl",
                "_features.json",
            ),
        }

    info = {
        "model_name": name,
        "test_RMSE": float(best["RMSE"]),
        "test_MAE": float(best["MAE"]),
        "files": files,
    }

    with open(MODELS_DIR / "best_model.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.success(
        f"\nBest model: {name} "
        f"(RMSE={best['RMSE']:.2f})"
    )

    return info


def train_pipeline(
    features_path: Path,
    labels_path: Path,
    test_features_path: Path | None = None,
    test_labels_path: Path | None = None,
    n_splits: int = N_SPLITS,
):
    # Load data
    features, y = load_data(
        features_path,
        labels_path,
    )

    # 2. GroupKFold
    results = cross_validate(
        features,
        y,
        n_splits,
    )

    save_cv_results(results)

    # 3. Train final models on the entire dataset
    X = prepare_data(features)

    models = train_final_models(
        X,
        y,
    )

    # LSTM entraîné séparément, sur le TRAIN (features complet, avec
    # unit/cycle -- surtout pas `X`, qui les a déjà supprimés), puis ajouté
    # au même dict que les autres modèles pour que evaluate_test les traite
    # tous ensemble.
    lstm_model = train_final_lstm(features, y)
    models["LSTM"] = lstm_model


    # 4. Official evaluation
    if test_features_path and test_labels_path:

        test_features, test_y = load_data(
            test_features_path,
            test_labels_path,
        )

        results = evaluate_test(
            models,
            test_features,
            test_y,
        )

        save_best_model(results)

    logger.success("\nTraining completed.")
