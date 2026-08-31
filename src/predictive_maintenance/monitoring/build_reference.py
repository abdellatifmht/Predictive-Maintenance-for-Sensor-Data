import json

import pandas as pd
import typer
from loguru import logger

from predictive_maintenance.config import INTERIM_DATA_DIR, RESULTS_DIR

app = typer.Typer()

MONITORING_DIR = RESULTS_DIR / "monitoring"
REFERENCE_PATH = MONITORING_DIR / "reference_profile.json"


def build_reference(train_preprocessed_path, output_path) -> None:
    if not train_preprocessed_path.exists():
        raise FileNotFoundError(f"{train_preprocessed_path} introuvable — lance `make train` d'abord.")

    train_df = pd.read_csv(train_preprocessed_path)
    sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]

    profile = {"features": {}, "n_reference_rows": len(train_df)}
    for feature in sensor_columns:
        values = train_df[feature].dropna()
        profile["features"][feature] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f)

    logger.success(
        f"Profil de référence sauvegardé dans {output_path} "
        f"({len(sensor_columns)} capteurs, {len(train_df):,} lignes de train)."
    )


@app.command()
def main(
    train_preprocessed_path=INTERIM_DATA_DIR / "train_preprocessed.csv",
    output_path=REFERENCE_PATH,
) -> None:
    build_reference(train_preprocessed_path, output_path)


if __name__ == "__main__":
    app()
