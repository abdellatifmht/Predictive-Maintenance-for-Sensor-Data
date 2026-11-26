"""Import the bundled NASA C-MAPSS FD001 text files."""

from pathlib import Path

import typer

from predictive_maintenance.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR
from predictive_maintenance.data.ingestion import ingest_dataset

app = typer.Typer()


@app.command()
def main(input_dir: Path = EXTERNAL_DATA_DIR, output_dir: Path = RAW_DATA_DIR) -> None:
    ingest_dataset(
        input_dir / "train_FD001.txt",
        input_dir / "test_FD001.txt",
        input_dir / "RUL_FD001.txt",
        output_dir,
    )


if __name__ == "__main__":
    app()
