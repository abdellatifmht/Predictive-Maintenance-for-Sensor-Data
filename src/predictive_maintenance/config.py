import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
import yaml

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(os.getenv("PM_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def load_settings(filename: str) -> dict:
    """Read workspace settings, with defaults when YAML files are absent."""
    path = PROJ_ROOT / "configs" / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    if not isinstance(settings, dict):
        raise TypeError(f"Expected a settings mapping in {path}")
    return settings


SETTINGS = load_settings("base.yaml")
MODEL_SETTINGS = load_settings("model.yaml")
TREND_WINDOW = SETTINGS.get("trend_window", 10)
RUL_CAP = SETTINGS.get("rul_cap", 125)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Columns
RAW_COLUMNS = (
    ["unit", "cycle"] + [f"os_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
)

REQUIRED_COLUMNS = ["unit", "cycle", "RUL"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
