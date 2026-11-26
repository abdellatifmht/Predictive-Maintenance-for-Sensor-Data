"""Keep notebook sources valid and prevent saved failing executions from returning."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "Training.ipynb",
    *[
        ROOT / "notebooks" / name
        for name in (
            "01_eda.ipynb",
            "02_data_cleaning.ipynb",
            "03_feature_engineering.ipynb",
            "04_model_training.ipynb",
            "05_model_evaluation.ipynb",
            "EDA_resultats_maintenance_predictive.ipynb",
        )
    ],
]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_sources_and_saved_outputs_are_valid(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        ast.parse("".join(cell["source"]), filename=path.name)
        assert not any(output["output_type"] == "error" for output in cell["outputs"])
