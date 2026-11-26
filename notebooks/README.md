# Notebook workflow

All notebooks use the package under `src/predictive_maintenance` and the bundled FD001 text files. They work with a kernel started from the repository root or from this directory.

Install and launch JupyterLab from the repository root:

```sh
uv sync --locked --extra notebooks
uv run python -m jupyterlab
```

Select the project's Python environment and use **Restart Kernel and Run All Cells**. Each notebook is independent of variables defined in other notebooks; later stages read the files produced by earlier stages.

| Notebook | Purpose | Prerequisite |
| --- | --- | --- |
| `01_eda.ipynb` | Inspect sensor values, engine lifetimes and correlations | Bundled source data only |
| `02_data_cleaning.ipynb` | Import, preprocess, construct RUL targets and validate both splits | Bundled source data only |
| `03_feature_engineering.ipynb` | Generate aligned features and labels; inspect causal trends | Stage 02 |
| `04_model_training.ipynb` | Train three classical models, run grouped validation and select the serving artifact | Stage 03 |
| `05_model_evaluation.ipynb` | Evaluate held-out engines, plot residuals and verify API feature equivalence | Stage 04 |
| `EDA_resultats_maintenance_predictive.ipynb` | Generate a report and export figures using measured artifact scores | Stages 02-05 or `pm-train` |
| `../Training.ipynb` | Run the entire maintained pipeline and inspect the results | Bundled source data only |

`uv run pm-train` can generate all prerequisites without opening notebooks. For the optional LSTM experiment, install both extras with `uv sync --locked --extra notebooks --extra deep-learning` and set `TRAIN_LSTM = True` in a training notebook. The default notebook workflow does not require PyTorch.

Execute all seven notebooks in fresh kernels from the command line:

```sh
uv run python scripts/run_notebooks.py
```

This runs stages 01-05, the results report, and the root training notebook in order. Successful executions are written under `.uv-cache/notebook-runs/`; failed executions are saved there with a `.failed.ipynb` suffix. To refresh the source notebooks' displayed tables and figures:

```sh
uv run python scripts/run_notebooks.py --in-place
```

Both commands regenerate local data/model artifacts and retrain models. To run a single notebook after its prerequisites are available, pass its path, for example `uv run python scripts/run_notebooks.py notebooks/05_model_evaluation.ipynb`.

Committed outputs are snapshots from an actual execution. Rerunning recomputes them from the current configuration. Performance scores are read from generated results rather than embedded as constants. Training cross-validation selects the serving model; test scores only report its held-out performance. Capped RUL metrics and heuristic API health classes are described in the main README.
