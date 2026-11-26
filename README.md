# Predictive Maintenance for Sensor Data

Estimate turbofan engine remaining useful life (RUL) from NASA C-MAPSS FD001 sensor histories. The project includes a reproducible training pipeline, a FastAPI prediction service, and a Streamlit drift dashboard.

## Quick start

Requires Python 3.10-3.13 and [uv](https://docs.astral.sh/uv/). Run from this repository's root; the three FD001 input files are already in `data/external/`.

```sh
uv sync --locked
uv run pm-train
uv run python -m uvicorn predictive_maintenance.api.main:app --port 8000
```

Open `http://localhost:8000/docs` for the interactive API. `GET /health` reports the selected model, evaluation metrics, and required history length. Before training, health and prediction requests return 503.

The default pipeline trains linear regression, XGBoost, and LightGBM. It imports the text files, validates and preprocesses them, builds causal sensor trends, runs five-fold cross-validation grouped by engine, saves all estimators, generates test predictions, and evaluates the final cycle of each test engine. It writes artifacts under `models/`, `data/processed/`, and `results/`.

Without uv, create and activate a virtual environment, run `python -m pip install -e .`, then run `python -m predictive_maintenance.pipeline`. Set `PM_PROJECT_ROOT` to the absolute workspace directory when running an installed package outside the repository.

## Predictions

Send `POST /predict` with one engine's consecutive readings in chronological order. Each reading contains `os_1`, `os_2`, and the retained sensors. The full schema is available in `/docs`; constant FD001 sensors are optional. `unit_id` is accepted as an alias for `unit`.

Generate a real request from the bundled data in a separate terminal after training:

```sh
uv run python examples/predict_engine.py --unit 1
```

The API needs at least 10 readings with the default trend window. Send the full available history or the last 10 consecutive readings. RUL is returned as a non-negative number of cycles, rounded to two decimals. Health classes are derived from RUL: `imminent_failure` at 10 cycles or below, `anomaly` above 10 through 30, and `healthy` above 30. These are heuristic labels, not a separately trained fault classifier.

Batch prediction and evaluation can also be rerun:

```sh
uv run pm-predict
uv run pm-evaluate
```

Every command supports `--help`. `pm-data` only imports the raw files. `pm-train --n-splits 3` changes the number of engine folds. Base settings live in `configs/base.yaml`; estimator parameters and seeds live in `configs/model.yaml`. The serving manifest records the feature window used during training, so later configuration edits do not alter a saved model's input transformation.

## Evaluation and model selection

The serving model is selected by mean validation RMSE from GroupKFold on training engines. Test metrics do not determine selection. Consequently, the selected model may differ from the lowest test RMSE in `results/test_evaluation.csv`.

- `groupkfold_results.csv`: MAE/RMSE for each training validation fold, across all validation cycles.
- `model_comparison.csv`: mean and standard deviation across folds.
- `test_evaluation.csv`: final-cycle MAE/RMSE for each trained model on the 100 test engines.
- `evaluation.csv`: final-cycle MAE, RMSE, R2 and PHM08 score for the serving model.
- `evaluation_by_unit.csv`: final-cycle error per engine.

RUL targets are capped at 125 cycles by default on both training and test data. Test targets use the supplied ground-truth RUL at the last observed cycle; they are not inferred from a truncated trajectory's end. Reported metrics use these capped targets. Features exclude engine identifiers, cycle numbers, and labels. Missing values are forward-filled within each engine; missing initial observations are rejected to avoid using future information.

## Optional LSTM experiment

```sh
uv sync --locked --extra deep-learning
uv run pm-train --train-lstm
```

The PyTorch LSTM uses causal windows, train-only scaling, a separate engine validation split, and the best validation epoch. Its held-out test metrics are included as an experiment. The serving champion is selected from the three classical models with comparable GroupKFold results; LSTM is not automatically promoted without equivalent model-selection validation. Core training and serving do not require PyTorch.

## Monitoring

```sh
uv sync --locked --extra monitoring
uv run pm-reference
# Keep the API running in another terminal.
uv run pm-stream --rate 2
uv run python -m streamlit run src/predictive_maintenance/monitoring/dashboard.py
```

Use `pm-stream --inject-drift --drift-after 50` to exercise alerts. The simulator records request results, latency, and final sensor readings in `results/monitoring/logs/predictions.jsonl`. The dashboard compares them with training sensor distributions using PSI and the two-sample KS test. This is a diagnostic demo: differences in lifecycle sampling can also cause alerts, and no ground-truth online performance or automatic retraining is provided. Requests from other clients are not logged by this simulator.

## Docker

Train first, then mount the generated models into the API container:

```sh
docker build -t predictive-maintenance-api .
docker run --rm -p 8000:8000 --mount "type=bind,source=/absolute/path/to/models,target=/app/models,readonly" predictive-maintenance-api
```

Replace the source with the absolute path of this repository's `models` directory (a Windows path is valid with Docker Desktop). The image installs the application and OpenMP runtime, runs as a non-root user, and checks `/health`. It serves classical artifacts; PyTorch is an optional host-side experiment.

## Development

```sh
uv sync --locked --extra docs
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mkdocs build --strict -f docs/mkdocs.yml
```

CI checks Python 3.10 and 3.13. Tests cover target construction, causality, schema alignment, engine separation, artifact round trips, model selection, API behavior, drift detection, and a complete small training run. Optional dashboard and LSTM tests run when their dependencies are installed.

All seven notebooks use the maintained package and measured artifact results. Start with `Training.ipynb` for the complete pipeline, or follow notebooks 01-05 stage by stage. The results notebook exports figures to `reports/figures/`. See the [notebook guide](notebooks/README.md) for prerequisites and execution commands. A separate CI job executes every notebook in a fresh kernel and saves the outputs as an artifact.

See [the project audit](docs/project-audit.md) for the completion changes and validation status. Local model and data outputs are regenerated rather than committed. Only load model files generated by this project from a trusted source.
