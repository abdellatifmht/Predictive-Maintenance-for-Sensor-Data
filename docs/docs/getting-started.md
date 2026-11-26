# Getting started

Use Python 3.10-3.13 and uv from the repository root. The FD001 train, test, and ground-truth files are included under `data/external`.

```sh
uv sync --locked
uv run pm-train
uv run python -m uvicorn predictive_maintenance.api.main:app --port 8000
```

Open `http://localhost:8000/docs`. Run a real example in another terminal:

```sh
uv run python examples/predict_engine.py --unit 1
```

The default command trains three classical models. To include the optional LSTM experiment:

```sh
uv sync --locked --extra deep-learning
uv run pm-train --train-lstm
```

For monitoring, install its extra, create a reference profile, generate traffic with the API running, and start the dashboard:

```sh
uv sync --locked --extra monitoring
uv run pm-reference
uv run pm-stream
uv run python -m streamlit run src/predictive_maintenance/monitoring/dashboard.py
```

Generated files are kept under `data/raw`, `data/interim`, `data/processed`, `models`, and `results`. The original files in `data/external` are never modified. If running the installed package elsewhere, set `PM_PROJECT_ROOT` to this repository's absolute path.
