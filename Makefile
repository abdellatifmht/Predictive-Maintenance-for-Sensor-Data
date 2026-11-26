.PHONY: install data train train-fast train-lstm predict evaluate api monitoring-reference monitoring-stream monitoring-dashboard docker-build docker-run test lint docs clean

PYTHON ?= uv run --no-sync python

install:
	uv sync --locked

data:
	$(PYTHON) -m predictive_maintenance.dataset

train:
	$(PYTHON) -m predictive_maintenance.pipeline

train-fast:
	$(PYTHON) -m predictive_maintenance.pipeline --no-train-lstm

train-lstm:
	$(PYTHON) -m predictive_maintenance.pipeline --train-lstm

predict:
	$(PYTHON) -m predictive_maintenance.models.predict

evaluate:
	$(PYTHON) -m predictive_maintenance.models.evaluate

api:
	$(PYTHON) -m uvicorn predictive_maintenance.api.main:app --reload --port 8000

monitoring-reference:
	$(PYTHON) -m predictive_maintenance.monitoring.build_reference

monitoring-stream:
	$(PYTHON) -m predictive_maintenance.monitoring.simulate_stream

monitoring-dashboard:
	$(PYTHON) -m streamlit run src/predictive_maintenance/monitoring/dashboard.py

docker-build:
	docker build -t predictive-maintenance-api .

docker-run: docker-build
	docker run --rm -p 8000:8000 -v "$(CURDIR)/models:/app/models:ro" predictive-maintenance-api

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

docs:
	$(PYTHON) -m mkdocs build --strict -f docs/mkdocs.yml

# Never remove source datasets or trained models as part of routine cleanup.
clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(p) for p in Path('.').glob('src/**/__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
