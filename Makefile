.PHONY: install data train predict evaluate api monitoring-reference monitoring-stream monitoring-dashboard docker-build docker-run test clean

PYTHON := python3

install:
	pip install -r requirements.txt

# ------------------------------------------------------------------
# Semaine 5 — pipeline complet, reproductible en une commande.
# ------------------------------------------------------------------
train:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.pipeline

train-fast:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.pipeline --no-train-lstm

predict:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.models.predict

evaluate:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.models.evaluate

# ------------------------------------------------------------------
# Semaine 6 — serving
# ------------------------------------------------------------------
api:
	PYTHONPATH=src uvicorn predictive_maintenance.api.main:app --reload --port 8000

docker-build:
	docker build -t predictive-maintenance-api .

docker-run: docker-build
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro predictive-maintenance-api

# ------------------------------------------------------------------
# Semaine 7 — monitoring
# ------------------------------------------------------------------
monitoring-reference:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.monitoring.build_reference

monitoring-stream:
	PYTHONPATH=src $(PYTHON) -m predictive_maintenance.monitoring.simulate_stream

monitoring-dashboard:
	PYTHONPATH=src streamlit run src/predictive_maintenance/monitoring/dashboard.py

# ------------------------------------------------------------------
test:
	PYTHONPATH=src pytest tests/ -v

clean:
	rm -rf data/raw data/interim data/processed models results
	find . -type d -name "__pycache__" -exec rm -rf {} +
