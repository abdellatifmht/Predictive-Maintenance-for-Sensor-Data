# Predictive Maintenance

This project trains and serves remaining useful life models for the bundled NASA C-MAPSS FD001 engine data.

The reproducible workflow is ingestion, causal preprocessing, sensor trend features, engine-grouped cross-validation, artifact generation, test prediction, and final-cycle evaluation. The FastAPI service shares training feature code and model artifacts. An optional Streamlit dashboard demonstrates distribution drift diagnostics.

Start with the [setup guide](getting-started.md), then inspect the [API and evaluation contract](workflow.md).
