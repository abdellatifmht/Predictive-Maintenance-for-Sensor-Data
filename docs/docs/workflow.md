# Workflow and API contract

## Data and features

FD001 rows contain an engine ID, cycle number, three operating settings, and 21 sensors. Constant sensor and operating-setting columns are identified from training data and the same columns are removed from test and API inputs. Engine and cycle identifiers are retained for grouping, but excluded from classical estimator inputs.

Training RUL is the last cycle minus the current cycle. Test RUL is the provided final-cycle ground truth plus the difference between the final and current cycles. Both are capped at 125 cycles by default. Sensor trends use a causal 10-cycle linear slope. Future observations and other engines never contribute to a cycle's features.

## Model selection and artifacts

GroupKFold keeps engines separate and measures MAE and RMSE across validation cycles. The classical model with the lowest mean validation RMSE becomes the serving model. Final-cycle test metrics are reporting-only. The optional LSTM has its own engine validation split and is not automatically selected against classical cross-validation scores.

`models/best_model.json` records model name, artifact filenames, validation and test metrics, and preprocessing settings. Feature lists, dropped-column metadata, and model preprocessing must accompany the estimator. Linear regression saves its scaler and estimator together. The API and batch command use these saved artifacts.

`results/evaluation.csv` reports MAE, RMSE, R2, and PHM08 score at the last labelled cycle of every engine. Unmatched or duplicated prediction keys fail evaluation rather than silently disappearing in a join.

## HTTP API

- `GET /`: service information.
- `GET /health`: model readiness, name, type, history requirement, and held-out test scores; 503 until artifacts are available.
- `POST /predict`: `unit` (optional string) and `readings` (chronological array of sensor objects). The legacy `unit_id` spelling is accepted.
- `GET /docs`: complete schema and interactive requests.

The default classical model requires at least 10 consecutive readings. Only the last window is used. Finite sensor values are required, and unknown fields, empty histories, or short histories return 422. Predictions preserve fractional cycles and are clipped at zero. Unexpected server failures return 500.

The health class is a rule applied to predicted RUL: at most 10 means imminent failure, above 10 through 30 means anomaly, and above 30 means healthy. These labels are not fault diagnoses.

## Monitoring scope

The simulator writes JSONL records for its requests. The dashboard tolerates incomplete trailing log records, small request counts, and error-only traffic. PSI and KS tests compare sensor distributions with training reference values. Lifecycle sampling changes can cause alerts; these diagnostics do not establish model accuracy, provide online truth labels, or trigger retraining.
