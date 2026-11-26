# Project completion audit

Reviewed on 26 November 2026. Scope: the Python package, bundled FD001 data workflow, training artifacts, API, monitoring demo, notebooks, packaging, container definition, tests, and documentation.

## Gaps found and changes made

| Area | Gap | Completed change |
| --- | --- | --- |
| Installation | Missing `requirements.txt`, Python 3.10-only constraint, unnecessary `lstm` dependency, missing runtime packages | Made `pyproject.toml` authoritative, regenerated `uv.lock`, supported Python 3.10-3.13, separated optional extras, added command entry points |
| Training | Skip-LSTM flag ignored; every import required PyTorch | Connected the flag and made PyTorch optional; classical training works independently |
| Model selection | Test scores selected the winner | Select the serving model from training GroupKFold validation RMSE; retain test scores for reporting |
| Linear regression | Persisted estimator lost its scaling transformation | Save a scikit-learn pipeline containing the scaler and estimator |
| Preprocessing | Possible identifier removal, future-value imputation, inconsistent test schema and index alignment | Preserve identifiers, causally forward-fill within engines, reuse training column metadata, align RUL after the unit merge |
| Data quality | Duplicate engine/cycle keys and non-finite or fractional identifiers could pass | Validate key uniqueness, numeric finite inputs, positive integer identifiers, non-empty data and non-negative targets |
| Features | Sorting features could misalign labels | Sort before separating features and labels; verify identifiers when loading training files |
| Serving | Independent feature code, insufficient history accepted, fractional predictions truncated | Share feature engineering, require the full trend window, preserve fractional cycles, validate finite inputs, save preprocessing settings in the manifest |
| API | Simulator used a different engine identifier field; generic exceptions became input errors | Align request fields, accept the legacy alias, return 422 for input errors and 503 when untrained |
| Evaluation | Unmatched prediction keys could disappear; cycle aggregation differed by estimator | Validate one-to-one keys and evaluate the final labelled cycle of every engine consistently |
| Monitoring | Missing dependencies and CLI path annotations; small or failed traffic could break dashboard | Add monitoring extra, typed paths, drift sample checks, constant-reference PSI bins, partial-log handling and small/error-only traffic guards |
| Notebooks | Three stage notebooks were empty invalid JSON files | Implement valid feature, training and evaluation notebooks using package functions |
| Containers | Missing requirements file and incorrect application module | Install locked dependencies, use the correct ASGI module, include OpenMP, run as non-root, mount models and check readiness |
| Maintenance | One deliberately failing test, empty test files and template documentation | Add regression and integration coverage, GitHub Actions, working quick start and API/evaluation documentation |

## Measured model results

A full default five-fold training run completed on the bundled FD001 data: 20,631 training rows and 100 test engines. LightGBM was selected by mean grouped validation RMSE of **17.8256 cycles**. Held-out metrics use the last recorded cycle of each test engine and the configured RUL cap of 125 cycles.

| Model | Test MAE | Test RMSE |
| --- | ---: | ---: |
| Linear regression | 16.0214 | 20.2548 |
| XGBoost | 12.3043 | 16.5277 |
| LightGBM | 11.8269 | 16.3770 |

The selected model also achieved R2 **0.8330** and PHM08 score **607.3937**. Models and intermediate outputs are generated locally and excluded from Git; regenerate them with `uv run pm-train`.

## Verification

- **40 core tests pass**, including a complete small classical training run and API/batch prediction equivalence. Optional dashboard and LSTM test modules skip when their dependencies are unavailable.
- The full FD001 ingestion, preprocessing, five-fold training, artifact loading, prediction and evaluation run completed.
- Live HTTP health and prediction checks succeeded. All 100 test engines matched saved batch predictions within API rounding (maximum difference 0.004976 cycles). Engine 1 returned RUL 119.17 using the last 10 of 31 readings.
- The training reference profile was generated for 15 retained sensors.
- Source lint and formatting checks pass. The new stage notebooks are valid JSON and their Python cells compile.
- The MkDocs documentation builds successfully with strict validation.

## Practical limits

Docker Desktop is installed but its daemon was stopped, so the image definition could not be built or run in this session. GitHub Actions is configured for Python 3.10 and 3.13; those hosted jobs run after pushing. Local checks used Python 3.13.

The optional LSTM has a focused checkpoint and short-history regression test that runs when PyTorch is installed; a full LSTM benchmark was not performed. It remains an experiment and is not automatically promoted against classical models without equivalent validation. The original analysis notebooks retain their historical outputs; maintained execution uses package commands and stage notebooks 03-05.

Monitoring is a distribution-drift demonstration, with logs produced by the simulator. It does not provide labelled online accuracy, automatic retraining, or comprehensive API request collection. Health labels are RUL thresholds rather than learned fault diagnoses.
