# CODEX MASTER PROMPT — Restaurant Demand Forecasting

You are acting as a Senior Data Scientist, Time-Series Researcher, Machine Learning Engineer, and MLOps-minded code reviewer. Your task is to audit, upgrade, execute, and document this repository into a rigorous, reproducible, portfolio-grade forecasting project.

## 1. Non-negotiable working rules

1. **All project artefacts must be in English**: notebook titles, Markdown, code comments, variable/function names, chart titles/axes/legends, reports, tests, and documentation.
2. Work directly in the existing repository. Do not create a second project or replace the supplied source data.
3. Inspect the complete repository before changing anything. Read the existing README, all 12 notebooks, source modules, tests, and data schemas.
4. Preserve the original notebook and original 28-day forecast workbook under `data/reference/`.
5. Never fabricate metrics, model rankings, business findings, or successful execution claims. Clearly label unexecuted experiments and unavailable dependencies.
6. Do not silently drop observations, replace missing customer counts with zero, remove outliers, or interpolate across train/test boundaries.
7. Never use random train/test splitting for time-series evaluation. No shuffling.
8. Prevent target leakage. Every feature at forecast origin t must be available at t. Rolling features must be shifted; scalers, imputers, encoders, selectors, and tuning must be fitted within each training fold.
9. Use reproducible seeds where applicable. Record package versions, data cutoff, forecast horizon, model parameters, and validation dates.
10. Make changes incrementally. After each meaningful change, run targeted tests and report what changed.
11. Do not commit, push, publish, or delete files. Stop before Git operations that change remote state.
12. If any assumption is uncertain, document it instead of inventing domain knowledge.

## 2. Business and data context

This project forecasts daily customer demand for two restaurants:
- Chimi & Churri
- Assa Frão

Source files:
- `data/raw/clientes_restaurantes.csv`
- `data/raw/datas_comemorativas.csv`

Reference artefacts:
- `data/reference/Project_Regression_original.ipynb`
- `data/reference/original_28_day_forecast.xlsx`

The supplied historical dataset is small (approximately 478 daily observations spanning about 16 months, subject to validation from the actual files). It includes two restaurant demand series and a separate commemorative-date calendar. Validate exact columns, date range, row count, missing values, duplicate dates, frequency, and calendar coverage directly from the files.

The project must be honest about sample limitations. In particular:
- Annual seasonality is not robustly identifiable from only about 16 months.
- Complex deep-learning models may overfit.
- Prophet yearly seasonality may be weakly identified and must be treated as an experiment, not assumed useful.
- Model performance must be compared against simple naive and seasonal-naive baselines.
- Report uncertainty and fold-to-fold variability, not only a single score.

## 3. Required time-series libraries and model families

The repository must explicitly include and use **Statsmodels** and **Prophet**, in addition to the existing scikit-learn / boosting / optional deep-learning stack.

### A. Statsmodels — required

Create a dedicated, well-documented statistical modelling workflow using `statsmodels`:

- ACF and PACF diagnostics (`statsmodels.graphics.tsaplots`)
- Stationarity tests: ADF and KPSS
- STL decomposition and, where appropriate, classical seasonal decomposition
- Autoregressive models (`AutoReg`)
- ARMA represented through `ARIMA(order=(p, 0, q))` where appropriate
- ARIMA
- SARIMA / SARIMAX (`SARIMAX`)
- Exogenous regressors only if they are known at forecast time
- Residual diagnostics (`plot_diagnostics`, residual ACF, residual mean/variance checks)
- Information-criterion model comparison (AIC/BIC) on training data only
- Small, constrained order search or information-criterion search; do not run an enormous brute-force grid
- Forecast intervals and comparison of predicted mean vs observed values

Be explicit that ARMA is the ARIMA special case with d=0. Do not use obsolete APIs or hard-code orders copied from unrelated datasets. Select candidate orders from diagnostics and training-fold evidence.

### B. Prophet — required

Add a dedicated Prophet notebook or a clearly separated section with:

- Prophet input contract (`ds`, `y`)
- Correct chronological train/validation/test split
- Trend and changepoint analysis
- Additive vs multiplicative seasonality as controlled candidates
- Weekly seasonality
- Yearly seasonality only as a cautious experiment, given the short history
- Holiday effects using the supplied commemorative calendar, only after checking date coverage and semantic meaning
- Custom regressors only when known in advance and available in production
- Forecast output including `yhat`, `yhat_lower`, `yhat_upper`
- Trend and seasonal component visualisations
- Prophet cross-validation (`prophet.diagnostics.cross_validation`) and `performance_metrics`
- Tuning of a small, defensible set of parameters such as `changepoint_prior_scale`, `seasonality_prior_scale`, `holidays_prior_scale`, and `seasonality_mode`
- Serialization / deserialization (`prophet.serialize`) with a model artefact and metadata
- A clear explanation of Prophet's uncertainty intervals and limitations

**Critical:** Prophet's built-in cross-validation must not be confused with a final untouched test set. Use cutoffs and horizons compatible with the available history. If the data are too short for a proposed Prophet CV configuration, reduce the horizon/initial period or mark the experiment as infeasible. Never remove outliers merely because they fall outside Prophet's in-sample interval. Investigate observations using source context and report sensitivity analysis separately.

### C. Additional Statsmodels and Prophet references

Two supplied notebooks are methodological references:
- `Farm_Project.ipynb`: demonstrates ACF/PACF, `AutoReg`, ARMA/ARIMA, SARIMA/SARIMAX, stationarity checks, residual diagnostics, and information-criterion grid search across multiple farm series.
- `Model_O3_prophet.ipynb`: demonstrates Prophet fitting, forecast intervals, components, changepoint/seasonality parameter experiments, temporal evaluation, Prophet cross-validation, performance metrics, and JSON serialization.

Use these notebooks as **method references only**. Do not copy their Portuguese text, farm/temperature data, pollution/O3 data, hard-coded model orders, external dataset URLs, or assumptions into the restaurant project. Re-implement the relevant methods in clean, modular English code using the restaurant datasets.

## 4. Repository architecture and notebooks

Retain the existing 12-notebook sequence, but revise it as needed to ensure that Statsmodels and Prophet are first-class citizens. A suitable structure is:

1. Data Acquisition, Integration & Quality Assessment
2. Advanced Exploratory Data Analysis
3. Statistical Time Series Analysis & Diagnostics
4. Advanced Feature Engineering
5. Classical Statistical Forecasting with Statsmodels
6. Prophet Forecasting
7. Machine Learning Forecasting
8. Deep Learning Forecasting (optional, capacity controlled)
9. Hyperparameter Optimisation
10. Walk-Forward Backtesting & Ensemble Forecasting
11. Explainability, Residual Diagnostics & Forecast Uncertainty
12. Business Intelligence & Forecasting Strategy

If you keep the current notebook names, ensure the content and ordering are coherent. Do not leave duplicated or contradictory modelling workflows.

Keep reusable code in `src/`, not duplicated across notebooks. Add or improve modules as needed, for example:
- `src/data.py`
- `src/features.py`
- `src/evaluation.py`
- `src/models/statsmodels_models.py`
- `src/models/prophet_models.py`
- `src/backtesting.py`
- `src/plotting.py`

Use clean functions with type hints and docstrings where useful. Avoid over-engineering.

## 5. Evaluation design — essential

Implement a defensible evaluation protocol for each restaurant independently and, where meaningful, for total demand.

At minimum:
- Chronological final holdout (if the data volume allows)
- Expanding-window or rolling-origin validation
- Same forecast origins and horizons for candidate comparisons
- Seasonal-naive benchmark (weekly period 7) and naive benchmark
- Horizon-specific performance (e.g. 1–7, 8–14, 15–21, 22–28 days where enough folds exist)
- Metrics: MAE, RMSE, WAPE, sMAPE, MASE, and R² (explain R² limitations for forecasting)
- Forecast bias / mean error
- Prediction interval coverage and width when intervals are available
- Per-restaurant metrics and total-demand metrics
- Fold-level results and dispersion (mean/std or confidence summaries)

Be careful with recursive forecasts: for future dates without actual observations, use prior predictions to build lag features. Do not leak actual test-period target values into a multi-step forecast unless simulating a rolling operational process where each actual becomes available before the next forecast is issued. State which deployment scenario is being simulated.

Prophet, Statsmodels, and ML models must be compared on aligned evaluation windows. If an algorithm cannot support the same window or exogenous variables, explain the difference rather than presenting an unfair ranking.

## 6. Data and feature engineering requirements

- Verify date frequency and duplicate dates.
- Preserve missingness flags.
- Treat missing target values as unknown, not zero.
- Fit any imputation using training data only.
- Separate target series per restaurant.
- Build calendar features: weekday, weekend, month, quarter, month boundaries, and cyclical encodings.
- Build lag features and shifted rolling statistics.
- Holiday features must be known for future dates and correctly aligned.
- Consider Fourier terms for weekly patterns; avoid unsupported yearly complexity.
- Do not include future target-derived features.
- Check for outliers, but do not automatically delete them.
- Explicitly document that no weather, reservation, opening-hours, promotion, staffing, or local-event variables were supplied unless the files prove otherwise.

## 7. Dependency management

Update `requirements.txt` to include a compatible Prophet package (`prophet`) and Statsmodels (`statsmodels`) explicitly. Keep optional dependencies separated where practical:
- `requirements-deep-learning.txt` for TensorFlow / neural models
- `requirements-optional.txt` for libraries such as XGBoost, LightGBM, CatBoost, SHAP, Optuna, if not all are core dependencies

Check package compatibility with the active Python version. Do not silently downgrade core packages. If installation fails, provide the exact error and a reproducible alternative. Prophet may require a working CmdStan backend; verify actual installation and model fitting in the current environment.

## 8. Tests and code quality

Add tests for:
- Date parsing and daily index validation
- Duplicate-date detection
- Missing-value policy
- No-leakage lag and rolling feature construction
- Chronological split behaviour
- Forecast metric edge cases
- Seasonal-naive benchmark
- Recursive forecast mechanics
- Prophet input schema (`ds`, `y`) and future dataframe generation (mock Prophet if package is unavailable)
- Statsmodels forecast output shape and index alignment

Run:
- `pytest -q`
- notebook syntax/structure validation
- import checks for core dependencies
- full notebook execution in order where feasible

For notebooks, clear stale outputs before committing, but preserve useful outputs only if they are genuine, reproducible results. Do not claim execution success if execution was skipped or failed.

## 9. Required reports and outputs

Create:
- `reports/data_quality_report.md`
- `reports/model_comparison.md` (only populated with genuinely executed results)
- `reports/validation_protocol.md`
- `reports/limitations_and_assumptions.md`
- saved forecast tables with date, restaurant, model, horizon, prediction, lower/upper interval where available
- model metadata including training cutoff, package versions, parameters, and validation method

Use professional, readable charts with labelled axes and units. Avoid decorative visualisations that do not answer a question.

## 10. Work plan

Execute these phases in order:

### Phase 1 — Repository and data audit
Inspect all repository files and the supplied data. Report issues before making broad changes.

### Phase 2 — Architecture and dependencies
Refactor dependencies and reusable modules. Ensure Prophet and Statsmodels are core forecasting tracks.

### Phase 3 — Notebook implementation
Revise all notebooks, integrate Statsmodels and Prophet, and keep all project text in English.

### Phase 4 — Validation and testing
Run unit tests, import checks, notebook validation, and notebook execution. Fix failures rather than hiding them.

### Phase 5 — Results and documentation
Generate reports from actual execution results. Clearly distinguish measured outcomes from hypotheses.

### Phase 6 — Final audit
Check consistency across README, notebooks, source code, dependencies, and reports. Ensure no Portuguese project content remains (except original source data labels and preserved original reference files, which must not be altered).

## 11. Final response required from you (Codex)

When finished, provide:
1. Summary of repository changes.
2. Final notebook list and purpose.
3. Statsmodels methods implemented.
4. Prophet methods implemented.
5. Tests executed and exact pass/fail counts.
6. Notebooks executed and any skipped/failed notebooks.
7. Actual model comparison results, only if successfully generated.
8. Known limitations and unresolved issues.
9. Exact shell commands needed to reproduce the environment and run the project.
10. Files changed and files created.

Start with Phase 1. Do not skip the audit. Do not claim the project is complete until the full audit, tests, and feasible notebook execution have been completed.
