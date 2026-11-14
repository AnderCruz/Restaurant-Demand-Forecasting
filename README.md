# Restaurant Demand Forecasting

A reproducible time-series project exploring how historical customer counts can support day-to-day planning at two restaurants: **Chimi & Churri** and **Assa Frão**. The project compares simple benchmarks with statistical, Prophet and machine-learning forecasts, then examines how forecast accuracy changes across time and forecast horizons.

This is a research and portfolio project, not a production staffing or inventory system. The available history is short, so every result is presented with its validation limits.

## The data

The supplied demand file contains **478 daily dates**, from 1 January 2016 to 22 April 2017. It has two customer-count series and three missing target values: two for Chimi & Churri and one for Assa Frão. Missing counts are unknown; they are not replaced with zero. Where a model requires a complete training series, internal gaps are interpolated using that training window only. Unknown actuals are excluded from scoring.

A separate calendar contains 17 labelled commemorative dates. The labels' business meaning and future availability have not been verified, so the main model comparisons do not assume holiday effects. No weather, bookings, opening hours, promotions, staffing or local-event data were supplied.

## What the project does

The twelve numbered notebooks take the analysis from data checks to forecasting and business interpretation:

1. Validate the source schemas, dates, missing values and calendar coverage.
2. Explore the demand series and describe their distributions.
3. Inspect autocorrelation, stationarity and weekly decomposition using Statsmodels.
4. Build calendar, lag and shifted rolling features without using future targets.
5. Fit constrained AutoReg, ARMA/ARIMA and weekly SARIMAX candidates; compare training-only information criteria and inspect forecast intervals and residuals.
6. Fit Prophet candidates, tune using development cross-validation, inspect components and intervals, and serialise the selected models with metadata.
7. Prepare a machine-learning forecasting workflow.
8. Compare optional MLP, GRU and LSTM models using recursive expanding-window validation. The neural-network experiment is skipped unless TensorFlow is installed.
9. Tune a small HistGradientBoosting parameter set using recursive expanding-window validation.
10. Compare forecasts on aligned development folds against naive and weekly seasonal-naive baselines.
11. Inspect holdout residuals, autocorrelation, interval coverage and interval width.
12. Summarise what the results can support and what additional operational data would be needed.

A concise project index is available in [`notebooks/project_index.ipynb`](notebooks/project_index.ipynb). Reusable loading, feature, evaluation, backtesting and model code lives in [`src/`](src/).

## Evaluation and current findings

Models are compared with 28-day forecasts from expanding training windows. The final 28 calendar dates are kept out of tuning and development comparison. Folds with unknown actuals in their scoring period are excluded; therefore, Chimi & Churri has six eligible folds, while Assa Frão and total demand each have five. Total demand is scored only on dates when both restaurant counts are known.

The executed development folds produced these lowest mean MAEs among the tested candidates:

| Series | Candidate with lowest mean MAE | Mean MAE (customers/day) | Weekly seasonal-naive mean MAE |
|---|---|---:|---:|
| Chimi & Churri | Prophet, additive | 11.67 | 16.85 |
| Assa Frão | Weekly SARIMAX | 12.15 | 18.75 |
| Total demand | Prophet, additive | 17.90 | 25.05 |

These are observations from a small historical sample, not a production ranking. The development report includes fold-to-fold variation, MAE, RMSE, WAPE, sMAPE, MASE, forecast bias, horizon-specific results, and interval coverage and width. The final 28-day holdout has now been scored once after development-based model selection; its results are reported separately and must not be used for further tuning.

The optional neural models improved on the weekly seasonal-naive baseline in development folds, but did not outperform the best classical model for any series. Their comparison and limitations are documented in the [model-comparison report](reports/model_comparison.md). Install `requirements-deep-learning.txt` to run the notebook's TensorFlow experiment.

## Project layout

- `data/raw/` — copies of the supplied demand and calendar CSV files.
- `data/reference/` — preserved original demand notebook, 28-day forecast workbook and method-reference notebooks.
- `notebooks/` — twelve numbered notebooks and the project index.
- `src/` — reusable Python modules for data handling, features, metrics, backtesting, Statsmodels and Prophet.
- `tests/` — tests for data policy, feature leakage, chronological splits, metrics, baselines and forecast alignment.
- `reports/` — data quality, validation protocol, model comparison and limitations.
- `artifacts/` — executed forecast tables, model files and metadata.
- `scripts/` — notebook runner and reproducible final-holdout forecast/scoring commands.
- `App.py` — optional Streamlit viewer for the historical demand data.

## Set up and reproduce

The core environment was tested with Python 3.12.3. Create a virtual environment and install the pinned core dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests and execute the twelve numbered notebooks in order. The runner also generates the final candidate forecasts and holdout metrics after development-based model selection:

```bash
pytest -q
python scripts/run_notebooks.py
```

The notebook runner starts each notebook from the project root, clears stale outputs and saves genuine execution results. Prophet requires a working CmdStan backend; fitting and serialisation were verified in the tested environment.

Deep learning and optional packages are separated into `requirements-deep-learning.txt` and `requirements-optional.txt`. To run the neural-network experiment in notebook 08, also install `requirements-deep-learning.txt` in the active environment before running the notebook runner. Without TensorFlow, notebook 08 reports that the optional experiment was skipped. To launch the optional historical-data viewer, install `requirements-optional.txt` and run:

```bash
streamlit run App.py
```

## Reports and saved outputs

- [`reports/data_quality_report.md`](reports/data_quality_report.md) records the source-data audit.
- [`reports/validation_protocol.md`](reports/validation_protocol.md) describes chronological evaluation and scoring rules.
- [`reports/model_comparison.md`](reports/model_comparison.md) contains executed development-fold results and uncertainty diagnostics.
- [`reports/final_holdout_metrics.csv`](reports/final_holdout_metrics.csv) contains the one-time final holdout scores; `artifacts/forecasts/final_holdout_forecasts.csv` and `artifacts/models/final_holdout_metadata.json` contain the associated forecasts and run metadata.
- [`reports/limitations_and_assumptions.md`](reports/limitations_and_assumptions.md) documents unresolved data and modelling assumptions.
- `reports/fold_level_model_comparison.csv` and `reports/horizon_level_model_comparison.csv` contain the underlying metric summaries.
- `artifacts/forecasts/` contains holdout and walk-forward forecast tables; `artifacts/models/` contains model files and run metadata.

## Limitations

The data span only about 16 months, which is not enough to identify annual seasonality reliably. Deep-learning models may overfit this sample. Prophet's intervals describe model uncertainty and do not guarantee real-world coverage. Calendar labels have not been verified, and the data do not include operational drivers such as bookings, weather, staffing or promotions. Read the reports before drawing business conclusions; the final holdout must not be reused to tune models.
