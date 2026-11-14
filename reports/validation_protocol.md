# Validation Protocol

Each restaurant target is modeled independently. Preserve unknown targets as missing and exclude them only from the affected fit or metric, with dates reported. Use chronological splits only.

The final holdout is the last 28 calendar dates where target availability permits. Model selection uses expanding-window origins within the earlier history. Candidate models share origins and horizons; report fold-level results and mean and standard deviation. Compare naive and weekly seasonal-naive forecasts on every eligible fold. Report MAE, RMSE, WAPE, sMAPE, MASE, R², and mean error. R² is descriptive and can be unstable on short, low-variance horizons. Report interval coverage and mean width when available. Also evaluate the combined target only on dates when both restaurant targets are observed.

For multi-step forecasts, forecast future lags recursively from predictions. Do not use realized holdout values to update a forecast. Prophet built-in cross-validation is an additional development diagnostic and does not replace the untouched final holdout. Any differing eligible windows or covariates are stated beside the comparison.
