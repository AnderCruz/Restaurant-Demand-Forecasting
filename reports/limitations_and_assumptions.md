# Limitations and Assumptions

## Data limitations

- The history covers 1 January 2016 to 22 April 2017 (478 daily observations). This is too short to estimate annual seasonality reliably or establish long-term performance.
- Three restaurant counts are unknown. Missing counts are not zero. Internal gaps are interpolated only inside each training segment; unknown scoring actuals are excluded.
- The commemorative calendar has 17 dates and incomplete coverage. Its labels, relevance to these restaurants, and future availability have not been verified, so it is not used as a forecasting regressor.
- The supplied files contain no bookings, opening hours, promotions, staffing, weather, local events, or other operational drivers.
- The original workbook's forecast method and provenance are unavailable; it is retained for reference and is not scored as a model.

## Model and evaluation limits

- The final holdout covers one 28-day period (26 March to 22 April 2017). It has now been scored once after model choices were fixed from development data. Do not use these holdout outcomes to tune or reselect models; doing so would turn the holdout into training information.
- Expanding-window folds overlap in their training histories, and only five or six eligible folds are available per series. Fold means and standard deviations are descriptive, not precise estimates of future performance.
- Deep-learning models are exploratory on this short sample. TensorFlow is optional and runs in a separate environment; installing only the core requirements intentionally skips neural training in notebook 08. Existing neural holdout forecasts were generated and recorded with their own development-fold selection metadata.
- Prediction intervals are available for Prophet and Statsmodels candidates only. Their empirical holdout coverage comes from 28 observations and is not a calibration guarantee. No intervals are estimated for baselines, HistGradientBoosting, or neural models.
- Model performance may change with different periods, restaurant operations, or data collection practices. No model is approved for staffing, purchasing, or other operational decisions without ongoing monitoring and business review.

## Reproduction status

- The core environment was installed from the pinned `requirements.txt`. All 14 tests passed.
- All 12 numbered notebooks completed in order. Notebook 08 completed in its documented optional mode because TensorFlow is not part of the core environment; the neural experiment was separately run in the TensorFlow environment and its outputs are retained.
- Final holdout forecast generation and scoring are reproducible with `scripts/build_final_holdout_forecasts.py` and `scripts/evaluate_holdout.py`, or through `scripts/run_notebooks.py`.
