"""Aligned recursive forecasts and simple time-series baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(float(history.dropna().iloc[-1]), index=index, name="naive")


def seasonal_naive_forecast(history: pd.Series, index: pd.DatetimeIndex, period: int = 7) -> pd.Series:
    """Repeat the latest complete seasonal cycle; recursively repeats if needed."""
    values = history.dropna().to_numpy(dtype=float)
    if len(values) < period:
        raise ValueError("History must contain at least one complete seasonal cycle")
    forecasts = np.resize(values[-period:], len(index))
    return pd.Series(forecasts, index=index, name=f"seasonal_naive_{period}")


def recursive_predict(model, history: pd.Series, future_index: pd.DatetimeIndex, feature_builder, target: str):
    """Iteratively predict and append predictions, never reading future actuals."""
    values = history.copy()
    predictions = []
    for date in future_index:
        extended = pd.concat([values, pd.Series([np.nan], index=pd.DatetimeIndex([date]))])
        row = feature_builder(extended).loc[[date]]
        prediction = float(model.predict(row)[0])
        values.loc[date] = prediction
        predictions.append(prediction)
    return pd.Series(predictions, index=future_index, name=target)
