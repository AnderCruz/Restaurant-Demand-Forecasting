"""Leakage-aware calendar and lag feature construction."""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Create features known at the forecast origin for the supplied dates."""
    d = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=d)
    out["day_of_week"] = d.dayofweek
    out["is_weekend"] = (d.dayofweek >= 5).astype(int)
    out["month"] = d.month
    out["quarter"] = d.quarter
    out["day_of_month"] = d.day
    out["is_month_start"] = d.is_month_start.astype(int)
    out["is_month_end"] = d.is_month_end.astype(int)
    out["week_sin"] = np.sin(2 * np.pi * d.dayofweek / 7)
    out["week_cos"] = np.cos(2 * np.pi * d.dayofweek / 7)
    return out


def make_lag_features(y: pd.Series, lags=(1, 7, 14, 28), windows=(7, 14)) -> pd.DataFrame:
    """Create one-step features using only values observed before each date."""
    out = pd.DataFrame(index=y.index)
    for lag in lags:
        out[f"lag_{lag}"] = y.shift(lag)
    for window in windows:
        prior = y.shift(1)
        out[f"rolling_mean_{window}"] = prior.rolling(window, min_periods=window).mean()
        out[f"rolling_std_{window}"] = prior.rolling(window, min_periods=window).std()
    return out
