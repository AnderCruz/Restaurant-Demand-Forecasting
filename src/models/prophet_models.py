"""Prophet input validation, fitting, CV, and serialization."""
from __future__ import annotations
import pandas as pd


def prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Convert a target series into Prophet's required ds/y schema."""
    frame = series.rename("y").rename_axis("ds").reset_index()
    if list(frame.columns) != ["ds", "y"]:
        raise ValueError("Prophet input must contain exactly ds and y")
    if frame["ds"].duplicated().any() or frame["y"].isna().any():
        raise ValueError("Prophet input dates must be unique and targets observed")
    return frame


def make_model(seasonality_mode: str = "additive", yearly_seasonality: bool = False,
               holidays=None, changepoint_prior_scale: float = 0.05,
               seasonality_prior_scale: float = 10.0, holidays_prior_scale: float = 10.0):
    """Create a compact Prophet candidate; yearly seasonality defaults off."""
    from prophet import Prophet
    return Prophet(weekly_seasonality=True, yearly_seasonality=yearly_seasonality,
                   seasonality_mode=seasonality_mode, holidays=holidays,
                   changepoint_prior_scale=changepoint_prior_scale,
                   seasonality_prior_scale=seasonality_prior_scale,
                   holidays_prior_scale=holidays_prior_scale)


def future_frame(model, periods: int, freq: str = "D") -> pd.DataFrame:
    if periods < 1:
        raise ValueError("periods must be positive")
    return model.make_future_dataframe(periods=periods, freq=freq)
