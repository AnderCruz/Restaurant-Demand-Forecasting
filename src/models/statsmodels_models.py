"""Small training-only Statsmodels order searches and forecast intervals."""
from __future__ import annotations
import warnings
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_candidates(train: pd.Series, seasonal_period: int = 7):
    """Fit constrained candidates and select AR/ARMA/ARIMA orders by training AIC.

    ARMA is the ARIMA special case with d=0. Missing training values are rejected;
    callers must document and fit any imputation on training data only.
    """
    y = train.asfreq("D")
    if y.isna().any():
        raise ValueError("Training target contains unknown values; impute only from training data or mark the candidate infeasible.")
    fitted = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            selection = ar_select_order(y, maxlag=min(14, max(1, len(y) // 5)), ic="aic", old_names=False)
            fitted["AutoReg_AIC"] = selection.model.fit()
        except Exception as exc:
            fitted["AutoReg_AIC"] = exc
        for label, d in (("ARMA_AIC", 0), ("ARIMA_AIC", 1)):
            orders = []
            for p in range(3):
                for q in range(3):
                    try:
                        result = ARIMA(y, order=(p, d, q)).fit()
                        orders.append((result.aic, result))
                    except Exception:
                        continue
            fitted[label] = min(orders, key=lambda item: item[0])[1] if orders else ValueError("No order converged")
        seasonal = []
        seasonal_orders = ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))
        for seasonal_p, seasonal_d, seasonal_q in seasonal_orders:
            try:
                result = SARIMAX(y, order=(1, 0, 1),
                                 seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                seasonal.append((result.aic, result))
            except Exception:
                continue
        fitted["SARIMAX_weekly_AIC"] = min(seasonal, key=lambda item: item[0])[1] if seasonal else ValueError("No seasonal order converged")
    return fitted


def forecast_frame(result, steps: int, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align mean and confidence bounds to an explicit daily future index."""
    if hasattr(result, "get_forecast"):
        frame = result.get_forecast(steps=steps).summary_frame()
    else:
        start = len(result.model.endog)
        prediction = result.get_prediction(start=start, end=start + steps - 1)
        mean = prediction.predicted_mean
        try:
            bounds = prediction.conf_int()
            frame = pd.DataFrame({"mean": mean, "mean_ci_lower": bounds.iloc[:, 0], "mean_ci_upper": bounds.iloc[:, 1]})
        except (AttributeError, ValueError, IndexError):
            frame = pd.DataFrame({"mean": mean, "mean_ci_lower": float("nan"), "mean_ci_upper": float("nan")})
    if len(frame) != len(future_index):
        raise ValueError("Forecast length does not match supplied future index")
    frame.index = future_index
    return frame.rename(columns={"mean": "yhat", "mean_ci_lower": "yhat_lower", "mean_ci_upper": "yhat_upper"})
