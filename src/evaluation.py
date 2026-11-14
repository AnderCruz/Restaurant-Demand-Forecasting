"""Forecast metrics and chronological evaluation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd


def forecast_metrics(actual, predicted, training=None, mase_period: int = 7) -> dict[str, float]:
    """Calculate point forecast metrics; MASE requires training observations."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        raise ValueError("No finite actual/predicted pairs.")
    err = p - y
    denom = np.abs(y) + np.abs(p)
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) else float("nan"),
        "smape": float(np.mean(np.divide(2 * np.abs(err), denom, out=np.zeros_like(err), where=denom != 0))),
        "mean_error": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 and np.sum((y-y.mean())**2) else float("nan"),
    }
    if training is not None:
        train = np.asarray(training, dtype=float)
        scale = np.nanmean(np.abs(train[mase_period:] - train[:-mase_period])) if len(train) > mase_period else float("nan")
        metrics["mase"] = float(metrics["mae"] / scale) if np.isfinite(scale) and scale else float("nan")
    return metrics


def chronological_holdout(frame: pd.DataFrame, horizon: int = 28):
    """Return a chronological train/test split with no shuffling."""
    if horizon < 1 or horizon >= len(frame):
        raise ValueError("horizon must be positive and smaller than the series")
    return frame.iloc[:-horizon].copy(), frame.iloc[-horizon:].copy()


def expanding_splits(n: int, initial: int, horizon: int, step: int | None = None):
    """Yield expanding-window positional train/test slices."""
    step = step or horizon
    origin = initial
    while origin + horizon <= n:
        yield slice(0, origin), slice(origin, origin + horizon)
        origin += step


def interval_metrics(actual, lower, upper) -> dict[str, float]:
    """Calculate empirical coverage and mean interval width on finite rows."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not valid.any():
        raise ValueError("No finite actual/interval rows.")
    return {"coverage": float(np.mean((y[valid] >= lo[valid]) & (y[valid] <= hi[valid]))),
            "mean_interval_width": float(np.mean(hi[valid] - lo[valid]))}

def horizon_band_metrics(actual, predicted, training=None) -> pd.DataFrame:
    """Summarize metrics for days 1-7, 8-14, 15-21, and 22-28."""
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(y) != len(p):
        raise ValueError("Actual and prediction lengths must match")
    rows = []
    for start in range(0, len(y), 7):
        stop = min(start + 7, len(y))
        rows.append({"horizon_start": start + 1, "horizon_end": stop,
                     **forecast_metrics(y[start:stop], p[start:stop], training)})
    return pd.DataFrame(rows)
