"""Score saved final-holdout forecasts once against their observed outcomes."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 28


def _read_actuals() -> tuple[pd.DataFrame, pd.Series]:
    demand = pd.read_csv(ROOT / "data/raw/clientes_restaurantes.csv", parse_dates=["data"], date_format="%m/%d/%Y")
    demand = demand.set_index("data").sort_index()
    demand["Total demand"] = demand[["Chimi & Churri", "Assa Frão"]].sum(axis=1, min_count=2)
    return demand, demand.index[-HORIZON:]


def _metrics(actual: np.ndarray, predicted: np.ndarray, training: np.ndarray, lower, upper) -> dict:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y, yhat = actual[valid], predicted[valid]
    error = yhat - y
    denominator = np.abs(y).sum()
    weekly_scale = np.abs(training[7:] - training[:-7]).mean() if len(training) > 7 else np.nan
    coverage = width = np.nan
    if lower is not None and upper is not None:
        lo, hi = np.asarray(lower, dtype=float)[valid], np.asarray(upper, dtype=float)[valid]
        interval_valid = np.isfinite(lo) & np.isfinite(hi)
        if interval_valid.any():
            coverage = float(((y[interval_valid] >= lo[interval_valid]) & (y[interval_valid] <= hi[interval_valid])).mean())
            width = float((hi[interval_valid] - lo[interval_valid]).mean())
    return {
        "n": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "smape": float(np.mean(2 * np.abs(error) / (np.abs(y) + np.abs(yhat) + 1e-8))),
        "mase": float(np.abs(error).mean() / weekly_scale) if weekly_scale and np.isfinite(weekly_scale) else np.nan,
        "mean_error": float(error.mean()),
        "r2": float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)) if len(y) > 1 and np.var(y) > 0 else np.nan,
        "interval_coverage": coverage,
        "interval_mean_width": width,
    }


def main() -> None:
    demand, holdout_dates = _read_actuals()
    holdout = demand.loc[holdout_dates]
    forecast_path = ROOT / "artifacts/forecasts/final_holdout_forecasts.csv"
    forecasts = pd.read_csv(forecast_path, parse_dates=["date"])
    forecasts = forecasts[forecasts["date"].isin(holdout_dates)].copy()
    actual_rows = []
    for (target, model), part in forecasts.groupby(["restaurant", "model"], sort=True):
        part = part.sort_values("date").drop_duplicates("date", keep="last")
        actual = holdout[target].reindex(part["date"]).to_numpy(dtype=float)
        train = demand.loc[demand.index < holdout_dates[0], target].interpolate(limit_area="inside").dropna().to_numpy(dtype=float)
        lower = part["lower"].to_numpy(dtype=float) if part["lower"].notna().any() else None
        upper = part["upper"].to_numpy(dtype=float) if part["upper"].notna().any() else None
        actual_rows.append({"restaurant": target, "model": model, **_metrics(actual, part["prediction"].to_numpy(dtype=float), train, lower, upper)})
    result = pd.DataFrame(actual_rows).sort_values(["restaurant", "mae"])
    result.to_csv(ROOT / "reports/final_holdout_metrics.csv", index=False)
    metadata = {
        "holdout_period": [str(holdout_dates.min().date()), str(holdout_dates.max().date())],
        "horizon_days": HORIZON,
        "training_cutoff": str(demand.index[-HORIZON - 1].date()),
        "selection_rule": "The final comparison contains model families/parameters selected using development folds before holdout scoring.",
        "scoring_rule": "Unknown actuals are excluded; total demand requires both restaurant actuals.",
        "warning": "Use these holdout results for final assessment; do not tune or reselect using them.",
        "forecast_file": str(forecast_path.relative_to(ROOT)),
    }
    (ROOT / "reports/final_holdout_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Saved reports/final_holdout_metrics.csv and reports/final_holdout_metadata.json")


if __name__ == "__main__":
    main()
