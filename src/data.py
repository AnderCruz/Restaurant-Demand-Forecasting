"""Data loading and validation for restaurant demand forecasting."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

TARGETS = ("Chimi & Churri", "Assa Frão")
DATE_COLUMN = "data"


def load_demand(path: str | Path) -> pd.DataFrame:
    """Load demand as a daily-indexed frame without imputing unknown targets."""
    frame = pd.read_csv(path, parse_dates=[DATE_COLUMN], date_format="%m/%d/%Y")
    missing = {DATE_COLUMN, *TARGETS} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[DATE_COLUMN].duplicated().any():
        duplicates = frame.loc[frame[DATE_COLUMN].duplicated(False), DATE_COLUMN].tolist()
        raise ValueError(f"Duplicate demand dates: {duplicates}")
    frame = frame.set_index(DATE_COLUMN).sort_index()
    frame.index.name = "date"
    if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq="D")):
        raise ValueError("Demand dates must form a complete daily index; gaps are not filled.")
    frame = frame.asfreq("D")
    for target in TARGETS:
        frame[f"{target}__missing"] = frame[target].isna()
    return frame


def load_calendar(path: str | Path) -> pd.DataFrame:
    """Load commemorative labels with source date coverage retained."""
    calendar = pd.read_csv(path, parse_dates=[DATE_COLUMN], date_format="%m/%d/%Y")
    if calendar[DATE_COLUMN].duplicated().any():
        raise ValueError("Calendar contains duplicate dates; resolve labels explicitly.")
    return calendar.set_index(DATE_COLUMN).sort_index()
