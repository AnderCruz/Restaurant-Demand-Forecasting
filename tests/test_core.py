from datetime import timedelta
import numpy as np
import pandas as pd
import pytest

from src.data import load_demand, load_calendar
from src.features import make_lag_features
from src.evaluation import forecast_metrics, chronological_holdout
from src.backtesting import seasonal_naive_forecast


def test_load_demand_daily_index_and_preserves_missingness(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('data,Chimi & Churri,Assa Frão\n1/1/2016,10,20\n1/2/2016,,21\n', encoding='utf-8')
    frame = load_demand(p)
    assert len(frame) == 2 and frame.index.freqstr == 'D'
    assert frame['Chimi & Churri'].isna().sum() == 1
    assert frame['Chimi & Churri__missing'].tolist() == [False, True]


def test_duplicate_demand_dates_raise(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('data,Chimi & Churri,Assa Frão\n1/1/2016,10,20\n1/1/2016,11,21\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        load_demand(p)


def test_calendar_duplicate_dates_raise(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text('data,feriado\n1/1/2016,New Year\n1/1/2016,Holiday\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        load_calendar(p)


def test_lag_and_rolling_features_use_past_only():
    y = pd.Series(range(1, 10), index=pd.date_range('2020-01-01', periods=9))
    features = make_lag_features(y, lags=(1,), windows=(3,))
    assert features.loc[y.index[4], 'lag_1'] == 4
    assert features.loc[y.index[4], 'rolling_mean_3'] == 3
    assert pd.isna(features.loc[y.index[0], 'lag_1'])


def test_holdout_is_chronological():
    frame = pd.DataFrame({'y': range(20)}, index=pd.date_range('2020-01-01', periods=20))
    train, test = chronological_holdout(frame, 4)
    assert train.index.max() < test.index.min() and len(test) == 4


def test_metrics_handle_zero_denominators_and_mase():
    result = forecast_metrics([0, 2, 4], [0, 3, 2], training=[1, 2, 3, 4, 5, 6, 7, 8], mase_period=1)
    assert result['mae'] == pytest.approx(1.0)
    assert result['mase'] == pytest.approx(1.0)
    assert np.isfinite(result['smape'])


def test_seasonal_naive_repeats_latest_week():
    idx = pd.date_range('2020-02-01', periods=10)
    hist = pd.Series(range(10), index=idx)
    future = pd.date_range(idx[-1] + timedelta(days=1), periods=9)
    result = seasonal_naive_forecast(hist, future, period=7)
    assert result.tolist() == [3, 4, 5, 6, 7, 8, 9, 3, 4]


def test_prophet_contract_and_future_frame(monkeypatch):
    from src.models.prophet_models import prophet_frame, future_frame
    series = pd.Series([1, 2], index=pd.date_range('2020-01-01', periods=2))
    assert list(prophet_frame(series).columns) == ['ds', 'y']
    class Dummy:
        def make_future_dataframe(self, periods, freq):
            return pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=2+periods, freq=freq)})
    assert len(future_frame(Dummy(), 3)) == 5


def test_statsmodels_forecast_frame_alignment():
    from src.models.statsmodels_models import forecast_frame
    class Forecast:
        def summary_frame(self):
            return pd.DataFrame({'mean': [2, 3], 'mean_ci_lower': [1, 2], 'mean_ci_upper': [3, 4]})
    class Result:
        def get_forecast(self, steps):
            assert steps == 2
            return Forecast()
    idx = pd.date_range('2020-01-03', periods=2)
    frame = forecast_frame(Result(), 2, idx)
    assert frame.index.equals(idx) and frame['yhat'].tolist() == [2, 3]


def test_recursive_forecast_uses_previous_predictions():
    from src.backtesting import recursive_predict
    class IncrementModel:
        def predict(self, features):
            return features["lag_1"].to_numpy() + 1
    def builder(values):
        return pd.DataFrame({"lag_1": values.shift(1)}, index=values.index)
    history = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2))
    future = pd.date_range("2020-01-03", periods=2)
    result = recursive_predict(IncrementModel(), history, future, builder, "target")
    assert result.tolist() == [3.0, 4.0]


def test_missing_demand_date_is_not_silently_filled(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('data,Chimi & Churri,Assa Frão\n1/1/2016,10,20\n1/3/2016,12,22\n', encoding='utf-8')
    with pytest.raises(ValueError, match='daily index'):
        load_demand(p)


def test_deep_learning_sequence_windows_are_chronological():
    from src.models.deep_learning_models import make_sequences
    x, y = make_sequences([1, 2, 3, 4, 5], lookback=3)
    assert x.shape == (2, 3, 1)
    assert x[:, :, 0].tolist() == [[1, 2, 3], [2, 3, 4]]
    assert y.tolist() == [4, 5]


def test_deep_learning_recursive_forecast_consumes_its_previous_prediction():
    from src.models.deep_learning_models import recursive_forecast
    class IncrementModel:
        def __call__(self, window, training=False):
            return window[:, -1, :] + 1
    forecast = recursive_forecast(IncrementModel(), [1, 2, 3], horizon=3, lookback=2)
    assert forecast.tolist() == pytest.approx([4, 5, 6])


def test_holdout_metric_contract_reports_error_and_intervals():
    from scripts.evaluate_holdout import _metrics
    result = _metrics(
        np.array([10.0, 12.0]), np.array([11.0, 10.0]),
        np.arange(1, 11, dtype=float), np.array([9.0, 9.0]), np.array([12.0, 13.0]),
    )
    assert result['n'] == 2
    assert result['mae'] == pytest.approx(1.5)
    assert result['mean_error'] == pytest.approx(-0.5)
    assert result['interval_coverage'] == pytest.approx(1.0)
