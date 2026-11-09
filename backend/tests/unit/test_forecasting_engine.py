"""Unit tests for forecasting engine."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import (
    calculate_confidence_intervals,
    calculate_technical_indicators,
    evaluate_forecast_metrics,
    prepare_features,
)


class TestForecastResult:
    """Test ForecastResult dataclass."""

    def test_forecast_result_creation(self):
        """Test creating ForecastResult."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        values = np.array([100.0 + i for i in range(10)])
        returns = np.array([0.01] * 10)

        result = ForecastResult(
            method="Test",
            forecast_dates=dates,
            forecast_values=values,
            forecast_returns=returns,
        )

        assert result.method == "Test"
        assert len(result.forecast_values) == 10
        assert result.final_value == 109.0
        assert result.success is True

    def test_forecast_result_to_dict(self):
        """Test converting ForecastResult to dictionary."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        values = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        returns = np.array([0.01] * 5)

        result = ForecastResult(
            method="Test",
            forecast_dates=dates,
            forecast_values=values,
            forecast_returns=returns,
        )

        result_dict = result.to_dict()
        assert result_dict["method"] == "Test"
        assert len(result_dict["forecast_values"]) == 5
        assert result_dict["final_value"] == 104.0


class TestForecastMetrics:
    """Test forecast quality metrics."""

    def test_evaluate_forecast_metrics_perfect(self):
        """Test metrics with perfect forecast."""
        forecast = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        actual = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

        metrics = evaluate_forecast_metrics(forecast, actual)

        assert metrics["mape"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r_squared"] == 1.0

    def test_evaluate_forecast_metrics_with_error(self):
        """Test metrics with forecast error."""
        forecast = np.array([100.0, 102.0, 104.0])
        actual = np.array([100.0, 101.0, 103.0])

        metrics = evaluate_forecast_metrics(forecast, actual)

        assert metrics["mape"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r_squared"] < 1.0

    def test_evaluate_forecast_metrics_direction_accuracy(self):
        """Test direction accuracy calculation."""
        forecast = np.array([100.0, 101.0, 100.5, 102.0])
        actual = np.array([100.0, 101.5, 100.0, 102.5])

        metrics = evaluate_forecast_metrics(forecast, actual)

        # Direction accuracy should be calculated
        assert "direction_accuracy" in metrics
        assert not np.isnan(metrics["direction_accuracy"])

    def test_evaluate_forecast_metrics_empty(self):
        """Test metrics with empty arrays."""
        forecast = np.array([])
        actual = np.array([])

        metrics = evaluate_forecast_metrics(forecast, actual)

        assert np.isnan(metrics["mape"])
        assert np.isnan(metrics["rmse"])


class TestConfidenceIntervals:
    """Test confidence interval calculation."""

    def test_calculate_confidence_intervals(self):
        """Test confidence interval calculation."""
        forecast = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        residuals = np.array([0.1, -0.1, 0.2, -0.2, 0.1])

        ci = calculate_confidence_intervals(
            forecast, residuals=residuals, confidence_level=0.95
        )

        assert "upper_95" in ci
        assert "lower_95" in ci
        assert len(ci["upper_95"]) == len(forecast)
        assert len(ci["lower_95"]) == len(forecast)
        # Upper should be greater than forecast
        assert (ci["upper_95"] > forecast).all()
        # Lower should be less than forecast
        assert (ci["lower_95"] < forecast).all()


class TestFeaturePreparation:
    """Test feature preparation for ML models."""

    def test_prepare_features(self):
        """Test feature preparation."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(100.0 + np.random.randn(50).cumsum(), index=dates)
        returns = prices.pct_change().dropna()

        features_df = prepare_features(
            prices=prices,
            returns=returns,
            lookback=5,
            include_technical=True,
        )

        assert not features_df.empty
        assert "target" in features_df.columns
        # Should have lagged returns
        assert any("return_lag" in col for col in features_df.columns)

    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(100.0 + np.random.randn(50).cumsum(), index=dates)
        returns = prices.pct_change().dropna()

        indicators = calculate_technical_indicators(prices, returns)

        assert not indicators.empty
        assert "sma_5" in indicators.columns
        assert "sma_10" in indicators.columns
        assert "volatility_5" in indicators.columns
        assert "rsi" in indicators.columns


class TestBaseForecaster:
    """Test BaseForecaster abstract class."""

    def test_base_forecaster_init(self):
        """Test BaseForecaster initialization."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(100.0 + np.random.randn(50).cumsum(), index=dates)

        # Create a simple concrete implementation for testing
        class TestForecaster(BaseForecaster):
            def forecast(self, horizon: int, **kwargs):
                return ForecastResult(
                    method="Test",
                    forecast_dates=pd.date_range(
                        "2024-02-01", periods=horizon, freq="D"
                    ),
                    forecast_values=np.array([100.0] * horizon),
                    forecast_returns=np.array([0.0] * horizon),
                )

        forecaster = TestForecaster(prices=prices)
        assert len(forecaster.prices) == 50
        assert len(forecaster.returns) > 0

    def test_base_forecaster_insufficient_data(self):
        """Test BaseForecaster with insufficient data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = pd.Series(100.0 + np.random.randn(10).cumsum(), index=dates)

        class TestForecaster(BaseForecaster):
            def forecast(self, horizon: int, **kwargs):
                pass

        with pytest.raises(InsufficientDataError):
            TestForecaster(prices=prices)

    def test_calculate_change_pct(self):
        """Test change percentage calculation."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = pd.Series(100.0 + np.random.randn(50).cumsum(), index=dates)

        class TestForecaster(BaseForecaster):
            def forecast(self, horizon: int, **kwargs):
                pass

        forecaster = TestForecaster(prices=prices)
        forecast_values = np.array([110.0, 115.0, 120.0])
        last_price = 100.0

        change_pct = forecaster._calculate_change_pct(forecast_values, last_price)
        assert change_pct == 20.0  # (120 - 100) / 100 * 100


class TestArimaIrregularIndex:
    """Market data skips holidays, so the DatetimeIndex has no frequency."""

    @staticmethod
    def _prices() -> pd.Series:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2023-01-02", periods=400).delete([10, 57, 120, 200])
        values = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx)))
        return pd.Series(values, index=idx)

    @pytest.mark.parametrize(
        "kwargs", [{"auto": True}, {"auto": False, "p": 1, "d": 0, "q": 1}]
    )
    def test_forecast_works_without_index_frequency(self, kwargs):
        from core.forecasting_engine.classical.arima import ARIMAForecaster

        result = ARIMAForecaster(self._prices()).forecast(horizon=10, **kwargs)
        assert result.success is True
        assert len(result.forecast_values) == 10
        assert np.all(np.isfinite(result.forecast_values))


@pytest.mark.parametrize(
    "module,cls",
    [
        ("core.forecasting_engine.deep_learning.lstm", "LSTMForecaster"),
        ("core.forecasting_engine.deep_learning.tcn", "TCNForecaster"),
    ],
)
def test_directional_accuracy_metrics_on_arrays(module, cls):
    """Was: `not <array>` raised, so every LSTM/TCN forecast with validation failed."""
    pytest.importorskip("tensorflow")
    import importlib

    forecaster = getattr(importlib.import_module(module), cls)(
        pd.Series(
            np.linspace(100, 120, 60), index=pd.bdate_range("2024-01-02", periods=60)
        )
    )
    y_true = np.array([1.0, 2.0, 1.5, 1.8, 1.2])
    y_pred = np.array([1.0, 1.5, 1.7, 1.6, 1.9])
    metrics = forecaster._calculate_directional_accuracy(y_true, y_pred)
    # true moves + - + -, predicted + + - +: tp=1, fp=2, fn=1
    assert metrics["directional_accuracy"] == pytest.approx(0.25)
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(0.5)


class TestArimaGarchIrregularIndex:
    @pytest.mark.parametrize("kwargs", [{"arima_auto": True}, {"arima_auto": False}])
    def test_forecast_works_without_index_frequency(self, kwargs):
        from core.forecasting_engine.classical.arima_garch import ARIMAGARCHForecaster

        prices = TestArimaIrregularIndex._prices()
        result = ARIMAGARCHForecaster(prices).forecast(horizon=10, **kwargs)
        assert result.success is True
        assert len(result.forecast_values) == 10
        assert np.all(np.isfinite(result.forecast_values))
