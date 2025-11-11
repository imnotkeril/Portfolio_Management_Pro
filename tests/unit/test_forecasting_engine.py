"""Unit tests for forecasting engine."""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from core.forecasting_engine.base import BaseForecaster, ForecastResult
from core.forecasting_engine.utils import (
    evaluate_forecast_metrics,
    calculate_confidence_intervals,
    prepare_features,
    calculate_technical_indicators,
)
from core.exceptions import InsufficientDataError, CalculationError


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

        ci = calculate_confidence_intervals(forecast, residuals=residuals, confidence_level=0.95)

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
                    forecast_dates=pd.date_range("2024-02-01", periods=horizon, freq="D"),
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

