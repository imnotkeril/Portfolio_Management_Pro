"""Unit tests for analytics engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from core.analytics_engine.engine import AnalyticsEngine
from core.exceptions import InsufficientDataError


@pytest.fixture
def analytics_engine() -> AnalyticsEngine:
    """Create analytics engine instance."""
    return AnalyticsEngine(risk_free_rate=0.04)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create sample returns series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 252),
        index=dates
    )
    return returns


@pytest.fixture
def sample_benchmark_returns() -> pd.Series:
    """Create sample benchmark returns series."""
    np.random.seed(43)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    returns = pd.Series(
        np.random.normal(0.0008, 0.015, 252),
        index=dates
    )
    return returns


@pytest.fixture
def sample_portfolio_values() -> pd.Series:
    """Create sample portfolio values series."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    base_value = 100000.0
    returns = np.random.normal(0.001, 0.02, 252)
    values = base_value * (1 + pd.Series(returns)).cumprod()
    return pd.Series(values, index=dates)


class TestAnalyticsEngine:
    """Test AnalyticsEngine class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        engine = AnalyticsEngine()
        assert engine.risk_free_rate == 0.0435

    def test_init_custom_rate(self) -> None:
        """Test initialization with custom risk-free rate."""
        engine = AnalyticsEngine(risk_free_rate=0.05)
        assert engine.risk_free_rate == 0.05

    def test_calculate_all_metrics_basic(
        self,
        analytics_engine: AnalyticsEngine,
        sample_returns: pd.Series,
    ) -> None:
        """Test calculating all metrics without benchmark."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        results = analytics_engine.calculate_all_metrics(
            portfolio_returns=sample_returns,
            start_date=start_date,
            end_date=end_date,
        )

        # Check structure
        assert "performance" in results
        assert "risk" in results
        assert "ratios" in results
        assert "market" in results
        assert "metadata" in results

        # Check performance metrics exist
        assert "total_return" in results["performance"]
        assert "annualized_return" in results["performance"]
        # CAGR may not be present if portfolio_values not provided

        # Check risk metrics exist (keys may vary, check that dict is not empty)
        assert len(results["risk"]) > 0
        assert "max_drawdown" in results["risk"] or "annual" in results["risk"]
        assert len(results["ratios"]) > 0
        assert "sharpe_ratio" in results["ratios"]

        # Check metadata
        assert "calculation_time_seconds" in results["metadata"]
        assert "data_points" in results["metadata"]

    def test_calculate_all_metrics_with_benchmark(
        self,
        analytics_engine: AnalyticsEngine,
        sample_returns: pd.Series,
        sample_benchmark_returns: pd.Series,
    ) -> None:
        """Test calculating all metrics with benchmark."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        results = analytics_engine.calculate_all_metrics(
            portfolio_returns=sample_returns,
            start_date=start_date,
            end_date=end_date,
            benchmark_returns=sample_benchmark_returns,
        )

        # Check market metrics exist when benchmark provided
        assert "beta" in results["market"]
        assert "alpha" in results["market"]
        assert "correlation" in results["market"]

    def test_calculate_all_metrics_with_portfolio_values(
        self,
        analytics_engine: AnalyticsEngine,
        sample_returns: pd.Series,
        sample_portfolio_values: pd.Series,
    ) -> None:
        """Test calculating metrics with portfolio values."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        results = analytics_engine.calculate_all_metrics(
            portfolio_returns=sample_returns,
            start_date=start_date,
            end_date=end_date,
            portfolio_values=sample_portfolio_values,
        )

        # Check period returns are calculated
        assert "ytd" in results["performance"]
        assert "1y" in results["performance"]

    def test_calculate_all_metrics_empty_returns(
        self,
        analytics_engine: AnalyticsEngine,
    ) -> None:
        """Test calculating metrics with empty returns raises error."""
        empty_returns = pd.Series([], dtype=float)
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        with pytest.raises(InsufficientDataError):
            analytics_engine.calculate_all_metrics(
                portfolio_returns=empty_returns,
                start_date=start_date,
                end_date=end_date,
            )

    def test_calculate_all_metrics_insufficient_data(
        self,
        analytics_engine: AnalyticsEngine,
    ) -> None:
        """Test calculating metrics with insufficient data."""
        # Only 10 data points (need more for some metrics)
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)

        results = analytics_engine.calculate_all_metrics(
            portfolio_returns=returns,
            start_date=start_date,
            end_date=end_date,
        )

        # Should still return structure, but some metrics may be None
        assert "performance" in results
        assert "risk" in results
        # Some metrics may be None due to insufficient data
        assert results["performance"]["total_return"] is not None

    def test_calculate_all_metrics_metadata(
        self,
        analytics_engine: AnalyticsEngine,
        sample_returns: pd.Series,
    ) -> None:
        """Test metadata in results."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        results = analytics_engine.calculate_all_metrics(
            portfolio_returns=sample_returns,
            start_date=start_date,
            end_date=end_date,
        )

        metadata = results["metadata"]
        assert "calculation_time_seconds" in metadata
        assert isinstance(metadata["calculation_time_seconds"], float)
        assert metadata["calculation_time_seconds"] > 0

        assert "data_points" in metadata
        assert metadata["data_points"] > 0

        assert "start_date" in metadata
        assert "end_date" in metadata

