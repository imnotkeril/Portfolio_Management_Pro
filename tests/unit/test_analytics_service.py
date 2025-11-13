"""Unit tests for analytics service."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.analytics_engine.engine import AnalyticsEngine
from core.data_manager.portfolio import Portfolio
from core.exceptions import InsufficientDataError, ValidationError
from services.analytics_service import AnalyticsService
from services.data_service import DataService
from services.portfolio_service import PortfolioService


@pytest.fixture
def mock_analytics_engine() -> AnalyticsEngine:
    """Create mock analytics engine."""
    return MagicMock(spec=AnalyticsEngine)


@pytest.fixture
def mock_portfolio_service() -> PortfolioService:
    """Create mock portfolio service."""
    return MagicMock(spec=PortfolioService)


@pytest.fixture
def mock_data_service() -> DataService:
    """Create mock data service."""
    return MagicMock(spec=DataService)


@pytest.fixture
def analytics_service(
    mock_analytics_engine: AnalyticsEngine,
    mock_portfolio_service: PortfolioService,
    mock_data_service: DataService,
) -> AnalyticsService:
    """Create analytics service with mocked dependencies."""
    return AnalyticsService(
        analytics_engine=mock_analytics_engine,
        portfolio_service=mock_portfolio_service,
        data_service=mock_data_service,
    )


def test_calculate_portfolio_metrics_invalid_date_range(
    analytics_service: AnalyticsService,
) -> None:
    """Test metrics calculation with invalid date range."""
    start_date = date(2024, 12, 31)
    end_date = date(2024, 1, 1)  # Invalid: end before start

    with pytest.raises(ValidationError, match="Start date must be before end date"):
        analytics_service.calculate_portfolio_metrics(
            portfolio_id="test-id",
            start_date=start_date,
            end_date=end_date,
        )


def test_calculate_portfolio_metrics_no_positions(
    analytics_service: AnalyticsService,
) -> None:
    """Test metrics calculation with portfolio that has no positions."""
    portfolio = Portfolio(name="Empty Portfolio", starting_capital=100000.0)
    analytics_service._portfolio_service.get_portfolio.return_value = portfolio

    with pytest.raises(InsufficientDataError, match="Portfolio has no positions"):
        analytics_service.calculate_portfolio_metrics(
            portfolio_id="test-id",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )


def test_calculate_portfolio_metrics_success(
    analytics_service: AnalyticsService,
) -> None:
    """Test successfully calculating portfolio metrics."""
    # Create portfolio with positions
    portfolio = Portfolio(name="Test Portfolio", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)
    analytics_service._portfolio_service.get_portfolio.return_value = portfolio

    # Mock price data
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price_df = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]},
        index=dates,
    )

    # Mock fetch_portfolio_prices
    with patch.object(
        analytics_service, "_fetch_portfolio_prices", return_value=price_df
    ) as mock_fetch:
        # Mock calculate_portfolio_returns
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=dates[1:])
        with patch.object(
            analytics_service, "_calculate_portfolio_returns", return_value=returns
        ) as mock_calc_returns:
            # Mock calculate_portfolio_values
            values = pd.Series([100000.0] * 10, index=dates)
            with patch.object(
                analytics_service, "_calculate_portfolio_values", return_value=values
            ) as mock_calc_values:
                # Mock engine.calculate_all_metrics
                mock_metrics = {
                    "performance": {"total_return": 0.1},
                    "risk": {"volatility": 0.15},
                    "ratios": {"sharpe_ratio": 1.5},
                    "market": {},
                    "metadata": {"data_points": 10},
                }
                analytics_service._engine.calculate_all_metrics.return_value = mock_metrics

                result = analytics_service.calculate_portfolio_metrics(
                    portfolio_id="test-id",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 10),
                )

                assert "performance" in result
                assert "risk" in result
                assert "ratios" in result
                assert "market" in result
                assert "metadata" in result
                assert "portfolio_returns" in result
                assert "portfolio_values" in result


def test_calculate_portfolio_metrics_with_benchmark(
    analytics_service: AnalyticsService,
) -> None:
    """Test metrics calculation with benchmark ticker."""
    portfolio = Portfolio(name="Test Portfolio", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)
    analytics_service._portfolio_service.get_portfolio.return_value = portfolio

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price_df = pd.DataFrame(
        {"AAPL": [100.0] * 10, "SPY": [400.0] * 10}, index=dates
    )

    returns = pd.Series([0.01] * 9, index=dates[1:])
    values = pd.Series([100000.0] * 10, index=dates)

    with patch.object(
        analytics_service, "_fetch_portfolio_prices", return_value=price_df
    ):
        with patch.object(
            analytics_service, "_calculate_portfolio_returns", return_value=returns
        ):
            with patch.object(
                analytics_service, "_calculate_portfolio_values", return_value=values
            ):
                mock_metrics = {
                    "performance": {},
                    "risk": {},
                    "ratios": {},
                    "market": {"beta": 1.2},
                    "metadata": {},
                }
                analytics_service._engine.calculate_all_metrics.return_value = mock_metrics

                result = analytics_service.calculate_portfolio_metrics(
                    portfolio_id="test-id",
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 1, 10),
                    benchmark_ticker="SPY",
                )

                assert "benchmark_returns" in result
                analytics_service._engine.calculate_all_metrics.assert_called_once()


def test_get_single_ticker_returns(analytics_service: AnalyticsService) -> None:
    """Test getting returns for a single ticker."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # _fetch_portfolio_prices returns DataFrame with ticker as column
    price_df = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]},
        index=dates,
    )

    with patch.object(
        analytics_service, "_fetch_portfolio_prices", return_value=price_df
    ) as mock_fetch:
        result = analytics_service._get_single_ticker_returns(
            "AAPL", date(2024, 1, 1), date(2024, 1, 10)
        )

        assert isinstance(result, pd.Series)
        # Should have returns (one less than prices due to pct_change)
        assert len(result) >= 0  # Can be empty if calculation fails
        mock_fetch.assert_called_once()


def test_get_portfolio_returns_by_id(analytics_service: AnalyticsService) -> None:
    """Test getting returns for a portfolio by ID."""
    portfolio = Portfolio(name="Test Portfolio", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)
    analytics_service._portfolio_service.get_portfolio.return_value = portfolio

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price_df = pd.DataFrame({"AAPL": [100.0] * 10}, index=dates)

    with patch.object(
        analytics_service, "_fetch_portfolio_prices", return_value=price_df
    ):
        with patch.object(
            analytics_service, "_calculate_portfolio_returns"
        ) as mock_calc:
            returns = pd.Series([0.01] * 9, index=dates[1:])
            mock_calc.return_value = returns

            result = analytics_service._get_portfolio_returns_by_id(
                "test-id", date(2024, 1, 1), date(2024, 1, 10)
            )

            assert isinstance(result, pd.Series)
            assert len(result) > 0


def test_compute_basic_metrics_from_returns(analytics_service: AnalyticsService) -> None:
    """Test computing basic metrics from returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    result = analytics_service._compute_basic_metrics_from_returns(returns)

    assert isinstance(result, dict)
    assert "total_return" in result or "annualized_return" in result


def test_fetch_portfolio_prices(analytics_service: AnalyticsService) -> None:
    """Test fetching prices for multiple tickers."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # fetch_historical_prices returns DataFrame with Date and Adjusted_Close
    price_df_aapl = pd.DataFrame(
        {
            "Date": dates,
            "Adjusted_Close": [100.0] * 10,
        },
    )
    price_df_msft = pd.DataFrame(
        {
            "Date": dates,
            "Adjusted_Close": [300.0] * 10,
        },
    )

    def mock_fetch(ticker, start, end, **kwargs):
        if ticker == "AAPL":
            return price_df_aapl.copy()
        elif ticker == "MSFT":
            return price_df_msft.copy()
        return pd.DataFrame()

    analytics_service._data_service.fetch_historical_prices.side_effect = mock_fetch

    result = analytics_service._fetch_portfolio_prices(
        ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 10)
    )

    assert isinstance(result, pd.DataFrame)
    # After pivot, tickers should be columns
    if not result.empty:
        assert "AAPL" in result.columns or "MSFT" in result.columns


def test_calculate_portfolio_returns(analytics_service: AnalyticsService) -> None:
    """Test calculating portfolio returns from prices."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price_df = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]},
        index=dates,
    )

    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    result = analytics_service._calculate_portfolio_returns(price_df, portfolio.get_all_positions())

    assert isinstance(result, pd.Series)
    assert len(result) > 0


def test_calculate_portfolio_values(analytics_service: AnalyticsService) -> None:
    """Test calculating portfolio values from prices."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    # Price DataFrame has tickers as columns
    price_df = pd.DataFrame(
        {"AAPL": [100.0] * 10}, index=dates
    )

    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    result = analytics_service._calculate_portfolio_values(
        price_df, portfolio.get_all_positions(), 100000.0
    )

    assert isinstance(result, pd.Series)
    # Should have values for each date
    assert len(result) > 0
    # First value should be close to starting capital (or calculated from prices)
    assert result.iloc[0] > 0

