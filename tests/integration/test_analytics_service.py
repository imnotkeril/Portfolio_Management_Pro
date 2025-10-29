"""Integration tests for analytics service."""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from services.analytics_service import AnalyticsService
from services.portfolio_service import PortfolioService
from services.data_service import DataService
from core.data_manager.portfolio import Portfolio


@pytest.fixture
def mock_portfolio_service() -> PortfolioService:
    """Create mock portfolio service."""
    service = PortfolioService()
    return service


@pytest.fixture
def mock_data_service() -> DataService:
    """Create mock data service."""
    service = DataService()

    # Mock fetch_historical_prices
    def mock_fetch(
        ticker: str, start: date, end: date, **kwargs
    ) -> pd.DataFrame:
        # Generate mock price data
        dates = pd.date_range(start, end, freq="D")
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        prices = prices[1:]  # Remove first base price

        df = pd.DataFrame({
            "Date": dates,
            "Adjusted_Close": prices,
        })
        return df

    service.fetch_historical_prices = MagicMock(side_effect=mock_fetch)
    return service


@pytest.fixture
def analytics_service(
    mock_portfolio_service: PortfolioService,
    mock_data_service: DataService,
) -> AnalyticsService:
    """Create analytics service with mocked dependencies."""
    return AnalyticsService(
        portfolio_service=mock_portfolio_service,
        data_service=mock_data_service,
    )


def test_calculate_portfolio_metrics_basic(
    analytics_service: AnalyticsService,
) -> None:
    """Test basic metrics calculation."""
    # Create a portfolio
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
    )
    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.add_position(ticker="MSFT", shares=50.0)

    # Mock portfolio service to return our portfolio
    with patch.object(
        analytics_service._portfolio_service,
        "get_portfolio",
    ) as mock_get:
        mock_get.return_value = portfolio

        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        # This will use mocked data service
        metrics = analytics_service.calculate_portfolio_metrics(
            portfolio_id="test-id",
            start_date=start_date,
            end_date=end_date,
        )

        assert "performance" in metrics
        assert "risk" in metrics
        assert "ratios" in metrics
        assert "market" in metrics
        assert "metadata" in metrics


def test_calculate_portfolio_metrics_with_benchmark(
    analytics_service: AnalyticsService,
) -> None:
    """Test metrics calculation with benchmark."""
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
    )
    portfolio.add_position(ticker="AAPL", shares=100.0)

    with patch.object(
        analytics_service._portfolio_service,
        "get_portfolio",
    ) as mock_get:
        mock_get.return_value = portfolio

        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)

        metrics = analytics_service.calculate_portfolio_metrics(
            portfolio_id="test-id",
            start_date=start_date,
            end_date=end_date,
            benchmark_ticker="SPY",
        )

        # With benchmark, market metrics should be populated
        assert "market" in metrics
        # Some market metrics should be calculated
        assert len(metrics["market"]) > 0


def test_calculate_portfolio_metrics_invalid_date_range(
    analytics_service: AnalyticsService,
) -> None:
    """Test metrics calculation with invalid date range."""
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
    )

    with patch.object(
        analytics_service._portfolio_service,
        "get_portfolio",
    ) as mock_get:
        mock_get.return_value = portfolio

        start_date = date(2024, 12, 31)
        end_date = date(2024, 1, 1)  # Invalid: end before start

        with pytest.raises(Exception):  # Should raise ValidationError
            analytics_service.calculate_portfolio_metrics(
                portfolio_id="test-id",
                start_date=start_date,
                end_date=end_date,
            )

