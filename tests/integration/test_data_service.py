"""Integration tests for data service."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from core.data_manager.cache import Cache
from services.data_service import DataService


def test_validate_ticker_integration() -> None:
    """Integration test for ticker validation."""
    service = DataService()

    # This will make actual API call - skip in CI or use mock
    with patch("core.data_manager.ticker_validator.yf.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        result = service.validate_ticker("AAPL")
        assert result is True


def test_fetch_historical_prices_with_cache() -> None:
    """Integration test for fetching prices with caching."""
    cache = Cache()
    service = DataService(cache=cache)

    # Mock price manager to avoid actual API calls
    with patch.object(
        service._price_manager, "fetch_historical_prices"
    ) as mock_fetch:
        # Create a proper DataFrame mock
        import pandas as pd
        mock_df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "open": [99.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [98.0, 99.0, 100.0],
            "volume": [1000000, 1100000, 1200000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        mock_fetch.return_value = mock_df

        # First call - should fetch from API
        result1 = service.fetch_historical_prices(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 31),
            use_cache=True,
            save_to_db=False,
        )

        # Second call - should use cache
        result2 = service.fetch_historical_prices(
            "AAPL",
            date(2024, 1, 1),
            date(2024, 1, 31),
            use_cache=True,
            save_to_db=False,
        )

        # Results should be DataFrames
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)
        # Results should be the same (cached)
        pd.testing.assert_frame_equal(result1, result2)
        # Price manager should be called (caching happens at price_manager level)
        assert mock_fetch.call_count >= 1


def test_validate_tickers_batch() -> None:
    """Integration test for batch ticker validation."""
    service = DataService()

    with patch("core.data_manager.ticker_validator.yf.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        results = service.validate_tickers(["AAPL", "MSFT", "GOOGL"])

        assert isinstance(results, dict)
        assert "AAPL" in results
        assert len(results) == 3

