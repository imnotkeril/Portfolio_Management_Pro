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
        mock_df = MagicMock()
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

        # Price manager should be called only once due to caching
        assert mock_fetch.call_count == 1


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

