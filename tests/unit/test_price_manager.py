"""Unit tests for price manager."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.data_manager.price_manager import PriceManager
from core.exceptions import DataFetchError, TickerNotFoundError


@patch("core.data_manager.price_manager.yf.Ticker")
def test_fetch_current_price(mock_ticker_class) -> None:
    """Test fetching current price."""
    manager = PriceManager()

    # Mock ticker and history
    mock_ticker = MagicMock()
    mock_history = pd.DataFrame(
        {
            "Close": [150.0],
        },
        index=pd.date_range("2024-01-01", periods=1),
    )
    mock_ticker.history.return_value = mock_history
    mock_ticker_class.return_value = mock_ticker

    price = manager.fetch_current_price("AAPL", use_cache=False)

    assert price == 150.0
    mock_ticker.history.assert_called_once_with(period="1d")


@patch("core.data_manager.price_manager.yf.Ticker")
def test_fetch_current_price_empty(mock_ticker_class) -> None:
    """Test fetching current price when no data available."""
    manager = PriceManager()

    # Mock ticker with empty history
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_class.return_value = mock_ticker

    with pytest.raises(TickerNotFoundError):
        manager.fetch_current_price("INVALID", use_cache=False)


@patch("core.data_manager.price_manager.yf.download")
def test_fetch_bulk_prices(mock_download) -> None:
    """Test fetching prices for multiple tickers."""
    manager = PriceManager()

    # Mock download data
    dates = pd.date_range("2024-01-01", periods=5)
    mock_data = pd.DataFrame(
        {
            ("AAPL", "Close"): [150.0, 151.0, 152.0, 153.0, 154.0],
            ("MSFT", "Close"): [300.0, 301.0, 302.0, 303.0, 304.0],
        },
        index=dates,
    )
    mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
    mock_download.return_value = mock_data

    result = manager.fetch_bulk_prices(
        ["AAPL", "MSFT"],
        date(2024, 1, 1),
        date(2024, 1, 5),
        use_cache=False,
    )

    assert not result.empty
    mock_download.assert_called_once()


def test_standardize_dataframe() -> None:
    """Test DataFrame standardization."""
    manager = PriceManager()

    # Create sample DataFrame similar to yfinance output
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000000, 1100000],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )

    result = manager._standardize_dataframe(df)

    assert "Open" in result.columns
    assert "High" in result.columns
    assert "Low" in result.columns
    assert "Close" in result.columns
    assert "Adjusted_Close" in result.columns
    assert "Volume" in result.columns


@patch("core.data_manager.price_manager.yf.Ticker")
def test_fetch_historical_prices_retry(mock_ticker_class) -> None:
    """Test retry logic for failed fetches."""
    manager = PriceManager()

    # Mock ticker that fails first time, succeeds second time
    mock_ticker = MagicMock()
    mock_ticker.history.side_effect = [
        Exception("Network error"),
        pd.DataFrame(
            {"Close": [150.0]},
            index=pd.date_range("2024-01-01", periods=1),
        ),
    ]
    mock_ticker.info = {"symbol": "AAPL"}
    mock_ticker_class.return_value = mock_ticker

    result = manager._fetch_from_yahoo(
        "AAPL", date(2024, 1, 1), date(2024, 1, 1)
    )

    assert not result.empty
    assert mock_ticker.history.call_count == 2

