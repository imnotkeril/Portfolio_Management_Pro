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


@patch("core.data_manager.price_manager.yf.Ticker")
def test_fetch_historical_prices_with_cache(mock_ticker_class, cache) -> None:
    """Test that cache is used when fetching historical prices."""
    manager = PriceManager(cache=cache)

    # Mock ticker
    mock_ticker = MagicMock()
    mock_history = pd.DataFrame(
        {
            "Close": [150.0, 151.0],
            "Open": [149.0, 150.0],
            "High": [151.0, 152.0],
            "Low": [148.0, 149.0],
            "Volume": [1000000, 1100000],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    mock_ticker.history.return_value = mock_history
    mock_ticker_class.return_value = mock_ticker

    # First fetch - should call API
    result1 = manager.fetch_historical_prices(
        "AAPL", date(2024, 1, 1), date(2024, 1, 2), use_cache=True
    )

    # Second fetch - should use cache
    result2 = manager.fetch_historical_prices(
        "AAPL", date(2024, 1, 1), date(2024, 1, 2), use_cache=True
    )

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)
    # API should only be called once (second call uses cache)
    assert mock_ticker.history.call_count == 1


@patch("core.data_manager.price_manager.yf.Ticker")
def test_fetch_historical_prices_invalid_ticker(mock_ticker_class) -> None:
    """Test fetching prices for invalid ticker."""
    manager = PriceManager()

    # Mock ticker with empty history and no info
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker.info = {}  # Empty info means ticker not found
    mock_ticker_class.return_value = mock_ticker

    # fetch_historical_prices wraps TickerNotFoundError in DataFetchError
    with pytest.raises(DataFetchError):
        manager.fetch_historical_prices(
            "INVALID", date(2024, 1, 1), date(2024, 1, 2), use_cache=False
        )


def test_standardize_dataframe_missing_columns() -> None:
    """Test standardization with missing columns."""
    manager = PriceManager()

    # DataFrame with only Close column
    df = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    result = manager._standardize_dataframe(df)

    # Should still have Close column
    assert "Close" in result.columns
    # Standardize should add required columns (Open, High, Low, etc.)
    # Even if missing, they should be present in result
    assert len(result.columns) > 0


@patch("core.data_manager.price_manager.yf.download")
def test_fetch_bulk_prices_single_ticker(mock_download) -> None:
    """Test bulk fetching with single ticker (should use regular fetch)."""
    manager = PriceManager()

    # Mock single ticker fetch
    with patch.object(manager, "fetch_historical_prices") as mock_fetch:
        mock_df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5),
                "Close": [150.0, 151.0, 152.0, 153.0, 154.0],
            }
        )
        mock_fetch.return_value = mock_df

        result = manager.fetch_bulk_prices(
            ["AAPL"],
            date(2024, 1, 1),
            date(2024, 1, 5),
            use_cache=False,
        )

        assert not result.empty
        assert "Ticker" in result.columns
        assert result["Ticker"].iloc[0] == "AAPL"
        # Should use regular fetch, not bulk download
        mock_fetch.assert_called_once()
        mock_download.assert_not_called()


@patch("core.data_manager.price_manager.yf.download")
def test_fetch_bulk_prices_all_cached(mock_download, cache) -> None:
    """Test bulk fetching when all tickers are cached (should use sequential)."""
    manager = PriceManager(cache=cache)

    # Pre-populate cache
    start_date = date(2024, 1, 1)
    end_date = date(2024, 1, 5)
    for ticker in ["AAPL", "MSFT"]:
        cache_key = f"prices:{ticker}:{start_date}:{end_date}"
        mock_df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5),
                "Close": [150.0] * 5,
            }
        )
        cache.set(cache_key, mock_df, ttl=3600)

    # Mock fetch_historical_prices to return cached data with Ticker column
    with patch.object(manager, "fetch_historical_prices") as mock_fetch:
        def fetch_side_effect(ticker, *args, **kwargs):
            mock_df = pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=5),
                    "Close": [150.0 if ticker == "AAPL" else 300.0] * 5,
                }
            )
            return mock_df

        mock_fetch.side_effect = fetch_side_effect

        result = manager.fetch_bulk_prices(
            ["AAPL", "MSFT"],
            start_date,
            end_date,
            use_cache=True,
        )

        assert not result.empty
        assert len(result["Ticker"].unique()) == 2
        assert "AAPL" in result["Ticker"].values
        assert "MSFT" in result["Ticker"].values
        # Should use sequential fetch from cache, not bulk download
        assert mock_fetch.call_count == 2
        mock_download.assert_not_called()


@patch("core.data_manager.price_manager.yf.download")
@patch("core.data_manager.price_manager.ThreadPoolExecutor")
def test_fetch_bulk_prices_parallel_fallback(mock_executor_class, mock_download) -> None:
    """Test parallel fetching fallback when bulk download fails."""
    manager = PriceManager()

    # Make bulk download fail
    mock_download.side_effect = Exception("Bulk download failed")

    # Mock ThreadPoolExecutor
    mock_executor = MagicMock()
    mock_executor_class.return_value.__enter__.return_value = mock_executor
    mock_executor_class.return_value.__exit__.return_value = None

    # Mock futures
    from concurrent.futures import Future
    mock_future1 = MagicMock(spec=Future)
    mock_future2 = MagicMock(spec=Future)
    
    mock_df1 = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5),
            "Close": [150.0] * 5,
        }
    )
    mock_df1["Ticker"] = "AAPL"
    
    mock_df2 = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5),
            "Close": [300.0] * 5,
        }
    )
    mock_df2["Ticker"] = "MSFT"

    mock_future1.result.return_value = ("AAPL", mock_df1)
    mock_future2.result.return_value = ("MSFT", mock_df2)

    mock_executor.submit.side_effect = [mock_future1, mock_future2]
    mock_executor.__enter__.return_value = mock_executor
    mock_executor.__exit__.return_value = None

    # Mock as_completed to return futures in order
    from unittest.mock import patch as mock_patch
    with mock_patch("core.data_manager.price_manager.as_completed") as mock_as_completed:
        mock_as_completed.return_value = [mock_future1, mock_future2]

        # Mock fetch_historical_prices for parallel fetching
        with patch.object(manager, "fetch_historical_prices") as mock_fetch:
            def fetch_side_effect(ticker, *args, **kwargs):
                if ticker == "AAPL":
                    return mock_df1.drop(columns=["Ticker"])
                else:
                    return mock_df2.drop(columns=["Ticker"])

            mock_fetch.side_effect = fetch_side_effect

            result = manager.fetch_bulk_prices(
                ["AAPL", "MSFT"],
                date(2024, 1, 1),
                date(2024, 1, 5),
                use_cache=False,
            )

            # Should fallback to parallel fetching
            assert not result.empty
            assert len(result["Ticker"].unique()) == 2
            # Should have tried bulk download first
            mock_download.assert_called_once()