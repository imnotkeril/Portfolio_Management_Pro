"""Unit tests for data service."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.data_manager.cache import Cache
from core.data_manager.price_manager import PriceManager
from core.data_manager.ticker_validator import TickerInfo, TickerValidator
from core.exceptions import DataFetchError, TickerNotFoundError, ValidationError
from services.data_service import DataService


@pytest.fixture
def mock_price_manager() -> PriceManager:
    """Create mock price manager."""
    return MagicMock(spec=PriceManager)


@pytest.fixture
def mock_ticker_validator() -> TickerValidator:
    """Create mock ticker validator."""
    return MagicMock(spec=TickerValidator)


@pytest.fixture
def data_service(
    cache: Cache,
    mock_price_manager: PriceManager,
    mock_ticker_validator: TickerValidator,
) -> DataService:
    """Create data service with mocked dependencies."""
    return DataService(
        cache=cache,
        price_manager=mock_price_manager,
        ticker_validator=mock_ticker_validator,
    )


def test_validate_ticker_valid(data_service: DataService) -> None:
    """Test validating a valid ticker."""
    data_service._ticker_validator.validate_ticker.return_value = True

    result = data_service.validate_ticker("AAPL")

    assert result is True
    data_service._ticker_validator.validate_ticker.assert_called_once_with("AAPL")


def test_validate_ticker_invalid(data_service: DataService) -> None:
    """Test validating an invalid ticker."""
    data_service._ticker_validator.validate_ticker.side_effect = ValidationError(
        "Invalid ticker format"
    )

    with pytest.raises(ValidationError):
        data_service.validate_ticker("INVALID")


def test_validate_tickers(data_service: DataService) -> None:
    """Test validating multiple tickers."""
    expected = {"AAPL": True, "MSFT": True, "INVALID": False}
    data_service._ticker_validator.validate_tickers.return_value = expected

    result = data_service.validate_tickers(["AAPL", "MSFT", "INVALID"])

    assert result == expected
    data_service._ticker_validator.validate_tickers.assert_called_once_with(
        ["AAPL", "MSFT", "INVALID"]
    )


def test_get_ticker_info(data_service: DataService) -> None:
    """Test getting ticker information."""
    ticker_info = TickerInfo(
        ticker="AAPL",
        name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
    )
    data_service._ticker_validator.get_ticker_info.return_value = ticker_info

    result = data_service.get_ticker_info("AAPL")

    assert result == ticker_info
    assert result.ticker == "AAPL"
    data_service._ticker_validator.get_ticker_info.assert_called_once_with("AAPL")


def test_fetch_historical_prices_success(data_service: DataService) -> None:
    """Test successfully fetching historical prices."""
    mock_df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "Open": [99.0, 100.0, 101.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [98.0, 99.0, 100.0],
            "Volume": [1000000, 1100000, 1200000],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    data_service._price_manager.fetch_historical_prices.return_value = mock_df

    result = data_service.fetch_historical_prices(
        "AAPL", date(2024, 1, 1), date(2024, 1, 3), use_cache=False, save_to_db=False
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    data_service._price_manager.fetch_historical_prices.assert_called_once()


def test_fetch_historical_prices_error(data_service: DataService) -> None:
    """Test fetching historical prices with error."""
    data_service._price_manager.fetch_historical_prices.side_effect = DataFetchError(
        "Failed to fetch data"
    )

    with pytest.raises(DataFetchError):
        data_service.fetch_historical_prices(
            "INVALID",
            date(2024, 1, 1),
            date(2024, 1, 3),
            use_cache=False,
            save_to_db=False,
        )


def test_fetch_current_price(data_service: DataService) -> None:
    """Test fetching current price."""
    data_service._price_manager.fetch_current_price.return_value = 150.0

    result = data_service.fetch_current_price("AAPL", use_cache=False)

    assert result == 150.0
    data_service._price_manager.fetch_current_price.assert_called_once_with(
        "AAPL", use_cache=False
    )


def test_get_latest_prices(data_service: DataService) -> None:
    """Test getting latest prices for multiple tickers."""
    # Mock fetch_current_price for each ticker
    def mock_fetch_current(ticker: str, **kwargs) -> float:
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}
        return prices.get(ticker, 0.0)

    data_service._price_manager.fetch_current_price.side_effect = mock_fetch_current

    result = data_service.get_latest_prices(["AAPL", "MSFT", "GOOGL"])

    assert isinstance(result, dict)
    assert result["AAPL"] == 150.0
    assert result["MSFT"] == 300.0
    assert result["GOOGL"] == 2500.0
    assert len(result) == 3


def test_get_latest_prices_single_ticker(data_service: DataService) -> None:
    """Test getting latest price for single ticker (should use regular fetch)."""
    data_service._price_manager.fetch_current_price.return_value = 150.0

    result = data_service.get_latest_prices(["AAPL"])

    assert isinstance(result, dict)
    assert result["AAPL"] == 150.0
    assert len(result) == 1
    # Should use regular fetch, not parallel
    data_service._price_manager.fetch_current_price.assert_called_once_with(
        "AAPL", use_cache=True
    )


def test_get_latest_prices_parallel(data_service: DataService) -> None:
    """Test parallel fetching for multiple tickers."""
    # Mock fetch_current_price for each ticker
    def mock_fetch_current(ticker: str, **kwargs) -> float:
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0, "AMZN": 3500.0}
        return prices.get(ticker, 0.0)

    data_service._price_manager.fetch_current_price.side_effect = mock_fetch_current

    result = data_service.get_latest_prices(["AAPL", "MSFT", "GOOGL", "AMZN"])

    assert isinstance(result, dict)
    assert len(result) == 4
    assert result["AAPL"] == 150.0
    assert result["MSFT"] == 300.0
    assert result["GOOGL"] == 2500.0
    assert result["AMZN"] == 3500.0
    # Should be called for each ticker
    assert data_service._price_manager.fetch_current_price.call_count == 4


def test_get_latest_prices_with_error(data_service: DataService) -> None:
    """Test getting latest prices with some errors."""
    def mock_fetch_current(ticker: str, **kwargs) -> float:
        if ticker == "INVALID":
            raise TickerNotFoundError("Ticker not found")
        return 100.0

    data_service._price_manager.fetch_current_price.side_effect = mock_fetch_current

    result = data_service.get_latest_prices(["AAPL", "INVALID", "MSFT"])

    # Should return prices for valid tickers only
    assert "AAPL" in result
    assert "MSFT" in result
    assert "INVALID" not in result


def test_fetch_bulk_prices(data_service: DataService) -> None:
    """Test fetching prices for multiple tickers."""
    mock_df = pd.DataFrame(
        {
            ("AAPL", "Close"): [100.0, 101.0],
            ("MSFT", "Close"): [300.0, 301.0],
        },
        index=pd.date_range("2024-01-01", periods=2),
    )
    mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns)
    data_service._price_manager.fetch_bulk_prices.return_value = mock_df

    result = data_service.fetch_bulk_prices(
        ["AAPL", "MSFT"], date(2024, 1, 1), date(2024, 1, 2), use_cache=False
    )

    assert isinstance(result, pd.DataFrame)
    data_service._price_manager.fetch_bulk_prices.assert_called_once()


@patch("services.data_service.get_db_session")
def test_fetch_historical_prices_with_db_cached_ticker(
    mock_get_db_session, data_service: DataService
) -> None:
    """Test fetching prices for DB-cached ticker (benchmark)."""
    # SPY is in _db_cached_tickers
    mock_df = pd.DataFrame(
        {
            "Close": [400.0, 401.0],
            "Open": [399.0, 400.0],
            "High": [401.0, 402.0],
            "Low": [398.0, 399.0],
            "Volume": [50000000, 51000000],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )
    data_service._price_manager.fetch_historical_prices.return_value = mock_df

    # Mock database session
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    mock_get_db_session.return_value.__exit__.return_value = None
    
    # Mock query to return None (no data in DB)
    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value.first.return_value = None

    result = data_service.fetch_historical_prices(
        "SPY", date(2024, 1, 1), date(2024, 1, 2), use_cache=False, save_to_db=True
    )

    assert isinstance(result, pd.DataFrame)
    # Should try to get from DB first (SPY is in _db_cached_tickers)
    assert mock_session.query.called

