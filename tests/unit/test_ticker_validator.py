"""Unit tests for ticker validator."""

import pytest
from unittest.mock import MagicMock, patch

from core.data_manager.ticker_validator import TickerValidator
from core.exceptions import TickerNotFoundError, ValidationError


def test_validate_ticker_format_invalid() -> None:
    """Test ticker format validation."""
    validator = TickerValidator()

    # Test invalid formats - should raise ValidationError before API call
    with pytest.raises(ValidationError):
        validator.validate_ticker("")

    # "AAPL-" ends with hyphen, but pattern allows hyphens
    # However, it should still be validated by format check
    # The pattern allows hyphens, so this might pass format check
    # but fail API validation. Let's test with clearly invalid format
    with pytest.raises(ValidationError):
        validator.validate_ticker("TOO-LONG-TICKER-SYMBOL")  # Too long

    # Test with invalid characters
    with pytest.raises(ValidationError):
        validator.validate_ticker("AAPL@")  # Invalid character


@patch("core.data_manager.ticker_validator.yf.Ticker")
def test_validate_ticker_normalization(mock_ticker_class) -> None:
    """Test ticker normalization (uppercase, strip)."""
    # Create validator without cache to avoid cache interference
    from core.data_manager.cache import Cache
    cache = Cache()
    validator = TickerValidator(cache=cache)

    # Mock ticker info
    mock_ticker = MagicMock()
    mock_ticker.info = {"symbol": "AAPL", "longName": "Apple Inc."}
    mock_ticker_class.return_value = mock_ticker

    # Clear cache to ensure fresh call
    cache.clear()

    # Should normalize to uppercase and strip whitespace
    result = validator.validate_ticker("  aapl  ")
    assert result is True

    # Verify it was called with normalized ticker
    mock_ticker_class.assert_called_with("AAPL")


@patch("core.data_manager.ticker_validator.yf.Ticker")
def test_validate_ticker_valid(mock_ticker_class) -> None:
    """Test validation of valid ticker."""
    validator = TickerValidator()

    # Mock ticker info
    mock_ticker = MagicMock()
    mock_ticker.info = {"symbol": "AAPL", "longName": "Apple Inc."}
    mock_ticker_class.return_value = mock_ticker

    result = validator.validate_ticker("AAPL")
    assert result is True


@patch("core.data_manager.ticker_validator.yf.Ticker")
def test_validate_ticker_invalid(mock_ticker_class) -> None:
    """Test validation of invalid ticker."""
    validator = TickerValidator()

    # Mock ticker with no info
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker_class.return_value = mock_ticker

    result = validator.validate_ticker("INVALID")
    assert result is False


def test_validate_tickers_multiple() -> None:
    """Test validation of multiple tickers."""
    validator = TickerValidator()

    with patch.object(validator, "validate_ticker") as mock_validate:
        mock_validate.side_effect = [True, False, True]

        results = validator.validate_tickers(["AAPL", "INVALID", "MSFT"])

        assert results == {"AAPL": True, "INVALID": False, "MSFT": True}
        assert mock_validate.call_count == 3


@patch("core.data_manager.ticker_validator.yf.Ticker")
def test_get_ticker_info(mock_ticker_class) -> None:
    """Test getting ticker information."""
    validator = TickerValidator()

    # Mock ticker info
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000,
        "currency": "USD",
    }
    mock_ticker_class.return_value = mock_ticker

    with patch.object(validator, "validate_ticker", return_value=True):
        info = validator.get_ticker_info("AAPL")

        assert info.ticker == "AAPL"
        assert info.name == "Apple Inc."
        assert info.sector == "Technology"
        assert info.currency == "USD"


@patch("core.data_manager.ticker_validator.yf.Ticker")
def test_get_ticker_info_not_found(mock_ticker_class) -> None:
    """Test getting info for non-existent ticker."""
    validator = TickerValidator()

    with patch.object(validator, "validate_ticker", return_value=False):
        with pytest.raises(TickerNotFoundError):
            validator.get_ticker_info("INVALID")

