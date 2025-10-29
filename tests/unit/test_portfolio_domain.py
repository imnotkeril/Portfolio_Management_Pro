"""Unit tests for portfolio domain model."""

from datetime import date

import pytest

from core.data_manager.portfolio import Portfolio, Position
from core.exceptions import ValidationError


def test_portfolio_creation() -> None:
    """Test portfolio creation."""
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
        description="Test description",
    )

    assert portfolio.name == "Test Portfolio"
    assert portfolio.starting_capital == 100000.0
    assert portfolio.description == "Test description"
    assert portfolio.base_currency == "USD"
    assert len(portfolio.get_all_positions()) == 0


def test_portfolio_creation_invalid_name() -> None:
    """Test portfolio creation with invalid name."""
    with pytest.raises(ValidationError):
        Portfolio(name="", starting_capital=100000.0)

    with pytest.raises(ValidationError):
        Portfolio(name="x" * 101, starting_capital=100000.0)


def test_portfolio_creation_invalid_capital() -> None:
    """Test portfolio creation with invalid capital."""
    with pytest.raises(ValidationError):
        Portfolio(name="Test", starting_capital=0)

    with pytest.raises(ValidationError):
        Portfolio(name="Test", starting_capital=-1000)


def test_position_creation() -> None:
    """Test position creation."""
    position = Position(
        ticker="AAPL",
        shares=100.0,
        weight_target=0.3,
        purchase_price=150.0,
        purchase_date=date(2024, 1, 1),
    )

    assert position.ticker == "AAPL"
    assert position.shares == 100.0
    assert position.weight_target == 0.3
    assert position.purchase_price == 150.0


def test_position_creation_invalid_shares() -> None:
    """Test position creation with invalid shares."""
    with pytest.raises(ValidationError):
        Position(ticker="AAPL", shares=0)

    with pytest.raises(ValidationError):
        Position(ticker="AAPL", shares=-10)


def test_position_creation_invalid_weight() -> None:
    """Test position creation with invalid weight."""
    with pytest.raises(ValidationError):
        Position(ticker="AAPL", shares=100, weight_target=-0.1)

    with pytest.raises(ValidationError):
        Position(ticker="AAPL", shares=100, weight_target=1.5)


def test_portfolio_add_position() -> None:
    """Test adding position to portfolio."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0, weight_target=0.3)

    assert len(portfolio.get_all_positions()) == 1
    position = portfolio.get_position("AAPL")
    assert position is not None
    assert position.shares == 100.0


def test_portfolio_add_duplicate_position() -> None:
    """Test adding duplicate position."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)

    with pytest.raises(ValidationError, match="already exists"):
        portfolio.add_position(ticker="AAPL", shares=50.0)


def test_portfolio_remove_position() -> None:
    """Test removing position from portfolio."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)
    assert len(portfolio.get_all_positions()) == 1

    portfolio.remove_position("AAPL")
    assert len(portfolio.get_all_positions()) == 0


def test_portfolio_remove_nonexistent_position() -> None:
    """Test removing nonexistent position."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    with pytest.raises(ValidationError, match="not found"):
        portfolio.remove_position("AAPL")


def test_portfolio_update_position() -> None:
    """Test updating position."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.update_position(ticker="AAPL", shares=150.0)

    position = portfolio.get_position("AAPL")
    assert position is not None
    assert position.shares == 150.0


def test_portfolio_calculate_current_value() -> None:
    """Test calculating current portfolio value."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.add_position(ticker="MSFT", shares=50.0)

    prices = {"AAPL": 150.0, "MSFT": 300.0}
    value = portfolio.calculate_current_value(prices)

    # 100 * 150 + 50 * 300 = 15000 + 15000 = 30000
    assert value == 30000.0


def test_portfolio_calculate_current_weights() -> None:
    """Test calculating current weights."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.add_position(ticker="MSFT", shares=50.0)

    prices = {"AAPL": 150.0, "MSFT": 300.0}
    weights = portfolio.calculate_current_weights(prices)

    # Total value: 30000
    # AAPL: 15000 / 30000 = 0.5
    # MSFT: 15000 / 30000 = 0.5
    assert weights["AAPL"] == 0.5
    assert weights["MSFT"] == 0.5
    assert sum(weights.values()) == 1.0


def test_portfolio_calculate_weights_missing_price() -> None:
    """Test calculating weights with missing price."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)

    prices = {}  # Missing price

    with pytest.raises(ValidationError, match="Price not available"):
        portfolio.calculate_current_weights(prices)


def test_portfolio_validate_weights() -> None:
    """Test weight validation."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0, weight_target=0.3)
    portfolio.add_position(ticker="MSFT", shares=50.0, weight_target=0.7)

    assert portfolio.validate_weights() is True


def test_portfolio_validate_weights_invalid_sum() -> None:
    """Test weight validation with invalid sum."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0, weight_target=0.3)
    portfolio.add_position(ticker="MSFT", shares=50.0, weight_target=0.5)

    # Sum = 0.8, not 1.0
    assert portfolio.validate_weights() is False

