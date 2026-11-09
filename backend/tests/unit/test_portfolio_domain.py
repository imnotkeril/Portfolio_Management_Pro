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


def test_portfolio_update_position_nonexistent() -> None:
    """Test updating nonexistent position."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    with pytest.raises(ValidationError, match="not found"):
        portfolio.update_position(ticker="AAPL", shares=100.0)


def test_portfolio_update_position_invalid_shares() -> None:
    """Test updating position with invalid shares."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    with pytest.raises(ValidationError, match="Shares must be greater than 0"):
        portfolio.update_position(ticker="AAPL", shares=0)

    with pytest.raises(ValidationError, match="Shares must be greater than 0"):
        portfolio.update_position(ticker="AAPL", shares=-10)


def test_portfolio_update_position_invalid_weight() -> None:
    """Test updating position with invalid weight."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    with pytest.raises(ValidationError, match="Weight target must be between"):
        portfolio.update_position(ticker="AAPL", weight_target=-0.1)

    with pytest.raises(ValidationError, match="Weight target must be between"):
        portfolio.update_position(ticker="AAPL", weight_target=1.5)


def test_portfolio_update_position_invalid_purchase_price() -> None:
    """Test updating position with invalid purchase price."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    with pytest.raises(ValidationError, match="Purchase price must be greater than 0"):
        portfolio.update_position(ticker="AAPL", purchase_price=0)

    with pytest.raises(ValidationError, match="Purchase price must be greater than 0"):
        portfolio.update_position(ticker="AAPL", purchase_price=-10)


def test_portfolio_update_position_purchase_date() -> None:
    """Test updating position purchase date."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    purchase_date = date(2024, 1, 15)
    portfolio.update_position(ticker="AAPL", purchase_date=purchase_date)

    position = portfolio.get_position("AAPL")
    assert position is not None
    assert position.purchase_date == purchase_date


def test_portfolio_calculate_current_value_zero_prices() -> None:
    """Test calculating value with zero prices."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    prices = {"AAPL": 0.0}
    value = portfolio.calculate_current_value(prices)

    assert value == 0.0


def test_portfolio_calculate_current_weights_empty_portfolio() -> None:
    """Test calculating weights for empty portfolio."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    weights = portfolio.calculate_current_weights({})

    assert weights == {}


def test_portfolio_calculate_current_weights_zero_total_value() -> None:
    """Test calculating weights when total value is zero."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)
    portfolio.add_position(ticker="MSFT", shares=50.0)

    prices = {"AAPL": 0.0, "MSFT": 0.0}
    weights = portfolio.calculate_current_weights(prices)

    # Should return zero weights for all positions
    assert weights["AAPL"] == 0.0
    assert weights["MSFT"] == 0.0


def test_portfolio_validate_weights_with_tolerance() -> None:
    """Test weight validation with custom tolerance."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0, weight_target=0.5)
    portfolio.add_position(ticker="MSFT", shares=50.0, weight_target=0.5)

    # Sum = 1.0, should pass
    assert portfolio.validate_weights() is True

    # With very strict tolerance, might fail due to floating point
    assert portfolio.validate_weights(tolerance=0.0000001) is True


def test_portfolio_validate_weights_no_weights() -> None:
    """Test weight validation when positions have no target weights."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    portfolio.add_position(ticker="AAPL", shares=100.0)  # No weight_target
    portfolio.add_position(ticker="MSFT", shares=50.0)  # No weight_target

    # Sum of None weights = 0.0, should fail
    assert portfolio.validate_weights() is False


def test_position_equality() -> None:
    """Test position equality comparison."""
    pos1 = Position(ticker="AAPL", shares=100.0)
    pos2 = Position(ticker="AAPL", shares=200.0)  # Different shares
    pos3 = Position(ticker="MSFT", shares=100.0)  # Different ticker

    # Positions are equal if ticker is the same (regardless of shares)
    assert pos1 == pos2  # Same ticker
    assert pos1 != pos3  # Different ticker
    assert pos1 != "not a position"  # Different type


def test_position_repr() -> None:
    """Test position string representation."""
    position = Position(ticker="AAPL", shares=100.0, weight_target=0.3)

    repr_str = repr(position)

    assert "AAPL" in repr_str
    assert "100" in repr_str
    assert "0.3" in repr_str or "weight" in repr_str.lower()


def test_portfolio_repr() -> None:
    """Test portfolio string representation."""
    portfolio = Portfolio(name="Test Portfolio", starting_capital=100000.0)

    repr_str = repr(portfolio)

    assert "Test Portfolio" in repr_str or "Portfolio" in repr_str


def test_portfolio_get_position_nonexistent() -> None:
    """Test getting nonexistent position."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)

    position = portfolio.get_position("AAPL")

    assert position is None


def test_portfolio_get_position_case_insensitive() -> None:
    """Test getting position with case-insensitive ticker."""
    portfolio = Portfolio(name="Test", starting_capital=100000.0)
    portfolio.add_position(ticker="AAPL", shares=100.0)

    position1 = portfolio.get_position("AAPL")
    position2 = portfolio.get_position("aapl")
    position3 = portfolio.get_position("  aapl  ")

    assert position1 is not None
    assert position2 is not None
    assert position3 is not None
    assert position1 == position2 == position3
