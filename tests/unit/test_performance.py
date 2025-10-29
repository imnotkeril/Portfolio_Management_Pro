"""Unit tests for performance metrics."""

import pytest
import pandas as pd

from core.analytics_engine.performance import (
    calculate_total_return,
    calculate_cagr,
    calculate_annualized_return,
    calculate_win_rate,
    calculate_payoff_ratio,
    calculate_profit_factor,
    calculate_expectancy,
)
from core.exceptions import InsufficientDataError


def test_calculate_total_return() -> None:
    """Test total return calculation."""
    start_value = 100000.0
    end_value = 125000.0
    expected = 0.25  # 25%

    result = calculate_total_return(start_value, end_value)
    assert result == pytest.approx(expected, rel=1e-6)


def test_calculate_total_return_negative() -> None:
    """Test total return with loss."""
    start_value = 100000.0
    end_value = 75000.0
    expected = -0.25  # -25%

    result = calculate_total_return(start_value, end_value)
    assert result == pytest.approx(expected, rel=1e-6)


def test_calculate_total_return_invalid_start() -> None:
    """Test total return with invalid start value."""
    with pytest.raises(ValueError, match="must be greater than 0"):
        calculate_total_return(0, 100000.0)


def test_calculate_cagr() -> None:
    """Test CAGR calculation."""
    start_value = 100000.0
    end_value = 200000.0
    years = 5.0
    # (200000/100000)^(1/5) - 1 = 2^0.2 - 1 ≈ 0.1487

    result = calculate_cagr(start_value, end_value, years)
    assert result == pytest.approx(0.1487, rel=0.01)


def test_calculate_cagr_loss() -> None:
    """Test CAGR with total loss."""
    start_value = 100000.0
    end_value = 0.0
    years = 2.0

    result = calculate_cagr(start_value, end_value, years)
    assert result == -1.0


def test_calculate_annualized_return() -> None:
    """Test annualized return calculation."""
    # Daily returns of 0.1% per day (approx 28% annualized)
    daily_returns = pd.Series([0.001] * 252)
    result = calculate_annualized_return(daily_returns)

    assert result > 0.20  # Should be around 28%
    assert result < 0.35


def test_calculate_annualized_return_empty() -> None:
    """Test annualized return with empty series."""
    returns = pd.Series([], dtype=float)
    with pytest.raises(InsufficientDataError):
        calculate_annualized_return(returns)


def test_calculate_win_rate() -> None:
    """Test win rate calculation."""
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03, -0.005])
    # 4 positive, 2 negative
    expected = 4 / 6  # 0.6667

    result = calculate_win_rate(returns)
    assert result == pytest.approx(expected, rel=1e-6)


def test_calculate_payoff_ratio() -> None:
    """Test payoff ratio calculation."""
    returns = pd.Series([0.05, -0.02, 0.04, -0.01, 0.03])
    # Avg win: (0.05 + 0.04 + 0.03) / 3 = 0.04
    # Avg loss: (0.02 + 0.01) / 2 = 0.015
    # Ratio: 0.04 / 0.015 = 2.67

    result = calculate_payoff_ratio(returns)
    assert result is not None
    assert result > 2.0
    assert result < 3.0


def test_calculate_profit_factor() -> None:
    """Test profit factor calculation."""
    returns = pd.Series([0.10, -0.05, 0.08, -0.03, 0.06])
    # Gross profits: 0.10 + 0.08 + 0.06 = 0.24
    # Gross losses: 0.05 + 0.03 = 0.08
    # Factor: 0.24 / 0.08 = 3.0

    result = calculate_profit_factor(returns)
    assert result is not None
    assert result == pytest.approx(3.0, rel=1e-6)


def test_calculate_expectancy() -> None:
    """Test expectancy calculation."""
    returns = pd.Series([0.02, -0.01, 0.015, -0.005])
    # Win rate: 2/4 = 0.5
    # Loss rate: 0.5
    # Avg win: (0.02 + 0.015) / 2 = 0.0175
    # Avg loss: (0.01 + 0.005) / 2 = 0.0075
    # Expectancy: 0.5 * 0.0175 - 0.5 * 0.0075 = 0.005

    result = calculate_expectancy(returns)
    assert result > 0
    assert result < 0.01

