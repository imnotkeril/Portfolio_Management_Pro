"""Unit tests for performance metrics."""

import pytest
import pandas as pd

from core.analytics_engine.performance import (
    calculate_total_return,
    calculate_cagr,
    calculate_annualized_return,
    calculate_period_returns,
    calculate_best_worst_periods,
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
    # 3 positive (0.01, 0.015, 0.03), 3 negative (-0.02, -0.01, -0.005)
    expected = 3 / 6  # 0.5

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


def test_calculate_period_returns() -> None:
    """Test period returns calculation."""
    from datetime import date
    import numpy as np
    
    # Create portfolio values over 2 years
    dates = pd.date_range("2023-01-01", periods=504, freq="D")  # 2 years
    np.random.seed(42)
    base_value = 100000.0
    returns = np.random.normal(0.001, 0.02, 504)
    values = base_value * (1 + pd.Series(returns)).cumprod()
    values.index = dates

    period_returns = calculate_period_returns(values)

    assert "ytd" in period_returns
    assert "mtd" in period_returns
    assert "qtd" in period_returns
    assert "1m" in period_returns
    assert "3m" in period_returns
    assert "6m" in period_returns
    assert "1y" in period_returns
    assert "3y" in period_returns
    assert "5y" in period_returns


def test_calculate_period_returns_empty() -> None:
    """Test period returns with empty values."""
    empty_values = pd.Series([], dtype=float)

    period_returns = calculate_period_returns(empty_values)

    # Should return dict with all None values
    assert all(v is None for v in period_returns.values())


def test_calculate_best_worst_periods() -> None:
    """Test best/worst periods calculation."""
    import numpy as np
    from datetime import date
    
    # Create returns with known best/worst months
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
    # Add a very good month
    returns.iloc[30:60] = 0.05
    # Add a very bad month
    returns.iloc[120:150] = -0.08

    best_worst = calculate_best_worst_periods(returns)

    assert "best_month" in best_worst
    assert "worst_month" in best_worst
    assert best_worst["best_month"] is not None
    assert best_worst["worst_month"] is not None
    assert best_worst["best_month"] > best_worst["worst_month"]


def test_calculate_best_worst_periods_empty() -> None:
    """Test best/worst periods with empty returns."""
    empty_returns = pd.Series([], dtype=float)

    result = calculate_best_worst_periods(empty_returns)

    assert result["best_month"] is None
    assert result["worst_month"] is None

