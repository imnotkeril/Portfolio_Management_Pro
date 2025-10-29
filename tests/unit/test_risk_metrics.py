"""Unit tests for risk metrics."""

import numpy as np
import pandas as pd

from core.analytics_engine.risk_metrics import (
    calculate_volatility,
    calculate_max_drawdown,
    calculate_current_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_skewness,
    calculate_kurtosis,
)


def test_calculate_volatility() -> None:
    """Test volatility calculation."""
    # Create returns with known volatility
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    result = calculate_volatility(returns)

    assert "daily" in result
    assert "annual" in result
    assert result["annual"] > 0.1  # Should be around 0.32
    assert result["annual"] < 0.5


def test_calculate_max_drawdown() -> None:
    """Test max drawdown calculation."""
    # Create returns that cause a drawdown
    returns = pd.Series([0.0] * 50)
    # Add a peak, then decline
    returns.iloc[10:30] = -0.15  # 15% decline
    returns.index = pd.date_range("2024-01-01", periods=50, freq="D")

    max_dd, peak_date, trough_date, duration = (
        calculate_max_drawdown(returns)
    )

    assert max_dd < 0
    assert abs(max_dd) > 0.10  # Should be around 15%
    assert duration > 0


def test_calculate_current_drawdown() -> None:
    """Test current drawdown calculation."""
    returns = pd.Series([0.0] * 50)
    returns.iloc[0:40] = 0.02  # Peak
    returns.iloc[40:50] = -0.05  # Decline
    returns.index = pd.date_range("2024-01-01", periods=50, freq="D")

    result = calculate_current_drawdown(returns)

    assert result < 0


def test_calculate_var_historical() -> None:
    """Test historical VaR calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    var_95 = calculate_var(returns, 0.95, "historical")

    # VaR should be negative (loss)
    assert var_95 < 0
    # For normal distribution, 95% VaR ≈ -1.645 * std
    assert abs(var_95) < 0.05  # Should be around -0.033


def test_calculate_cvar() -> None:
    """Test CVaR calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    cvar_95 = calculate_cvar(returns, 0.95)

    # CVaR should be more negative than VaR
    var_95 = calculate_var(returns, 0.95, "historical")
    assert cvar_95 < 0
    assert cvar_95 <= var_95  # CVaR is more extreme


def test_calculate_skewness() -> None:
    """Test skewness calculation."""
    # Create positively skewed returns
    returns = pd.Series([0.05] * 50 + [-0.01] * 50)

    result = calculate_skewness(returns)

    assert np.isfinite(result)


def test_calculate_kurtosis() -> None:
    """Test kurtosis calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    result = calculate_kurtosis(returns)

    # For normal distribution, kurtosis ≈ 3
    assert np.isfinite(result)
    assert 2 < result < 4  # Close to 3

