"""Unit tests for risk-adjusted ratios."""

import numpy as np
import pandas as pd

from core.analytics_engine.ratios import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio,
)


def test_calculate_sharpe_ratio() -> None:
    """Test Sharpe ratio calculation."""
    # Positive returns with low volatility
    returns = pd.Series([0.001] * 252)  # 0.1% daily ≈ 28% annual

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04)

    assert sharpe is not None
    assert sharpe > 0  # Should be positive for positive returns


def test_calculate_sharpe_ratio_zero_volatility() -> None:
    """Test Sharpe ratio with zero volatility."""
    returns = pd.Series([0.01] * 100)  # Constant returns

    sharpe = calculate_sharpe_ratio(returns)

    assert sharpe is None  # Cannot calculate with zero volatility


def test_calculate_sortino_ratio() -> None:
    """Test Sortino ratio calculation."""
    # Returns with positive trend but some downside
    np.random.seed(42)
    base_returns = np.random.normal(0.001, 0.02, 252)
    returns = pd.Series(base_returns)

    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.04)

    assert sortino is not None
    assert sortino > 0


def test_calculate_calmar_ratio() -> None:
    """Test Calmar ratio calculation."""
    # Create returns with drawdown
    returns = pd.Series([0.001] * 200)  # Small positive returns
    returns.iloc[50:100] = -0.10  # 10% drawdown
    returns.index = pd.date_range("2024-01-01", periods=200, freq="D")

    calmar = calculate_calmar_ratio(returns)

    assert calmar is not None


def test_calculate_omega_ratio() -> None:
    """Test Omega ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    omega = calculate_omega_ratio(returns, threshold=0.0)

    assert omega is not None
    assert omega > 0

