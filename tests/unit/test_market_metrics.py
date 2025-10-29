"""Unit tests for market metrics."""

import numpy as np
import pandas as pd

from core.analytics_engine.market_metrics import (
    calculate_beta,
    calculate_alpha,
    calculate_correlation,
    calculate_up_capture,
    calculate_down_capture,
)


def test_calculate_beta() -> None:
    """Test beta calculation."""
    np.random.seed(42)
    # Create correlated returns
    market_return = np.random.normal(0.001, 0.02, 252)
    portfolio_return = market_return * 1.2 + np.random.normal(
        0, 0.005, 252
    )  # Beta ≈ 1.2

    portfolio_returns = pd.Series(portfolio_return)
    benchmark_returns = pd.Series(market_return)

    beta = calculate_beta(portfolio_returns, benchmark_returns)

    assert beta is not None
    assert beta > 1.0  # Should be around 1.2
    assert beta < 1.5


def test_calculate_alpha() -> None:
    """Test alpha calculation."""
    np.random.seed(42)
    # Portfolio outperforms market
    market_return = np.random.normal(0.001, 0.02, 252)
    portfolio_return = market_return + 0.002  # +0.2% daily alpha

    portfolio_returns = pd.Series(portfolio_return)
    benchmark_returns = pd.Series(market_return)

    alpha = calculate_alpha(
        portfolio_returns, benchmark_returns, risk_free_rate=0.04
    )

    assert alpha is not None
    assert alpha > 0  # Positive alpha (outperformance)


def test_calculate_correlation() -> None:
    """Test correlation calculation."""
    np.random.seed(42)
    market_return = np.random.normal(0.0, 0.02, 100)
    portfolio_return = market_return * 0.8 + np.random.normal(0, 0.01, 100)

    portfolio_returns = pd.Series(portfolio_return)
    benchmark_returns = pd.Series(market_return)

    correlation = calculate_correlation(
        portfolio_returns, benchmark_returns
    )

    assert correlation is not None
    assert -1.0 <= correlation <= 1.0
    assert correlation > 0.5  # Should be correlated


def test_calculate_up_capture() -> None:
    """Test up capture calculation."""
    # Benchmark up days
    benchmark = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03])
    # Portfolio captures 150% of up moves
    portfolio = pd.Series([0.03, -0.01, 0.0225, -0.005, 0.045])

    up_capture = calculate_up_capture(portfolio, benchmark)

    assert up_capture is not None
    assert up_capture > 1.0  # Should be around 1.5
    assert up_capture < 2.0


def test_calculate_down_capture() -> None:
    """Test down capture calculation."""
    # Benchmark down days
    benchmark = pd.Series([0.02, -0.02, 0.015, -0.01, 0.03])
    # Portfolio captures 50% of down moves (defensive)
    portfolio = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03])

    down_capture = calculate_down_capture(portfolio, benchmark)

    assert down_capture is not None
    # Down capture should be positive (less negative)
    assert down_capture > -1.0
    assert down_capture < 0  # Still negative but less than benchmark

