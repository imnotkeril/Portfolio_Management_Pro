"""Unit tests for market metrics."""

import numpy as np
import pandas as pd

import pytest
from core.analytics_engine.market_metrics import (
    calculate_beta,
    calculate_alpha,
    calculate_r_squared,
    calculate_correlation,
    calculate_tracking_error,
    calculate_active_return,
    calculate_up_capture,
    calculate_down_capture,
    calculate_capture_ratio,
    calculate_jensens_alpha,
    calculate_active_share,
    calculate_batting_average,
    calculate_benchmark_relative_return,
    calculate_rolling_beta,
    calculate_market_timing_ratio,
)
from core.exceptions import InsufficientDataError


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
    # Down capture 0.5 means portfolio loses only 50% of benchmark losses (good)
    # Formula: avg(portfolio_down) / avg(benchmark_down) = (-0.0075) / (-0.015) = 0.5
    assert down_capture > 0  # Positive means less loss than benchmark
    assert down_capture < 1.0  # Less than 1.0 means defensive


def test_calculate_r_squared() -> None:
    """Test R-squared calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark * 1.2 + np.random.normal(0, 0.005, 252)

    r_squared = calculate_r_squared(portfolio, benchmark)

    assert r_squared is not None
    assert 0 <= r_squared <= 1.0
    assert r_squared > 0.5  # Should be well correlated


def test_calculate_tracking_error() -> None:
    """Test tracking error calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark + np.random.normal(0, 0.005, 252)  # Small tracking error

    tracking_error = calculate_tracking_error(portfolio, benchmark)

    assert tracking_error is not None
    assert tracking_error > 0


def test_calculate_active_return() -> None:
    """Test active return calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark + 0.002  # +0.2% daily active return

    active_return = calculate_active_return(portfolio, benchmark)

    assert active_return is not None
    assert active_return > 0


def test_calculate_capture_ratio() -> None:
    """Test capture ratio calculation."""
    benchmark = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03])
    portfolio = pd.Series([0.03, -0.005, 0.0225, -0.0025, 0.045])  # 1.5x capture

    capture_ratio = calculate_capture_ratio(portfolio, benchmark)

    assert capture_ratio is not None
    assert capture_ratio > 1.0  # Should be around 1.5


def test_calculate_jensens_alpha() -> None:
    """Test Jensen's alpha calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark + 0.002  # Positive alpha

    jensens_alpha = calculate_jensens_alpha(
        portfolio, benchmark, risk_free_rate=0.04
    )

    assert jensens_alpha is not None
    assert jensens_alpha > 0


def test_calculate_active_share() -> None:
    """Test active share calculation."""
    # Active share requires weight dictionaries, not returns
    portfolio_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}
    benchmark_weights = {"AAPL": 0.3, "MSFT": 0.4, "GOOGL": 0.2, "AMZN": 0.1}

    active_share = calculate_active_share(portfolio_weights, benchmark_weights)

    assert active_share is not None
    assert 0 <= active_share <= 1.0


def test_calculate_batting_average() -> None:
    """Test batting average calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark + 0.001  # Outperforms in most periods

    batting_avg = calculate_batting_average(portfolio, benchmark)

    assert batting_avg is not None
    assert 0 <= batting_avg <= 1.0


def test_calculate_benchmark_relative_return() -> None:
    """Test benchmark relative return calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark * 1.1  # 10% better than benchmark

    relative_return = calculate_benchmark_relative_return(
        portfolio, benchmark
    )

    assert relative_return is not None


def test_calculate_rolling_beta() -> None:
    """Test rolling beta calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark * 1.2 + np.random.normal(0, 0.005, 252)
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    benchmark.index = dates
    portfolio.index = dates

    rolling_beta = calculate_rolling_beta(
        portfolio, benchmark, window=60
    )

    # Rolling beta returns average beta as float
    assert rolling_beta is not None
    assert isinstance(rolling_beta, float)
    assert np.isfinite(rolling_beta)


def test_calculate_market_timing_ratio() -> None:
    """Test market timing ratio calculation."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))

    timing_ratio = calculate_market_timing_ratio(portfolio, benchmark)

    assert timing_ratio is None or np.isfinite(timing_ratio)


def test_market_metrics_empty_returns() -> None:
    """Test that market metrics raise error with empty returns."""
    empty_returns = pd.Series([], dtype=float)
    benchmark = pd.Series([0.01] * 100)

    with pytest.raises(InsufficientDataError):
        calculate_beta(empty_returns, benchmark)

    with pytest.raises(InsufficientDataError):
        calculate_alpha(empty_returns, benchmark)

    with pytest.raises(InsufficientDataError):
        calculate_correlation(empty_returns, benchmark)


def test_calculate_beta_zero_benchmark_variance() -> None:
    """Test beta calculation with zero benchmark variance."""
    portfolio = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    benchmark = pd.Series([0.01] * 252)  # Constant benchmark

    beta = calculate_beta(portfolio, benchmark)

    # Should return None if benchmark has zero variance
    assert beta is None


def test_calculate_alpha_negative_alpha() -> None:
    """Test alpha calculation with underperforming portfolio."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark - 0.002  # Underperforms by 0.2% daily

    alpha = calculate_alpha(portfolio, benchmark, risk_free_rate=0.04)

    assert alpha is not None
    assert alpha < 0  # Negative alpha (underperformance)


def test_calculate_r_squared_perfect_correlation() -> None:
    """Test R-squared with perfectly correlated returns."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark * 1.2  # Perfectly correlated, 1.2x leverage

    r_squared = calculate_r_squared(portfolio, benchmark)

    assert r_squared is not None
    assert r_squared > 0.9  # Should be very high (close to 1.0)


def test_calculate_tracking_error_identical() -> None:
    """Test tracking error with identical returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = returns
    benchmark = returns.copy()

    tracking_error = calculate_tracking_error(portfolio, benchmark)

    # Should be close to zero
    assert tracking_error is not None
    assert tracking_error < 0.001


def test_calculate_active_return_negative() -> None:
    """Test active return with underperforming portfolio."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = benchmark - 0.001  # Underperforms

    active_return = calculate_active_return(portfolio, benchmark)

    assert active_return is not None
    assert active_return < 0


def test_calculate_batting_average_all_positive() -> None:
    """Test batting average when portfolio always outperforms."""
    benchmark = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    portfolio = benchmark + 0.005  # Always outperforms

    batting_avg = calculate_batting_average(portfolio, benchmark)

    assert batting_avg is not None
    assert batting_avg == 1.0  # 100% batting average


def test_calculate_batting_average_all_negative() -> None:
    """Test batting average when portfolio always underperforms."""
    benchmark = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    portfolio = benchmark - 0.005  # Always underperforms

    batting_avg = calculate_batting_average(portfolio, benchmark)

    assert batting_avg is not None
    assert batting_avg == 0.0  # 0% batting average


def test_calculate_rolling_beta_insufficient_data() -> None:
    """Test rolling beta with insufficient data."""
    benchmark = pd.Series([0.01] * 10)  # Only 10 days
    portfolio = pd.Series([0.02] * 10)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    benchmark.index = dates
    portfolio.index = dates

    rolling_beta = calculate_rolling_beta(portfolio, benchmark, window=60)

    # Should return None if window > data length
    assert rolling_beta is None


def test_calculate_market_timing_ratio_no_up_down() -> None:
    """Test market timing ratio with no clear up/down periods."""
    np.random.seed(42)
    benchmark = pd.Series(np.random.normal(0.0, 0.01, 252))  # Small moves
    portfolio = pd.Series(np.random.normal(0.0, 0.01, 252))

    timing_ratio = calculate_market_timing_ratio(portfolio, benchmark)

    assert timing_ratio is None or np.isfinite(timing_ratio)

