"""Unit tests for risk metrics."""

import numpy as np
import pandas as pd

import pytest
from core.analytics_engine.risk_metrics import (
    calculate_volatility,
    calculate_max_drawdown,
    calculate_current_drawdown,
    calculate_average_drawdown,
    calculate_drawdown_duration,
    calculate_recovery_time,
    calculate_ulcer_index,
    calculate_pain_index,
    calculate_var,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_semi_deviation,
    calculate_skewness,
    calculate_kurtosis,
    calculate_top_drawdowns,
)
from core.exceptions import InsufficientDataError


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

    # pandas kurtosis() returns excess kurtosis (kurtosis - 3)
    # For normal distribution, excess kurtosis ≈ 0
    assert np.isfinite(result)
    assert -1 < result < 1  # Close to 0 (excess kurtosis)


def test_calculate_average_drawdown() -> None:
    """Test average drawdown calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:30] = -0.10  # First drawdown
    returns.iloc[50:70] = -0.05  # Second drawdown
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    avg_dd = calculate_average_drawdown(returns)

    assert avg_dd < 0
    assert avg_dd <= 0  # Should be negative or zero


def test_calculate_drawdown_duration() -> None:
    """Test drawdown duration calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:30] = -0.10  # 20-day drawdown
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    duration = calculate_drawdown_duration(returns)

    assert "max_duration_days" in duration
    assert "avg_duration_days" in duration
    assert duration["max_duration_days"] > 0


def test_calculate_recovery_time() -> None:
    """Test recovery time calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:20] = -0.10  # Drawdown
    returns.iloc[20:30] = 0.02  # Recovery
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    recovery = calculate_recovery_time(returns)

    assert recovery is None or recovery >= 0


def test_calculate_ulcer_index() -> None:
    """Test Ulcer Index calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:30] = -0.10  # Drawdown period
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    ulcer = calculate_ulcer_index(returns)

    assert ulcer >= 0
    assert np.isfinite(ulcer)


def test_calculate_pain_index() -> None:
    """Test Pain Index calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:30] = -0.10  # Drawdown period
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    pain = calculate_pain_index(returns)

    assert pain >= 0
    assert np.isfinite(pain)


def test_calculate_downside_deviation() -> None:
    """Test downside deviation calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    downside_dev = calculate_downside_deviation(returns)

    assert downside_dev >= 0
    assert np.isfinite(downside_dev)


def test_calculate_semi_deviation() -> None:
    """Test semi-deviation calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    semi_dev = calculate_semi_deviation(returns)

    assert semi_dev >= 0
    assert np.isfinite(semi_dev)


def test_calculate_top_drawdowns() -> None:
    """Test top drawdowns calculation."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:20] = -0.15  # First drawdown
    returns.iloc[50:60] = -0.10  # Second drawdown
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    top_dds = calculate_top_drawdowns(returns, top_n=5)

    assert isinstance(top_dds, list)
    assert len(top_dds) > 0
    assert len(top_dds) <= 5
    # First drawdown should be the largest
    if len(top_dds) > 1:
        assert abs(top_dds[0]["drawdown"]) >= abs(top_dds[1]["drawdown"])


def test_calculate_var_parametric() -> None:
    """Test parametric VaR calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    var_95 = calculate_var(returns, 0.95, "parametric")

    assert var_95 < 0
    assert np.isfinite(var_95)


def test_calculate_var_cornish_fisher() -> None:
    """Test Cornish-Fisher VaR calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    var_95 = calculate_var(returns, 0.95, "cornish_fisher")

    assert var_95 is None or (var_95 < 0 and np.isfinite(var_95))


def test_risk_metrics_empty_returns() -> None:
    """Test that risk metrics raise error with empty returns."""
    empty_returns = pd.Series([], dtype=float)

    with pytest.raises(InsufficientDataError):
        calculate_volatility(empty_returns)

    with pytest.raises(InsufficientDataError):
        calculate_max_drawdown(empty_returns)

    with pytest.raises(InsufficientDataError):
        calculate_current_drawdown(empty_returns)


def test_calculate_volatility_single_value() -> None:
    """Test volatility calculation with single return value."""
    returns = pd.Series([0.01])

    result = calculate_volatility(returns)

    # Should handle single value (volatility would be 0 or NaN)
    assert "daily" in result
    assert "annual" in result


def test_calculate_max_drawdown_no_drawdown() -> None:
    """Test max drawdown with no drawdown (all positive)."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    returns.index = pd.date_range("2024-01-01", periods=252, freq="D")

    max_dd, peak_date, trough_date, duration = calculate_max_drawdown(returns)

    # Should return 0 or very small drawdown
    assert max_dd <= 0
    assert duration >= 0


def test_calculate_current_drawdown_no_drawdown() -> None:
    """Test current drawdown with no drawdown."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    returns.index = pd.date_range("2024-01-01", periods=252, freq="D")

    result = calculate_current_drawdown(returns)

    # Should be 0 or positive if at peak
    assert result <= 0


def test_calculate_var_confidence_levels() -> None:
    """Test VaR calculation with different confidence levels."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0, 0.02, 1000))

    var_90 = calculate_var(returns, 0.90, "historical")
    var_95 = calculate_var(returns, 0.95, "historical")
    var_99 = calculate_var(returns, 0.99, "historical")

    assert var_90 < 0
    assert var_95 < 0
    assert var_99 < 0
    # Higher confidence = more negative VaR
    assert var_99 <= var_95 <= var_90


def test_calculate_cvar_extreme_losses() -> None:
    """Test CVaR with extreme loss scenarios."""
    # Create returns with extreme losses
    returns = pd.Series([0.01] * 900 + [-0.20, -0.15, -0.10] * 33)  # Mix with extreme losses

    cvar_95 = calculate_cvar(returns, 0.95)

    assert cvar_95 < 0
    # CVaR should be more negative than VaR
    var_95 = calculate_var(returns, 0.95, "historical")
    assert cvar_95 <= var_95


def test_calculate_downside_deviation_all_positive() -> None:
    """Test downside deviation with all positive returns."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)

    downside_dev = calculate_downside_deviation(returns)

    # Should be 0 if all returns are positive (no negative returns)
    assert downside_dev == 0.0


def test_calculate_semi_deviation_all_positive() -> None:
    """Test semi-deviation with all positive returns."""
    # Semi-deviation measures deviation below mean, not below zero
    # Even if all positive, some will be below mean
    returns = pd.Series([0.01] * 252)  # Constant returns (all equal to mean)

    semi_dev = calculate_semi_deviation(returns)

    # Should be 0 if all returns equal the mean (no deviation below mean)
    assert semi_dev == 0.0


def test_calculate_top_drawdowns_single_drawdown() -> None:
    """Test top drawdowns with single drawdown."""
    returns = pd.Series([0.0] * 100)
    returns.iloc[10:30] = -0.10  # Single drawdown
    returns.index = pd.date_range("2024-01-01", periods=100, freq="D")

    top_dds = calculate_top_drawdowns(returns, top_n=5)

    assert isinstance(top_dds, list)
    assert len(top_dds) == 1  # Only one drawdown


def test_calculate_top_drawdowns_no_drawdowns() -> None:
    """Test top drawdowns with no drawdowns."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)
    returns.index = pd.date_range("2024-01-01", periods=252, freq="D")

    top_dds = calculate_top_drawdowns(returns, top_n=5)

    # Should return empty list if no drawdowns
    assert isinstance(top_dds, list)
    assert len(top_dds) == 0

