"""Unit tests for risk-adjusted ratios."""

import pytest
import numpy as np
import pandas as pd

from core.analytics_engine.ratios import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_sterling_ratio,
    calculate_burke_ratio,
    calculate_treynor_ratio,
    calculate_information_ratio,
    calculate_modigliani_m2,
    calculate_omega_ratio,
    calculate_kappa_3,
    calculate_gain_pain_ratio,
    calculate_martin_ratio,
    calculate_tail_ratio,
    calculate_common_sense_ratio,
    calculate_rachev_ratio,
)
from core.exceptions import InsufficientDataError


def test_calculate_sharpe_ratio() -> None:
    """Test Sharpe ratio calculation."""
    # Positive returns with some volatility
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))  # Mean 0.1% daily, std 1%

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04)

    assert sharpe is not None
    assert np.isfinite(sharpe)


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


def test_calculate_sterling_ratio() -> None:
    """Test Sterling ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    sterling = calculate_sterling_ratio(returns)

    assert sterling is None or np.isfinite(sterling)


def test_calculate_burke_ratio() -> None:
    """Test Burke ratio calculation."""
    # Create returns with drawdowns
    returns = pd.Series([0.001] * 200)
    returns.iloc[50:100] = -0.05  # Drawdown period

    burke = calculate_burke_ratio(returns)

    assert burke is None or np.isfinite(burke)


def test_calculate_treynor_ratio() -> None:
    """Test Treynor ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    beta = 1.2

    treynor = calculate_treynor_ratio(returns, beta=beta, risk_free_rate=0.04)

    assert treynor is not None
    assert np.isfinite(treynor)


def test_calculate_treynor_ratio_zero_beta() -> None:
    """Test Treynor ratio with zero beta."""
    returns = pd.Series([0.01] * 100)
    beta = 0.0

    treynor = calculate_treynor_ratio(returns, beta=beta)

    assert treynor is None


def test_calculate_information_ratio() -> None:
    """Test Information ratio calculation."""
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))

    info_ratio = calculate_information_ratio(portfolio, benchmark)

    assert info_ratio is None or np.isfinite(info_ratio)


def test_calculate_modigliani_m2() -> None:
    """Test Modigliani M2 calculation."""
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))

    m2 = calculate_modigliani_m2(portfolio, benchmark, risk_free_rate=0.04)

    assert m2 is None or np.isfinite(m2)


def test_calculate_kappa_3() -> None:
    """Test Kappa 3 ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    kappa3 = calculate_kappa_3(returns)

    assert kappa3 is None or np.isfinite(kappa3)


def test_calculate_gain_pain_ratio() -> None:
    """Test Gain-Pain ratio calculation."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015, -0.01])

    gain_pain = calculate_gain_pain_ratio(returns)

    assert gain_pain is not None
    assert gain_pain > 0


def test_calculate_gain_pain_ratio_no_losses() -> None:
    """Test Gain-Pain ratio with no losses."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])  # All positive

    gain_pain = calculate_gain_pain_ratio(returns)

    assert gain_pain is None


def test_calculate_martin_ratio() -> None:
    """Test Martin ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    martin = calculate_martin_ratio(returns)

    assert martin is None or np.isfinite(martin)


def test_calculate_tail_ratio() -> None:
    """Test Tail ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    tail = calculate_tail_ratio(returns)

    assert tail is None or np.isfinite(tail)


def test_calculate_common_sense_ratio() -> None:
    """Test Common Sense ratio calculation."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    common_sense = calculate_common_sense_ratio(returns)

    assert common_sense is None or np.isfinite(common_sense)


def test_calculate_rachev_ratio() -> None:
    """Test Rachev ratio calculation."""
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.015, 252))

    rachev = calculate_rachev_ratio(portfolio, benchmark, alpha=0.05)

    assert rachev is None or np.isfinite(rachev)


def test_calculate_ratios_empty_returns() -> None:
    """Test that ratios raise error with empty returns."""
    empty_returns = pd.Series([], dtype=float)

    with pytest.raises(InsufficientDataError):
        calculate_sharpe_ratio(empty_returns)

    with pytest.raises(InsufficientDataError):
        calculate_sortino_ratio(empty_returns)

    with pytest.raises(InsufficientDataError):
        calculate_calmar_ratio(empty_returns)


def test_calculate_sortino_ratio_zero_downside() -> None:
    """Test Sortino ratio with zero downside deviation."""
    # All positive returns - no downside
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])

    sortino = calculate_sortino_ratio(returns)

    # May return None if downside deviation is zero
    assert sortino is None or np.isfinite(sortino)


def test_calculate_calmar_ratio_no_drawdown() -> None:
    """Test Calmar ratio with no drawdown."""
    # All positive returns - no drawdown
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)  # ~252 days

    calmar = calculate_calmar_ratio(returns)

    # May return None if max drawdown is zero
    assert calmar is None or np.isfinite(calmar)


def test_calculate_treynor_ratio_negative_beta() -> None:
    """Test Treynor ratio with negative beta."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    beta = -0.5  # Negative beta

    treynor = calculate_treynor_ratio(returns, beta=beta)

    # Should handle negative beta
    assert treynor is None or np.isfinite(treynor)


def test_calculate_information_ratio_identical_returns() -> None:
    """Test Information ratio with identical portfolio and benchmark."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = returns
    benchmark = returns.copy()

    info_ratio = calculate_information_ratio(portfolio, benchmark)

    # Should be close to 0 or None (no tracking error)
    assert info_ratio is None or abs(info_ratio) < 0.1


def test_calculate_modigliani_m2_identical_volatility() -> None:
    """Test M2 with identical portfolio and benchmark volatility."""
    np.random.seed(42)
    portfolio = pd.Series(np.random.normal(0.001, 0.02, 252))
    benchmark = pd.Series(np.random.normal(0.0008, 0.02, 252))  # Same volatility

    m2 = calculate_modigliani_m2(portfolio, benchmark)

    assert m2 is None or np.isfinite(m2)


def test_calculate_omega_ratio_threshold() -> None:
    """Test Omega ratio with different thresholds."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))

    omega_zero = calculate_omega_ratio(returns, threshold=0.0)
    omega_positive = calculate_omega_ratio(returns, threshold=0.01)

    assert omega_zero is not None
    assert omega_positive is None or np.isfinite(omega_positive)


def test_calculate_kappa_3_no_drawdown() -> None:
    """Test Kappa 3 with no drawdown."""
    # All positive returns
    returns = pd.Series([0.01, 0.02, 0.015, 0.03] * 63)

    kappa3 = calculate_kappa_3(returns)

    assert kappa3 is None or np.isfinite(kappa3)


def test_calculate_tail_ratio_extreme_values() -> None:
    """Test Tail ratio with extreme return values."""
    # Mix of extreme positive and negative returns
    returns = pd.Series([0.10, -0.08, 0.05, -0.06, 0.03, -0.04] * 42)

    tail = calculate_tail_ratio(returns)

    assert tail is None or np.isfinite(tail)


def test_calculate_rachev_ratio_identical_returns() -> None:
    """Test Rachev ratio with identical portfolio and benchmark."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    portfolio = returns
    benchmark = returns.copy()

    rachev = calculate_rachev_ratio(portfolio, benchmark, alpha=0.05)

    assert rachev is None or np.isfinite(rachev)

