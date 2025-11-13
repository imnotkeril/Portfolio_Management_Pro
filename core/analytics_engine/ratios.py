"""Risk-adjusted ratios calculation (15 ratios)."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.analytics_engine.performance import calculate_annualized_return
from core.analytics_engine.risk_metrics import (
    calculate_cvar,
    calculate_max_drawdown,
    calculate_ulcer_index,
    calculate_downside_deviation,
)
from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Sharpe ratio.

    Formula: (Return - Risk Free Rate) / Volatility

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods per year

    Returns:
        Sharpe ratio or None if volatility is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(
            returns, periods_per_year
        )
        volatility = float(returns.std() * np.sqrt(periods_per_year))

        # Check for zero or near-zero volatility (with tolerance for floating point errors)
        if volatility < 1e-10:
            return None

        excess_return = annual_return - risk_free_rate
        sharpe = excess_return / volatility

        if not np.isfinite(sharpe):
            return None

        return float(sharpe)
    except Exception as e:
        logger.warning(f"Error calculating Sharpe ratio: {e}")
        return None


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Sortino ratio.

    Formula: (Return - Risk Free Rate) / Downside Deviation

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Sortino ratio or None if downside deviation is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(
            returns, periods_per_year
        )
        downside_dev = calculate_downside_deviation(
            returns, periods_per_year
        )

        if downside_dev == 0:
            return None

        excess_return = annual_return - risk_free_rate
        sortino = excess_return / downside_dev

        if not np.isfinite(sortino):
            return None

        return float(sortino)
    except Exception as e:
        logger.warning(f"Error calculating Sortino ratio: {e}")
        return None


def calculate_calmar_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Calmar ratio.

    Formula: Annualized Return / Max Drawdown (absolute)

    Args:
        returns: Series of returns

    Returns:
        Calmar ratio or None if max drawdown is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(returns)
        max_dd, _, _, _ = calculate_max_drawdown(returns)

        if abs(max_dd) == 0:
            return None

        calmar = annual_return / abs(max_dd)

        if not np.isfinite(calmar):
            return None

        return float(calmar)
    except Exception as e:
        logger.warning(f"Error calculating Calmar ratio: {e}")
        return None


def calculate_sterling_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Sterling ratio.

    Formula: Return / Average Drawdown (absolute)

    Args:
        returns: Series of returns

    Returns:
        Sterling ratio or None if average drawdown is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        from core.analytics_engine.risk_metrics import (
            calculate_average_drawdown,
        )

        annual_return = calculate_annualized_return(returns)
        avg_dd = calculate_average_drawdown(returns)

        if abs(avg_dd) == 0:
            return None

        sterling = annual_return / abs(avg_dd)

        if not np.isfinite(sterling):
            return None

        return float(sterling)
    except Exception as e:
        logger.warning(f"Error calculating Sterling ratio: {e}")
        return None


def calculate_burke_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Burke ratio.

    Formula: Return / sqrt(Sum of Squared Drawdowns)

    Args:
        returns: Series of returns

    Returns:
        Burke ratio or None if no drawdowns

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(returns)

        # Calculate squared drawdowns
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        negative_dd = drawdowns[drawdowns < 0]

        if negative_dd.empty:
            return None

        sum_squared_dd = float((negative_dd**2).sum())
        if sum_squared_dd == 0:
            return None

        burke = annual_return / np.sqrt(sum_squared_dd)

        if not np.isfinite(burke):
            return None

        return float(burke)
    except Exception as e:
        logger.warning(f"Error calculating Burke ratio: {e}")
        return None


def calculate_treynor_ratio(
    returns: pd.Series,
    beta: float,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """
    Calculate Treynor ratio.

    Formula: (Return - Risk Free Rate) / Beta

    Args:
        returns: Series of returns
        beta: Beta coefficient (from market metrics)
        risk_free_rate: Annual risk-free rate

    Returns:
        Treynor ratio or None if beta is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    if beta == 0:
        return None

    try:
        annual_return = calculate_annualized_return(returns)
        excess_return = annual_return - risk_free_rate

        treynor = excess_return / beta

        if not np.isfinite(treynor):
            return None

        return float(treynor)
    except Exception as e:
        logger.warning(f"Error calculating Treynor ratio: {e}")
        return None


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Information ratio.

    Formula: Active Return / Tracking Error

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns (aligned by date)

    Returns:
        Information ratio or None if tracking error is zero

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        # Align by date
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if aligned.empty:
            return None

        active_returns = (
            aligned["portfolio"] - aligned["benchmark"]
        )
        tracking_error = float(active_returns.std())

        if tracking_error == 0:
            return None

        active_return_annualized = float(
            active_returns.mean() * TRADING_DAYS_PER_YEAR
        )

        info_ratio = active_return_annualized / tracking_error

        if not np.isfinite(info_ratio):
            return None

        return float(info_ratio)
    except Exception as e:
        logger.warning(f"Error calculating Information ratio: {e}")
        return None


def calculate_modigliani_m2(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """
    Calculate Modigliani M² ratio.

    Formula: Risk-adjusted return vs benchmark

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate

    Returns:
        M² ratio

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        portfolio_return = calculate_annualized_return(
            portfolio_returns
        )

        portfolio_vol = float(
            portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        benchmark_vol = float(
            benchmark_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        if benchmark_vol == 0:
            return None

        # M² = Risk-free + (Portfolio Return - Risk-free) *
        #     (Benchmark Vol / Portfolio Vol)
        m2 = risk_free_rate + (
            (portfolio_return - risk_free_rate)
            * (benchmark_vol / portfolio_vol)
        )

        if not np.isfinite(m2):
            return None

        return float(m2)
    except Exception as e:
        logger.warning(f"Error calculating M² ratio: {e}")
        return None


def calculate_omega_ratio(
    returns: pd.Series, threshold: float = 0.0
) -> Optional[float]:
    """
    Calculate Omega ratio.

    Formula: Sum(probability-weighted gains above threshold) /
             Sum(probability-weighted losses below threshold)

    Args:
        returns: Series of returns
        threshold: Threshold return (default: 0.0)

    Returns:
        Omega ratio or None if no losses

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if losses.empty or losses.sum() == 0:
            return None

        gains_sum = float(gains.sum())
        losses_sum = float(losses.sum())

        omega = gains_sum / losses_sum

        if not np.isfinite(omega):
            return None

        return float(omega)
    except Exception as e:
        logger.warning(f"Error calculating Omega ratio: {e}")
        return None


def calculate_kappa_3(returns: pd.Series) -> Optional[float]:
    """
    Calculate Kappa 3 ratio.

    Formula: Return / (Downside Risk)^3

    Args:
        returns: Series of returns

    Returns:
        Kappa 3 ratio or None if downside risk is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(returns)
        downside_dev = calculate_downside_deviation(returns)

        if downside_dev == 0:
            return None

        kappa3 = annual_return / (downside_dev**3)

        if not np.isfinite(kappa3):
            return None

        return float(kappa3)
    except Exception as e:
        logger.warning(f"Error calculating Kappa 3 ratio: {e}")
        return None


def calculate_gain_pain_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Gain-Pain ratio.

    Formula: Sum(Gains) / Sum(Losses) [absolute values]

    Args:
        returns: Series of returns

    Returns:
        Gain-Pain ratio or None if no losses

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    gains = returns[returns > 0]
    losses = returns[returns < 0]

    if losses.empty or losses.sum() == 0:
        return None

    gains_sum = float(abs(gains.sum()))
    losses_sum = float(abs(losses.sum()))

    if losses_sum == 0:
        return None

    return float(gains_sum / losses_sum)


def calculate_martin_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Martin ratio.

    Formula: Annualized Return / Ulcer Index

    Args:
        returns: Series of returns

    Returns:
        Martin ratio or None if ulcer index is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        annual_return = calculate_annualized_return(returns)
        ulcer_index = calculate_ulcer_index(returns)

        if ulcer_index == 0:
            return None

        martin = annual_return / ulcer_index

        if not np.isfinite(martin):
            return None

        return float(martin)
    except Exception as e:
        logger.warning(f"Error calculating Martin ratio: {e}")
        return None


def calculate_tail_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Tail ratio.

    Formula: 95th percentile / 5th percentile (absolute returns)

    Args:
        returns: Series of returns

    Returns:
        Tail ratio or None if 5th percentile is zero

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        p95 = float(np.percentile(returns, 95))
        p5 = float(np.percentile(returns, 5))

        if abs(p5) == 0:
            return None

        tail = abs(p95) / abs(p5)

        if not np.isfinite(tail):
            return None

        return float(tail)
    except Exception as e:
        logger.warning(f"Error calculating Tail ratio: {e}")
        return None


def calculate_common_sense_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate Common Sense ratio.

    Formula: Profit Factor × Tail Ratio

    Args:
        returns: Series of returns

    Returns:
        Common Sense ratio or None if components unavailable

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        from core.analytics_engine.performance import (
            calculate_profit_factor,
        )

        profit_factor = calculate_profit_factor(returns)
        tail_ratio = calculate_tail_ratio(returns)

        if profit_factor is None or tail_ratio is None:
            return None

        common_sense = profit_factor * tail_ratio

        if not np.isfinite(common_sense):
            return None

        return float(common_sense)
    except Exception as e:
        logger.warning(
            f"Error calculating Common Sense ratio: {e}"
        )
        return None


def calculate_rachev_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha: float = 0.05,
) -> Optional[float]:
    """
    Calculate Rachev ratio.

    Formula: CVaR(alpha) of benchmark / CVaR(alpha) of portfolio

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        alpha: Confidence level for CVaR (default: 0.05)

    Returns:
        Rachev ratio or None if CVaR unavailable

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        portfolio_cvar = calculate_cvar(
            portfolio_returns, confidence_level=1 - alpha
        )
        benchmark_cvar = calculate_cvar(
            benchmark_returns, confidence_level=1 - alpha
        )

        if abs(portfolio_cvar) == 0:
            return None

        rachev = abs(benchmark_cvar) / abs(portfolio_cvar)

        if not np.isfinite(rachev):
            return None

        return float(rachev)
    except Exception as e:
        logger.warning(f"Error calculating Rachev ratio: {e}")
        return None

