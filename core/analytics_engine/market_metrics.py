"""Market-related metrics calculation (15 metrics)."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Beta (systematic risk vs benchmark).

    Formula: Cov(portfolio, benchmark) / Var(benchmark)

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns (aligned by date)

    Returns:
        Beta coefficient or None if insufficient data

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

        if len(aligned) < 2:
            return None

        cov = float(aligned["portfolio"].cov(aligned["benchmark"]))
        benchmark_var = float(aligned["benchmark"].var())

        if benchmark_var == 0:
            return None

        beta = cov / benchmark_var

        if not np.isfinite(beta):
            return None

        return float(beta)
    except Exception as e:
        logger.warning(f"Error calculating Beta: {e}")
        return None


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """
    Calculate Alpha (excess return above expected, CAPM).

    Formula: Portfolio Return - (Risk Free + Beta × (Benchmark - Risk Free))

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Alpha (annualized) or None if beta unavailable

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        from core.analytics_engine.performance import (
            calculate_annualized_return,
        )

        portfolio_return = calculate_annualized_return(
            portfolio_returns
        )
        benchmark_return = calculate_annualized_return(
            benchmark_returns
        )

        beta = calculate_beta(portfolio_returns, benchmark_returns)
        if beta is None:
            return None

        # CAPM: Expected return = Rf + Beta × (Rm - Rf)
        expected_return = risk_free_rate + beta * (
            benchmark_return - risk_free_rate
        )
        alpha = portfolio_return - expected_return

        if not np.isfinite(alpha):
            return None

        return float(alpha)
    except Exception as e:
        logger.warning(f"Error calculating Alpha: {e}")
        return None


def calculate_r_squared(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate R-squared (correlation to benchmark).

    Formula: (Correlation)^2

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        R-squared (0.0 to 1.0) or None if insufficient data

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        correlation = calculate_correlation(
            portfolio_returns, benchmark_returns
        )

        if correlation is None:
            return None

        r_squared = correlation**2

        if not np.isfinite(r_squared):
            return None

        return float(r_squared)
    except Exception as e:
        logger.warning(f"Error calculating R-squared: {e}")
        return None


def calculate_correlation(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate correlation between portfolio and benchmark.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Correlation coefficient (-1.0 to 1.0) or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if len(aligned) < 2:
            return None

        correlation = float(
            aligned["portfolio"].corr(aligned["benchmark"])
        )

        if not np.isfinite(correlation):
            return None

        return float(correlation)
    except Exception as e:
        logger.warning(f"Error calculating Correlation: {e}")
        return None


def calculate_tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate tracking error (standard deviation of active returns).

    Formula: Std Dev(Portfolio Returns - Benchmark Returns)

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Tracking error (annualized) or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if aligned.empty:
            return None

        active_returns = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(
            active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        if not np.isfinite(tracking_error):
            return None

        return float(tracking_error)
    except Exception as e:
        logger.warning(f"Error calculating Tracking Error: {e}")
        return None


def calculate_active_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate active return (return above benchmark).

    Formula: Portfolio Return - Benchmark Return

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Active return (annualized) or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        from core.analytics_engine.performance import (
            calculate_annualized_return,
        )

        portfolio_return = calculate_annualized_return(
            portfolio_returns
        )
        benchmark_return = calculate_annualized_return(
            benchmark_returns
        )

        active_return = portfolio_return - benchmark_return

        if not np.isfinite(active_return):
            return None

        return float(active_return)
    except Exception as e:
        logger.warning(f"Error calculating Active Return: {e}")
        return None


def calculate_up_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Up Capture ratio.

    Formula: Avg(Portfolio Return in Up Markets) /
             Avg(Benchmark Return in Up Markets)

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Up Capture ratio or None if no up markets

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        # Up markets: benchmark return > 0
        up_markets = aligned[aligned["benchmark"] > 0]

        if up_markets.empty:
            return None

        portfolio_up_avg = float(up_markets["portfolio"].mean())
        benchmark_up_avg = float(up_markets["benchmark"].mean())

        if benchmark_up_avg == 0:
            return None

        up_capture = portfolio_up_avg / benchmark_up_avg

        if not np.isfinite(up_capture):
            return None

        return float(up_capture)
    except Exception as e:
        logger.warning(f"Error calculating Up Capture: {e}")
        return None


def calculate_down_capture(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Down Capture ratio.

    Formula: Avg(Portfolio Return in Down Markets) /
             Avg(Benchmark Return in Down Markets)

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Down Capture ratio or None if no down markets

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        # Down markets: benchmark return < 0
        down_markets = aligned[aligned["benchmark"] < 0]

        if down_markets.empty:
            return None

        portfolio_down_avg = float(down_markets["portfolio"].mean())
        benchmark_down_avg = float(down_markets["benchmark"].mean())

        if benchmark_down_avg == 0:
            return None

        down_capture = portfolio_down_avg / benchmark_down_avg

        if not np.isfinite(down_capture):
            return None

        return float(down_capture)
    except Exception as e:
        logger.warning(f"Error calculating Down Capture: {e}")
        return None


def calculate_capture_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Up/Down Capture Ratio.

    Formula: Up Capture / Down Capture

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Capture ratio or None if components unavailable

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    up_capture = calculate_up_capture(
        portfolio_returns, benchmark_returns
    )
    down_capture = calculate_down_capture(
        portfolio_returns, benchmark_returns
    )

    if up_capture is None or down_capture is None:
        return None

    if abs(down_capture) == 0:
        return None

    capture_ratio = up_capture / abs(down_capture)

    if not np.isfinite(capture_ratio):
        return None

    return float(capture_ratio)


def calculate_jensens_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Optional[float]:
    """
    Calculate Jensen's Alpha (risk-adjusted excess return).

    Same as Alpha calculation (CAPM-based).

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Jensen's Alpha (annualized)

    Raises:
        InsufficientDataError: If returns series are empty
    """
    return calculate_alpha(
        portfolio_returns, benchmark_returns, risk_free_rate
    )


def calculate_active_share(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
) -> Optional[float]:
    """
    Calculate Active Share (% of holdings different from benchmark).

    Formula: 0.5 × Sum(abs(Portfolio Weight - Benchmark Weight))

    Args:
        portfolio_weights: Dictionary mapping ticker to weight
        benchmark_weights: Dictionary mapping ticker to weight

    Returns:
        Active Share (0.0 to 1.0) or None

    Raises:
        InsufficientDataError: If weights dictionaries are empty
    """
    if not portfolio_weights or not benchmark_weights:
        raise InsufficientDataError("Weights dictionaries are empty")

    try:
        # Get union of all tickers
        all_tickers = set(portfolio_weights.keys()) | set(
            benchmark_weights.keys()
        )

        total_diff = 0.0
        for ticker in all_tickers:
            portfolio_weight = portfolio_weights.get(ticker, 0.0)
            benchmark_weight = benchmark_weights.get(ticker, 0.0)
            total_diff += abs(portfolio_weight - benchmark_weight)

        active_share = 0.5 * total_diff

        if not np.isfinite(active_share):
            return None

        return float(active_share)
    except Exception as e:
        logger.warning(f"Error calculating Active Share: {e}")
        return None


def calculate_batting_average(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Batting Average (% of periods beating benchmark).

    Formula: Count(Portfolio Return > Benchmark Return) / Total Periods

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Batting average (0.0 to 1.0) or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if aligned.empty:
            return None

        beats_count = int(
            (aligned["portfolio"] > aligned["benchmark"]).sum()
        )
        total_periods = len(aligned)

        batting_avg = beats_count / total_periods

        if not np.isfinite(batting_avg):
            return None

        return float(batting_avg)
    except Exception as e:
        logger.warning(f"Error calculating Batting Average: {e}")
        return None


def calculate_benchmark_relative_return(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Benchmark Relative Return (cumulative outperformance).

    Formula: Cumulative Portfolio Return - Cumulative Benchmark Return

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Relative return or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if aligned.empty:
            return None

        portfolio_cum = float((1 + aligned["portfolio"]).prod() - 1)
        benchmark_cum = float((1 + aligned["benchmark"]).prod() - 1)

        relative_return = portfolio_cum - benchmark_cum

        if not np.isfinite(relative_return):
            return None

        return float(relative_return)
    except Exception as e:
        logger.warning(
            f"Error calculating Benchmark Relative Return: {e}"
        )
        return None


def calculate_rolling_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 30,
) -> Optional[float]:
    """
    Calculate average rolling Beta.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size in days (default: 30)

    Returns:
        Average rolling beta or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        raise InsufficientDataError("Returns series are empty")

    try:
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if len(aligned) < window:
            return None

        # Calculate rolling betas
        rolling_betas = []
        for i in range(window, len(aligned) + 1):
            window_data = aligned.iloc[i - window:i]
            cov = float(
                window_data["portfolio"].cov(window_data["benchmark"])
            )
            benchmark_var = float(window_data["benchmark"].var())

            if benchmark_var > 0:
                beta = cov / benchmark_var
                if np.isfinite(beta):
                    rolling_betas.append(beta)

        if not rolling_betas:
            return None

        avg_beta = float(np.mean(rolling_betas))

        if not np.isfinite(avg_beta):
            return None

        return float(avg_beta)
    except Exception as e:
        logger.warning(f"Error calculating Rolling Beta: {e}")
        return None


def calculate_market_timing_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[float]:
    """
    Calculate Market Timing Ratio.

    Formula: Up Capture / abs(Down Capture)

    Similar to Capture Ratio but focuses on timing ability.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Market Timing Ratio or None

    Raises:
        InsufficientDataError: If returns series are empty
    """
    # Market Timing Ratio is same as Capture Ratio
    return calculate_capture_ratio(portfolio_returns, benchmark_returns)

