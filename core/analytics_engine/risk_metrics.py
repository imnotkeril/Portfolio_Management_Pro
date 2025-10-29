"""Risk metrics calculation (22 metrics)."""

import logging
from datetime import date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.exceptions import InsufficientDataError, NumericalError

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """
    Calculate volatility for multiple timeframes.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of trading periods per year

    Returns:
        Dictionary with:
        - daily: Daily volatility
        - weekly: Weekly volatility (if weekly data available)
        - monthly: Monthly volatility (if monthly data available)
        - annual: Annualized volatility

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    daily_vol = float(returns.std())

    results: Dict[str, float] = {
        "daily": daily_vol,
        "annual": daily_vol * np.sqrt(periods_per_year),
    }

    # Calculate weekly if we have enough daily data
    if len(returns) >= 5:
        try:
            weekly_returns = returns.resample("W").apply(
                lambda x: (1 + x).prod() - 1
            )
            if len(weekly_returns) >= 2:
                results["weekly"] = float(weekly_returns.std())
        except Exception:
            pass

    # Calculate monthly if we have enough daily data
    if len(returns) >= 20:
        try:
            monthly_returns = returns.resample("M").apply(
                lambda x: (1 + x).prod() - 1
            )
            if len(monthly_returns) >= 2:
                results["monthly"] = float(monthly_returns.std())
        except Exception:
            pass

    return results


def calculate_max_drawdown(
    returns: pd.Series,
) -> Tuple[float, Optional[date], Optional[date], int]:
    """
    Calculate maximum drawdown with dates and duration.

    Formula: Max((Peak - Trough) / Peak)

    Args:
        returns: Series of returns indexed by date

    Returns:
        Tuple of:
        - max_drawdown: Maximum drawdown as decimal (negative)
        - peak_date: Date of peak
        - trough_date: Date of trough
        - duration_days: Duration in days

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max

    max_dd = float(drawdowns.min())
    max_dd_idx = drawdowns.idxmin()

    # Find peak date (last date before max drawdown where peak occurred)
    peak_idx = cum_returns[:max_dd_idx].idxmax()
    peak_date = peak_idx.date() if hasattr(peak_idx, "date") else None
    trough_date = max_dd_idx.date() if hasattr(max_dd_idx, "date") else None

    # Calculate duration
    if peak_date and trough_date:
        duration = (trough_date - peak_date).days
    else:
        duration = 0

    return max_dd, peak_date, trough_date, duration


def calculate_current_drawdown(returns: pd.Series) -> float:
    """
    Calculate current drawdown from recent peak.

    Args:
        returns: Series of returns indexed by date

    Returns:
        Current drawdown as decimal (negative)

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    cum_returns = (1 + returns).cumprod()
    current_value = float(cum_returns.iloc[-1])
    peak_value = float(cum_returns.max())

    if peak_value == 0:
        return 0.0

    return float((current_value - peak_value) / peak_value)


def calculate_average_drawdown(returns: pd.Series) -> float:
    """
    Calculate average drawdown.

    Args:
        returns: Series of returns

    Returns:
        Average drawdown as decimal (negative)

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max

    # Average of all drawdown periods
    negative_drawdowns = drawdowns[drawdowns < 0]

    if negative_drawdowns.empty:
        return 0.0

    return float(negative_drawdowns.mean())


def calculate_drawdown_duration(
    returns: pd.Series,
) -> Dict[str, Optional[float]]:
    """
    Calculate maximum and average drawdown duration.

    Args:
        returns: Series of returns indexed by date

    Returns:
        Dictionary with:
        - max_duration_days: Maximum drawdown duration in days
        - avg_duration_days: Average drawdown duration in days
    """
    if returns.empty:
        return {
            "max_duration_days": None,
            "avg_duration_days": None,
        }

    try:
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max

        # Find drawdown periods (when drawdown < 0)
        in_drawdown = drawdowns < 0
        drawdown_periods = []

        if in_drawdown.any():
            # Group consecutive drawdown periods
            groups = (in_drawdown != in_drawdown.shift()).cumsum()
            for group_id, group_data in in_drawdown.groupby(groups):
                if group_data.iloc[0]:  # Is drawdown period
                    period_dates = group_data.index
                    if len(period_dates) >= 2:
                        duration = (
                            period_dates[-1] - period_dates[0]
                        ).days
                        drawdown_periods.append(duration)

        if drawdown_periods:
            max_duration = max(drawdown_periods)
            avg_duration = sum(drawdown_periods) / len(drawdown_periods)
            return {
                "max_duration_days": float(max_duration),
                "avg_duration_days": float(avg_duration),
            }
        else:
            return {
                "max_duration_days": 0.0,
                "avg_duration_days": 0.0,
            }
    except Exception as e:
        logger.warning(f"Error calculating drawdown duration: {e}")
        return {
            "max_duration_days": None,
            "avg_duration_days": None,
        }


def calculate_recovery_time(returns: pd.Series) -> Optional[int]:
    """
    Calculate recovery time from max drawdown (days to recover).

    Args:
        returns: Series of returns indexed by date

    Returns:
        Recovery time in days or None if not recovered yet
    """
    if returns.empty:
        return None

    try:
        max_dd, peak_date, trough_date, _ = calculate_max_drawdown(
            returns
        )

        if not trough_date:
            return None

        # Find when we recovered (first date after trough where
        # value >= peak value)
        cum_returns = (1 + returns).cumprod()
        peak_value = cum_returns.loc[peak_date] if peak_date else None

        if peak_value is None:
            return None

        # Find recovery point
        after_trough = cum_returns[cum_returns.index > trough_date]
        recovery_mask = after_trough >= peak_value

        if recovery_mask.any():
            recovery_date = after_trough[recovery_mask].index[0]
            recovery_days = (recovery_date.date() - trough_date).days
            return int(recovery_days)
        else:
            # Not recovered yet
            return None
    except Exception as e:
        logger.warning(f"Error calculating recovery time: {e}")
        return None


def calculate_ulcer_index(returns: pd.Series) -> float:
    """
    Calculate Ulcer Index.

    Formula: sqrt(mean((drawdown^2)))

    Args:
        returns: Series of returns

    Returns:
        Ulcer Index (higher = more stress)

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - running_max) / running_max

    # Only consider negative drawdowns
    negative_dd = drawdowns[drawdowns < 0]

    if negative_dd.empty:
        return 0.0

    # Mean of squared drawdowns
    mean_squared_dd = float((negative_dd**2).mean())

    return float(np.sqrt(mean_squared_dd))


def calculate_pain_index(returns: pd.Series) -> float:
    """
    Calculate Pain Index (cumulative drawdown measure).

    Formula: Mean of all drawdowns (not squared)

    Args:
        returns: Series of returns

    Returns:
        Pain Index

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    return float(abs(calculate_average_drawdown(returns)))


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (0.90, 0.95, 0.99)
        method: Method: 'historical', 'parametric', 'cornish_fisher'

    Returns:
        VaR as decimal (negative)

    Raises:
        InsufficientDataError: If returns series is empty
        ValueError: If invalid method or confidence level
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    if confidence_level not in [0.90, 0.95, 0.99]:
        raise ValueError(
            "Confidence level must be 0.90, 0.95, or 0.99"
        )

    alpha = 1.0 - confidence_level

    if method == "historical":
        # Historical VaR: Percentile of returns
        return float(np.percentile(returns, alpha * 100))

    elif method == "parametric":
        # Parametric VaR: Mean - (Z-score × Std Dev)
        mean_return = float(returns.mean())
        std_return = float(returns.std())
        z_score = stats.norm.ppf(alpha)

        return float(mean_return + z_score * std_return)

    elif method == "cornish_fisher":
        # Cornish-Fisher VaR (accounts for skewness/kurtosis)
        mean_return = float(returns.mean())
        std_return = float(returns.std())
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())

        z_normal = stats.norm.ppf(alpha)
        z_cf = (
            z_normal
            + (1.0 / 6.0) * (z_normal**2 - 1) * skew
            + (1.0 / 24.0)
            * (z_normal**3 - 3 * z_normal)
            * (kurt - 3)
            - (1.0 / 36.0) * (2 * z_normal**3 - 5 * z_normal) * skew**2
        )

        return float(mean_return + z_cf * std_return)

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            "Use 'historical', 'parametric', or 'cornish_fisher'"
        )


def calculate_cvar(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional VaR (CVaR / Expected Shortfall).

    Formula: Mean of returns below VaR threshold

    Args:
        returns: Series of returns
        confidence_level: Confidence level (0.90, 0.95, 0.99)

    Returns:
        CVaR as decimal (negative)

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    # First calculate VaR
    var = calculate_var(returns, confidence_level, "historical")

    # CVaR is mean of returns below VaR
    tail_returns = returns[returns <= var]

    if tail_returns.empty:
        return var

    return float(tail_returns.mean())


def calculate_downside_deviation(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate downside deviation (volatility of negative returns).

    Args:
        returns: Series of returns
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized downside deviation

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    negative_returns = returns[returns < 0]

    if negative_returns.empty:
        return 0.0

    downside_std = float(negative_returns.std())
    return float(downside_std * np.sqrt(periods_per_year))


def calculate_semi_deviation(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Calculate semi-deviation (volatility of returns below mean).

    Args:
        returns: Series of returns
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized semi-deviation

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    mean_return = float(returns.mean())
    below_mean = returns[returns < mean_return]

    if below_mean.empty:
        return 0.0

    semi_std = float(below_mean.std())
    return float(semi_std * np.sqrt(periods_per_year))


def calculate_skewness(returns: pd.Series) -> float:
    """
    Calculate skewness of return distribution.

    Formula: E[(X - μ)³] / σ³

    Args:
        returns: Series of returns

    Returns:
        Skewness coefficient

    Raises:
        InsufficientDataError: If returns series is empty
        NumericalError: If calculation fails
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        skew = float(returns.skew())
        if not np.isfinite(skew):
            raise NumericalError("Skewness calculation resulted in NaN/Inf")
        return skew
    except Exception as e:
        raise NumericalError(f"Error calculating skewness: {e}") from e


def calculate_kurtosis(returns: pd.Series) -> float:
    """
    Calculate kurtosis of return distribution.

    Formula: E[(X - μ)⁴] / σ⁴

    Args:
        returns: Series of returns

    Returns:
        Kurtosis coefficient (normal distribution = 3)

    Raises:
        InsufficientDataError: If returns series is empty
        NumericalError: If calculation fails
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    try:
        kurt = float(returns.kurtosis())
        if not np.isfinite(kurt):
            raise NumericalError(
                "Kurtosis calculation resulted in NaN/Inf"
            )
        return kurt
    except Exception as e:
        raise NumericalError(f"Error calculating kurtosis: {e}") from e

