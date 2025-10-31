"""Performance metrics calculation (18 metrics)."""

import logging
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd

from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252


def calculate_total_return(
    start_value: float, end_value: float
) -> float:
    """
    Calculate total return over period.

    Formula: (End Value - Start Value) / Start Value

    Args:
        start_value: Starting portfolio value
        end_value: Ending portfolio value

    Returns:
        Total return as decimal (e.g., 0.2534 = 25.34%)

    Raises:
        ValidationError: If start_value <= 0
    """
    if start_value <= 0:
        raise ValueError("Start value must be greater than 0")

    return (end_value - start_value) / start_value


def calculate_cagr(
    start_value: float, end_value: float, years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    Formula: ((End Value / Start Value)^(1/Years) - 1)

    Args:
        start_value: Starting portfolio value
        end_value: Ending portfolio value
        years: Number of years (can be fractional)

    Returns:
        CAGR as decimal (e.g., 0.3125 = 31.25% per year)

    Raises:
        InsufficientDataError: If years <= 0 or start_value <= 0
    """
    if years <= 0:
        raise InsufficientDataError(
            "Years must be greater than 0 for CAGR"
        )
    if start_value <= 0:
        raise ValueError("Start value must be greater than 0")

    if end_value <= 0:
        return float(-1.0)  # Total loss

    return float((end_value / start_value) ** (1.0 / years) - 1.0)


def calculate_annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized return from periodic returns.

    Formula: (1 + Mean Return) * periods_per_year - 1

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized return as decimal

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError(
            "Returns series is empty"
        )

    mean_return = float(returns.mean())
    return float((1.0 + mean_return) ** periods_per_year - 1.0)


def calculate_period_returns(
    portfolio_values: pd.Series,
    periods: Optional[Dict[str, date]] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate returns for various periods.

    Args:
        portfolio_values: Series of portfolio values indexed by date
        periods: Optional dict with period boundaries

    Returns:
        Dictionary with period returns:
        - ytd, mtd, qtd, 1m, 3m, 6m, 1y, 3y, 5y
        - Returns None if insufficient data
    """
    if portfolio_values.empty:
        return {
            "ytd": None,
            "mtd": None,
            "qtd": None,
            "1m": None,
            "3m": None,
            "6m": None,
            "1y": None,
            "3y": None,
            "5y": None,
        }

    results: Dict[str, Optional[float]] = {}
    end_date_raw = portfolio_values.index[-1]
    # Normalize end_date to date
    if hasattr(end_date_raw, 'date'):
        end_date = end_date_raw.date()
    elif hasattr(end_date_raw, 'to_pydatetime'):
        end_date = end_date_raw.to_pydatetime().date()
    else:
        end_date = pd.to_datetime(end_date_raw).date()
    end_value = float(portfolio_values.iloc[-1])

    # YTD (Year-to-Date)
    try:
        ytd_start = date(end_date.year, 1, 1)
        # Find closest date >= ytd_start
        mask = portfolio_values.index >= pd.Timestamp(ytd_start)
        if mask.any():
            start_value = float(portfolio_values[mask].iloc[0])
            results["ytd"] = calculate_total_return(
                start_value, end_value
            )
        else:
            results["ytd"] = None
    except Exception:
        results["ytd"] = None

    # MTD (Month-to-Date)
    try:
        mtd_start = date(end_date.year, end_date.month, 1)
        # Find closest date >= mtd_start
        mask = portfolio_values.index >= pd.Timestamp(mtd_start)
        if mask.any():
            start_value = float(portfolio_values[mask].iloc[0])
            results["mtd"] = calculate_total_return(
                start_value, end_value
            )
        else:
            results["mtd"] = None
    except Exception:
        results["mtd"] = None

    # QTD (Quarter-to-Date)
    try:
        quarter = (end_date.month - 1) // 3 + 1
        qtd_start = date(end_date.year, (quarter - 1) * 3 + 1, 1)
        # Find closest date >= qtd_start
        mask = portfolio_values.index >= pd.Timestamp(qtd_start)
        if mask.any():
            start_value = float(portfolio_values[mask].iloc[0])
            results["qtd"] = calculate_total_return(
                start_value, end_value
            )
        else:
            results["qtd"] = None
    except Exception:
        results["qtd"] = None

    # Fixed periods (1M, 3M, 6M, 1Y, 3Y, 5Y)
    period_map = {
        "1m": pd.DateOffset(months=1),
        "3m": pd.DateOffset(months=3),
        "6m": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
    }

    for period_name, offset in period_map.items():
        try:
            period_start = end_date - offset
            # Find closest date >= period_start
            mask = portfolio_values.index >= period_start
            period_data = portfolio_values[mask]

            if len(period_data) >= 2:
                start_value = float(period_data.iloc[0])
                results[period_name] = calculate_total_return(
                    start_value, end_value
                )
            else:
                results[period_name] = None
        except Exception:
            results[period_name] = None

    return results


def calculate_best_worst_periods(
    returns: pd.Series,
) -> Dict[str, Optional[Tuple[float, str]]]:
    """
    Calculate best and worst periods.

    Args:
        returns: Series of periodic returns indexed by date

    Returns:
        Dictionary with:
        - best_month: (return, month_string) or None
        - worst_month: (return, month_string) or None
    """
    if returns.empty:
        return {"best_month": None, "worst_month": None}

    try:
        # Resample to monthly
        monthly_returns = returns.resample("M").apply(
            lambda x: (1 + x).prod() - 1
        )

        if monthly_returns.empty:
            return {"best_month": None, "worst_month": None}

        best_month_idx = monthly_returns.idxmax()
        worst_month_idx = monthly_returns.idxmin()

        best_month_return = float(monthly_returns.max())
        worst_month_return = float(monthly_returns.min())

        best_month_str = best_month_idx.strftime("%b %Y")
        worst_month_str = worst_month_idx.strftime("%b %Y")

        return {
            "best_month": (best_month_return, best_month_str),
            "worst_month": (worst_month_return, worst_month_str),
        }
    except Exception as e:
        logger.warning(f"Error calculating best/worst periods: {e}")
        return {"best_month": None, "worst_month": None}


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive periods).

    Formula: Count(Positive Periods) / Count(Total Periods)

    Args:
        returns: Series of returns

    Returns:
        Win rate as decimal (e.g., 0.6471 = 64.71%)

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    positive_count = int((returns > 0).sum())
    total_count = len(returns)

    return float(positive_count / total_count)


def calculate_payoff_ratio(returns: pd.Series) -> Optional[float]:
    """
    Calculate payoff ratio (average win / average loss).

    Formula: Average Win / Average Loss (absolute value)

    Args:
        returns: Series of returns

    Returns:
        Payoff ratio or None if no losses/wins

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if wins.empty or losses.empty:
        return None

    avg_win = float(wins.mean())
    avg_loss = float(abs(losses.mean()))

    if avg_loss == 0:
        return None

    return float(avg_win / avg_loss)


def calculate_profit_factor(returns: pd.Series) -> Optional[float]:
    """
    Calculate profit factor (gross profits / gross losses).

    Formula: Gross Profits / Gross Losses

    Args:
        returns: Series of returns

    Returns:
        Profit factor or None if no losses

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    gross_profits = float(returns[returns > 0].sum())
    gross_losses = float(abs(returns[returns < 0].sum()))

    if gross_losses == 0:
        return None

    return float(gross_profits / gross_losses)


def calculate_expectancy(returns: pd.Series) -> float:
    """
    Calculate expectancy (expected value per period).

    Formula: (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

    Args:
        returns: Series of returns

    Returns:
        Expectancy as decimal

    Raises:
        InsufficientDataError: If returns series is empty
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    win_rate = calculate_win_rate(returns)
    loss_rate = 1.0 - win_rate

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0

    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

    return float(expectancy)

