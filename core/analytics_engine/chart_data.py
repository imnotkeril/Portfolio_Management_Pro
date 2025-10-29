"""Chart data preparation functions for portfolio visualization."""

import logging
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def get_cumulative_returns_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    Prepare data for cumulative returns chart.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns

    Returns:
        Dictionary with 'portfolio' and optionally 'benchmark' Series
    """
    # Calculate cumulative returns (1 + returns).cumprod()
    portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
    result = {"portfolio": portfolio_cumulative}

    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align dates
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index,
            method="ffill"
        )
        aligned_benchmark = aligned_benchmark.fillna(0)
        benchmark_cumulative = (1 + aligned_benchmark).cumprod() - 1
        result["benchmark"] = benchmark_cumulative

    return result


def get_drawdown_data(
    portfolio_values: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Prepare data for drawdown chart.

    Args:
        portfolio_values: Series of portfolio values indexed by date

    Returns:
        Dictionary with 'drawdown' Series and 'peak' Series
    """
    if portfolio_values.empty:
        empty_series = pd.Series(dtype=float)
        return {"drawdown": empty_series, "peak": empty_series}

    # Calculate running maximum
    peak = portfolio_values.expanding().max()

    # Calculate drawdown as percentage
    drawdown = (portfolio_values - peak) / peak

    # Find max drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_date = max_dd_idx if isinstance(max_dd_idx, date) else None
    max_dd_value = drawdown.min()

    return {
        "drawdown": drawdown,
        "peak": peak,
        "max_drawdown": {
            "date": max_dd_date,
            "value": float(max_dd_value),
        },
    }


def get_rolling_metric_data(
    portfolio_returns: pd.Series,
    metric_name: str,
    window: int = 30,
) -> Dict[str, pd.Series]:
    """
    Prepare data for rolling metric chart.

    Args:
        portfolio_returns: Series of portfolio returns
        metric_name: Metric to calculate ('sharpe', 'volatility', 'returns')
        window: Rolling window size in days

    Returns:
        Dictionary with rolling metric Series
    """
    if portfolio_returns.empty:
        return {"metric": pd.Series(dtype=float)}

    metric_series = pd.Series(dtype=float, index=portfolio_returns.index)

    if metric_name.lower() == "sharpe":
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        metric_series = (
            rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).fillna(0)
    elif metric_name.lower() == "volatility":
        metric_series = (
            portfolio_returns.rolling(window).std()
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        ).fillna(0)
    elif metric_name.lower() == "returns":
        metric_series = portfolio_returns.rolling(window).mean().fillna(0)
    else:
        logger.warning(f"Unknown metric: {metric_name}, using returns")
        metric_series = portfolio_returns.rolling(window).mean().fillna(0)

    return {"metric": metric_series}


def get_return_distribution_data(
    portfolio_returns: pd.Series,
    bins: int = 50,
) -> Dict[str, any]:
    """
    Prepare data for return distribution histogram.

    Args:
        portfolio_returns: Series of portfolio returns
        bins: Number of bins for histogram

    Returns:
        Dictionary with histogram data and statistics
    """
    if portfolio_returns.empty:
        return {
            "counts": np.array([]),
            "edges": np.array([]),
            "mean": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    returns_array = portfolio_returns.values
    counts, edges = np.histogram(returns_array, bins=bins)

    # Calculate statistics
    mean = float(portfolio_returns.mean())
    std = float(portfolio_returns.std())
    skewness = float(portfolio_returns.skew())
    kurtosis = float(portfolio_returns.kurtosis())

    # Calculate VaR levels
    var_90 = float(np.percentile(returns_array, 10))
    var_95 = float(np.percentile(returns_array, 5))
    var_99 = float(np.percentile(returns_array, 1))

    return {
        "counts": counts,
        "edges": edges,
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "var_90": var_90,
        "var_95": var_95,
        "var_99": var_99,
    }


def get_qq_plot_data(
    portfolio_returns: pd.Series,
) -> Dict[str, np.ndarray]:
    """
    Prepare data for Q-Q plot (quantile-quantile plot).

    Args:
        portfolio_returns: Series of portfolio returns

    Returns:
        Dictionary with theoretical and sample quantiles
    """
    if portfolio_returns.empty:
        return {"theoretical": np.array([]), "sample": np.array([])}

    returns_array = portfolio_returns.dropna().values

    if len(returns_array) == 0:
        return {"theoretical": np.array([]), "sample": np.array([])}

    # Fit normal distribution
    mean = returns_array.mean()
    std = returns_array.std()

    if std == 0:
        return {"theoretical": np.array([]), "sample": np.array([])}

    # Calculate theoretical quantiles (normal distribution)
    theoretical_quantiles = np.linspace(
        stats.norm.ppf(0.01, loc=mean, scale=std),
        stats.norm.ppf(0.99, loc=mean, scale=std),
        len(returns_array),
    )

    # Sort sample returns
    sample_quantiles = np.sort(returns_array)

    return {
        "theoretical": theoretical_quantiles,
        "sample": sample_quantiles,
        "mean": float(mean),
        "std": float(std),
    }


def get_monthly_heatmap_data(
    portfolio_returns: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for monthly returns heatmap.

    Args:
        portfolio_returns: Series of portfolio returns

    Returns:
        Dictionary with DataFrame indexed by year/month
    """
    if portfolio_returns.empty:
        return {"heatmap": pd.DataFrame()}

    # Convert returns to DataFrame with date index
    df = pd.DataFrame({"returns": portfolio_returns})

    # Extract year and month
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Calculate monthly returns
    # (sum of daily returns for each month)
    monthly_returns = (
        df.groupby(["year", "month"])["returns"].sum() * 100
    )  # Convert to %

    # Pivot to create heatmap (years as rows, months as columns)
    heatmap = monthly_returns.unstack(level=1)

    return {"heatmap": heatmap}

