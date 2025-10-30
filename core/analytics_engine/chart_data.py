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


def get_rolling_sharpe_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 126,  # ~6 months
    risk_free_rate: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    Prepare data for rolling Sharpe ratio chart.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with rolling Sharpe series
    """
    if portfolio_returns.empty:
        return {"portfolio": pd.Series(dtype=float)}

    # Calculate rolling Sharpe
    rolling_mean = portfolio_returns.rolling(window).mean()
    rolling_std = portfolio_returns.rolling(window).std()
    
    # Annualize
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    portfolio_sharpe = ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

    result = {"portfolio": portfolio_sharpe.fillna(0)}

    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align dates
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index, method="ffill"
        ).fillna(0)
        
        bench_mean = aligned_benchmark.rolling(window).mean()
        bench_std = aligned_benchmark.rolling(window).std()
        benchmark_sharpe = ((bench_mean - daily_rf) / bench_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
        result["benchmark"] = benchmark_sharpe.fillna(0)

    return result


def get_rolling_sortino_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 126,
    risk_free_rate: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    Prepare data for rolling Sortino ratio chart.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with rolling Sortino series
    """
    if portfolio_returns.empty:
        return {"portfolio": pd.Series(dtype=float)}

    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    def rolling_sortino(returns, window_size):
        result = pd.Series(index=returns.index, dtype=float)
        for i in range(window_size - 1, len(returns)):
            window_returns = returns.iloc[i - window_size + 1 : i + 1]
            excess = window_returns - daily_rf
            mean_excess = excess.mean()
            downside = excess[excess < 0].std()
            if downside > 0:
                result.iloc[i] = (mean_excess / downside) * np.sqrt(TRADING_DAYS_PER_YEAR)
            else:
                result.iloc[i] = 0.0
        return result

    portfolio_sortino = rolling_sortino(portfolio_returns, window)
    result = {"portfolio": portfolio_sortino.fillna(0)}

    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index, method="ffill"
        ).fillna(0)
        benchmark_sortino = rolling_sortino(aligned_benchmark, window)
        result["benchmark"] = benchmark_sortino.fillna(0)

    return result


def get_rolling_beta_alpha_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.0,
) -> Dict[str, pd.Series]:
    """
    Prepare data for rolling Beta and Alpha chart.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with rolling Beta and Alpha series
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {"beta": pd.Series(dtype=float), "alpha": pd.Series(dtype=float)}

    # Align dates
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < window:
        return {"beta": pd.Series(dtype=float), "alpha": pd.Series(dtype=float)}

    rolling_beta = aligned['portfolio'].rolling(window).cov(aligned['benchmark']) / \
                   aligned['benchmark'].rolling(window).var()

    # Calculate rolling alpha
    portfolio_mean = aligned['portfolio'].rolling(window).mean() * TRADING_DAYS_PER_YEAR
    benchmark_mean = aligned['benchmark'].rolling(window).mean() * TRADING_DAYS_PER_YEAR
    rolling_alpha = portfolio_mean - (risk_free_rate + rolling_beta * (benchmark_mean - risk_free_rate))

    return {
        "beta": rolling_beta.fillna(0),
        "alpha": rolling_alpha.fillna(0),
    }


def get_underwater_plot_data(
    portfolio_values: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    Prepare data for underwater plot (drawdown from peak).

    Args:
        portfolio_values: Series of portfolio values
        benchmark_values: Optional benchmark values for comparison

    Returns:
        Dictionary with underwater series (drawdown %) for portfolio and optionally benchmark
    """
    result = {}
    
    if not portfolio_values.empty:
        # Calculate running maximum
        peak = portfolio_values.expanding().max()
        # Calculate drawdown as percentage
        underwater = (portfolio_values / peak - 1) * 100
        result["underwater"] = underwater
    else:
        result["underwater"] = pd.Series(dtype=float)
    
    if benchmark_values is not None and not benchmark_values.empty:
        # Align dates with portfolio
        aligned_benchmark = benchmark_values.reindex(
            portfolio_values.index, method="ffill"
        ).bfill().ffill()
        
        if not aligned_benchmark.empty:
            # Calculate running maximum for benchmark
            benchmark_peak = aligned_benchmark.expanding().max()
            # Calculate drawdown as percentage
            benchmark_underwater = (aligned_benchmark / benchmark_peak - 1) * 100
            result["benchmark"] = benchmark_underwater
    
    return result


def get_best_worst_periods_data(
    portfolio_returns: pd.Series,
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for best and worst periods tables.

    Args:
        portfolio_returns: Series of portfolio returns
        top_n: Number of top/bottom periods to return

    Returns:
        Dictionary with DataFrames for best and worst days/weeks/months
    """
    if portfolio_returns.empty:
        return {
            "best_days": pd.DataFrame(),
            "worst_days": pd.DataFrame(),
        }

    # Best and worst days
    best_days = portfolio_returns.nlargest(top_n).to_frame('Return')
    best_days['Return'] = best_days['Return'] * 100  # Convert to %
    best_days.index.name = 'Date'

    worst_days = portfolio_returns.nsmallest(top_n).to_frame('Return')
    worst_days['Return'] = worst_days['Return'] * 100
    worst_days.index.name = 'Date'

    return {
        "best_days": best_days,
        "worst_days": worst_days,
    }


def get_comparison_stats_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Prepare comprehensive comparison statistics.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns

    Returns:
        Dictionary with comparison metrics
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {}

    # Align dates
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return {}

    # Calculate correlation
    correlation = float(aligned['portfolio'].corr(aligned['benchmark']))

    # Calculate beta
    covariance = float(aligned['portfolio'].cov(aligned['benchmark']))
    benchmark_variance = float(aligned['benchmark'].var())
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0.0

    # Win rate (portfolio outperforms benchmark)
    outperformance = aligned['portfolio'] > aligned['benchmark']
    win_rate = float(outperformance.sum() / len(aligned))

    # Average outperformance on winning days
    winning_days = aligned[outperformance]
    avg_win = float((winning_days['portfolio'] - winning_days['benchmark']).mean()) if len(winning_days) > 0 else 0.0

    # Average underperformance on losing days
    losing_days = aligned[~outperformance]
    avg_loss = float((losing_days['portfolio'] - losing_days['benchmark']).mean()) if len(losing_days) > 0 else 0.0

    return {
        "correlation": correlation,
        "beta": beta,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
    }


def get_yearly_returns_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare yearly returns comparison data.

    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Optional benchmark returns

    Returns:
        Dictionary with yearly returns DataFrame
    """
    if portfolio_returns.empty:
        return {"yearly": pd.DataFrame()}

    df = pd.DataFrame({"Portfolio": portfolio_returns})
    df['Year'] = df.index.year

    # Calculate yearly returns
    yearly_portfolio = ((1 + df['Portfolio']).groupby(df['Year']).prod() - 1) * 100

    result_df = pd.DataFrame({"Portfolio": yearly_portfolio})

    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_df = pd.DataFrame({"Benchmark": benchmark_returns})
        bench_df['Year'] = bench_df.index.year
        yearly_benchmark = ((1 + bench_df['Benchmark']).groupby(bench_df['Year']).prod() - 1) * 100
        result_df['Benchmark'] = yearly_benchmark
        result_df['Difference'] = result_df['Portfolio'] - result_df['Benchmark']

    return {"yearly": result_df}