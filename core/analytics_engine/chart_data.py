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

    # Calculate VaR levels (lower tail - negative)
    var_90 = float(np.percentile(returns_array, 10))
    var_95 = float(np.percentile(returns_array, 5))
    var_99 = float(np.percentile(returns_array, 1))
    
    # Calculate positive VaR (upper tail)
    var_90_pos = float(np.percentile(returns_array, 90))
    var_95_pos = float(np.percentile(returns_array, 95))
    var_99_pos = float(np.percentile(returns_array, 99))

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
        "var_90_pos": var_90_pos,
        "var_95_pos": var_95_pos,
        "var_99_pos": var_99_pos,
        "returns_array": returns_array,  # For percentile calculations
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

    # Ensure index is DatetimeIndex
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns = portfolio_returns.copy()
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    # Convert returns to DataFrame with date index
    df = pd.DataFrame({"returns": portfolio_returns})

    # Extract year and month
    df["year"] = df.index.year
    df["month"] = df.index.month

    # Calculate monthly returns
    # (compound daily returns for each month)
    monthly_returns = (
        (1 + df["returns"]).groupby([df["year"], df["month"]]).prod() - 1
    ) * 100  # Convert to %

    # Pivot to create heatmap (years as rows, months as columns)
    heatmap = monthly_returns.unstack(level=1)
    heatmap.columns.name = "Month"
    heatmap.index.name = "Year"

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
) -> Dict[str, any]:
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
        
        # Find max drawdown point
        max_dd_idx = underwater.idxmin()
        result["max_drawdown"] = {
            "date": max_dd_idx,
            "value": float(underwater.min())
        }
        
        # Check if currently in drawdown
        current_dd = underwater.iloc[-1]
        if current_dd < -0.1:  # More than 0.1% from peak
            result["current_drawdown"] = {
                "date": underwater.index[-1],
                "value": float(current_dd)
            }
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


def get_drawdown_periods_data(
    portfolio_values: pd.Series,
    threshold: float = 0.05,
) -> Dict[str, any]:
    """
    Prepare data for drawdown periods chart with shaded zones.

    Args:
        portfolio_values: Series of portfolio values
        threshold: Minimum drawdown depth to include (e.g., 0.05 = 5%)

    Returns:
        Dictionary with cumulative returns and drawdown zones
    """
    if portfolio_values.empty:
        return {
            "cumulative_returns": pd.Series(dtype=float),
            "drawdown_zones": []
        }
    
    # Calculate cumulative returns (for display)
    cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
    
    # Calculate drawdowns
    peak = portfolio_values.expanding().max()
    drawdowns = (portfolio_values / peak - 1)
    
    # Find drawdown periods exceeding threshold
    in_drawdown = drawdowns < -threshold
    drawdown_zones = []
    
    if in_drawdown.any():
        # Group consecutive drawdown periods
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        
        for group_id, group_data in in_drawdown.groupby(groups):
            if not group_data.iloc[0]:  # Skip non-drawdown periods
                continue
                
            period_dates = group_data[group_data].index
            
            if len(period_dates) == 0:
                continue
            
            # Find min drawdown in this period
            period_drawdowns = drawdowns[period_dates]
            min_dd = float(period_drawdowns.min())
            
            zone = {
                "start": period_dates[0],
                "end": period_dates[-1],
                "depth": min_dd
            }
            drawdown_zones.append(zone)
    
    return {
        "cumulative_returns": cumulative_returns,
        "drawdown_zones": drawdown_zones
    }


def get_drawdown_recovery_data(
    returns: pd.Series,
    top_n: int = 3,
) -> list[dict]:
    """
    Prepare data for drawdown recovery timeline visualization.

    Args:
        returns: Series of returns
        top_n: Number of top drawdowns to show

    Returns:
        List of dictionaries with recovery timeline information
    """
    from core.analytics_engine.risk_metrics import calculate_top_drawdowns
    
    if returns.empty:
        return []
    
    # Get top drawdowns
    top_drawdowns = calculate_top_drawdowns(returns, top_n=top_n)
    
    if not top_drawdowns:
        return []
    
    # Calculate cumulative returns for value calculation
    cum_returns = (1 + returns).cumprod()
    
    result = []
    for i, dd in enumerate(top_drawdowns, 1):
        # Get peak value
        peak_value = cum_returns.loc[dd['start_date']] if dd['start_date'] in cum_returns.index else None
        trough_value = cum_returns.loc[dd['bottom_date']] if dd['bottom_date'] in cum_returns.index else None
        recovery_value = cum_returns.loc[dd['recovery_date']] if dd['recovery_date'] and dd['recovery_date'] in cum_returns.index else None
        
        timeline_data = {
            "number": i,
            "start_date": dd['start_date'],
            "bottom_date": dd['bottom_date'],
            "recovery_date": dd['recovery_date'],
            "depth": dd['depth'],
            "duration_days": dd['duration_days'],
            "recovery_days": dd['recovery_days'],
            "peak_value": float(peak_value) if peak_value is not None else None,
            "trough_value": float(trough_value) if trough_value is not None else None,
            "recovery_value": float(recovery_value) if recovery_value is not None else None,
        }
        
        result.append(timeline_data)
    
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

    # Ensure index is DatetimeIndex
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns = portfolio_returns.copy()
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    df = pd.DataFrame({"Portfolio": portfolio_returns})
    df['Year'] = df.index.year

    # Calculate yearly returns (compound daily returns for each year)
    yearly_portfolio = ((1 + df['Portfolio']).groupby(df['Year']).prod() - 1) * 100

    result_df = pd.DataFrame({"Portfolio": yearly_portfolio})
    result_df.index.name = "Year"

    if benchmark_returns is not None and not benchmark_returns.empty:
        # Ensure index is DatetimeIndex
        if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            benchmark_returns = benchmark_returns.copy()
            benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        bench_df = pd.DataFrame({"Benchmark": benchmark_returns})
        bench_df['Year'] = bench_df.index.year
        yearly_benchmark = ((1 + bench_df['Benchmark']).groupby(bench_df['Year']).prod() - 1) * 100
        # Align indices
        result_df['Benchmark'] = yearly_benchmark.reindex(result_df.index)
        result_df['Difference'] = result_df['Portfolio'] - result_df['Benchmark']

    return {"yearly": result_df}


def get_period_returns_comparison_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    portfolio_values: Optional[pd.Series] = None,
    benchmark_values: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate returns for various periods (MTD, YTD, 1M, 3M, 6M, 1Y, 3Y, 5Y).
    
    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns series
        portfolio_values: Optional portfolio values series (for period returns)
        benchmark_values: Optional benchmark values series
        
    Returns:
        DataFrame with period returns for portfolio and benchmark
    """
    from core.analytics_engine.performance import calculate_period_returns
    
    periods = ["MTD", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"]
    period_map = {
        "MTD": "mtd",
        "YTD": "ytd",
        "1M": "1m",
        "3M": "3m",
        "6M": "6m",
        "1Y": "1y",
        "3Y": "3y",
        "5Y": "5y",
    }
    
    portfolio_periods = {}
    benchmark_periods = {}
    
    # Calculate portfolio periods from values if available, else from returns
    if portfolio_values is not None and not portfolio_values.empty:
        try:
            port_periods = calculate_period_returns(portfolio_values)
            for display_name, key in period_map.items():
                val = port_periods.get(key, None)
                portfolio_periods[display_name] = val if val is not None else None
        except Exception as e:
            logger.warning(f"Error calculating portfolio periods from values: {e}")
            portfolio_periods = {p: None for p in periods}
    else:
        # Fallback: calculate from returns (less accurate)
        end_date_ts = pd.to_datetime(portfolio_returns.index[-1])
        end_date = end_date_ts.date()
        
        for display_name, key in period_map.items():
            if key == "mtd":
                from datetime import date as dt_date
                start_date = dt_date(end_date.year, end_date.month, 1)
            elif key == "ytd":
                from datetime import date as dt_date
                start_date = dt_date(end_date.year, 1, 1)
            elif key == "1m":
                start_date = (end_date_ts - pd.DateOffset(months=1)).date()
            elif key == "3m":
                start_date = (end_date_ts - pd.DateOffset(months=3)).date()
            elif key == "6m":
                start_date = (end_date_ts - pd.DateOffset(months=6)).date()
            elif key == "1y":
                start_date = (end_date_ts - pd.DateOffset(years=1)).date()
            elif key == "3y":
                start_date = (end_date_ts - pd.DateOffset(years=3)).date()
            elif key == "5y":
                start_date = (end_date_ts - pd.DateOffset(years=5)).date()
            else:
                start_date = None
            
            if start_date:
                # Normalize index to date for comparison
                try:
                    idx_normalized = pd.to_datetime(portfolio_returns.index).normalize()
                    start_ts = pd.Timestamp(start_date)
                    mask = idx_normalized >= start_ts
                    period_returns = portfolio_returns[mask]
                    if len(period_returns) > 0:
                        portfolio_periods[display_name] = float((1 + period_returns).prod() - 1)
                    else:
                        portfolio_periods[display_name] = None
                except Exception as e:
                    logger.warning(f"Error calculating {key} for portfolio: {e}")
                    portfolio_periods[display_name] = None
            else:
                portfolio_periods[display_name] = None
    
    # Calculate benchmark periods
    if benchmark_values is not None and not benchmark_values.empty:
        try:
            bench_periods = calculate_period_returns(benchmark_values)
            for display_name, key in period_map.items():
                val = bench_periods.get(key, None)
                benchmark_periods[display_name] = val if val is not None else None
        except Exception as e:
            logger.warning(f"Error calculating benchmark periods from values: {e}")
            benchmark_periods = {p: None for p in periods}
    elif benchmark_returns is not None and not benchmark_returns.empty:
        end_date_ts = pd.to_datetime(benchmark_returns.index[-1])
        end_date = end_date_ts.date()
        
        for display_name, key in period_map.items():
            if key == "mtd":
                from datetime import date as dt_date
                start_date = dt_date(end_date.year, end_date.month, 1)
            elif key == "ytd":
                from datetime import date as dt_date
                start_date = dt_date(end_date.year, 1, 1)
            elif key == "1m":
                start_date = (end_date_ts - pd.DateOffset(months=1)).date()
            elif key == "3m":
                start_date = (end_date_ts - pd.DateOffset(months=3)).date()
            elif key == "6m":
                start_date = (end_date_ts - pd.DateOffset(months=6)).date()
            elif key == "1y":
                start_date = (end_date_ts - pd.DateOffset(years=1)).date()
            elif key == "3y":
                start_date = (end_date_ts - pd.DateOffset(years=3)).date()
            elif key == "5y":
                start_date = (end_date_ts - pd.DateOffset(years=5)).date()
            else:
                start_date = None
            
            if start_date:
                # Normalize index to date for comparison
                try:
                    idx_normalized = pd.to_datetime(benchmark_returns.index).normalize()
                    start_ts = pd.Timestamp(start_date)
                    mask = idx_normalized >= start_ts
                    period_returns = benchmark_returns[mask]
                    if len(period_returns) > 0:
                        benchmark_periods[display_name] = float((1 + period_returns).prod() - 1)
                    else:
                        benchmark_periods[display_name] = None
                except Exception as e:
                    logger.warning(f"Error calculating {key} for benchmark: {e}")
                    benchmark_periods[display_name] = None
            else:
                benchmark_periods[display_name] = None
    else:
        benchmark_periods = {p: None for p in periods}
    
    # Build DataFrame
    data = {
        "Period": periods,
        "Portfolio": [portfolio_periods.get(p, None) for p in periods],
        "Benchmark": [benchmark_periods.get(p, None) for p in periods],
    }
    df = pd.DataFrame(data)
    # Calculate difference (only where both are not None)
    df["Difference"] = df.apply(
        lambda row: (row["Portfolio"] - row["Benchmark"]) 
        if pd.notna(row["Portfolio"]) and pd.notna(row["Benchmark"]) 
        else None, axis=1
    )
    df["Better"] = df["Difference"] > 0
    
    return {"periods": df}


def get_three_month_rolling_periods_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    top_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate best and worst 3-month rolling periods.
    
    Args:
        portfolio_returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns
        top_n: Number of top/bottom periods to return
        
    Returns:
        Dictionary with best and worst DataFrames
    """
    if portfolio_returns.empty:
        return {"best": pd.DataFrame(), "worst": pd.DataFrame()}
    
    # Calculate 3-month rolling returns (63 trading days)
    window = 63
    portfolio_rolling = portfolio_returns.rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=True)
    
    results = []
    
    for i in range(window - 1, len(portfolio_rolling)):
        if pd.isna(portfolio_rolling.iloc[i]):
            continue
            
        start_idx = i - window + 1
        end_idx = i
        
        start_date = portfolio_returns.index[start_idx]
        end_date = portfolio_returns.index[end_idx]
        port_return = float(portfolio_rolling.iloc[i])
        
        bench_return = 0.0
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                # Align benchmark dates
                aligned_bench = benchmark_returns.loc[
                    (benchmark_returns.index >= start_date) & 
                    (benchmark_returns.index <= end_date)
                ]
                if len(aligned_bench) > 0:
                    bench_return = float((1 + aligned_bench).prod() - 1)
            except Exception:
                pass
        
        results.append({
            "Start": start_date,
            "End": end_date,
            "Portfolio": port_return,
            "Benchmark": bench_return,
            "Difference": port_return - bench_return,
        })
    
    if not results:
        return {"best": pd.DataFrame(), "worst": pd.DataFrame()}
    
    df = pd.DataFrame(results)
    
    def filter_non_overlapping(df_sorted, top_n):
        """Filter out overlapping periods."""
        selected = []
        df_sorted = df_sorted.copy()
        
        while len(selected) < top_n and len(df_sorted) > 0:
            # Take the best/worst remaining
            current = df_sorted.iloc[0]
            selected.append(current)
            
            # Remove overlapping periods (periods that overlap with current)
            current_start = pd.to_datetime(current["Start"])
            current_end = pd.to_datetime(current["End"])
            
            # Remove periods that overlap with current
            mask = df_sorted.apply(
                lambda row: not (
                    (pd.to_datetime(row["Start"]) <= current_end and 
                     pd.to_datetime(row["End"]) >= current_start)
                ),
                axis=1
            )
            df_sorted = df_sorted[mask]
        
        if selected:
            return pd.DataFrame(selected)[["Start", "End", "Portfolio", "Benchmark", "Difference"]]
        else:
            return pd.DataFrame(columns=["Start", "End", "Portfolio", "Benchmark", "Difference"])
    
    # Sort by Portfolio return
    best_sorted = df.nlargest(len(df), "Portfolio")
    worst_sorted = df.nsmallest(len(df), "Portfolio")
    
    best_df = filter_non_overlapping(best_sorted, top_n)
    worst_df = filter_non_overlapping(worst_sorted, top_n)
    
    return {"best": best_df, "worst": worst_df}


def get_seasonal_analysis_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate seasonal analysis (day of week, month, quarter).
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary with DataFrames for day of week, month, quarter
    """
    if portfolio_returns.empty:
        return {
            "day_of_week": pd.DataFrame(),
            "month": pd.DataFrame(),
            "quarter": pd.DataFrame(),
        }
    
    # Day of week (0=Monday, 6=Sunday)
    portfolio_returns_df = pd.DataFrame({"returns": portfolio_returns})
    portfolio_returns_df["day_of_week"] = portfolio_returns_df.index.dayofweek
    portfolio_returns_df["month"] = portfolio_returns_df.index.month
    portfolio_returns_df["quarter"] = portfolio_returns_df.index.quarter
    
    day_avg_port = portfolio_returns_df.groupby("day_of_week")["returns"].mean() * 100
    month_avg_port = portfolio_returns_df.groupby("month")["returns"].mean() * 100
    quarter_avg_port = portfolio_returns_df.groupby("quarter")["returns"].mean() * 100
    
    day_df = pd.DataFrame({"Portfolio": day_avg_port})
    month_df = pd.DataFrame({"Portfolio": month_avg_port})
    quarter_df = pd.DataFrame({"Portfolio": quarter_avg_port})
    
    # Add benchmark if available
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_df = pd.DataFrame({"returns": benchmark_returns})
        bench_df["day_of_week"] = bench_df.index.dayofweek
        bench_df["month"] = bench_df.index.month
        bench_df["quarter"] = bench_df.index.quarter
        
        day_avg_bench = bench_df.groupby("day_of_week")["returns"].mean() * 100
        month_avg_bench = bench_df.groupby("month")["returns"].mean() * 100
        quarter_avg_bench = bench_df.groupby("quarter")["returns"].mean() * 100
        
        day_df["Benchmark"] = day_avg_bench
        month_df["Benchmark"] = month_avg_bench
        quarter_df["Benchmark"] = quarter_avg_bench
    
    # Map day numbers to names
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_df.index = [day_names[i] for i in day_df.index]
    
    # Map month numbers to names
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_df.index = [month_names[i-1] for i in month_df.index]
    
    # Map quarter numbers
    quarter_df.index = [f"Q{i}" for i in quarter_df.index]
    
    return {
        "day_of_week": day_df,
        "month": month_df,
        "quarter": quarter_df,
    }


def get_monthly_active_returns_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate monthly active returns (portfolio - benchmark).
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        DataFrame for heatmap (similar to monthly heatmap)
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {"heatmap": pd.DataFrame()}
    
    # Ensure indices are DatetimeIndex
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns = portfolio_returns.copy()
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
        benchmark_returns = benchmark_returns.copy()
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    
    # Align series
    aligned = pd.DataFrame({
        "portfolio": portfolio_returns,
        "benchmark": benchmark_returns
    }).dropna()
    
    if aligned.empty:
        return {"heatmap": pd.DataFrame()}
    
    # Calculate active returns
    aligned["active"] = aligned["portfolio"] - aligned["benchmark"]
    
    # Extract year and month
    aligned["year"] = aligned.index.year
    aligned["month"] = aligned.index.month
    
    # Calculate monthly active returns (sum of daily active returns)
    monthly_active = aligned.groupby([aligned["year"], aligned["month"]])["active"].sum() * 100
    
    # Pivot to create heatmap
    heatmap = monthly_active.unstack(level=1)
    heatmap.columns.name = "Month"
    heatmap.index.name = "Year"
    
    return {"heatmap": heatmap}


def get_win_rate_statistics_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """
    Calculate comprehensive win rate statistics.
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary with win rate metrics and rolling chart data
    """
    if portfolio_returns.empty:
        return {"stats": {}, "rolling": pd.Series()}
    
    from core.analytics_engine.advanced_metrics import calculate_win_rate_stats
    
    # Daily win rate
    daily_stats = calculate_win_rate_stats(portfolio_returns, "daily")
    
    # Weekly win rate
    weekly_returns = portfolio_returns.resample("W").apply(lambda x: (1 + x).prod() - 1)
    weekly_stats = calculate_win_rate_stats(weekly_returns, "weekly") if not weekly_returns.empty else {}
    
    # Monthly win rate
    monthly_returns = portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_stats = calculate_win_rate_stats(monthly_returns, "monthly") if not monthly_returns.empty else {}
    
    # Quarterly win rate
    quarterly_returns = portfolio_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
    quarterly_stats = calculate_win_rate_stats(quarterly_returns, "quarterly") if not quarterly_returns.empty else {}
    
    # Yearly win rate
    yearly_returns = portfolio_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
    yearly_stats = calculate_win_rate_stats(yearly_returns, "yearly") if not yearly_returns.empty else {}
    
    # Calculate rolling 12-month win rate
    # Use daily returns with 252-day window (~12 months), then calculate monthly win rate
    rolling_win_rate = pd.Series(dtype=float, index=portfolio_returns.index)
    window_days = 252  # ~12 months of trading days
    
    # Pre-calculate monthly returns for efficiency
    monthly_returns_full = portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    
    # For each day, get returns from last 12 months (252 trading days)
    for i in range(len(portfolio_returns)):
        date_idx = portfolio_returns.index[i]
        
        # Get returns from the past 252 trading days (or all available if less)
        window_start_idx = max(0, i - window_days + 1)
        window_returns = portfolio_returns.iloc[window_start_idx : i + 1]
        
        if len(window_returns) >= 20:  # At least 20 days to have meaningful data
            # Calculate monthly returns in this window
            monthly_in_window = window_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
            
            # Need at least 3 months to calculate meaningful win rate
            if len(monthly_in_window) >= 3:
                win_rate = (monthly_in_window > 0).sum() / len(monthly_in_window) * 100
                rolling_win_rate.iloc[i] = win_rate
    
    # Benchmark comparison - calculate all metrics
    bench_stats = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_daily = calculate_win_rate_stats(benchmark_returns, "daily")
        bench_weekly_returns = benchmark_returns.resample("W").apply(lambda x: (1 + x).prod() - 1)
        bench_weekly = calculate_win_rate_stats(bench_weekly_returns, "weekly") if not bench_weekly_returns.empty else {}
        bench_monthly_returns = benchmark_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        bench_monthly = calculate_win_rate_stats(bench_monthly_returns, "monthly") if not bench_monthly_returns.empty else {}
        bench_quarterly_returns = benchmark_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
        bench_quarterly = calculate_win_rate_stats(bench_quarterly_returns, "quarterly") if not bench_quarterly_returns.empty else {}
        bench_yearly_returns = benchmark_returns.resample("Y").apply(lambda x: (1 + x).prod() - 1)
        bench_yearly = calculate_win_rate_stats(bench_yearly_returns, "yearly") if not bench_yearly_returns.empty else {}
        
        bench_stats = {
            "win_rate_daily": bench_daily.get("win_rate_daily", 0) if bench_daily else 0,
            "win_rate_weekly": bench_weekly.get("win_rate_weekly", 0) if bench_weekly else 0,
            "win_rate_monthly": bench_monthly.get("win_rate_monthly", 0) if bench_monthly else 0,
            "win_rate_quarterly": bench_quarterly.get("win_rate_quarterly", 0) if bench_quarterly else 0,
            "win_rate_yearly": bench_yearly.get("win_rate_yearly", 0) if bench_yearly else 0,
            "avg_win_daily": bench_daily.get("avg_win_daily", 0) if bench_daily else 0,
            "avg_loss_daily": bench_daily.get("avg_loss_daily", 0) if bench_daily else 0,
            "avg_win_monthly": bench_monthly.get("avg_win_monthly", 0) if bench_monthly else 0,
            "avg_loss_monthly": bench_monthly.get("avg_loss_monthly", 0) if bench_monthly else 0,
            "best_daily": bench_daily.get("best_daily", 0) if bench_daily else 0,
            "worst_daily": bench_daily.get("worst_daily", 0) if bench_daily else 0,
            "best_monthly": bench_monthly.get("best_monthly", 0) if bench_monthly else 0,
            "worst_monthly": bench_monthly.get("worst_monthly", 0) if bench_monthly else 0,
        }
    
    stats = {
        "portfolio": {
            "win_days_pct": daily_stats.get("win_rate_daily", 0) * 100 if daily_stats and daily_stats.get("win_rate_daily") is not None else 0,
            "win_weeks_pct": weekly_stats.get("win_rate_weekly", 0) * 100 if weekly_stats and weekly_stats.get("win_rate_weekly") is not None else 0,
            "win_months_pct": monthly_stats.get("win_rate_monthly", 0) * 100 if monthly_stats and monthly_stats.get("win_rate_monthly") is not None else 0,
            "win_quarters_pct": quarterly_stats.get("win_rate_quarterly", 0) * 100 if quarterly_stats and quarterly_stats.get("win_rate_quarterly") is not None else 0,
            "win_years_pct": yearly_stats.get("win_rate_yearly", 0) * 100 if yearly_stats and yearly_stats.get("win_rate_yearly") is not None else 0,
            "avg_up_day": daily_stats.get("avg_win_daily", 0) * 100 if daily_stats and daily_stats.get("avg_win_daily") is not None else 0,
            "avg_down_day": daily_stats.get("avg_loss_daily", 0) * 100 if daily_stats and daily_stats.get("avg_loss_daily") is not None else 0,
            "avg_up_month": monthly_stats.get("avg_win_monthly", 0) * 100 if monthly_stats and monthly_stats.get("avg_win_monthly") is not None else 0,
            "avg_down_month": monthly_stats.get("avg_loss_monthly", 0) * 100 if monthly_stats and monthly_stats.get("avg_loss_monthly") is not None else 0,
            "best_day": daily_stats.get("best_daily", 0) * 100 if daily_stats and daily_stats.get("best_daily") is not None else 0,
            "worst_day": daily_stats.get("worst_daily", 0) * 100 if daily_stats and daily_stats.get("worst_daily") is not None else 0,
            "best_month": monthly_stats.get("best_monthly", 0) * 100 if monthly_stats and monthly_stats.get("best_monthly") is not None else 0,
            "worst_month": monthly_stats.get("worst_monthly", 0) * 100 if monthly_stats and monthly_stats.get("worst_monthly") is not None else 0,
        },
        "benchmark": bench_stats,
    }
    
    return {"stats": stats, "rolling": rolling_win_rate}


def get_outlier_analysis_data(
    portfolio_returns: pd.Series,
    outlier_threshold: float = 2.0,
) -> Dict[str, any]:
    """
    Analyze outliers in returns (beyond N standard deviations).
    
    Args:
        portfolio_returns: Portfolio returns
        outlier_threshold: Number of standard deviations for outlier definition
        
    Returns:
        Dictionary with outlier statistics and scatter plot data
    """
    if portfolio_returns.empty:
        return {
            "stats": {},
            "outliers": pd.DataFrame(),
        }
    
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    if std_return == 0:
        return {
            "stats": {},
            "outliers": pd.DataFrame(),
        }
    
    # Identify outliers
    z_scores = (portfolio_returns - mean_return) / std_return
    outliers_mask = abs(z_scores) > outlier_threshold
    
    outlier_returns = portfolio_returns[outliers_mask]
    
    # Separate wins and losses
    outlier_wins = outlier_returns[outlier_returns > 0]
    outlier_losses = outlier_returns[outlier_returns < 0]
    
    # Normal returns
    normal_returns = portfolio_returns[~outliers_mask]
    normal_wins = normal_returns[normal_returns > 0]
    normal_losses = normal_returns[normal_returns < 0]
    
    # Calculate ratios
    outlier_win_ratio = 0.0
    outlier_loss_ratio = 0.0
    
    if len(normal_wins) > 0 and len(outlier_wins) > 0:
        avg_outlier_win = outlier_wins.mean()
        avg_normal_win = normal_wins.mean()
        outlier_win_ratio = avg_outlier_win / avg_normal_win if avg_normal_win != 0 else 0.0
    
    if len(normal_losses) > 0 and len(outlier_losses) > 0:
        avg_outlier_loss = abs(outlier_losses.mean())
        avg_normal_loss = abs(normal_losses.mean())
        outlier_loss_ratio = avg_outlier_loss / avg_normal_loss if avg_normal_loss != 0 else 0.0
    
    # Prepare scatter plot data
    outlier_df = pd.DataFrame({
        "Date": outlier_returns.index,
        "Return": outlier_returns.values * 100,
        "ZScore": z_scores[outliers_mask].values,
    })
    
    stats = {
        "outlier_win_ratio": float(outlier_win_ratio),
        "outlier_loss_ratio": float(outlier_loss_ratio),
        "outlier_count": int(outliers_mask.sum()),
        "total_count": len(portfolio_returns),
        "mean": float(mean_return),
        "std": float(std_return),
        "threshold": outlier_threshold,
    }
    
    return {
        "stats": stats,
        "outliers": outlier_df,
        "z_scores": z_scores,
        "mean": mean_return,
        "std": std_return,
    }


def get_statistical_tests_data(
    portfolio_returns: pd.Series,
) -> Dict[str, any]:
    """
    Calculate statistical tests for distribution normality.
    
    Args:
        portfolio_returns: Portfolio returns
        
    Returns:
        Dictionary with test results including sample size
    """
    if portfolio_returns.empty:
        return {
            "shapiro_wilk": {"statistic": 0.0, "pvalue": 1.0},
            "jarque_bera": {"statistic": 0.0, "pvalue": 1.0},
            "skewness": 0.0,
            "kurtosis": 0.0,
            "sample_size": 0,
        }
    
    returns_array = portfolio_returns.dropna().values
    total_size = len(returns_array)
    
    if total_size < 3:
        return {
            "shapiro_wilk": {"statistic": 0.0, "pvalue": 1.0},
            "jarque_bera": {"statistic": 0.0, "pvalue": 1.0},
            "skewness": 0.0,
            "kurtosis": 0.0,
            "sample_size": total_size,
        }
    
    # Shapiro-Wilk test (max 5000 samples, use random sampling if needed)
    sample_size = min(total_size, 5000)
    if total_size > 5000:
        # Use random sampling instead of first N elements
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        sample_indices = rng.choice(total_size, size=sample_size, replace=False)
        sample_data = returns_array[sample_indices]
    else:
        sample_data = returns_array
    
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
    
    # Jarque-Bera test (uses all data)
    jb_stat, jb_p = stats.jarque_bera(returns_array)
    
    # Skewness and Kurtosis
    skewness = float(stats.skew(returns_array))
    kurtosis = float(stats.kurtosis(returns_array))
    
    return {
        "shapiro_wilk": {"statistic": float(shapiro_stat), "pvalue": float(shapiro_p), "sample_size": sample_size},
        "jarque_bera": {"statistic": float(jb_stat), "pvalue": float(jb_p)},
        "skewness": skewness,
        "kurtosis": kurtosis,
        "sample_size": total_size,
    }


def get_capture_ratio_data(
    up_capture: Optional[float],
    down_capture: Optional[float],
) -> Optional[Dict[str, float]]:
    """
    Prepare data for capture ratio visualization.
    
    Args:
        up_capture: Up capture ratio (e.g., 1.05 = 105%)
        down_capture: Down capture ratio (e.g., 0.85 = 85%)
        
    Returns:
        Dictionary with capture ratio data or None
    """
    if up_capture is None or down_capture is None:
        return None
    
    capture_ratio = up_capture / down_capture if down_capture != 0 else 0.0
    
    return {
        "up_capture": float(up_capture),
        "down_capture": float(down_capture),
        "capture_ratio": float(capture_ratio),
    }


def get_risk_return_scatter_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for risk/return scatter plot.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with scatter plot data or None
    """
    if portfolio_returns.empty:
        return None
    
    try:
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.risk_metrics import calculate_volatility
        
        # Calculate portfolio metrics
        port_return = calculate_annualized_return(portfolio_returns, periods_per_year)
        port_vol_dict = calculate_volatility(portfolio_returns, periods_per_year)
        port_vol = port_vol_dict.get("annual", 0.0) if isinstance(port_vol_dict, dict) else port_vol_dict
        
        result = {
            "portfolio": {
                "return": float(port_return),
                "volatility": float(port_vol),
                "label": "Portfolio",
            },
            "risk_free_rate": float(risk_free_rate),
        }
        
        # Calculate benchmark metrics if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_return = calculate_annualized_return(benchmark_returns, periods_per_year)
            bench_vol_dict = calculate_volatility(benchmark_returns, periods_per_year)
            bench_vol = bench_vol_dict.get("annual", 0.0) if isinstance(bench_vol_dict, dict) else bench_vol_dict
            
            result["benchmark"] = {
                "return": float(bench_return),
                "volatility": float(bench_vol),
                "label": "Benchmark",
            }
        
        # Calculate Capital Market Line (CML) points
        # CML: E(R) = Rf + Sharpe * σ
        if port_vol > 0:
            sharpe = (port_return - risk_free_rate) / port_vol
            # Generate line points from 0 to max volatility * 1.2
            max_vol = port_vol * 1.2
            if benchmark_returns is not None and not benchmark_returns.empty:
                max_vol = max(port_vol, bench_vol) * 1.2
            
            vol_points = np.linspace(0, max_vol, 50)
            return_points = risk_free_rate + sharpe * vol_points
            
            result["cml"] = {
                "volatility": vol_points.tolist(),
                "return": return_points.tolist(),
            }
        
        return result
        
    except Exception as e:
        logger.warning(f"Error preparing risk/return scatter data: {e}")
        return None


def get_rolling_var_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 63,
    confidence_level: float = 0.95,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for rolling VaR chart.

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        window: Rolling window size in days
        confidence_level: Confidence level (0.90, 0.95, or 0.99)

    Returns:
        Dictionary with rolling VaR series and statistics or None
    """
    if portfolio_returns.empty:
        return None

    try:
        from core.analytics_engine.risk_metrics import calculate_var

        # Calculate rolling VaR
        portfolio_var = portfolio_returns.rolling(window=window).apply(
            lambda x: calculate_var(x, confidence_level, "historical")
            if len(x) >= window else np.nan,
            raw=False,
        )

        result = {
            "portfolio": portfolio_var,
            "window": window,
            "confidence_level": confidence_level,
            "portfolio_stats": {
                "avg": float(portfolio_var.mean()),
                "median": float(portfolio_var.median()),
                "min": float(portfolio_var.min()),
                "max": float(portfolio_var.max()),
            },
        }

        # Calculate benchmark rolling VaR if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned_benchmark = benchmark_returns.reindex(
                portfolio_returns.index, method="ffill"
            ).fillna(0)

            benchmark_var = aligned_benchmark.rolling(
                window=window
            ).apply(
                lambda x: calculate_var(x, confidence_level, "historical")
                if len(x) >= window else np.nan,
                raw=False,
            )

            result["benchmark"] = benchmark_var
            result["benchmark_stats"] = {
                "avg": float(benchmark_var.mean()),
                "median": float(benchmark_var.median()),
                "min": float(benchmark_var.min()),
                "max": float(benchmark_var.max()),
            }

        return result

    except Exception as e:
        logger.warning(f"Error preparing rolling VaR data: {e}")
        return None


def get_rolling_volatility_data(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for rolling volatility chart with statistics table.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Optional benchmark returns
        window: Rolling window size in days
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with rolling volatility series and statistics or None
    """
    if portfolio_returns.empty:
        return None
    
    try:
        # Calculate rolling volatility (annualized)
        portfolio_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(periods_per_year)
        
        result = {
            "portfolio": portfolio_vol,
            "window": window,
            "portfolio_stats": {
                "avg": float(portfolio_vol.mean()),
                "median": float(portfolio_vol.median()),
                "min": float(portfolio_vol.min()),
                "max": float(portfolio_vol.max()),
            }
        }
        
        # Calculate benchmark rolling volatility if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned_benchmark = benchmark_returns.reindex(
                portfolio_returns.index,
                method="ffill"
            ).fillna(0)
            
            benchmark_vol = aligned_benchmark.rolling(window=window).std() * np.sqrt(periods_per_year)
            
            result["benchmark"] = benchmark_vol
            result["benchmark_stats"] = {
                "avg": float(benchmark_vol.mean()),
                "median": float(benchmark_vol.median()),
                "min": float(benchmark_vol.min()),
                "max": float(benchmark_vol.max()),
            }
        
        return result
        
    except Exception as e:
        logger.warning(f"Error preparing rolling volatility data: {e}")
        return None


def get_rolling_beta_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for rolling beta chart with zones.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size in days
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with rolling beta series or None
    """
    if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return None
    
    try:
        # Align benchmark with portfolio
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index,
            method="ffill"
        ).fillna(0)
        
        # Calculate rolling beta
        def calculate_beta(port_window, bench_window):
            if len(port_window) < 2 or len(bench_window) < 2:
                return np.nan
            
            cov = np.cov(port_window, bench_window)[0, 1]
            var = np.var(bench_window, ddof=1)
            
            if var <= 0:
                return np.nan
            
            return cov / var
        
        rolling_beta = pd.Series(index=portfolio_returns.index, dtype=float)
        
        for i in range(window, len(portfolio_returns) + 1):
            port_window = portfolio_returns.iloc[i-window:i].values
            bench_window = aligned_benchmark.iloc[i-window:i].values
            rolling_beta.iloc[i-1] = calculate_beta(port_window, bench_window)
        
        return {
            "beta": rolling_beta,
            "window": window,
        }
        
    except Exception as e:
        logger.warning(f"Error preparing rolling beta data: {e}")
        return None


def get_rolling_alpha_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for rolling alpha chart.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size in days
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with rolling alpha series or None
    """
    if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return None
    
    try:
        # Align benchmark with portfolio
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index,
            method="ffill"
        ).fillna(0)
        
        # Calculate rolling alpha (annualized)
        def calculate_alpha(port_window, bench_window):
            if len(port_window) < 2 or len(bench_window) < 2:
                return np.nan
            
            # Calculate beta
            cov = np.cov(port_window, bench_window)[0, 1]
            var = np.var(bench_window, ddof=1)
            
            if var <= 0:
                beta = 0
            else:
                beta = cov / var
            
            # Calculate mean returns (annualized)
            port_mean_annual = port_window.mean() * periods_per_year
            bench_mean_annual = bench_window.mean() * periods_per_year
            
            # Alpha = Portfolio Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
            alpha = port_mean_annual - (risk_free_rate + beta * (bench_mean_annual - risk_free_rate))
            
            return alpha
        
        rolling_alpha = pd.Series(index=portfolio_returns.index, dtype=float)
        
        for i in range(window, len(portfolio_returns) + 1):
            port_window = portfolio_returns.iloc[i-window:i].values
            bench_window = aligned_benchmark.iloc[i-window:i].values
            rolling_alpha.iloc[i-1] = calculate_alpha(port_window, bench_window)
        
        return {
            "alpha": rolling_alpha,
            "window": window,
        }
        
    except Exception as e:
        logger.warning(f"Error preparing rolling alpha data: {e}")
        return None


def get_rolling_active_return_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for rolling active return area chart.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size in days
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with rolling active return series and statistics or None
    """
    if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return None
    
    try:
        # Align benchmark with portfolio
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index,
            method="ffill"
        ).fillna(0)
        
        # Calculate rolling active return (annualized)
        rolling_port_return = portfolio_returns.rolling(window=window).mean() * periods_per_year
        rolling_bench_return = aligned_benchmark.rolling(window=window).mean() * periods_per_year
        rolling_active_return = rolling_port_return - rolling_bench_return
        
        # Calculate statistics
        avg_active = rolling_active_return.mean()
        positive_periods = (rolling_active_return > 0).sum()
        total_periods = rolling_active_return.notna().sum()
        pct_positive = (positive_periods / total_periods * 100) if total_periods > 0 else 0
        
        return {
            "active_return": rolling_active_return,
            "window": window,
            "stats": {
                "avg": float(avg_active),
                "pct_positive": float(pct_positive),
                "max": float(rolling_active_return.max()),
                "min": float(rolling_active_return.min()),
            }
        }
        
    except Exception as e:
        logger.warning(f"Error preparing rolling active return data: {e}")
        return None


def get_bull_bear_analysis_data(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for bull/bear market analysis.
    
    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Benchmark returns
        window: Rolling window size for beta calculation
        periods_per_year: Number of trading periods per year
        
    Returns:
        Dictionary with bull/bear analysis data or None
    """
    if portfolio_returns.empty or benchmark_returns is None or benchmark_returns.empty:
        return None
    
    try:
        # Align benchmark with portfolio
        aligned_benchmark = benchmark_returns.reindex(
            portfolio_returns.index,
            method="ffill"
        ).fillna(0)
        
        # Identify bull and bear periods (benchmark return > 0 = bull)
        bull_mask = aligned_benchmark > 0
        bear_mask = aligned_benchmark <= 0
        
        # Calculate returns for bull and bear periods
        bull_port_returns = portfolio_returns[bull_mask]
        bull_bench_returns = aligned_benchmark[bull_mask]
        bear_port_returns = portfolio_returns[bear_mask]
        bear_bench_returns = aligned_benchmark[bear_mask]
        
        # Calculate AVERAGE DAILY return during bull/bear periods
        # Shows typical daily performance in different market conditions
        def calculate_avg_daily_return(returns_series):
            if len(returns_series) == 0:
                return 0.0
            
            # Use median (more robust than mean)
            median_daily = returns_series.median()
            
            # Convert to percentage
            return float(median_daily * 100)
        
        bull_port_cum = calculate_avg_daily_return(bull_port_returns)
        bull_bench_cum = calculate_avg_daily_return(bull_bench_returns)
        bear_port_cum = calculate_avg_daily_return(bear_port_returns)
        bear_bench_cum = calculate_avg_daily_return(bear_bench_returns)
        
        # Calculate beta for each period
        def calc_beta(port, bench):
            if len(port) < 2 or len(bench) < 2:
                return 0.0
            cov = np.cov(port.values, bench.values)[0, 1]
            var = np.var(bench.values, ddof=1)
            return cov / var if var > 0 else 0.0
        
        bull_beta = calc_beta(bull_port_returns, bull_bench_returns)
        bear_beta = calc_beta(bear_port_returns, bear_bench_returns)
        
        # Calculate rolling beta for bull/bear periods
        rolling_bull_beta = pd.Series(index=portfolio_returns.index, dtype=float)
        rolling_bear_beta = pd.Series(index=portfolio_returns.index, dtype=float)
        
        for i in range(window, len(portfolio_returns) + 1):
            port_window = portfolio_returns.iloc[i-window:i]
            bench_window = aligned_benchmark.iloc[i-window:i]
            
            bull_mask_window = bench_window > 0
            bear_mask_window = bench_window <= 0
            
            if bull_mask_window.sum() >= 2:
                rolling_bull_beta.iloc[i-1] = calc_beta(
                    port_window[bull_mask_window],
                    bench_window[bull_mask_window]
                )
            
            if bear_mask_window.sum() >= 2:
                rolling_bear_beta.iloc[i-1] = calc_beta(
                    port_window[bear_mask_window],
                    bench_window[bear_mask_window]
                )
        
        return {
            "bull": {
                "portfolio_return": float(bull_port_cum),
                "benchmark_return": float(bull_bench_cum),
                "beta": float(bull_beta),
                "difference": float(bull_port_cum - bull_bench_cum),
            },
            "bear": {
                "portfolio_return": float(bear_port_cum),
                "benchmark_return": float(bear_bench_cum),
                "beta": float(bear_beta),
                "difference": float(bear_port_cum - bear_bench_cum),
            },
            "rolling_beta": {
                "bull": rolling_bull_beta,
                "bear": rolling_bear_beta,
                "window": window,
            }
        }
        
    except Exception as e:
        logger.warning(f"Error preparing bull/bear analysis data: {e}")
        return None


def get_asset_metrics_data(
    positions: list,
    price_data: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Prepare extended asset information table data.
    
    Args:
        positions: List of Position objects
        price_data: Optional DataFrame with recent price data
        
    Returns:
        DataFrame with columns: ticker, weight, name, sector, industry, 
                                currency, price, change_pct
    """
    if not positions:
        return None
    
    try:
        import yfinance as yf
        
        # Build base data from positions
        data = []
        
        for pos in positions:
            ticker = pos.ticker
            weight = pos.weight_target if hasattr(pos, 'weight_target') and pos.weight_target else 0.0
            shares = pos.shares if hasattr(pos, 'shares') else 0.0
            
            # Skip CASH special handling
            if ticker == "CASH":
                data.append({
                    "ticker": ticker,
                    "weight": weight * 100,
                    "shares": shares,
                    "name": "Cash Position",
                    "sector": "Cash",
                    "industry": "Cash",
                    "currency": "USD",
                    "price": 1.0,
                    "change_pct": 0.0,
                })
                continue
            
            # Fetch ticker info from yfinance
            try:
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Get price change from price_data if available
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
                change_pct = 0.0
                
                if price_data is not None and ticker in price_data.columns:
                    prices = price_data[ticker].dropna()
                    if len(prices) >= 2:
                        current_price = float(prices.iloc[-1])
                        prev_price = float(prices.iloc[-2])
                        if prev_price > 0:
                            change_pct = ((current_price - prev_price) / prev_price) * 100
                
                data.append({
                    "ticker": ticker,
                    "weight": weight * 100,
                    "shares": shares,
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "sector": info.get("sector") or "N/A",
                    "industry": info.get("industry") or "N/A",
                    "currency": info.get("currency") or "USD",
                    "price": float(current_price),
                    "change_pct": float(change_pct),
                })
                
            except Exception as e:
                logger.warning(f"Error fetching info for {ticker}: {e}")
                # Fallback data
                data.append({
                    "ticker": ticker,
                    "weight": weight * 100,
                    "shares": shares,
                    "name": ticker,
                    "sector": "N/A",
                    "industry": "N/A",
                    "currency": "USD",
                    "price": 0.0,
                    "change_pct": 0.0,
                })
        
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        logger.warning(f"Error preparing asset metrics data: {e}")
        return None


def get_asset_impact_on_return_data(
    positions: list,
    price_data: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate each asset's contribution to total portfolio return.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data (dates × tickers)
        start_date: Optional start date for analysis
        end_date: Optional end date for analysis
        
    Returns:
        Dictionary with asset contributions data
    """
    if not positions or price_data.empty:
        return None
    
    try:
        if price_data.empty:
            logger.warning("Price data is empty")
            return None
        
        # Filter by date range if provided
        filtered_data = price_data.copy()
        if start_date is not None:
            # Convert date to datetime if needed for comparison
            if isinstance(filtered_data.index, pd.DatetimeIndex):
                start_dt = pd.Timestamp(start_date)
                filtered_data = filtered_data[filtered_data.index >= start_dt]
            else:
                # Try to convert index to datetime
                try:
                    filtered_data.index = pd.to_datetime(filtered_data.index)
                    start_dt = pd.Timestamp(start_date)
                    filtered_data = filtered_data[filtered_data.index >= start_dt]
                except Exception:
                    logger.warning(
                        f"Could not filter by start_date {start_date}"
                    )
        
        if end_date is not None:
            if isinstance(filtered_data.index, pd.DatetimeIndex):
                end_dt = pd.Timestamp(end_date)
                filtered_data = filtered_data[filtered_data.index <= end_dt]
            else:
                try:
                    filtered_data.index = pd.to_datetime(filtered_data.index)
                    end_dt = pd.Timestamp(end_date)
                    filtered_data = filtered_data[filtered_data.index <= end_dt]
                except Exception:
                    logger.warning(
                        f"Could not filter by end_date {end_date}"
                    )
        
        if filtered_data.empty:
            logger.warning("Price data is empty after date filtering")
            return None
        
        # Calculate weights based on initial portfolio value
        total_shares_value = {}
        for pos in positions:
            ticker = pos.ticker
            if ticker == "CASH":
                # Cash always 1.0
                total_shares_value[ticker] = pos.shares * 1.0
            elif ticker in filtered_data.columns:
                # Get first non-null price
                prices_series = filtered_data[ticker].dropna()
                if not prices_series.empty:
                    first_price = float(prices_series.iloc[0])
                    total_shares_value[ticker] = pos.shares * first_price
                else:
                    logger.warning(f"No price data for {ticker}")
                    continue
            else:
                logger.warning(
                    f"Ticker {ticker} not in price_data columns"
                )
                continue
        
        if not total_shares_value:
            logger.warning("No valid positions with price data")
            return None
        
        total_value = sum(total_shares_value.values())
        if total_value == 0:
            logger.warning("Total portfolio value is zero")
            return None
        
        weights = {
            ticker: value / total_value 
            for ticker, value in total_shares_value.items()
        }
        
        # Calculate returns for each asset
        asset_returns = {}
        for ticker in weights.keys():
            if ticker == "CASH":
                asset_returns[ticker] = 0.0  # Cash has no return
            elif ticker in filtered_data.columns:
                prices_series = filtered_data[ticker].dropna()
                if len(prices_series) >= 2:
                    first_price = float(prices_series.iloc[0])
                    last_price = float(prices_series.iloc[-1])
                    if first_price > 0:
                        total_return = (last_price / first_price) - 1
                        asset_returns[ticker] = float(total_return)
                    else:
                        asset_returns[ticker] = 0.0
                else:
                    asset_returns[ticker] = 0.0
            else:
                asset_returns[ticker] = 0.0
        
        # Calculate weighted contribution
        contributions = {}
        for ticker in weights.keys():
            weighted_return = (
                weights[ticker] * asset_returns.get(ticker, 0.0)
            )
            contributions[ticker] = float(weighted_return * 100)
        
        # Sort by contribution
        sorted_tickers = sorted(
            contributions.keys(), 
            key=lambda t: contributions[t], 
            reverse=True
        )
        
        return {
            "tickers": sorted_tickers,
            "contributions": [
                contributions[t] for t in sorted_tickers
            ],
            "returns": [
                asset_returns.get(t, 0.0) * 100 
                for t in sorted_tickers
            ],
            "weights": [weights[t] * 100 for t in sorted_tickers],
        }
        
    except Exception as e:
        logger.warning(
            f"Error calculating asset impact on return: {e}"
        )
        import traceback
        logger.warning(traceback.format_exc())
        return None


def get_asset_impact_on_risk_data(
    positions: list,
    price_data: pd.DataFrame,
) -> Optional[Dict[str, any]]:
    """
    Calculate each asset's contribution to total portfolio risk (volatility).
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data (dates × tickers)
        
    Returns:
        Dictionary with risk contribution data
    """
    if not positions or price_data.empty:
        return None
    
    try:
        # Calculate weights
        total_shares_value = {}
        for pos in positions:
            ticker = pos.ticker
            if ticker in price_data.columns:
                first_price = price_data[ticker].dropna().iloc[0] if not price_data[ticker].dropna().empty else 1.0
                total_shares_value[ticker] = pos.shares * first_price
        
        total_value = sum(total_shares_value.values())
        weights = {ticker: value / total_value for ticker, value in total_shares_value.items()}
        
        # Calculate returns for each asset
        returns_data = {}
        for ticker in weights.keys():
            if ticker in price_data.columns:
                prices = price_data[ticker].dropna()
                if len(prices) >= 2:
                    returns = prices.pct_change().dropna()
                    returns_data[ticker] = returns
        
        # Build returns matrix
        common_dates = None
        for returns in returns_data.values():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)
        
        if common_dates is None or len(common_dates) < 2:
            return None
        
        # Align returns
        aligned_returns = pd.DataFrame({
            ticker: returns_data[ticker].reindex(common_dates)
            for ticker in returns_data.keys()
        })
        
        # Calculate covariance matrix
        cov_matrix = aligned_returns.cov() * TRADING_DAYS_PER_YEAR
        
        # Calculate portfolio variance
        weights_array = np.array([weights.get(ticker, 0.0) for ticker in aligned_returns.columns])
        portfolio_variance = np.dot(weights_array, np.dot(cov_matrix.values, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate marginal contribution to risk (MCR)
        mcr = np.dot(cov_matrix.values, weights_array) / portfolio_volatility
        
        # Calculate contribution to risk (CTR)
        ctr = weights_array * mcr
        
        # Calculate percentage contribution
        risk_contribution_pct = (ctr / portfolio_volatility) * 100
        
        # Prepare results
        risk_contributions = {}
        for i, ticker in enumerate(aligned_returns.columns):
            risk_contributions[ticker] = float(risk_contribution_pct[i])
        
        # Sort by contribution
        sorted_tickers = sorted(risk_contributions.keys(), key=lambda t: risk_contributions[t], reverse=True)
        
        return {
            "tickers": sorted_tickers,
            "risk_contributions": [risk_contributions[t] for t in sorted_tickers],
            "weights": [weights[t] * 100 for t in sorted_tickers],
        }
        
    except Exception as e:
        logger.warning(f"Error calculating asset impact on risk: {e}")
        return None


def get_risk_vs_weight_comparison_data(
    positions: list,
    price_data: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[Dict[str, any]]:
    """
    Prepare data for comparing risk contribution, return impact vs portfolio weight.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        start_date: Optional start date for return calculation
        end_date: Optional end date for return calculation
        
    Returns:
        Dictionary with comparison data: tickers, risk_impact, return_impact, weights
    """
    # Get risk impact data
    risk_data = get_asset_impact_on_risk_data(positions, price_data)
    
    if risk_data is None:
        return None
    
    # Get return impact data
    return_data = get_asset_impact_on_return_data(
        positions, price_data, start_date, end_date
    )
    
    # Create dictionaries for easy lookup
    risk_dict = dict(zip(risk_data["tickers"], risk_data["risk_contributions"]))
    weight_dict = dict(zip(risk_data["tickers"], risk_data["weights"]))
    
    # Get return impact if available
    return_dict = {}
    if return_data and return_data.get("tickers"):
        return_dict = dict(
            zip(return_data["tickers"], return_data["contributions"])
        )
    
    # Align all tickers (use risk_data tickers as base)
    tickers = risk_data["tickers"]
    risk_impact = []
    return_impact = []
    weights = []
    
    for ticker in tickers:
        risk_impact.append(risk_dict.get(ticker, 0.0))
        weights.append(weight_dict.get(ticker, 0.0))
        # Return impact: use absolute value for comparison
        return_val = return_dict.get(ticker, 0.0)
        return_impact.append(abs(return_val))  # Use absolute for comparison
    
    return {
        "tickers": tickers,
        "risk_impact": risk_impact,
        "return_impact": return_impact,
        "weights": weights,
    }


def get_diversification_coefficient_data(
    positions: list,
    price_data: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    """
    Calculate diversification coefficient.
    
    Formula: Weighted sum of individual volatilities / Portfolio volatility
    
    A value > 1.0 indicates positive diversification effect.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        
    Returns:
        Dictionary with diversification metrics
    """
    if not positions or price_data.empty:
        return None
    
    try:
        # Calculate weights
        total_shares_value = {}
        for pos in positions:
            ticker = pos.ticker
            if ticker in price_data.columns:
                first_price = price_data[ticker].dropna().iloc[0] if not price_data[ticker].dropna().empty else 1.0
                total_shares_value[ticker] = pos.shares * first_price
        
        total_value = sum(total_shares_value.values())
        weights = {ticker: value / total_value for ticker, value in total_shares_value.items()}
        
        # Calculate individual volatilities
        individual_vols = {}
        returns_data = {}
        
        for ticker in weights.keys():
            if ticker in price_data.columns:
                prices = price_data[ticker].dropna()
                if len(prices) >= 2:
                    returns = prices.pct_change().dropna()
                    returns_data[ticker] = returns
                    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                    individual_vols[ticker] = vol
        
        # Calculate weighted sum of individual volatilities
        weighted_sum_vols = sum(weights[ticker] * individual_vols.get(ticker, 0.0) 
                                for ticker in weights.keys())
        
        # Calculate portfolio returns
        common_dates = None
        for returns in returns_data.values():
            if common_dates is None:
                common_dates = returns.index
            else:
                common_dates = common_dates.intersection(returns.index)
        
        if common_dates is None or len(common_dates) < 2:
            return None
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=common_dates)
        for ticker in returns_data.keys():
            aligned_returns = returns_data[ticker].reindex(common_dates).fillna(0)
            portfolio_returns += weights[ticker] * aligned_returns
        
        # Calculate portfolio volatility
        portfolio_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Calculate diversification coefficient
        if portfolio_vol > 0:
            div_coefficient = weighted_sum_vols / portfolio_vol
        else:
            div_coefficient = 1.0
        
        # Calculate volatility reduction
        vol_reduction = (div_coefficient - 1.0) * 100
        
        return {
            "diversification_coefficient": float(div_coefficient),
            "weighted_sum_volatilities": float(weighted_sum_vols),
            "portfolio_volatility": float(portfolio_vol),
            "volatility_reduction_pct": float(vol_reduction),
            "is_diversified": div_coefficient > 1.0,
        }
        
    except Exception as e:
        logger.warning(f"Error calculating diversification coefficient: {e}")
        return None


def get_correlation_matrix_data(
    positions: list,
    price_data: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate correlation matrix for all assets + benchmark.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data (dates x tickers)
        benchmark_returns: Optional benchmark returns series
        
    Returns:
        Dictionary with correlation matrix and metadata
    """
    if not positions or price_data.empty:
        return None
    
    try:
        # Calculate returns for all assets
        returns_df = price_data.pct_change().dropna()
        
        if returns_df.empty or len(returns_df) < 2:
            return None
        
        # Exclude CASH from correlation matrix (it's always 1.0, no real correlation)
        if "CASH" in returns_df.columns:
            returns_df = returns_df.drop(columns=["CASH"])
        
        if returns_df.empty or len(returns_df.columns) < 2:
            return None
        
        # Add benchmark if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Align benchmark with asset returns
            aligned_benchmark = benchmark_returns.reindex(
                returns_df.index, method="ffill"
            ).dropna()
            
            if not aligned_benchmark.empty:
                # Ensure we have common dates
                common_dates = returns_df.index.intersection(aligned_benchmark.index)
                if len(common_dates) >= 2:
                    returns_df = returns_df.loc[common_dates]
                    aligned_benchmark = aligned_benchmark.loc[common_dates]
                    returns_df["SPY"] = aligned_benchmark
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Get ticker list
        tickers = list(correlation_matrix.columns)
        
        return {
            "correlation_matrix": correlation_matrix,
            "tickers": tickers,
            "n_assets": len(tickers),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating correlation matrix: {e}")
        return None


def get_correlation_statistics_data(
    correlation_matrix: pd.DataFrame,
) -> Optional[Dict[str, any]]:
    """
    Calculate correlation statistics from correlation matrix.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        
    Returns:
        Dictionary with correlation statistics
    """
    if correlation_matrix.empty:
        return None
    
    try:
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_triangle = correlation_matrix.where(mask).stack()
        
        # Calculate statistics
        correlations = upper_triangle.values
        avg_correlation = float(np.nanmean(correlations))
        median_correlation = float(np.nanmedian(correlations))
        min_correlation = float(np.nanmin(correlations))
        max_correlation = float(np.nanmax(correlations))
        
        # Find min/max pairs
        min_idx = np.nanargmin(correlations)
        max_idx = np.nanargmax(correlations)
        min_pair = upper_triangle.index[min_idx]
        max_pair = upper_triangle.index[max_idx]
        
        # Count high/low correlations
        high_corr_count = int(np.sum(correlations > 0.8))
        low_corr_count = int(np.sum(correlations < 0.2))
        
        return {
            "average_correlation": avg_correlation,
            "median_correlation": median_correlation,
            "min_correlation": min_correlation,
            "max_correlation": max_correlation,
            "min_pair": (min_pair[0], min_pair[1]),
            "max_pair": (max_pair[0], max_pair[1]),
            "high_corr_count": high_corr_count,
            "low_corr_count": low_corr_count,
            "total_pairs": len(correlations),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating correlation statistics: {e}")
        return None


def get_correlation_with_benchmark_data(
    positions: list,
    price_data: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> Optional[Dict[str, any]]:
    """
    Calculate correlation and beta for each asset with benchmark.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        benchmark_returns: Benchmark returns series
        
    Returns:
        Dictionary with correlation and beta data for each asset
    """
    if not positions or price_data.empty or benchmark_returns.empty:
        return None
    
    try:
        # Calculate asset returns
        asset_returns = price_data.pct_change().dropna()
        
        if asset_returns.empty:
            return None
        
        # Align benchmark
        common_dates = asset_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 10:
            return None
        
        asset_returns_aligned = asset_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate correlation and beta for each asset
        results = []
        tickers = [pos.ticker for pos in positions if pos.ticker in asset_returns_aligned.columns]
        
        for ticker in tickers:
            asset_ret = asset_returns_aligned[ticker].dropna()
            # Align with benchmark
            aligned = pd.DataFrame({
                "asset": asset_ret,
                "benchmark": benchmark_aligned
            }).dropna()
            
            if len(aligned) >= 10:
                # Correlation
                corr = aligned["asset"].corr(aligned["benchmark"])
                
                # Beta
                cov = aligned["asset"].cov(aligned["benchmark"])
                bench_var = aligned["benchmark"].var()
                beta = cov / bench_var if bench_var > 0 else 0.0
                
                results.append({
                    "ticker": ticker,
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "beta": float(beta) if not np.isnan(beta) else 0.0,
                })
        
        # Sort by correlation descending
        results.sort(key=lambda x: x["correlation"], reverse=True)
        
        return {
            "data": results,
            "tickers": [r["ticker"] for r in results],
            "correlations": [r["correlation"] for r in results],
            "betas": [r["beta"] for r in results],
        }
        
    except Exception as e:
        logger.warning(f"Error calculating correlation with benchmark: {e}")
        return None


def get_cluster_analysis_data(
    correlation_matrix: pd.DataFrame,
) -> Optional[Dict[str, any]]:
    """
    Perform hierarchical clustering on correlation matrix.
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        
    Returns:
        Dictionary with clustered matrix and linkage data
    """
    if correlation_matrix.empty:
        return None
    
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1 - correlation_matrix.values
        
        # Convert to condensed form
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method="ward")
        
        # Get optimal leaf order
        optimal_leaves = leaves_list(linkage_matrix)
        
        # Reorder correlation matrix
        tickers = list(correlation_matrix.columns)
        reordered_tickers = [tickers[i] for i in optimal_leaves]
        reordered_matrix = correlation_matrix.loc[reordered_tickers, reordered_tickers]
        
        # Determine number of clusters using more intelligent method
        # Try to find optimal number of clusters using distance-based approach
        num_assets = len(tickers)
        
        if num_assets <= 3:
            n_clusters = num_assets
        elif num_assets <= 6:
            # For 4-6 assets, try 2-3 clusters
            n_clusters = max(2, min(3, num_assets // 2))
        elif num_assets <= 10:
            # For 7-10 assets, try 3-4 clusters
            n_clusters = max(3, min(4, num_assets // 2))
        else:
            # For more assets, use 4-5 clusters
            n_clusters = min(5, max(4, num_assets // 3))
        
        # Ensure we don't exceed number of assets
        n_clusters = min(n_clusters, num_assets)
        
        return {
            "clustered_matrix": reordered_matrix,
            "original_matrix": correlation_matrix,
            "linkage_matrix": linkage_matrix,
            "reordered_tickers": reordered_tickers,
            "n_clusters": n_clusters,
        }
        
    except Exception as e:
        logger.warning(f"Error performing cluster analysis: {e}")
        return None


def get_asset_price_dynamics_data(
    positions: list,
    price_data: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    selected_tickers: Optional[list] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate normalized price dynamics for selected assets.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        benchmark_returns: Optional benchmark returns
        selected_tickers: List of selected tickers to analyze
        
    Returns:
        Dictionary with normalized price series
    """
    if not positions or price_data.empty:
        return None
    
    try:
        # Use selected tickers or all tickers
        if selected_tickers is None:
            selected_tickers = [pos.ticker for pos in positions 
                             if pos.ticker in price_data.columns]
        
        # Filter to selected tickers
        selected_tickers = [t for t in selected_tickers if t in price_data.columns]
        
        if not selected_tickers:
            return None
        
        # Get price series for selected assets
        price_series = {}
        for ticker in selected_tickers:
            prices = price_data[ticker].dropna()
            if len(prices) >= 2:
                # Normalize to start at 0%
                first_price = prices.iloc[0]
                normalized = ((prices / first_price) - 1) * 100
                price_series[ticker] = normalized
        
        # Add benchmark if available
        if benchmark_returns is not None and not benchmark_returns.empty:
            # Convert returns to cumulative normalized prices
            cumulative = (1 + benchmark_returns).cumprod()
            if len(cumulative) >= 2:
                first_val = cumulative.iloc[0]
                normalized_bench = ((cumulative / first_val) - 1) * 100
                price_series["SPY"] = normalized_bench
        
        return {
            "price_series": price_series,
            "tickers": list(price_series.keys()),
            "dates": price_series[list(price_series.keys())[0]].index if price_series else None,
        }
        
    except Exception as e:
        logger.warning(f"Error calculating asset price dynamics: {e}")
        return None


def get_rolling_correlations_data(
    positions: list,
    price_data: pd.DataFrame,
    window: int = 60,
    selected_pairs: Optional[list] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate rolling correlations between asset pairs.

    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        window: Rolling window size in days
        selected_pairs: Optional list of (ticker1, ticker2) tuples

    Returns:
        Dictionary with rolling correlation series for each pair
    """
    if not positions or price_data.empty:
        return None

    try:
        # Calculate returns
        returns_df = price_data.pct_change().dropna()

        if returns_df.empty or len(returns_df) < window:
            return None

        # Exclude CASH
        if "CASH" in returns_df.columns:
            returns_df = returns_df.drop(columns=["CASH"])

        tickers = [pos.ticker for pos in positions
                   if pos.ticker in returns_df.columns]

        if len(tickers) < 2:
            return None

        # If no pairs specified, use all pairs
        if selected_pairs is None:
            selected_pairs = []
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:
                    selected_pairs.append((ticker1, ticker2))

        rolling_correlations = {}

        for ticker1, ticker2 in selected_pairs:
            if ticker1 not in returns_df.columns:
                continue
            if ticker2 not in returns_df.columns:
                continue

            # Calculate rolling correlation manually
            rolling_corr_list = []
            rolling_dates = []

            for i in range(window, len(returns_df) + 1):
                window_data = returns_df.iloc[i-window:i]
                asset1 = window_data[ticker1].dropna()
                asset2 = window_data[ticker2].dropna()

                # Align data
                aligned = pd.DataFrame({
                    "asset1": asset1,
                    "asset2": asset2,
                }).dropna()

                if len(aligned) >= window // 2:
                    corr = aligned["asset1"].corr(aligned["asset2"])
                    if not np.isnan(corr):
                        rolling_corr_list.append(corr)
                        rolling_dates.append(returns_df.index[i-1])

            if rolling_corr_list:
                pair_name = f"{ticker1}-{ticker2}"
                rolling_correlations[pair_name] = pd.Series(
                    rolling_corr_list, index=rolling_dates
                )

        if not rolling_correlations:
            return None

        return {
            "rolling_correlations": rolling_correlations,
            "window": window,
            "pairs": selected_pairs,
        }

    except Exception as e:
        logger.warning(f"Error calculating rolling correlations: {e}")
        return None


def get_average_correlation_to_portfolio_data(
    positions: list,
    price_data: pd.DataFrame,
) -> Optional[Dict[str, any]]:
    """
    Calculate average correlation of each asset to the rest of portfolio.

    Args:
        positions: List of Position objects
        price_data: DataFrame with price data

    Returns:
        Dictionary with average correlations and diversification scores
    """
    if not positions or price_data.empty:
        return None

    try:
        # Calculate returns
        returns_df = price_data.pct_change().dropna()

        if returns_df.empty:
            return None

        # Exclude CASH
        if "CASH" in returns_df.columns:
            returns_df = returns_df.drop(columns=["CASH"])

        tickers = [pos.ticker for pos in positions
                   if pos.ticker in returns_df.columns]

        if len(tickers) < 2:
            return None

        # Calculate correlation matrix
        corr_matrix = returns_df[tickers].corr()

        # For each asset, calculate average correlation to others
        avg_correlations = {}
        diversification_scores = {}

        for ticker in tickers:
            if ticker not in corr_matrix.columns:
                continue

            # Get correlations to all other assets
            other_tickers = [t for t in tickers if t != ticker]
            correlations_to_others = [
                corr_matrix.loc[ticker, other]
                for other in other_tickers
                if other in corr_matrix.columns
            ]

            if correlations_to_others:
                avg_corr = float(np.nanmean(correlations_to_others))
                avg_correlations[ticker] = avg_corr
                # Diversification score = 1 - avg_correlation
                diversification_scores[ticker] = 1.0 - avg_corr

        if not avg_correlations:
            return None

        # Sort by diversification score (highest first)
        sorted_tickers = sorted(
            diversification_scores.keys(),
            key=lambda t: diversification_scores[t],
            reverse=True
        )

        return {
            "tickers": sorted_tickers,
            "avg_correlations": [
                avg_correlations[t] for t in sorted_tickers
            ],
            "diversification_scores": [
                diversification_scores[t] for t in sorted_tickers
            ],
        }

    except Exception as e:
        logger.warning(
            f"Error calculating average correlation to portfolio: {e}"
        )
        return None


def get_rolling_correlation_with_benchmark_data(
    positions: list,
    price_data: pd.DataFrame,
    benchmark_returns: pd.Series,
    portfolio_returns: Optional[pd.Series] = None,
    window: int = 60,
    selected_tickers: Optional[list] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate rolling correlation with benchmark for selected assets.
    
    Args:
        positions: List of Position objects
        price_data: DataFrame with price data
        benchmark_returns: Benchmark returns series
        portfolio_returns: Optional portfolio returns for weighted average
        window: Rolling window size in days
        selected_tickers: List of selected tickers
        
    Returns:
        Dictionary with rolling correlation series
    """
    if not positions or price_data.empty or benchmark_returns.empty:
        return None
    
    try:
        # Use selected tickers or all tickers
        if selected_tickers is None:
            selected_tickers = [pos.ticker for pos in positions 
                             if pos.ticker in price_data.columns]
        
        selected_tickers = [t for t in selected_tickers if t in price_data.columns]
        
        if not selected_tickers:
            return None
        
        # Calculate asset returns
        asset_returns = price_data.pct_change().dropna()
        
        # Align dates
        common_dates = asset_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < window:
            return None
        
        asset_returns_aligned = asset_returns.loc[common_dates]
        benchmark_aligned = benchmark_returns.loc[common_dates]
        
        # Calculate rolling correlations for each asset
        rolling_correlations = {}
        
        for ticker in selected_tickers:
            if ticker not in asset_returns_aligned.columns:
                continue
            
            asset_ret = asset_returns_aligned[ticker].dropna()
            
            # Calculate rolling correlation
            rolling_corr = []
            rolling_dates = []
            
            for i in range(window, len(asset_ret) + 1):
                window_asset = asset_ret.iloc[i-window:i]
                window_bench = benchmark_aligned.iloc[i-window:i]
                
                # Align windows
                aligned = pd.DataFrame({
                    "asset": window_asset,
                    "benchmark": window_bench
                }).dropna()
                
                if len(aligned) >= window // 2:  # At least half window size
                    corr = aligned["asset"].corr(aligned["benchmark"])
                    if not np.isnan(corr):
                        rolling_corr.append(corr)
                        rolling_dates.append(asset_ret.index[i-1])
            
            if rolling_corr:
                rolling_correlations[ticker] = pd.Series(
                    rolling_corr, index=rolling_dates
                )
        
        # Calculate weighted portfolio average if portfolio_returns available
        portfolio_avg_corr = None
        if portfolio_returns is not None and not portfolio_returns.empty:
            # Calculate portfolio correlation with benchmark
            aligned_port = portfolio_returns.reindex(common_dates).dropna()
            aligned_bench = benchmark_returns.reindex(common_dates).dropna()
            
            if len(aligned_port) >= window:
                portfolio_rolling_corr = []
                portfolio_rolling_dates = []
                
                for i in range(window, len(aligned_port) + 1):
                    window_port = aligned_port.iloc[i-window:i]
                    window_bench = aligned_bench.iloc[i-window:i]
                    
                    aligned = pd.DataFrame({
                        "portfolio": window_port,
                        "benchmark": window_bench
                    }).dropna()
                    
                    if len(aligned) >= window // 2:
                        corr = aligned["portfolio"].corr(aligned["benchmark"])
                        if not np.isnan(corr):
                            portfolio_rolling_corr.append(corr)
                            portfolio_rolling_dates.append(aligned_port.index[i-1])
                
                if portfolio_rolling_corr:
                    portfolio_avg_corr = pd.Series(
                        portfolio_rolling_corr, index=portfolio_rolling_dates
                    )
        
        return {
            "rolling_correlations": rolling_correlations,
            "portfolio_avg_correlation": portfolio_avg_corr,
            "window": window,
            "tickers": list(rolling_correlations.keys()),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating rolling correlation: {e}")
        return None


def get_detailed_asset_analysis_data(
    ticker: str,
    positions: list,
    price_data: pd.DataFrame,
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Optional[Dict[str, any]]:
    """
    Calculate detailed analysis data for a single asset.
    
    Args:
        ticker: Asset ticker to analyze
        positions: List of Position objects
        price_data: DataFrame with price data
        portfolio_returns: Portfolio returns series
        benchmark_returns: Optional benchmark returns
        
    Returns:
        Dictionary with detailed asset metrics and comparison data
    """
    if ticker not in price_data.columns:
        return None
    
    try:
        # Get asset prices
        asset_prices = price_data[ticker].dropna()
        
        if len(asset_prices) < 2:
            return None
        
        # Calculate asset returns
        asset_returns = asset_prices.pct_change().dropna()
        
        # Calculate metrics
        total_return = (asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1
        trading_days = len(asset_returns)
        years = trading_days / TRADING_DAYS_PER_YEAR
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        volatility = asset_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0
        
        # Max drawdown
        cumulative = (1 + asset_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Portfolio metrics for comparison
        portfolio_total_return = (1 + portfolio_returns).prod() - 1
        portfolio_years = len(portfolio_returns) / TRADING_DAYS_PER_YEAR
        portfolio_annual_return = (1 + portfolio_total_return) ** (1 / portfolio_years) - 1 if portfolio_years > 0 else 0.0
        portfolio_volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        portfolio_sharpe = portfolio_annual_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # Beta and correlation with portfolio
        aligned = pd.DataFrame({
            "asset": asset_returns,
            "portfolio": portfolio_returns
        }).dropna()
        
        beta = None
        correlation = None
        if len(aligned) >= 10:
            cov = aligned["asset"].cov(aligned["portfolio"])
            port_var = aligned["portfolio"].var()
            beta = cov / port_var if port_var > 0 else 0.0
            correlation = aligned["asset"].corr(aligned["portfolio"])
        
        # Correlation with other assets
        other_correlations = {}
        other_tickers = [pos.ticker for pos in positions 
                        if pos.ticker != ticker and pos.ticker in price_data.columns]
        
        for other_ticker in other_tickers:
            other_returns = price_data[other_ticker].pct_change().dropna()
            aligned_other = pd.DataFrame({
                "asset": asset_returns,
                "other": other_returns
            }).dropna()
            
            if len(aligned_other) >= 10:
                corr = aligned_other["asset"].corr(aligned_other["other"])
                if not np.isnan(corr):
                    other_correlations[other_ticker] = float(corr)
        
        # Cumulative returns for comparison
        asset_cumulative = (1 + asset_returns).cumprod() - 1
        portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        
        # Benchmark cumulative if available
        benchmark_cumulative = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            aligned_bench = benchmark_returns.reindex(asset_returns.index, method="ffill").dropna()
            if len(aligned_bench) >= 2:
                benchmark_cumulative = (1 + aligned_bench).cumprod() - 1
        
        # Moving averages (50-day, 200-day)
        ma50 = asset_prices.rolling(50).mean() if len(asset_prices) >= 50 else None
        ma200 = asset_prices.rolling(200).mean() if len(asset_prices) >= 200 else None
        
        return {
            "ticker": ticker,
            "metrics": {
                "total_return": float(total_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "beta": float(beta) if beta is not None else None,
                "correlation": float(correlation) if correlation is not None else None,
            },
            "portfolio_metrics": {
                "total_return": float(portfolio_total_return),
                "annual_return": float(portfolio_annual_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(portfolio_sharpe),
            },
            "other_correlations": other_correlations,
            "cumulative_returns": {
                "asset": asset_cumulative,
                "portfolio": portfolio_cumulative,
                "benchmark": benchmark_cumulative,
            },
            "prices": asset_prices,
            "returns": asset_returns,
            "ma50": ma50,
            "ma200": ma200,
        }
        
    except Exception as e:
        logger.warning(f"Error calculating detailed asset analysis: {e}")
        return None