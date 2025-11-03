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