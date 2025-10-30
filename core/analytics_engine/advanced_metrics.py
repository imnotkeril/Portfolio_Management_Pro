"""Advanced portfolio metrics (PSR, Smart Sharpe, Kelly, etc.)."""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.analytics_engine.ratios import calculate_sharpe_ratio, calculate_sortino_ratio
from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def calculate_probabilistic_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_sharpe: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    
    PSR measures the probability that observed Sharpe > benchmark Sharpe.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        benchmark_sharpe: Benchmark Sharpe ratio (default: 1.0)
        periods_per_year: Number of trading periods per year
        
    Returns:
        PSR (0-1) or None if insufficient data
    """
    if returns.empty or len(returns) < 30:
        return None
        
    try:
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        if sharpe is None:
            return None
            
        n = len(returns)
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())
        
        # Standard error of Sharpe ratio
        sr_std = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / n)
        
        # Z-score
        z_score = (sharpe - benchmark_sharpe) / sr_std
        
        # Probability (CDF of normal distribution)
        psr = stats.norm.cdf(z_score)
        
        return float(psr)
        
    except Exception as e:
        logger.warning(f"Error calculating PSR: {e}")
        return None


def calculate_smart_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Smart Sharpe (adjusted for autocorrelation).
    
    Adjusts standard Sharpe for autocorrelation in returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Smart Sharpe ratio or None
    """
    if returns.empty or len(returns) < 30:
        return None
        
    try:
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        if sharpe is None:
            return None
            
        # Calculate autocorrelation at lag 1
        autocorr = float(returns.autocorr(lag=1))
        
        # Adjust for autocorrelation
        # If positive autocorrelation, Sharpe is overstated
        adjustment_factor = np.sqrt(1 - autocorr) if autocorr > 0 else 1.0
        
        smart_sharpe = sharpe * adjustment_factor
        
        return float(smart_sharpe)
        
    except Exception as e:
        logger.warning(f"Error calculating Smart Sharpe: {e}")
        return None


def calculate_smart_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[float]:
    """
    Calculate Smart Sortino (adjusted for autocorrelation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year
        
    Returns:
        Smart Sortino ratio or None
    """
    if returns.empty or len(returns) < 30:
        return None
        
    try:
        sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        if sortino is None:
            return None
            
        # Calculate autocorrelation
        autocorr = float(returns.autocorr(lag=1))
        
        # Adjust for autocorrelation
        adjustment_factor = np.sqrt(1 - autocorr) if autocorr > 0 else 1.0
        
        smart_sortino = sortino * adjustment_factor
        
        return float(smart_sortino)
        
    except Exception as e:
        logger.warning(f"Error calculating Smart Sortino: {e}")
        return None


def calculate_kelly_criterion(
    returns: pd.Series,
) -> Optional[Dict[str, float]]:
    """
    Calculate Kelly Criterion for optimal position sizing.
    
    Formula: f* = (p*b - q) / b
    where p = prob of win, q = prob of loss, b = win/loss ratio
    
    Args:
        returns: Series of returns
        
    Returns:
        Dict with full, half, and quarter Kelly percentages
    """
    if returns.empty or len(returns) < 30:
        return None
        
    try:
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return None
            
        # Win probability
        p = len(wins) / len(returns)
        q = 1 - p
        
        # Average win / Average loss ratio
        b = abs(wins.mean() / losses.mean())
        
        # Kelly percentage
        kelly_full = (p * b - q) / b
        
        # Cap at reasonable values (0-50%)
        kelly_full = max(0.0, min(kelly_full, 0.50))
        
        return {
            "kelly_full": float(kelly_full),
            "kelly_half": float(kelly_full / 2),
            "kelly_quarter": float(kelly_full / 4),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating Kelly Criterion: {e}")
        return None


def calculate_risk_of_ruin(
    returns: pd.Series,
    drawdown_levels: list = [0.10, 0.20, 0.25, 0.30, 0.50],
) -> Optional[Dict[str, float]]:
    """
    Calculate probability of reaching various drawdown levels.
    
    Uses historical returns to estimate probabilities.
    
    Args:
        returns: Series of returns
        drawdown_levels: List of drawdown levels to check
        
    Returns:
        Dict mapping drawdown level to probability
    """
    if returns.empty or len(returns) < 100:
        return None
        
    try:
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        results = {}
        for level in drawdown_levels:
            # Count how many times drawdown exceeded level
            exceeded = (drawdowns <= -level).sum()
            probability = exceeded / len(drawdowns)
            results[f"ruin_{int(level*100)}pct"] = float(probability)
            
        return results
        
    except Exception as e:
        logger.warning(f"Error calculating Risk of Ruin: {e}")
        return None


def calculate_win_rate_stats(
    returns: pd.Series,
    period_type: str = "daily",
) -> Optional[Dict[str, float]]:
    """
    Calculate comprehensive win rate statistics.
    
    Args:
        returns: Series of returns
        period_type: Type of period ('daily', 'monthly', etc.)
        
    Returns:
        Dict with win rate stats
    """
    if returns.empty:
        return None
        
    try:
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        total = len(returns)
        positive_count = len(positive_returns)
        negative_count = len(negative_returns)
        
        results = {
            f"win_rate_{period_type}": positive_count / total if total > 0 else 0.0,
            f"avg_win_{period_type}": float(positive_returns.mean()) if len(positive_returns) > 0 else 0.0,
            f"avg_loss_{period_type}": float(negative_returns.mean()) if len(negative_returns) > 0 else 0.0,
            f"best_{period_type}": float(returns.max()) if len(returns) > 0 else 0.0,
            f"worst_{period_type}": float(returns.min()) if len(returns) > 0 else 0.0,
        }
        
        return results
        
    except Exception as e:
        logger.warning(f"Error calculating win rate stats: {e}")
        return None


def calculate_outlier_analysis(
    returns: pd.Series,
    std_threshold: float = 2.0,
) -> Optional[Dict[str, float]]:
    """
    Analyze outlier returns (beyond N standard deviations).
    
    Args:
        returns: Series of returns
        std_threshold: Number of standard deviations for outlier
        
    Returns:
        Dict with outlier statistics
    """
    if returns.empty or len(returns) < 30:
        return None
        
    try:
        mean = returns.mean()
        std = returns.std()
        
        # Define outliers
        upper_bound = mean + std_threshold * std
        lower_bound = mean - std_threshold * std
        
        outlier_wins = returns[returns > upper_bound]
        outlier_losses = returns[returns < lower_bound]
        normal_wins = returns[(returns > mean) & (returns <= upper_bound)]
        normal_losses = returns[(returns < mean) & (returns >= lower_bound)]
        
        # Calculate ratios
        outlier_win_ratio = (
            abs(outlier_wins.mean() / normal_wins.mean())
            if len(outlier_wins) > 0 and len(normal_wins) > 0 and normal_wins.mean() != 0
            else 1.0
        )
        
        outlier_loss_ratio = (
            abs(outlier_losses.mean() / normal_losses.mean())
            if len(outlier_losses) > 0 and len(normal_losses) > 0 and normal_losses.mean() != 0
            else 1.0
        )
        
        return {
            "outlier_win_ratio": float(outlier_win_ratio),
            "outlier_loss_ratio": float(outlier_loss_ratio),
            "outlier_count": int(len(outlier_wins) + len(outlier_losses)),
            "outlier_pct": float((len(outlier_wins) + len(outlier_losses)) / len(returns)),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating outlier analysis: {e}")
        return None


def calculate_common_performance_periods(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Optional[Dict[str, float]]:
    """
    Calculate Common Performance Periods (CPP) index.
    
    Measures how often portfolio and benchmark move in same direction.
    
    Args:
        portfolio_returns: Portfolio returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Dict with CPP statistics
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return None
        
    try:
        # Align series
        aligned = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 30:
            return None
            
        # Count same direction moves
        same_direction = ((aligned['portfolio'] > 0) & (aligned['benchmark'] > 0)) | \
                        ((aligned['portfolio'] < 0) & (aligned['benchmark'] < 0))
        
        same_direction_pct = same_direction.sum() / len(aligned)
        
        # CPP Index (correlation of directions)
        port_direction = (aligned['portfolio'] > 0).astype(int)
        bench_direction = (aligned['benchmark'] > 0).astype(int)
        cpp_index = port_direction.corr(bench_direction)
        
        return {
            "same_direction_pct": float(same_direction_pct),
            "cpp_index": float(cpp_index),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating CPP: {e}")
        return None


def calculate_expected_returns(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Optional[Dict[str, float]]:
    """
    Calculate expected returns for different timeframes.
    
    Simple arithmetic mean extrapolation.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year
        
    Returns:
        Dict with expected returns (daily, monthly, yearly)
    """
    if returns.empty:
        return None
        
    try:
        mean_return = returns.mean()
        
        return {
            "expected_daily": float(mean_return),
            "expected_weekly": float(mean_return * 5),  # 5 trading days
            "expected_monthly": float(mean_return * 21),  # ~21 trading days
            "expected_quarterly": float(mean_return * 63),  # ~63 trading days
            "expected_yearly": float(mean_return * periods_per_year),
        }
        
    except Exception as e:
        logger.warning(f"Error calculating expected returns: {e}")
        return None

