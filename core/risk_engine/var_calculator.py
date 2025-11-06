"""
Value at Risk (VaR) calculator with multiple methods.

This module provides comprehensive VaR calculation methods including
Historical, Parametric, Monte Carlo, and Cornish-Fisher approaches.
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.analytics_engine.risk_metrics import (
    calculate_var as calculate_var_base,
)
from core.exceptions import InsufficientDataError


def calculate_var(
    returns: pd.Series,  # noqa: F821
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR) using specified method.

    This function wraps the existing risk_metrics.calculate_var and adds
    Monte Carlo method support.

    Args:
        returns: Series of portfolio returns
        confidence_level: Confidence level (0.90, 0.95, or 0.99)
        method: Method to use:
            - 'historical': Percentile of historical returns
            - 'parametric': Normal distribution assumption
            - 'cornish_fisher': Adjusted for skewness/kurtosis
            - 'monte_carlo': Monte Carlo simulation (requires num_simulations)

    Returns:
        VaR as decimal (negative value)

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

    # Use existing methods for historical, parametric, cornish_fisher
    if method in ["historical", "parametric", "cornish_fisher"]:
        return calculate_var_base(returns, confidence_level, method)

    # Monte Carlo method requires separate function
    if method == "monte_carlo":
        raise ValueError(
            "Monte Carlo VaR requires calculate_var_monte_carlo() function "
            "with num_simulations parameter"
        )

    raise ValueError(
        f"Unknown method: {method}. "
        "Use 'historical', 'parametric', 'cornish_fisher', or 'monte_carlo'"
    )


def calculate_var_monte_carlo(
    returns: pd.Series,
    confidence_level: float = 0.95,
    num_simulations: int = 10000,
    time_horizon: int = 1,
    random_seed: Optional[int] = None,
) -> float:
    """
    Calculate VaR using Monte Carlo simulation.

    Generates random portfolio paths based on historical return distribution
    and calculates VaR from the simulated distribution.

    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (0.90, 0.95, or 0.99)
        num_simulations: Number of Monte Carlo simulations (default: 10,000)
        time_horizon: Time horizon in days (default: 1)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        VaR as decimal (negative value)

    Raises:
        InsufficientDataError: If returns series is empty
        ValueError: If invalid parameters
    """
    if returns.empty:
        raise InsufficientDataError("Returns series is empty")

    if confidence_level not in [0.90, 0.95, 0.99]:
        raise ValueError(
            "Confidence level must be 0.90, 0.95, or 0.99"
        )

    if num_simulations < 1000:
        raise ValueError(
            "Number of simulations must be at least 1,000"
        )

    if time_horizon < 1:
        raise ValueError("Time horizon must be at least 1 day")

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate parameters from historical returns
    mean_return = float(returns.mean())
    std_return = float(returns.std())

    # Simulate returns for time horizon
    # For multi-day horizon, we simulate cumulative returns
    simulated_returns = []

    for _ in range(num_simulations):
        # Generate random returns for each day in horizon
        daily_returns = np.random.normal(
            mean_return,
            std_return,
            size=time_horizon
        )
        # Calculate cumulative return over horizon
        cumulative_return = np.prod(1 + daily_returns) - 1
        simulated_returns.append(cumulative_return)

    # Convert to numpy array for percentile calculation
    simulated_returns = np.array(simulated_returns)

    # Calculate VaR as percentile
    alpha = 1.0 - confidence_level
    var = float(np.percentile(simulated_returns, alpha * 100))

    return var


def calculate_var_all_methods(
    returns: pd.Series,
    confidence_level: float = 0.95,
    num_simulations: int = 10000,
    time_horizon: int = 1,
) -> dict:
    """
    Calculate VaR using all available methods for comparison.

    Args:
        returns: Series of historical returns
        confidence_level: Confidence level (0.90, 0.95, or 0.99)
        num_simulations: Number of Monte Carlo simulations
        time_horizon: Time horizon in days

    Returns:
        Dictionary with VaR values for each method:
        {
            'historical': float,
            'parametric': float,
            'cornish_fisher': float,
            'monte_carlo': float
        }
    """
    results = {}

    try:
        results["historical"] = calculate_var(
            returns, confidence_level, "historical"
        )
    except Exception:
        results["historical"] = None

    try:
        results["parametric"] = calculate_var(
            returns, confidence_level, "parametric"
        )
    except Exception:
        results["parametric"] = None

    try:
        results["cornish_fisher"] = calculate_var(
            returns, confidence_level, "cornish_fisher"
        )
    except Exception:
        results["cornish_fisher"] = None

    try:
        results["monte_carlo"] = calculate_var_monte_carlo(
            returns, confidence_level, num_simulations, time_horizon
        )
    except Exception:
        results["monte_carlo"] = None

    return results


