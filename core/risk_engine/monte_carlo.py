"""
Monte Carlo simulation for portfolio risk analysis.

This module provides Monte Carlo simulation capabilities for portfolio
path forecasting, percentile analysis, and distribution generation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional
from dataclasses import dataclass

from core.exceptions import InsufficientDataError


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    simulated_paths: np.ndarray  # Shape: (num_simulations, time_horizon)
    final_values: np.ndarray  # Shape: (num_simulations,)
    percentiles: Dict[float, float]  # Percentile -> value mapping
    statistics: Dict[str, float]  # Mean, median, std, min, max, etc.


class MonteCarloSimulator:
    """
    Monte Carlo simulator for portfolio risk analysis.

    Supports multiple models:
    - Geometric Brownian Motion (GBM)
    - Jump Diffusion
    - GARCH (if implemented)
    """

    def __init__(
        self,
        model: str = "gbm",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            model: Model type ('gbm', 'jump_diffusion')
            random_seed: Random seed for reproducibility
        """
        self.model = model
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate(
        self,
        returns: pd.Series,
        time_horizon: int,
        num_simulations: int = 10000,
        initial_value: float = 1.0,
    ) -> MonteCarloResult:
        """
        Simulate portfolio paths using Monte Carlo method.

        Args:
            returns: Historical returns series
            time_horizon: Number of days to simulate
            num_simulations: Number of simulation paths
            initial_value: Starting portfolio value (default: 1.0)

        Returns:
            MonteCarloResult with paths, final values, and statistics

        Raises:
            InsufficientDataError: If returns series is empty
        """
        if returns.empty:
            raise InsufficientDataError("Returns series is empty")

        if self.model == "gbm":
            return self._simulate_gbm(
                returns, time_horizon, num_simulations, initial_value
            )
        elif self.model == "jump_diffusion":
            return self._simulate_jump_diffusion(
                returns, time_horizon, num_simulations, initial_value
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _simulate_gbm(
        self,
        returns: pd.Series,
        time_horizon: int,
        num_simulations: int,
        initial_value: float,
    ) -> MonteCarloResult:
        """Simulate using Geometric Brownian Motion model."""
        # Calculate parameters from historical returns
        mean_return = float(returns.mean())
        std_return = float(returns.std())

        # Annualize if needed (assuming daily returns)
        # For daily returns: mean_daily, std_daily
        # For GBM: drift = mean_return, volatility = std_return

        # Generate random paths
        # Shape: (num_simulations, time_horizon)
        random_shocks = np.random.normal(
            0, 1, size=(num_simulations, time_horizon)
        )

        # Calculate daily returns for each path
        # GBM: dS = S * (mu * dt + sigma * dW)
        # For daily: return = mean_return + std_return * random_shock
        daily_returns = mean_return + std_return * random_shocks

        # Calculate cumulative returns and values
        # Each row is a simulation path
        simulated_paths = np.zeros((num_simulations, time_horizon))
        final_values = np.zeros(num_simulations)

        for i in range(num_simulations):
            path = initial_value
            path_values = []

            for day in range(time_horizon):
                path *= 1 + daily_returns[i, day]
                path_values.append(path)

            simulated_paths[i] = path_values
            final_values[i] = path

        # Calculate percentiles
        percentiles = {
            5.0: float(np.percentile(final_values, 5.0)),
            10.0: float(np.percentile(final_values, 10.0)),
            25.0: float(np.percentile(final_values, 25.0)),
            50.0: float(np.percentile(final_values, 50.0)),
            75.0: float(np.percentile(final_values, 75.0)),
            90.0: float(np.percentile(final_values, 90.0)),
            95.0: float(np.percentile(final_values, 95.0)),
        }

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(final_values)),
            "median": float(np.median(final_values)),
            "std": float(np.std(final_values)),
            "min": float(np.min(final_values)),
            "max": float(np.max(final_values)),
            "skewness": float(stats.skew(final_values)),
            "kurtosis": float(stats.kurtosis(final_values)),
        }

        return MonteCarloResult(
            simulated_paths=simulated_paths,
            final_values=final_values,
            percentiles=percentiles,
            statistics=statistics,
        )

    def _simulate_jump_diffusion(
        self,
        returns: pd.Series,
        time_horizon: int,
        num_simulations: int,
        initial_value: float,
    ) -> MonteCarloResult:
        """
        Simulate using Jump Diffusion model.

        This is a simplified version. Full implementation would include
        jump intensity, jump size distribution, etc.
        """
        # Calculate parameters from historical returns
        mean_return = float(returns.mean())
        std_return = float(returns.std())

        # Estimate jump parameters (simplified)
        # In practice, use more sophisticated estimation
        jump_intensity = 0.1  # Probability of jump per day
        jump_mean = -0.02  # Mean jump size
        jump_std = 0.05  # Jump size std

        # Generate random paths
        random_shocks = np.random.normal(
            0, 1, size=(num_simulations, time_horizon)
        )
        jump_events = np.random.binomial(
            1, jump_intensity, size=(num_simulations, time_horizon)
        )
        jump_sizes = np.random.normal(
            jump_mean, jump_std, size=(num_simulations, time_horizon)
        )

        # Calculate daily returns with jumps
        daily_returns = (
            mean_return
            + std_return * random_shocks
            + jump_events * jump_sizes
        )

        # Calculate cumulative returns and values
        simulated_paths = np.zeros((num_simulations, time_horizon))
        final_values = np.zeros(num_simulations)

        for i in range(num_simulations):
            path = initial_value
            path_values = []

            for day in range(time_horizon):
                path *= 1 + daily_returns[i, day]
                path_values.append(path)

            simulated_paths[i] = path_values
            final_values[i] = path

        # Calculate percentiles
        percentiles = {
            5.0: float(np.percentile(final_values, 5.0)),
            10.0: float(np.percentile(final_values, 10.0)),
            25.0: float(np.percentile(final_values, 25.0)),
            50.0: float(np.percentile(final_values, 50.0)),
            75.0: float(np.percentile(final_values, 75.0)),
            90.0: float(np.percentile(final_values, 90.0)),
            95.0: float(np.percentile(final_values, 95.0)),
        }

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(final_values)),
            "median": float(np.median(final_values)),
            "std": float(np.std(final_values)),
            "min": float(np.min(final_values)),
            "max": float(np.max(final_values)),
            "skewness": float(stats.skew(final_values)),
            "kurtosis": float(stats.kurtosis(final_values)),
        }

        return MonteCarloResult(
            simulated_paths=simulated_paths,
            final_values=final_values,
            percentiles=percentiles,
            statistics=statistics,
        )


def simulate_portfolio_paths(
    returns: pd.Series,
    time_horizon: int,
    num_simulations: int = 10000,
    initial_value: float = 1.0,
    model: str = "gbm",
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Convenience function to simulate portfolio paths.

    Args:
        returns: Historical returns series
        time_horizon: Number of days to simulate
        num_simulations: Number of simulation paths (default: 10,000)
        initial_value: Starting portfolio value (default: 1.0)
        model: Model type ('gbm' or 'jump_diffusion')
        random_seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with paths, final values, and statistics
    """
    simulator = MonteCarloSimulator(model=model, random_seed=random_seed)
    return simulator.simulate(
        returns, time_horizon, num_simulations, initial_value
    )

