"""Efficient frontier generation."""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Suppress numpy array comparison warnings before importing project modules
warnings.filterwarnings("ignore", category=FutureWarning)

from core.optimization_engine.max_sharpe import MaxSharpeOptimizer  # noqa: E402
from core.optimization_engine.mean_variance import MeanVarianceOptimizer  # noqa: E402
from core.optimization_engine.min_variance import MinVarianceOptimizer  # noqa: E402

logger = logging.getLogger(__name__)


class EfficientFrontier:
    """Generate efficient frontier for portfolio optimization."""

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize efficient frontier generator.

        Args:
            returns: Returns DataFrame with tickers as columns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.returns = returns
        self.tickers = returns.columns.tolist()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # Calculate statistics
        mean_returns = returns.mean() * periods_per_year
        cov_matrix = returns.cov() * periods_per_year

        self._mean_returns = mean_returns
        self._cov_matrix = cov_matrix

        # Find min and max returns
        self._min_return = mean_returns.min()
        self._max_return = mean_returns.max()

        # Find min variance portfolio (without constraints for initialization)
        # This is just for initial bounds, actual frontier will use constraints
        min_var_optimizer = MinVarianceOptimizer(
            returns, risk_free_rate, periods_per_year
        )
        min_var_result = min_var_optimizer.optimize()
        self._min_volatility = min_var_result.volatility or 0.0
        self._min_return_at_min_vol = min_var_result.expected_return or 0.0

    def generate_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """
        Generate efficient frontier points.

        Uses target return approach for smoother curve
        (as in Markowitz theory).

        Args:
            n_points: Number of points on frontier
            constraints: Optional constraints dictionary

        Returns:
            Tuple of (returns, volatilities, portfolios)
            portfolios is list of dicts with weights and metrics
        """
        optimizer = MeanVarianceOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )

        # First, find min and max returns by optimizing
        # Get min variance portfolio (lowest return on frontier)
        from core.optimization_engine.min_variance import MinVarianceOptimizer

        min_var_opt = MinVarianceOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )
        min_var_result = min_var_opt.optimize(constraints=constraints)
        min_ret = float(min_var_result.expected_return or self._min_return_at_min_vol)

        # Get max return portfolio (highest return on frontier)
        from core.optimization_engine.max_return import MaxReturnOptimizer

        max_ret_opt = MaxReturnOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )
        max_ret_result = max_ret_opt.optimize(constraints=constraints)
        max_ret = float(max_ret_result.expected_return or self._max_return)

        # Generate only the efficient branch:
        # target returns from min-variance return to max-return portfolio.
        if max_ret <= min_ret:
            all_target_returns = np.array([min_ret], dtype=float)
        else:
            all_target_returns = np.linspace(min_ret, max_ret, n_points)

        returns_list = []
        volatilities_list = []
        portfolios_list = []

        for target_ret in all_target_returns:
            try:
                # Convert target_ret to float to avoid numpy scalar issues
                target_ret_float = float(target_ret)
                result = optimizer.optimize(
                    constraints=constraints,
                    target_return=target_ret_float,
                    target_return_as_floor=True,
                )

                # Check success and volatility with explicit None check
                # Use bool() to ensure we're comparing boolean, not array
                if bool(result.success):
                    vol_value = result.volatility
                    # Explicit check for None and ensure it's not an array
                    if vol_value is not None:
                        try:
                            # Convert to float to ensure it's a scalar
                            vol_float = float(vol_value)
                            ret_value = result.expected_return or 0.0
                            ret_float = float(ret_value)

                            returns_list.append(ret_float)
                            volatilities_list.append(vol_float)
                            portfolios_list.append(
                                {
                                    "weights": result.weights,
                                    "tickers": result.tickers,
                                    "expected_return": ret_float,
                                    "volatility": vol_float,
                                    "sharpe_ratio": (
                                        float(result.sharpe_ratio)
                                        if result.sharpe_ratio is not None
                                        else None
                                    ),
                                }
                            )
                        except (ValueError, TypeError) as conv_e:
                            logger.warning(
                                f"Could not convert values to float: {conv_e}"
                            )
                            continue
            except Exception as e:
                logger.warning(
                    f"Failed to generate frontier point for "
                    f"return {target_ret:.2%}: {e}"
                )
                continue

        # Sort and keep only Pareto-efficient upper envelope.
        if len(volatilities_list) > 0:
            indexed_data = list(
                enumerate(
                    zip(
                        [float(v) for v in volatilities_list],
                        [float(r) for r in returns_list],
                    )
                )
            )
            sorted_indexed = sorted(indexed_data, key=lambda x: x[1][0])

            sorted_vols = [v for _, (v, _) in sorted_indexed]
            sorted_rets = [r for _, (_, r) in sorted_indexed]
            sorted_ports = [portfolios_list[i] for i, _ in sorted_indexed]

            efficient_vols: list[float] = []
            efficient_rets: list[float] = []
            efficient_ports: list[dict] = []
            best_ret = -np.inf
            eps = 1e-8
            for v, r, p in zip(sorted_vols, sorted_rets, sorted_ports):
                if r > best_ret + eps:
                    efficient_vols.append(v)
                    efficient_rets.append(r)
                    efficient_ports.append(p)
                    best_ret = r

            if len(efficient_vols) >= 2:
                volatilities_list = efficient_vols
                returns_list = efficient_rets
                portfolios_list = efficient_ports
            else:
                volatilities_list = sorted_vols
                returns_list = sorted_rets
                portfolios_list = sorted_ports

        return (
            np.array(returns_list),
            np.array(volatilities_list),
            portfolios_list,
        )

    def get_tangency_portfolio(
        self,
        constraints: Optional[dict] = None,
    ) -> dict:
        """
        Get tangency portfolio (maximum Sharpe ratio).

        Args:
            constraints: Optional constraints dictionary

        Returns:
            Dictionary with portfolio information
        """
        optimizer = MaxSharpeOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )

        result = optimizer.optimize(constraints=constraints)

        return {
            "weights": result.weights,
            "tickers": result.tickers,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "method": "Tangency Portfolio (Max Sharpe)",
        }

    def get_min_variance_portfolio(
        self,
        constraints: Optional[dict] = None,
    ) -> dict:
        """
        Get minimum variance portfolio.

        Args:
            constraints: Optional constraints dictionary

        Returns:
            Dictionary with portfolio information
        """
        optimizer = MinVarianceOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )

        result = optimizer.optimize(constraints=constraints)

        return {
            "weights": result.weights,
            "tickers": result.tickers,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "method": "Minimum Variance Portfolio",
        }

    def find_portfolio_on_frontier(
        self,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraints: Optional[dict] = None,
    ) -> dict:
        """
        Find portfolio on efficient frontier with target return or risk.

        Args:
            target_return: Target annualized return
            target_risk: Target annualized volatility
            constraints: Optional constraints dictionary

        Returns:
            Dictionary with portfolio information
        """
        optimizer = MeanVarianceOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )

        result = optimizer.optimize(
            constraints=constraints,
            target_return=target_return,
            target_risk=target_risk,
        )

        return {
            "weights": result.weights,
            "tickers": result.tickers,
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "method": "Efficient Frontier Portfolio",
        }
