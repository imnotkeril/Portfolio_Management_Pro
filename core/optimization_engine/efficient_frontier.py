"""Efficient frontier generation."""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress numpy array comparison warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from core.optimization_engine.max_sharpe import MaxSharpeOptimizer
from core.optimization_engine.mean_variance import MeanVarianceOptimizer
from core.optimization_engine.min_variance import MinVarianceOptimizer

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
        self._min_return_at_min_vol = (
            min_var_result.expected_return or 0.0
        )
    
    def generate_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
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
        min_ret = float(
            min_var_result.expected_return or self._min_return_at_min_vol
        )
        
        # Get max return portfolio (highest return on frontier)
        from core.optimization_engine.max_return import MaxReturnOptimizer
        max_ret_opt = MaxReturnOptimizer(
            self.returns,
            self.risk_free_rate,
            self.periods_per_year,
        )
        max_ret_result = max_ret_opt.optimize(constraints=constraints)
        max_ret = float(max_ret_result.expected_return or self._max_return)
        
        # Generate efficient frontier with both efficient and inefficient parts
        # Efficient part: from min variance to max return (minimize variance)
        # Inefficient part: from min return to min variance (same optimization,
        # but these are below efficient frontier)
        
        # For efficient part: target returns from min_ret to max_ret
        # This will generate the full curve including inefficient part
        # We'll filter later to separate efficient from inefficient
        
        # Generate target returns following ChatGPT example approach
        # Use np.linspace for uniform distribution across return range
        # This creates a smooth curve when plotted
        
        # Find the return of individual assets to find lower bound
        # This helps create the "rounded bottom" of the curve
        individual_returns = self._mean_returns.values
        # Convert to list/array and ensure all are floats
        if isinstance(individual_returns, (pd.Series, np.ndarray)):
            individual_returns_list = [float(x) for x in individual_returns]
        else:
            individual_returns_list = [float(x) for x in individual_returns]
        min_individual_ret = float(min(individual_returns_list))
        
        # Extend range slightly below min_ret for "rounded bottom" effect
        # Use a small buffer (1% of range) to ensure smooth curve
        ret_range = max_ret - min_ret
        lower_bound = min(min_individual_ret, min_ret - 0.01 * ret_range)
        
        # Generate target returns uniformly across full range
        # This ensures smooth curve without interpolation artifacts
        # Use all n_points for the full range (min to max)
        all_target_returns = np.linspace(lower_bound, max_ret, n_points)
        
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
                            portfolios_list.append({
                                "weights": result.weights,
                                "tickers": result.tickers,
                                "expected_return": ret_float,
                                "volatility": vol_float,
                                "sharpe_ratio": (
                                    float(result.sharpe_ratio)
                                    if result.sharpe_ratio is not None
                                    else None
                                ),
                            })
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
        
        # Sort by volatility to ensure smooth curve
        # Ensure all values are floats (not arrays) before sorting
        if len(volatilities_list) > 0:
            # Convert all to floats explicitly to avoid array comparison issues
            volatilities_float = [float(v) for v in volatilities_list]
            returns_float = [float(r) for r in returns_list]
            
            # Create list of tuples with index for stable sorting
            # Use index in portfolios_list to avoid comparing dicts/arrays
            indexed_data = list(
                enumerate(zip(volatilities_float, returns_float))
            )
            # Sort by volatility only (second element of tuple)
            sorted_indexed = sorted(
                indexed_data,
                key=lambda x: x[1][0]
                # Sort by volatility (first element of tuple)
            )
            
            # Rebuild lists in sorted order
            volatilities_list = [v for _, (v, _) in sorted_indexed]
            returns_list = [r for _, (_, r) in sorted_indexed]
            portfolios_list = [portfolios_list[i] for i, _ in sorted_indexed]
        
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

