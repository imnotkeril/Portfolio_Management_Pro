"""Maximum alpha optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MaxAlphaOptimizer(BaseOptimizer):
    """
    Maximum alpha optimizer.
    
    Maximizes portfolio alpha (excess return) relative to a benchmark.
    Requires benchmark returns as input.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize optimizer with benchmark.
        
        Args:
            returns: Portfolio returns DataFrame
            benchmark_returns: Benchmark returns Series (aligned)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        
        if len(benchmark_returns) != len(returns):
            raise ValueError(
                "Benchmark returns must have same length as portfolio returns"
            )
        
        benchmark_aligned = benchmark_returns.reindex(returns.index)
        if benchmark_aligned.isna().any():
            raise ValueError("Benchmark returns have missing values")
        
        self.benchmark_returns = benchmark_aligned * self.periods_per_year
        self._benchmark_mean = benchmark_aligned.mean() * periods_per_year
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to maximize alpha.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with maximum alpha weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Objective: maximize alpha = portfolio_return - benchmark_return
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, self._mean_returns)
            alpha = portfolio_return - self._benchmark_mean
            return -alpha  # Minimize negative = maximize
        
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        x0 = np.ones(n) / n
        
        try:
            result = scipy_opt.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(min_bounds, max_bounds)),
                constraints=constraints_list,
                options={"maxiter": 1000},
            )
            
            if not result.success:
                raise CalculationError(
                    f"Optimization failed: {result.message}"
                )
            
            weights = self._normalize_weights(result.x)
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate alpha
            portfolio_return = metrics["expected_return"]
            alpha = portfolio_return - self._benchmark_mean
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Maximum Alpha",
                success=True,
                message=f"Maximized alpha vs benchmark: {alpha:.2%}",
                metadata={
                    "iterations": result.nit,
                    "alpha": float(alpha),
                    "benchmark_return": float(self._benchmark_mean),
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Max alpha optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Maximum Alpha",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        constraints_obj = OptimizationConstraints(self.tickers)
        
        if constraints:
            # Call set_weight_bounds once with all parameters
            constraints_obj.set_weight_bounds(
                min_weight=constraints.get("min_weight"),
                max_weight=constraints.get("max_weight"),
                long_only=constraints.get("long_only", True),
            )
        
        return constraints_obj

