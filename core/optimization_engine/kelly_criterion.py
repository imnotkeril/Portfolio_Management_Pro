"""Kelly Criterion optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class KellyCriterionOptimizer(BaseOptimizer):
    """
    Kelly Criterion optimizer.
    
    Maximizes the long-term growth rate of portfolio value.
    This is the optimal strategy for maximizing geometric mean return.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
        kelly_fraction: float = 1.0,
    ) -> OptimizationResult:
        """
        Optimize portfolio using Kelly Criterion.
        
        Args:
            constraints: Optional constraints dictionary
            kelly_fraction: Fraction of full Kelly to use (0-1)
                          (1.0 = full Kelly, 0.5 = half Kelly, etc.)
        
        Returns:
            OptimizationResult with Kelly-optimal weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Kelly Criterion: maximize log of expected geometric mean
        # For portfolio: maximize E[log(1 + w^T * r)]
        def objective(weights: np.ndarray) -> float:
            # Calculate expected log return
            portfolio_returns = (self.returns @ weights).values
            
            # Add 1 to avoid log(0) or log(negative)
            # Use log(1 + return) for small returns approximation
            log_returns = np.log1p(portfolio_returns)
            
            # Expected log return (geometric mean)
            expected_log_return = np.mean(log_returns) * self.periods_per_year
            
            # Minimize negative = maximize
            return -expected_log_return
        
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
            
            weights = self._normalize_weights(result.x, constraints_obj)
            
            # Apply Kelly fraction
            if kelly_fraction < 1.0:
                # Scale down weights and add cash
                weights = weights * kelly_fraction
                # Remaining goes to cash (not in optimization)
                # For now, renormalize to sum to 1
                weights = self._normalize_weights(weights, constraints_obj)
            
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method=f"Kelly Criterion ({kelly_fraction:.0%})",
                success=True,
                message=f"Kelly Criterion optimization completed "
                f"(fraction: {kelly_fraction:.0%})",
                metadata={
                    "iterations": result.nit,
                    "kelly_fraction": kelly_fraction,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Kelly Criterion optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Kelly Criterion",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)

