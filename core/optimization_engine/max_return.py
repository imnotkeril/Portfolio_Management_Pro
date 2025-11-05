"""Maximum return optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MaxReturnOptimizer(BaseOptimizer):
    """
    Maximum return optimizer.
    
    Optimizes portfolio to maximize expected return without considering risk.
    This typically results in concentrating weights in highest-return assets.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to maximize expected return.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with maximum return weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Objective: maximize return = minimize negative return
        def objective(weights: np.ndarray) -> float:
            return -float(np.dot(weights, self._mean_returns))
        
        # Constraint: weights sum to 1
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Initial guess: equal weights
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
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Maximum Return",
                success=True,
                message="Maximized expected return",
                metadata={
                    "iterations": result.nit,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Max return optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Maximum Return",
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

