"""Minimum variance optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MinVarianceOptimizer(BaseOptimizer):
    """
    Minimum variance portfolio optimizer.
    
    Minimizes portfolio variance (risk) without considering expected returns.
    This is a special case of mean-variance optimization.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to minimize variance.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with minimum variance weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Objective: minimize portfolio variance = w^T * Cov * w
        def objective(weights: np.ndarray) -> float:
            return float(
                weights.T @ self._cov_matrix.values @ weights
            )
        
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
            
            weights = self._normalize_weights(result.x, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Minimum Variance",
                success=True,
                message="Minimized portfolio variance",
                metadata={
                    "iterations": result.nit,
                    "variance": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Min variance optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Minimum Variance",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)

