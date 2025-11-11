"""Minimum Correlation optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MinCorrelationOptimizer(BaseOptimizer):
    """
    Minimum Correlation optimizer.
    
    Minimizes the average pairwise correlation between assets in the
    portfolio. This results in assets that move independently,
    providing better diversification and crisis resistance.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to minimize average pairwise correlation.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with minimum correlation weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Handle CASH: set minimum volatility to avoid division by zero
        cov_matrix = self._cov_matrix.values.copy()
        cash_indices = [
            i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
        ]
        # Set minimum volatility for CASH to avoid numerical issues
        for cash_idx in cash_indices:
            if cov_matrix[cash_idx, cash_idx] < 1e-8:
                cov_matrix[cash_idx, cash_idx] = 1e-8
        
        # Build correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Objective: minimize average pairwise correlation
        # Average correlation = Σ_i Σ_j>i w_i * w_j * corr(i,j) / pairs
        def objective(weights: np.ndarray) -> float:
            # Calculate weighted average correlation
            # Sum of weighted correlations for all pairs
            total_corr = 0.0
            pair_count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    pair_weight = weights[i] * weights[j]
                    pair_corr = corr_matrix[i, j]
                    total_corr += pair_weight * pair_corr
                    pair_count += 1
            
            if pair_count == 0:
                return 0.0
            
            avg_corr = total_corr / pair_count
            return float(avg_corr)
        
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Return constraint (if specified)
        if constraints_obj.min_return is not None:
            mean_returns = self._mean_returns.values
            constraints_list.append({
                "type": "ineq",
                "fun": lambda w: np.dot(mean_returns, w) - constraints_obj.min_return,
            })
        
        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints_list.append({
                    "type": "ineq",
                    "fun": lambda w: float(
                        constraints_obj.max_cash_weight - sum(w[i] for i in cash_indices)
                    ),
                })

        # Initial guess: equal weights
        x0 = np.ones(n) / n
        
        try:
            result = scipy_opt.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(min_bounds, max_bounds)),
                constraints=constraints_list,
                options={"maxiter": 2000, "ftol": 1e-9},
            )
            
            if not result.success:
                logger.warning(
                    f"Min correlation optimization did not fully converge: "
                    f"{result.message}. Using best result."
                )
            
            weights = result.x
            weights = self._normalize_weights(weights, constraints_obj)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate actual average correlation (non-CASH only)
            avg_corr = objective(result.x)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Minimum Correlation",
                success=True,
                message="Minimum correlation optimization completed",
                metadata={
                    "average_correlation": float(avg_corr),
                    "iterations": result.nit,
                },
            )
        except Exception as e:
            logger.error(f"Min correlation optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Minimum Correlation",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

