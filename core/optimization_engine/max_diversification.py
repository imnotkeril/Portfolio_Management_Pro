"""Maximum Diversification optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MaxDiversificationOptimizer(BaseOptimizer):
    """
    Maximum Diversification optimizer.
    
    Maximizes the diversification ratio, which measures the benefit
    from diversification.
    
    Formula: max (Σ Weight[i] × Vol[i]) / Portfolio Vol
    
    Higher ratio indicates better diversification benefit.
    
    Note: CASH is excluded from optimization as it has zero volatility
    and would distort the diversification ratio.
    """

    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to maximize diversification ratio.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with maximum diversification weights
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

        # Calculate individual asset volatilities
        individual_vols = np.sqrt(np.diag(cov_matrix))

        # Objective: maximize diversification ratio
        # Ratio = (Σ w_i * σ_i) / σ_portfolio
        # Maximize ratio = minimize negative ratio
        def objective(weights: np.ndarray) -> float:
            # Weighted sum of individual volatilities
            weighted_sum_vols = np.dot(weights, individual_vols)

            # Portfolio volatility
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_variance)

            if portfolio_vol < 1e-8:
                return 1e10

            # Diversification ratio
            div_ratio = weighted_sum_vols / portfolio_vol

            # Minimize negative = maximize
            return -div_ratio

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

        # Initial guess: inverse volatility weights
        inv_vols = 1.0 / (individual_vols + 1e-6)
        x0 = inv_vols / inv_vols.sum()

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
                    f"Max diversification optimization did not fully converge: "
                    f"{result.message}. Using best result."
                )

            weights = result.x

            # Normalize to ensure sum = 1.0
            weights = self._normalize_weights(weights, constraints_obj)

            # Apply bounds to full weights array
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            metrics = self._calculate_portfolio_metrics(weights)

            # Calculate diversification ratio for metadata
            # Use full covariance matrix for calculation
            full_individual_vols = np.sqrt(np.diag(self._cov_matrix.values))
            weighted_sum_vols = np.dot(weights, full_individual_vols)
            portfolio_vol = metrics["volatility"]
            div_ratio = (
                weighted_sum_vols / portfolio_vol
                if portfolio_vol > 0
                else 0.0
            )

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Maximum Diversification",
                success=True,
                message="Maximum diversification optimization completed",
                metadata={
                    "diversification_ratio": float(div_ratio),
                    "iterations": result.nit,
                },
            )
        except Exception as e:
            logger.error(f"Max diversification optimization failed: {e}")
            # Fallback to inverse volatility weights (excluding CASH)
            weights = np.zeros(n)
            individual_vols = np.sqrt(np.diag(self._cov_matrix.values))
            inv_vols = 1.0 / (individual_vols + 1e-6)
            # Set CASH weight to zero
            for i, ticker in enumerate(self.tickers):
                if ticker == "CASH":
                    inv_vols[i] = 0.0
            if inv_vols.sum() > 0:
                weights = inv_vols / inv_vols.sum()
            else:
                weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Maximum Diversification",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )

    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)
