"""Minimum Correlation optimization."""

import logging
from typing import Optional

import numpy as np
import scipy.optimize as scipy_opt

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
        constraints: Optional[dict[str, any]] = None,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
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

        # Build correlation matrix from returns (not from shrunk covariance),
        # consistent with notebook objective w'Rw on train correlation.
        corr_df = self.returns.corr().copy()
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        corr_matrix = corr_df.values.astype(float)

        # Handle CASH: enforce near-zero correlation to other assets and 1.0 on diagonal
        # to avoid spurious numerical effects from constant return series.
        cash_indices = [i for i, ticker in enumerate(self.tickers) if ticker == "CASH"]
        for cash_idx in cash_indices:
            corr_matrix[cash_idx, :] = 0.0
            corr_matrix[:, cash_idx] = 0.0
            corr_matrix[cash_idx, cash_idx] = 1.0
        corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Objective: minimize weighted portfolio correlation score:
        # min w'Rw  (not pairwise-average approximation).
        def objective(weights: np.ndarray) -> float:
            return float(weights.T @ corr_matrix @ weights)

        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]

        # Return constraint (if specified)
        if constraints_obj.min_return is not None:
            mean_returns = self._mean_returns.values
            constraints_list.append(
                {
                    "type": "ineq",
                    "fun": lambda w: np.dot(mean_returns, w)
                    - constraints_obj.min_return,
                }
            )

        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints_list.append(
                    {
                        "type": "ineq",
                        "fun": lambda w: float(
                            constraints_obj.max_cash_weight
                            - sum(w[i] for i in cash_indices)
                        ),
                    }
                )

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
                    "weighted_corr_score": float(avg_corr),
                    # Legacy alias.
                    "average_correlation": float(avg_corr),
                    "iterations": result.nit,
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
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
        self, constraints: Optional[dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)
