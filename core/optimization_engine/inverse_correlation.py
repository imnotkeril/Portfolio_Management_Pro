"""Inverse Correlation Weighting optimization."""

import logging
from typing import Optional

import numpy as np

from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class InverseCorrelationOptimizer(BaseOptimizer):
    """
    Inverse Correlation Weighting optimizer.

    Analytical method (no optimization needed) that allocates weights
    inversely proportional to average correlation with other assets.

    Algorithm:
    1. Calculate average correlation to other assets: avg_corr_i
    2. Calculate diversification score: div_score_i = 1 - avg_corr_i
    3. Normalize weights: w_i = div_score_i / Σ div_score_j
    """

    def optimize(
        self,
        constraints: Optional[dict[str, any]] = None,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
    ) -> OptimizationResult:
        """
        Optimize portfolio using inverse correlation weighting.

        Args:
            constraints: Optional constraints dictionary
            (Note: This method is analytical, constraints are applied
            after calculation)

        Returns:
            OptimizationResult with inverse correlation weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()

        n = len(self.tickers)

        try:
            # Build correlation matrix directly from returns (train),
            # consistent with notebook inverse-correlation heuristic.
            corr_df = self.returns.corr().copy()
            corr_df = corr_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            corr_matrix = corr_df.values.astype(float)
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            for cash_idx in cash_indices:
                # Keep CASH neutral in correlation-based weighting.
                corr_matrix[cash_idx, :] = 0.0
                corr_matrix[:, cash_idx] = 0.0
                corr_matrix[cash_idx, cash_idx] = 1.0
            corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
            corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

            # Notebook-consistent inverse-correlation score:
            # mean_row_i = sum_j rho_ij / n
            # w_i ∝ 1 / mean_row_i
            mean_row = corr_matrix.sum(axis=1) / max(n, 1)
            inv_scores = 1.0 / np.maximum(mean_row, 1e-10)

            # Exclude CASH from heuristic signal; allocation to CASH is handled by bounds.
            for idx in cash_indices:
                inv_scores[idx] = 0.0

            if inv_scores.sum() <= 1e-20:
                # Fallback to equal weights if all scores are zero
                weights = np.ones(n) / n
            else:
                weights = inv_scores / inv_scores.sum()

            # Apply constraints (clip to bounds)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)

            # Calculate average correlation for metadata
            avg_corr = float(np.mean(mean_row))

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Inverse Correlation Weighting",
                success=True,
                message="Inverse correlation weighting completed",
                metadata={
                    "average_correlation": avg_corr,
                    "mean_row_correlation": mean_row.tolist(),
                    "inverse_correlation_scores": inv_scores.tolist(),
                    # Legacy alias.
                    "diversification_scores": inv_scores.tolist(),
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Inverse correlation weighting failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Inverse Correlation Weighting",
                success=False,
                message=f"Calculation failed: {str(e)}",
            )

    def _build_constraints(
        self, constraints: Optional[dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)
