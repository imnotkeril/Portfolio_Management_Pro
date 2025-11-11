"""Inverse Correlation Weighting optimization."""

import logging
from typing import Dict, Optional

import numpy as np

from core.exceptions import CalculationError
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
        constraints: Optional[Dict[str, any]] = None,
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
            
            # Step 1: Calculate average correlation for each asset
            avg_corrs = np.zeros(n)
            for i in range(n):
                # Average correlation with all other assets
                other_corrs = [
                    corr_matrix[i, j] for j in range(n) if j != i
                ]
                if len(other_corrs) > 0:
                    avg_corrs[i] = np.mean(other_corrs)
            
            # Step 2: Calculate diversification score
            # div_score = 1 - avg_corr (higher = more diversification)
            div_scores = 1.0 - avg_corrs
            
            # Ensure non-negative scores
            div_scores = np.maximum(div_scores, 0.0)
            
            # Step 3: Normalize to get weights
            if div_scores.sum() == 0:
                # Fallback to equal weights if all scores are zero
                weights = np.ones(n) / n
            else:
                weights = div_scores / div_scores.sum()
            
            # Apply constraints (clip to bounds)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate average correlation for metadata
            avg_corr = float(np.mean(avg_corrs))
            
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
                    "diversification_scores": div_scores.tolist(),
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
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

