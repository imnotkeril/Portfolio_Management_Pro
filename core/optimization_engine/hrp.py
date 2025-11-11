"""Hierarchical Risk Parity (HRP) optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class HRPOptimizer(BaseOptimizer):
    """
    Hierarchical Risk Parity (HRP) optimizer.
    
    Implements the HRP algorithm from López de Prado (2016).
    HRP uses hierarchical clustering to form a quasi-diagonal covariance
    matrix, then allocates weights using recursive bisection.
    
    Advantages:
    - More stable than standard risk parity
    - Less sensitive to estimation error
    - Better for large portfolios (20+ assets)
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using Hierarchical Risk Parity.
        
        Args:
            constraints: Optional constraints dictionary
            (Note: HRP doesn't fully support all constraints due to
            its hierarchical nature, but basic weight bounds are applied)
        
        Returns:
            OptimizationResult with HRP weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        if n < 2:
            # For single asset, return 100% weight to it
            weights = np.ones(n)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Hierarchical Risk Parity",
                success=True,
                message="Single asset portfolio",
            )
        
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
            
            # Work with full covariance matrix
            cov_df = self._cov_matrix.copy()
            for cash_idx in cash_indices:
                if cov_df.iloc[cash_idx, cash_idx] < 1e-8:
                    cov_df.iloc[cash_idx, cash_idx] = 1e-8
            
            # Step 1: Build correlation matrix
            std_devs = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Step 2: Hierarchical clustering
            # Convert correlation to distance (1 - corr)
            distance_matrix = 1 - corr_matrix
            # Make symmetric and ensure non-negative
            distance_matrix = (
                distance_matrix + distance_matrix.T
            ) / 2
            distance_matrix = np.maximum(distance_matrix, 0.0)
            
            # Convert to condensed distance matrix for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method="ward")
            
            # Step 3: Quasi-diagonalization (reorder by cluster)
            ordered_indices = leaves_list(linkage_matrix)
            
            # Reorder covariance matrix
            cov_ordered = cov_df.iloc[
                ordered_indices, ordered_indices
            ].values
            
            # Step 4: Recursive bisection for weights
            weights_ordered = self._recursive_bisection(
                cov_ordered,
                list(range(n)),
            )
            
            # Reorder weights back to original order
            weights = np.zeros(n)
            for i, orig_idx in enumerate(ordered_indices):
                weights[orig_idx] = weights_ordered[i]
            
            # Apply constraints (clip to bounds)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Validate constraints
            if constraints_obj.long_only and (weights < 0).any():
                logger.warning(
                    "HRP produced negative weights, clipping to zero"
                )
                weights = np.maximum(weights, 0.0)
                weights = self._normalize_weights(weights, constraints_obj)
            
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Hierarchical Risk Parity",
                success=True,
                message="HRP optimization completed",
                metadata={
                    "n_assets": n,
                },
            )
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Hierarchical Risk Parity",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_correlation_matrix(self) -> pd.DataFrame:
        """Build correlation matrix from covariance matrix."""
        # Convert covariance to correlation
        # corr = cov / (std_i * std_j)
        std_devs = np.sqrt(np.diag(self._cov_matrix.values))
        corr_matrix = self._cov_matrix.values / np.outer(std_devs, std_devs)
        
        # Handle division by zero
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure symmetric and valid correlation values
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)  # Clip to valid range
        
        return pd.DataFrame(
            corr_matrix,
            index=self.tickers,
            columns=self.tickers,
        )
    
    def _cluster_assets(
        self,
        corr_matrix: pd.DataFrame,
    ) -> np.ndarray:
        """
        Perform hierarchical clustering on correlation matrix.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
        
        Returns:
            Linkage matrix from scipy.cluster.hierarchy
        """
        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1 - corr_matrix.values
        
        # Convert to condensed form for linkage
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering using Ward's method
        # Ward minimizes variance within clusters
        linkage_matrix = linkage(
            condensed_distances,
            method="ward",
        )
        
        return linkage_matrix
    
    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """
        Recursively bisect portfolio to allocate weights.
        
        This is the core of HRP algorithm. It recursively splits
        the portfolio into two clusters and allocates weights
        inversely proportional to cluster variance.
        
        Args:
            cov_matrix: Covariance matrix (ordered)
            indices: Asset indices in current cluster
        
        Returns:
            Weights array for assets in this cluster
        """
        n = len(indices)
        
        if n == 1:
            # Base case: single asset gets full weight
            return np.array([1.0])
        
        if n == 2:
            # Base case: two assets, allocate inversely to variance
            var1 = cov_matrix[0, 0]
            var2 = cov_matrix[1, 1]
            
            # Inverse variance weighting
            w1 = 1.0 / var1 if var1 > 0 else 0.5
            w2 = 1.0 / var2 if var2 > 0 else 0.5
            
            weights = np.array([w1, w2])
            return weights / weights.sum()
        
        # For more than 2 assets, split into two clusters
        # Find split point that minimizes inter-cluster correlation
        # Simple approach: split at midpoint
        split_point = n // 2
        
        # Split indices
        left_indices = indices[:split_point]
        right_indices = indices[split_point:]
        
        # Get sub-covariance matrices
        left_cov = cov_matrix[:split_point, :split_point]
        right_cov = cov_matrix[split_point:, split_point:]
        
        # Calculate cluster variances
        left_var = np.trace(left_cov) / len(left_indices)
        right_var = np.trace(right_cov) / len(right_indices)
        
        # Allocate weight inversely to variance
        if left_var > 0 and right_var > 0:
            left_weight = 1.0 / left_var
            right_weight = 1.0 / right_var
        else:
            # Equal weights if variance is zero
            left_weight = 1.0
            right_weight = 1.0
        
        total_weight = left_weight + right_weight
        left_weight /= total_weight
        right_weight /= total_weight
        
        # Recursively allocate within each cluster
        left_weights = self._recursive_bisection(
            left_cov,
            np.arange(len(left_indices)),
        )
        right_weights = self._recursive_bisection(
            right_cov,
            np.arange(len(right_indices)),
        )
        
        # Combine weights
        weights = np.concatenate([
            left_weights * left_weight,
            right_weights * right_weight,
        ])
        
        return weights
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

