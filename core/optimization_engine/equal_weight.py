"""Equal weight (1/N) optimization."""

import logging
from typing import Dict, Optional

import numpy as np

from core.optimization_engine.base import BaseOptimizer, OptimizationResult

logger = logging.getLogger(__name__)


class EqualWeightOptimizer(BaseOptimizer):
    """
    Equal weight (1/N) portfolio optimizer.
    
    This is the simplest optimization method - assigns equal weights
    to all assets. Often used as a benchmark.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio with equal weights.
        
        Args:
            constraints: Optional constraints (ignored for equal weight)
        
        Returns:
            OptimizationResult with equal weights
        """
        n_assets = len(self.tickers)
        
        # Equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights)
        
        return OptimizationResult(
            weights=weights,
            tickers=self.tickers,
            expected_return=metrics["expected_return"],
            volatility=metrics["volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            method="Equal Weight (1/N)",
            success=True,
            message="Equal weight portfolio created successfully",
            metadata={"n_assets": n_assets},
        )

