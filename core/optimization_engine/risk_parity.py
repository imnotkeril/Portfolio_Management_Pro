"""Risk Parity optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity optimizer.
    
    Allocates weights so that each asset contributes equally to
    portfolio risk. This typically results in better diversification
    than equal weights.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using risk parity.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with risk parity weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Risk parity: minimize sum of squared differences in risk
        # contributions
        def objective(weights: np.ndarray) -> float:
            # Calculate risk contributions
            portfolio_vol = np.sqrt(
                weights.T @ self._cov_matrix @ weights
            )
            
            if portfolio_vol == 0:
                return 1e10
            
            # Marginal contribution to risk
            mcr = (self._cov_matrix @ weights) / portfolio_vol
            
            # Risk contribution per asset
            rc = weights * mcr
            
            # Target: equal risk contribution
            target_rc = portfolio_vol / n
            
            # Minimize sum of squared differences
            diff = rc - target_rc
            return float(np.sum(diff ** 2))
        
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
            
            weights = self._normalize_weights(result.x)
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Risk Parity",
                success=True,
                message="Risk parity optimization completed",
                metadata={
                    "iterations": result.nit,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Risk Parity",
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

