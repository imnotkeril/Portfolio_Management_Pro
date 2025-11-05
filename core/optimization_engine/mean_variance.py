"""Mean-Variance (Markowitz) optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MeanVarianceOptimizer(BaseOptimizer):
    """
    Markowitz Mean-Variance optimization.
    
    Optimizes portfolio to maximize expected return for a given risk level,
    or minimize risk for a given return level.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        risk_aversion: float = 0.5,
    ) -> OptimizationResult:
        """
        Optimize portfolio using mean-variance optimization.
        
        Args:
            constraints: Optional constraints dictionary
            target_return: Target annualized return (if None, maximize Sharpe)
            target_risk: Target annualized volatility (if None, minimize risk)
            risk_aversion: Risk aversion parameter (0-1, higher = more risk averse)
        
        Returns:
            OptimizationResult with optimal weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        # Objective function: minimize risk-adjusted return
        # If target_return is specified, minimize variance subject to return
        # If target_risk is specified, maximize return subject to risk
        # Otherwise, maximize Sharpe ratio (risk-adjusted return)
        
        if target_return is not None:
            # Minimize variance subject to target return
            return self._minimize_variance_with_return(
                target_return, min_bounds, max_bounds
            )
        elif target_risk is not None:
            # Maximize return subject to target risk
            return self._maximize_return_with_risk(
                target_risk, min_bounds, max_bounds
            )
        else:
            # Maximize Sharpe ratio
            return self._maximize_sharpe(min_bounds, max_bounds)
    
    def _minimize_variance_with_return(
        self,
        target_return: float,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
    ) -> OptimizationResult:
        """Minimize variance subject to target return."""
        n = len(self.tickers)
        
        # Objective: minimize portfolio variance = w^T * Cov * w
        def objective(weights: np.ndarray) -> float:
            return float(weights.T @ self._cov_matrix @ weights)
        
        # Constraint: weights sum to 1
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
            {
                "type": "eq",
                "fun": lambda w: np.dot(w, self._mean_returns) - target_return,
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
                constraints=constraints,
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
                method="Mean-Variance (Min Variance)",
                success=True,
                message=f"Minimized variance for target return {target_return:.2%}",
                metadata={
                    "target_return": target_return,
                    "iterations": result.nit,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Mean-Variance (Min Variance)",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _maximize_return_with_risk(
        self,
        target_risk: float,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
    ) -> OptimizationResult:
        """Maximize return subject to target risk."""
        n = len(self.tickers)
        
        # Objective: maximize return = -w^T * mean_returns (minimize negative)
        def objective(weights: np.ndarray) -> float:
            return -float(np.dot(weights, self._mean_returns))
        
        # Constraint: weights sum to 1, volatility <= target_risk
        def risk_constraint(weights: np.ndarray) -> float:
            variance = float(weights.T @ self._cov_matrix @ weights)
            volatility = np.sqrt(variance)
            return target_risk - volatility  # Must be >= 0
        
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
            {
                "type": "ineq",
                "fun": risk_constraint,
            },
        ]
        
        x0 = np.ones(n) / n
        
        try:
            result = scipy_opt.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(min_bounds, max_bounds)),
                constraints=constraints,
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
                method="Mean-Variance (Max Return)",
                success=True,
                message=f"Maximized return for target risk {target_risk:.2%}",
                metadata={
                    "target_risk": target_risk,
                    "iterations": result.nit,
                },
            )
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Mean-Variance (Max Return)",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _maximize_sharpe(
        self,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
    ) -> OptimizationResult:
        """Maximize Sharpe ratio."""
        n = len(self.tickers)
        
        # Objective: minimize negative Sharpe ratio
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, self._mean_returns)
            portfolio_variance = weights.T @ self._cov_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_variance)
            
            if portfolio_vol == 0:
                return 1e10  # Penalty for zero volatility
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative = maximize Sharpe
        
        constraints = [
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
                constraints=constraints,
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
                method="Mean-Variance (Max Sharpe)",
                success=True,
                message="Maximized Sharpe ratio",
                metadata={
                    "iterations": result.nit,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Mean-Variance (Max Sharpe)",
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
            # to avoid overwriting previous values
            constraints_obj.set_weight_bounds(
                min_weight=constraints.get("min_weight"),
                max_weight=constraints.get("max_weight"),
                long_only=constraints.get("long_only", True),
            )
        
        return constraints_obj

