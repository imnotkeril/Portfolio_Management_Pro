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
        objective: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using mean-variance optimization.
        
        Args:
            constraints: Optional constraints dictionary
            target_return: Target annualized return (if None, maximize Sharpe)
            target_risk: Target annualized volatility (if None, minimize risk)
            risk_aversion: Risk aversion parameter (0-1, higher = more risk averse)
            objective: Objective function: "maximize_sharpe", "minimize_volatility", 
                     "maximize_return" (overrides target_return/target_risk)
        
        Returns:
            OptimizationResult with optimal weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        # Use objective parameter if provided
        if objective == "maximize_sharpe":
            return self._maximize_sharpe(
                min_bounds, max_bounds, constraints_obj
            )
        elif objective == "minimize_volatility":
            return self._minimize_variance_with_return(
                target_return=None,  # Will minimize variance without return constraint
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                constraints_obj=constraints_obj,
            )
        elif objective == "maximize_return":
            # Maximize return without risk constraint
            return self._maximize_return_with_risk(
                target_risk=None,  # Will maximize return without risk constraint
                min_bounds=min_bounds,
                max_bounds=max_bounds,
                constraints_obj=constraints_obj,
            )
        
        # Legacy behavior: use target_return/target_risk if objective not specified
        # Objective function: minimize risk-adjusted return
        # If target_return is specified, minimize variance subject to return
        # If target_risk is specified, maximize return subject to risk
        # Otherwise, maximize Sharpe ratio (risk-adjusted return)
        
        if target_return is not None:
            # Minimize variance subject to target return
            return self._minimize_variance_with_return(
                target_return, min_bounds, max_bounds, constraints_obj
            )
        elif target_risk is not None:
            # Maximize return subject to target risk
            return self._maximize_return_with_risk(
                target_risk, min_bounds, max_bounds, constraints_obj
            )
        else:
            # Maximize Sharpe ratio (default)
            return self._maximize_sharpe(min_bounds, max_bounds, constraints_obj)
    
    def _minimize_variance_with_return(
        self,
        target_return: Optional[float],
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        constraints_obj: OptimizationConstraints,
    ) -> OptimizationResult:
        """Minimize variance subject to target return (or just minimize variance if target_return is None)."""
        n = len(self.tickers)
        
        # Get diversification regularization parameter
        lambda_div = constraints_obj.diversification_lambda or 0.0
        
        # Objective: minimize portfolio variance = w^T * Cov * w
        # Add diversification penalty if specified
        def objective(weights: np.ndarray) -> float:
            variance = float(
                weights.T @ self._cov_matrix.values @ weights
            )
            # Add diversification penalty: lambda * sum(w_i^2)
            # This encourages more equal weights (higher diversification)
            # Scale by 10 to make it more effective
            div_penalty = lambda_div * 10.0 * np.sum(weights ** 2)
            return variance + div_penalty
        
        # Constraint: weights sum to 1
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Add return constraint only if target_return is specified
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: float(
                    np.dot(w, self._mean_returns) - target_return
                ),
            })
        
        # Add min_return constraint if specified
        if constraints_obj.min_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(
                    np.dot(w, self._mean_returns) - constraints_obj.min_return
                ),
            })
        
        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints.append({
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
                constraints=constraints,
                options={"maxiter": 1000},
            )
            
            if not result.success:
                error_msg = (
                    result.message
                    if result.message
                    else "Optimization did not converge"
                )
                raise CalculationError(f"Optimization failed: {error_msg}")
            
            weights = self._normalize_weights(result.x, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Create message based on whether target_return is specified
            if target_return is not None:
                message = f"Minimized variance for target return {target_return:.2%}"
            else:
                message = "Minimized portfolio variance"
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Mean-Variance (Min Variance)",
                success=True,
                message=message,
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
        target_risk: Optional[float],
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        constraints_obj: OptimizationConstraints,
    ) -> OptimizationResult:
        """Maximize return subject to target risk (or just maximize return if target_risk is None)."""
        n = len(self.tickers)
        
        # Get diversification regularization parameter
        lambda_div = constraints_obj.diversification_lambda or 0.0
        
        # Objective: maximize return = -w^T * mean_returns (minimize negative)
        # Add diversification penalty if specified
        def objective(weights: np.ndarray) -> float:
            expected_return = float(np.dot(weights, self._mean_returns))
            # Add diversification penalty: lambda * sum(w_i^2)
            div_penalty = lambda_div * 10.0 * np.sum(weights ** 2)
            return -expected_return + div_penalty
        
        # Constraint: weights sum to 1
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Add risk constraint only if target_risk is specified
        if target_risk is not None:
            def risk_constraint(weights: np.ndarray) -> float:
                variance = float(
                    weights.T @ self._cov_matrix.values @ weights
                )
                volatility = np.sqrt(variance)
                return target_risk - volatility  # Must be >= 0
            
            constraints.append({
                "type": "ineq",
                "fun": risk_constraint,
            })
        
        # Add min_return constraint if specified
        if constraints_obj.min_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(
                    np.dot(w, self._mean_returns) - constraints_obj.min_return
                ),
            })
        
        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w: float(
                        constraints_obj.max_cash_weight - sum(w[i] for i in cash_indices)
                    ),
                })
        
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
                error_msg = (
                    result.message
                    if result.message
                    else "Optimization did not converge"
                )
                raise CalculationError(f"Optimization failed: {error_msg}")
            
            weights = self._normalize_weights(result.x, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Create message based on whether target_risk is specified
            if target_risk is not None:
                message = f"Maximized return for target risk {target_risk:.2%}"
            else:
                message = "Maximized expected return"
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Mean-Variance (Max Return)",
                success=True,
                message=message,
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
        constraints_obj: OptimizationConstraints,
    ) -> OptimizationResult:
        """Maximize Sharpe ratio."""
        n = len(self.tickers)
        
        # Get diversification regularization parameter
        lambda_div = constraints_obj.diversification_lambda or 0.0
        
        # Objective: minimize negative Sharpe ratio
        # Add diversification penalty if specified
        def objective(weights: np.ndarray) -> float:
            portfolio_return = np.dot(weights, self._mean_returns)
            portfolio_variance = float(
                weights.T @ self._cov_matrix.values @ weights
            )
            portfolio_vol = np.sqrt(portfolio_variance)
            
            if portfolio_vol < 1e-8:
                return 1e10  # Penalty for zero or near-zero volatility
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            # Add diversification penalty: lambda * sum(w_i^2)
            div_penalty = lambda_div * 10.0 * np.sum(weights ** 2)
            return -sharpe + div_penalty  # Minimize negative = maximize Sharpe
        
        constraints = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Add min_return constraint if specified
        if constraints_obj.min_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(
                    np.dot(w, self._mean_returns) - constraints_obj.min_return
                ),
            })
        
        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w: float(
                        constraints_obj.max_cash_weight - sum(w[i] for i in cash_indices)
                    ),
                })
        
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
                error_msg = (
                    result.message
                    if result.message
                    else "Optimization did not converge"
                )
                raise CalculationError(f"Optimization failed: {error_msg}")
            
            weights = self._normalize_weights(result.x, constraints_obj)
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
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)

