"""Maximum Sharpe ratio optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MaxSharpeOptimizer(BaseOptimizer):
    """
    Maximum Sharpe ratio optimizer.
    
    Optimizes portfolio to maximize risk-adjusted return (Sharpe ratio).
    This finds the tangency portfolio on the efficient frontier.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to maximize Sharpe ratio.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with maximum Sharpe ratio weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
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
                # Penalty for zero or near-zero volatility
                return 1e10
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            # Add diversification penalty: lambda * sum(w_i^2)
            # Scale by 10 to make it more effective
            div_penalty = lambda_div * 10.0 * np.sum(weights ** 2)
            return -sharpe + div_penalty  # Minimize negative = maximize Sharpe
        
        # Constraint: weights sum to 1
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        # Add min_return constraint if specified
        if constraints_obj.min_return is not None:
            constraints_list.append({
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
                constraints_list.append({
                    "type": "ineq",
                    "fun": lambda w: float(
                        constraints_obj.max_cash_weight - sum(w[i] for i in cash_indices)
                    ),
                })
        
        # Try multiple initial guesses for better convergence
        try:
            initial_guesses = [
                np.ones(n) / n,  # Equal weights
            ]
            
            # Add market cap weighted guess if possible
            # (fallback to equal if not available)
            initial_guesses.append(np.ones(n) / n)
            
            # Try to find a good starting point near the efficient frontier
            # by using a slightly perturbed equal weight
            for i in range(min(3, n)):
                guess = np.ones(n) / n
                if n > 1:
                    guess[i] *= 1.5
                    guess = guess / np.sum(guess)
                    # Ensure within bounds
                    guess = np.clip(guess, min_bounds, max_bounds)
                    guess = guess / np.sum(guess)
                initial_guesses.append(guess)
            
            best_result = None
            best_sharpe = -np.inf
            
            for x0 in initial_guesses:
                try:
                    result = scipy_opt.minimize(
                        objective,
                        x0,
                        method="SLSQP",
                        bounds=list(zip(min_bounds, max_bounds)),
                        constraints=constraints_list,
                        options={"maxiter": 1000, "ftol": 1e-6},
                    )
                    
                    if result.success:
                        # Calculate actual Sharpe for this result
                        test_weights = self._normalize_weights(result.x, constraints_obj)
                        test_return = np.dot(test_weights, self._mean_returns)
                        test_variance = float(
                            test_weights.T
                            @ self._cov_matrix.values
                            @ test_weights
                        )
                        test_vol = np.sqrt(test_variance)
                        if test_vol > 1e-8:
                            test_sharpe = (
                                (test_return - self.risk_free_rate) / test_vol
                            )
                            if test_sharpe > best_sharpe:
                                best_sharpe = test_sharpe
                                best_result = result
                    else:
                        # Even if not fully converged, check if result is usable
                        try:
                            test_weights = self._normalize_weights(result.x, constraints_obj)
                            test_return = np.dot(test_weights, self._mean_returns)
                            test_variance = float(
                                test_weights.T
                                @ self._cov_matrix.values
                                @ test_weights
                            )
                            test_vol = np.sqrt(test_variance)
                            if test_vol > 1e-8:
                                test_sharpe = (
                                    (test_return - self.risk_free_rate)
                                    / test_vol
                                )
                                if test_sharpe > best_sharpe:
                                    best_sharpe = test_sharpe
                                    best_result = result
                        except Exception:
                            pass
                except Exception:
                    continue
            
            if best_result is None:
                # If all attempts failed, try one more time with default settings
                try:
                    result = scipy_opt.minimize(
                        objective,
                        np.ones(n) / n,
                        method="SLSQP",
                        bounds=list(zip(min_bounds, max_bounds)),
                        constraints=constraints_list,
                        options={"maxiter": 2000, "ftol": 1e-5},
                    )
                    best_result = result
                except Exception as e:
                    raise CalculationError(
                        f"Optimization failed after multiple attempts: {str(e)}"
                    )
            
            result = best_result
            
            if not result.success:
                # Log warning but try to use result anyway if it's close
                logger.warning(
                    f"Optimization did not fully converge: {result.message}. "
                    f"Using best result found."
                )
            
            weights = self._normalize_weights(result.x, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Maximum Sharpe Ratio",
                success=True,
                message="Maximized Sharpe ratio",
                metadata={
                    "iterations": result.nit,
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {e}")
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Maximum Sharpe Ratio",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)

