"""Robust optimization with uncertainty sets."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class RobustOptimizer(BaseOptimizer):
    """
    Robust optimizer with uncertainty sets.
    
    Accounts for parameter uncertainty by optimizing over uncertainty sets
    for returns and covariance. This results in more stable weights that
    are less sensitive to estimation error.
    
    Method: Min-max optimization
    - Min over weights
    - Max over uncertainty set
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
        uncertainty_radius_returns: float = 0.1,
        uncertainty_radius_cov: float = 0.1,
        objective: Optional[str] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using robust optimization.
        
        Args:
            constraints: Optional constraints dictionary
            uncertainty_radius_returns: Uncertainty radius for returns
                                       (0.0 to 1.0, default: 0.1 = 10%)
            uncertainty_radius_cov: Uncertainty radius for covariance
                                   (0.0 to 1.0, default: 0.1 = 10%)
        
        Returns:
            OptimizationResult with robust optimal weights
        """
        if not CVXPY_AVAILABLE:
            # Fallback to mean-variance if CVXPy not available
            logger.warning(
                "CVXPy not available, using mean-variance as fallback"
            )
            from core.optimization_engine.mean_variance import (
                MeanVarianceOptimizer,
            )
            fallback = MeanVarianceOptimizer(
                self.returns,
                self.risk_free_rate,
                self.periods_per_year,
            )
            return fallback.optimize(constraints=constraints, objective=objective)
        
        # If objective is specified and not maximize_sharpe, use mean-variance
        # Robust optimization currently only supports maximize_sharpe
        if objective and objective != "maximize_sharpe":
            logger.info(
                f"Robust optimization only supports maximize_sharpe, "
                f"falling back to mean-variance for objective: {objective}"
            )
            from core.optimization_engine.mean_variance import (
                MeanVarianceOptimizer,
            )
            fallback = MeanVarianceOptimizer(
                self.returns,
                self.risk_free_rate,
                self.periods_per_year,
            )
            return fallback.optimize(constraints=constraints, objective=objective)
        
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        try:
            # Robust optimization: minimize worst-case risk
            # Subject to worst-case return constraint
            
            # Decision variable
            w = cp.Variable(n)
            
            # Nominal (estimated) values
            mu_nominal = self._mean_returns.values
            Sigma_nominal = self._cov_matrix.values
            
            # Uncertainty sets
            # Returns: mu in {mu_nominal ± uncertainty_radius}
            # Covariance: Sigma in {Sigma_nominal ± uncertainty_radius}
            
            # Worst-case return (minimum over uncertainty set)
            # For ellipsoidal uncertainty: min mu^T w - ||w||_2 * radius
            worst_case_return = (
                mu_nominal @ w
                - uncertainty_radius_returns * cp.norm(w, 2)
            )
            
            # Worst-case variance (maximum over uncertainty set)
            # For covariance uncertainty: max w^T Sigma w
            # Simplified: use nominal + uncertainty adjustment
            worst_case_variance = cp.quad_form(
                w,
                Sigma_nominal * (1.0 + uncertainty_radius_cov),
            )
            
            # Objective: minimize worst-case variance
            # Subject to worst-case return >= target
            # For simplicity, we maximize worst-case Sharpe-like ratio
            # Or minimize worst-case variance with return constraint
            
            # Use risk-adjusted return: max (return - lambda * risk)
            # This implements maximize_sharpe for robust optimization
            # Note: We use variance directly instead of sqrt(variance) for DCP compliance
            # This is equivalent to maximizing (return - lambda * variance) instead of Sharpe
            risk_aversion = 1.0
            cp_objective = cp.Maximize(
                worst_case_return
                - risk_aversion * worst_case_variance
            )
            
            # Constraints
            constraints_list = [
                # Weights sum to 1
                cp.sum(w) == 1.0,
                
                # Weight bounds
                w >= min_bounds,
                w <= max_bounds,
                
                # Worst-case return >= minimum acceptable
                worst_case_return >= mu_nominal.min() * 0.5,
            ]
            
            # Return constraint (if specified)
            if constraints_obj.min_return is not None:
                constraints_list.append(
                    worst_case_return >= constraints_obj.min_return
                )
            
            # Cash constraint (if specified)
            if constraints_obj.max_cash_weight is not None:
                cash_indices = [
                    i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
                ]
                if cash_indices:
                    # Constraint: sum of CASH weights <= max_cash_weight
                    constraints_list.append(
                        cp.sum([w[i] for i in cash_indices]) <= constraints_obj.max_cash_weight
                    )
            
            # Risk constraints
            if constraints_obj.max_volatility is not None:
                # Use variance constraint instead of volatility for DCP compliance
                # max_volatility^2 = max_variance
                max_variance = constraints_obj.max_volatility ** 2
                constraints_list.append(
                    worst_case_variance <= max_variance
                )
            
            # Solve
            problem = cp.Problem(cp_objective, constraints_list)
            # Try multiple solvers
            solvers_to_try = [
                cp.ECOS,
                cp.OSQP,
                cp.SCS,
                cp.CLARABEL,
            ]
            
            solved = False
            last_status = None
            last_exception = None
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    last_status = problem.status
                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        solved = True
                        break
                except Exception as e:
                    last_exception = e
                    logger.debug(f"Solver {solver} failed: {e}")
                    continue
            
            if not solved:
                # If infeasible or status is None, try relaxing constraints
                if last_status == "infeasible" or last_status is None:
                    logger.warning(
                        "Robust optimization is infeasible or failed to solve. "
                        "This may be due to conflicting constraints. "
                        "Try relaxing min_return or max_cash_weight constraints."
                    )
                    # Try without min_return constraint if it exists
                    if constraints_obj.min_return is not None:
                        logger.info("Retrying without min_return constraint...")
                        # Rebuild constraints without min_return
                        relaxed_constraints = [
                            cp.sum(w) == 1.0,
                            w >= min_bounds,
                            w <= max_bounds,
                            worst_case_return >= mu_nominal.min() * 0.5,
                        ]
                        if constraints_obj.max_cash_weight is not None:
                            cash_indices = [
                                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
                            ]
                            if cash_indices:
                                relaxed_constraints.append(
                                    cp.sum([w[i] for i in cash_indices]) <= constraints_obj.max_cash_weight
                                )
                        if constraints_obj.max_volatility is not None:
                            relaxed_constraints.append(
                                cp.sqrt(worst_case_variance) <= constraints_obj.max_volatility
                            )
                        
                        relaxed_problem = cp.Problem(cp_objective, relaxed_constraints)
                        for solver in solvers_to_try:
                            try:
                                relaxed_problem.solve(solver=solver, verbose=False)
                                if relaxed_problem.status in ["optimal", "optimal_inaccurate"]:
                                    problem = relaxed_problem
                                    solved = True
                                    logger.info("Successfully solved with relaxed constraints")
                                    break
                            except Exception as e:
                                logger.debug(f"Relaxed solver {solver} failed: {e}")
                                continue
                
                if not solved:
                    status_msg = last_status if last_status else "unknown (solver error)"
                    error_msg = f"Robust optimization failed: {status_msg}."
                    if last_exception:
                        error_msg += f" Last error: {str(last_exception)}"
                    error_msg += " Tried solvers: ECOS, OSQP, SCS, CLARABEL. "
                    error_msg += "Try relaxing constraints (min_return, max_cash_weight)."
                    raise CalculationError(error_msg)
            
            weights = np.array(w.value)
            
            # Handle numerical issues
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Calculate metrics using nominal (not worst-case) values
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Robust Optimization",
                success=True,
                message=(
                    "Robust optimization completed "
                    f"(uncertainty: {uncertainty_radius_returns:.0%})"
                ),
                metadata={
                    "uncertainty_radius_returns": uncertainty_radius_returns,
                    "uncertainty_radius_cov": uncertainty_radius_cov,
                    "problem_status": problem.status,
                },
            )
        except Exception as e:
            logger.error(f"Robust optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Robust Optimization",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

