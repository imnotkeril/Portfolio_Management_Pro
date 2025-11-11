"""Mean-CVaR optimization."""

import logging
from typing import Dict, Optional

import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MeanCVaROptimizer(BaseOptimizer):
    """
    Mean-CVaR optimizer.
    
    Maximizes the ratio of expected return to CVaR, providing optimal
    trade-off between return and tail risk.
    
    Objective: max (Expected Return / CVaR)
    
    Uses convex optimization (CVXPy) for optimization.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
        confidence_level: float = 0.95,
    ) -> OptimizationResult:
        """
        Optimize portfolio to maximize Return / CVaR ratio.
        
        Args:
            constraints: Optional constraints dictionary
            confidence_level: Confidence level for CVaR (0.90, 0.95, or 0.99)
        
        Returns:
            OptimizationResult with Mean-CVaR optimal weights
        """
        if not CVXPY_AVAILABLE:
            raise CalculationError(
                "CVXPy is required for Mean-CVaR optimization. "
                "Install with: pip install cvxpy>=1.4.0"
            )
        
        if confidence_level not in [0.90, 0.95, 0.99]:
            raise ValueError(
                "Confidence level must be 0.90, 0.95, or 0.99"
            )
        
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        T = len(self.returns)  # Number of historical periods
        
        if T < 30:
            raise CalculationError(
                "Insufficient data for Mean-CVaR optimization "
                f"(need at least 30 periods, got {T})"
            )
        
        try:
            # Prepare data
            returns_matrix = self.returns.values  # T × n matrix
            mean_returns = self._mean_returns.values  # n vector
            
            # Decision variables
            w = cp.Variable(n)  # Portfolio weights
            alpha = cp.Variable()  # VaR (auxiliary variable)
            u = cp.Variable(T)  # Auxiliary variables for CVaR
            
            # Confidence level parameter
            beta = 1.0 - confidence_level  # Tail probability
            
            # Expected return
            expected_return = mean_returns @ w
            
            # CVaR = alpha + (1/(beta*T)) * sum(u)
            cvar = alpha + (1.0 / (beta * T)) * cp.sum(u)
            
            # Objective: maximize Return / CVaR
            # For numerical stability, we maximize Return - lambda * CVaR
            # where lambda is a risk aversion parameter
            risk_aversion = 1.0
            objective = cp.Maximize(expected_return - risk_aversion * cvar)
            
            # Constraints
            constraints_list = [
                # Weights sum to 1
                cp.sum(w) == 1.0,
                
                # Weight bounds
                w >= min_bounds,
                w <= max_bounds,
                
                # CVaR auxiliary constraints
                u >= -returns_matrix @ w - alpha,
                u >= 0,
            ]
            
            # Note: We don't enforce cvar <= 0 as a hard constraint
            # because it may make the problem infeasible if all assets
            # have positive expected returns. Instead, we let the objective
            # function naturally minimize CVaR.
            
            # Return constraint (if specified)
            if constraints_obj.min_return is not None:
                constraints_list.append(
                    expected_return >= constraints_obj.min_return
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
            
            # Risk constraints (if specified)
            if constraints_obj.max_volatility is not None:
                portfolio_vol = cp.quad_form(w, self._cov_matrix.values)
                constraints_list.append(
                    cp.sqrt(portfolio_vol) <= constraints_obj.max_volatility
                )
            
            # Solve - try multiple solvers
            problem = cp.Problem(objective, constraints_list)
            
            # Try solvers in order of preference
            solvers_to_try = [
                cp.ECOS,
                cp.OSQP,
                cp.SCS,
                cp.CLARABEL,
            ]
            
            solved = False
            last_status = None
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    last_status = problem.status
                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        solved = True
                        break
                except Exception as e:
                    logger.debug(f"Solver {solver} failed: {e}")
                    continue
            
            if not solved:
                # If infeasible, try relaxing constraints
                if last_status == "infeasible":
                    logger.warning(
                        "Mean-CVaR optimization is infeasible. "
                        "This may be due to conflicting constraints. "
                        "Try relaxing min_return or max_cash_weight constraints."
                    )
                    # Try without min_return constraint if it exists
                    if constraints_obj.min_return is not None:
                        logger.info("Retrying without min_return constraint...")
                        relaxed_constraints = [
                            c for c in constraints_list 
                            if not (isinstance(c, cp.constraints.Inequality) and 
                                   hasattr(c, 'args') and len(c.args) > 0 and
                                   str(c.args[0]).find('expected_return') >= 0)
                        ]
                        # Rebuild constraints without min_return
                        relaxed_constraints = [
                            cp.sum(w) == 1.0,
                            w >= min_bounds,
                            w <= max_bounds,
                            u >= -returns_matrix @ w - alpha,
                            u >= 0,
                        ]
                        if constraints_obj.max_cash_weight is not None:
                            cash_indices = [
                                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
                            ]
                            if cash_indices:
                                relaxed_constraints.append(
                                    cp.sum([w[i] for i in cash_indices]) <= constraints_obj.max_cash_weight
                                )
                        
                        relaxed_problem = cp.Problem(objective, relaxed_constraints)
                        for solver in solvers_to_try:
                            try:
                                relaxed_problem.solve(solver=solver, verbose=False)
                                if relaxed_problem.status in ["optimal", "optimal_inaccurate"]:
                                    problem = relaxed_problem
                                    solved = True
                                    logger.info("Successfully solved with relaxed constraints")
                                    break
                            except Exception:
                                continue
                
                if not solved:
                    raise CalculationError(
                        f"Mean-CVaR optimization failed: {last_status or 'unknown'}. "
                        "Tried solvers: ECOS, OSQP, SCS, CLARABEL. "
                        "Try relaxing constraints (min_return, max_cash_weight)."
                    )
            
            weights = np.array(w.value)
            
            # Handle numerical issues
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate actual CVaR for the optimized portfolio
            portfolio_returns = (self.returns @ weights).values
            var_value = np.percentile(
                portfolio_returns,
                (1.0 - confidence_level) * 100
            )
            tail_returns = portfolio_returns[portfolio_returns <= var_value]
            cvar_value = float(tail_returns.mean()) if len(tail_returns) > 0 else var_value
            
            # Calculate Mean-CVaR ratio
            mean_cvar_ratio = None
            if cvar_value < 0 and abs(cvar_value) > 1e-6:
                mean_cvar_ratio = metrics["expected_return"] / abs(cvar_value)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Mean-CVaR",
                success=True,
                message=(
                    f"Mean-CVaR optimization completed "
                    f"(confidence: {confidence_level:.0%})"
                ),
                metadata={
                    "cvar": cvar_value,
                    "var": var_value,
                    "mean_cvar_ratio": mean_cvar_ratio,
                    "confidence_level": confidence_level,
                    "problem_status": problem.status,
                },
            )
        except Exception as e:
            logger.error(f"Mean-CVaR optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Mean-CVaR",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

