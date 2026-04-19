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
        cvar_cap_relax: float = 1.08,
        optimization_mode: str = "cvar_cap",
        risk_aversion: float = 1.0,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
    ) -> OptimizationResult:
        """
        Optimize Mean-CVaR portfolio.
        
        Args:
            constraints: Optional constraints dictionary
            confidence_level: Confidence level for CVaR in (0, 1)
            cvar_cap_relax: CVaR cap multiplier over min-CVaR optimum
                           (used in cvar_cap mode)
            optimization_mode:
                - "cvar_cap": maximize return with CVaR <= relax * min_CVaR
                - "penalty": maximize return - lambda * CVaR (legacy fallback)
            risk_aversion: Lambda for legacy penalty mode
        
        Returns:
            OptimizationResult with Mean-CVaR optimal weights
        """
        if not CVXPY_AVAILABLE:
            raise CalculationError(
                "CVXPy is required for Mean-CVaR optimization. "
                "Install with: pip install cvxpy>=1.4.0"
            )
        
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if cvar_cap_relax <= 0:
            raise ValueError("cvar_cap_relax must be positive")
        if optimization_mode not in ["cvar_cap", "penalty"]:
            raise ValueError("optimization_mode must be 'cvar_cap' or 'penalty'")
        
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        effective_cov = self._estimate_covariance_matrix(
            covariance_method=covariance_method,
            shrinkage_alpha=shrinkage_alpha,
        )
        
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
            # Use daily expected returns to keep units consistent with CVaR,
            # which is computed from daily scenario returns.
            mean_returns_daily = self.returns.mean().values  # n vector (daily)
            
            # Common variables and CVaR terms.
            w = cp.Variable(n)
            alpha = cp.Variable()
            u = cp.Variable(T)
            tail_prob = 1.0 - confidence_level
            expected_return_daily = mean_returns_daily @ w
            cvar = alpha + (1.0 / (tail_prob * T)) * cp.sum(u)

            constraints_list = [
                cp.sum(w) == 1.0,
                w >= min_bounds,
                w <= max_bounds,
                u >= -returns_matrix @ w - alpha,
                u >= 0,
            ]
            
            # Note: We don't enforce cvar <= 0 as a hard constraint
            # because it may make the problem infeasible if all assets
            # have positive expected returns. Instead, we let the objective
            # function naturally minimize CVaR.
            
            # Return constraint (if specified)
            if constraints_obj.min_return is not None:
                # min_return is annualized in constraints; convert to daily
                min_return_daily = (1.0 + constraints_obj.min_return) ** (1.0 / self.periods_per_year) - 1.0
                constraints_list.append(
                    expected_return_daily >= min_return_daily
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
                portfolio_vol = cp.quad_form(w, effective_cov.values)
                constraints_list.append(
                    cp.sqrt(portfolio_vol) <= constraints_obj.max_volatility
                )

            # Notebook-consistent default:
            # 1) Solve min-CVaR LP.
            # 2) Maximize expected return under CVaR cap = relax * min_CVaR.
            if optimization_mode == "cvar_cap":
                min_cvar_problem = cp.Problem(cp.Minimize(cvar), constraints_list)
                min_cvar_problem.solve(solver=cp.SCS, verbose=False)
                if min_cvar_problem.status not in ["optimal", "optimal_inaccurate"]:
                    raise CalculationError(
                        f"Mean-CVaR min-CVaR phase failed: {min_cvar_problem.status}"
                    )
                min_cvar_value = float(min_cvar_problem.value)
                cvar_cap = float(cvar_cap_relax) * min_cvar_value
                cap_constraints = list(constraints_list) + [cvar <= cvar_cap]
                objective = cp.Maximize(expected_return_daily)
                problem = cp.Problem(objective, cap_constraints)
            else:
                objective = cp.Maximize(expected_return_daily - risk_aversion * cvar)
                min_cvar_value = None
                cvar_cap = None
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
                        relaxed_constraints = [cp.sum(w) == 1.0, w >= min_bounds, w <= max_bounds, u >= -returns_matrix @ w - alpha, u >= 0]
                        if constraints_obj.max_cash_weight is not None:
                            cash_indices = [
                                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
                            ]
                            if cash_indices:
                                relaxed_constraints.append(
                                    cp.sum([w[i] for i in cash_indices]) <= constraints_obj.max_cash_weight
                                )
                        
                        if optimization_mode == "cvar_cap":
                            relaxed_min_problem = cp.Problem(cp.Minimize(cvar), relaxed_constraints)
                            relaxed_min_problem.solve(solver=cp.SCS, verbose=False)
                            if relaxed_min_problem.status in ["optimal", "optimal_inaccurate"]:
                                relaxed_min_cvar_value = float(relaxed_min_problem.value)
                                relaxed_cvar_cap = float(cvar_cap_relax) * relaxed_min_cvar_value
                                relaxed_cap_constraints = list(relaxed_constraints) + [cvar <= relaxed_cvar_cap]
                                relaxed_problem = cp.Problem(cp.Maximize(expected_return_daily), relaxed_cap_constraints)
                            else:
                                relaxed_problem = cp.Problem(cp.Maximize(expected_return_daily - risk_aversion * cvar), relaxed_constraints)
                        else:
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
            
            # Calculate Mean-CVaR ratio in consistent annualized units
            mean_cvar_ratio = None
            annualized_cvar = abs(cvar_value) * np.sqrt(self.periods_per_year)
            if annualized_cvar > 1e-6:
                mean_cvar_ratio = metrics["expected_return"] / annualized_cvar
            
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
                    "cvar_loss_daily": cvar_value,
                    "var_loss_daily": var_value,
                    # Legacy aliases.
                    "cvar": cvar_value,
                    "annualized_cvar": annualized_cvar,
                    "var": var_value,
                    "mean_cvar_ratio": mean_cvar_ratio,
                    "confidence_level": confidence_level,
                    "tail_probability": tail_prob,
                    "scenario_count": T,
                    "recommended_scenario_count": 500,
                    "scenario_count_warning": (
                        "Guide recommends 500+ scenarios for stable CVaR estimates"
                        if T < 500
                        else None
                    ),
                    "optimization_mode": optimization_mode,
                    "cvar_cap_relax": (
                        float(cvar_cap_relax) if optimization_mode == "cvar_cap" else None
                    ),
                    "min_cvar_lp_value": (
                        float(min_cvar_value) if optimization_mode == "cvar_cap" and min_cvar_value is not None else None
                    ),
                    "cvar_cap": (
                        float(cvar_cap) if optimization_mode == "cvar_cap" and cvar_cap is not None else None
                    ),
                    "problem_status": problem.status,
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
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

