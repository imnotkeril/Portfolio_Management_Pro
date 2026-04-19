"""CVaR (Conditional Value at Risk) optimization."""

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


class CVaROptimizer(BaseOptimizer):
    """
    CVaR (Conditional Value at Risk) optimizer.
    
    Minimizes Conditional Value at Risk, which is the expected loss
    given that the loss exceeds VaR. This focuses on tail risk management.
    
    Formula: Min CVaR = Min E[Loss | Loss > VaR]
    
    Uses linear programming (CVXPy) for optimization.
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
        confidence_level: float = 0.95,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
    ) -> OptimizationResult:
        """
        Optimize portfolio to minimize CVaR.
        
        Args:
            constraints: Optional constraints dictionary
            confidence_level: Confidence level for CVaR (0.90, 0.95, or 0.99)
        
        Returns:
            OptimizationResult with CVaR-optimal weights
        """
        if not CVXPY_AVAILABLE:
            raise CalculationError(
                "CVXPy is required for CVaR optimization. "
                "Install with: pip install cvxpy>=1.4.0"
            )
        
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        
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
                "Insufficient data for CVaR optimization "
                f"(need at least 30 periods, got {T})"
            )
        
        try:
            # Prepare data
            returns_matrix = self.returns.values  # T × n matrix
            
            # CVaR optimization using Rockafellar-Uryasev LP formulation
            # on scenario losses L_t = -r_t^T w
            
            # Decision variables
            w = cp.Variable(n)  # Portfolio weights
            alpha = cp.Variable()  # VaR of loss (auxiliary variable)
            u = cp.Variable(T)  # Auxiliary variables for CVaR
            
            # Confidence level parameter
            tail_prob = 1.0 - confidence_level  # Tail mass
            
            # Objective: minimize CVaR of losses
            # CVaR = alpha + (1/(tail_prob*T)) * sum(u_t),
            # u_t >= L_t - alpha, u_t >= 0, L_t = -r_t^T w
            objective = cp.Minimize(
                alpha + (1.0 / (tail_prob * T)) * cp.sum(u)
            )
            
            # Constraints
            constraints_list = [
                # Weights sum to 1
                cp.sum(w) == 1.0,
                
                # Weight bounds
                w >= min_bounds,
                w <= max_bounds,
                
                # CVaR auxiliary constraints over losses
                # u_t >= L_t - alpha, L_t = -r_t^T w
                u >= -returns_matrix @ w - alpha,
                u >= 0,
            ]
            
            # Return constraint (if specified)
            if constraints_obj.min_return is not None:
                mean_returns = self._mean_returns.values
                constraints_list.append(
                    mean_returns @ w >= constraints_obj.min_return
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
            for solver in solvers_to_try:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        solved = True
                        break
                except Exception:
                    continue
            
            if not solved:
                raise CalculationError(
                    f"CVaR optimization failed: {problem.status}. "
                    "Tried solvers: ECOS, OSQP, SCS, CLARABEL"
                )
            
            weights = np.array(w.value)
            
            # Handle numerical issues
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate actual empirical tail loss for metadata (daily units)
            portfolio_returns = (self.returns @ weights).values
            losses = -portfolio_returns
            var_loss = float(np.quantile(losses, confidence_level))
            tail_losses = losses[losses >= var_loss]
            cvar_loss = (
                float(tail_losses.mean())
                if len(tail_losses) > 0
                else var_loss
            )
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="CVaR Optimization",
                success=True,
                message=(
                    f"CVaR optimization completed "
                    f"(confidence: {confidence_level:.0%})"
                ),
                metadata={
                    "cvar_loss_daily": cvar_loss,
                    "var_loss_daily": var_loss,
                    # Legacy aliases for backward compatibility.
                    "cvar": cvar_loss,
                    "var": var_loss,
                    "confidence_level": confidence_level,
                    "tail_probability": tail_prob,
                    "scenario_count": T,
                    "recommended_scenario_count": 500,
                    "scenario_count_warning": (
                        "Guide recommends 500+ scenarios for stable CVaR estimates"
                        if T < 500
                        else None
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
            logger.error(f"CVaR optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="CVaR Optimization",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

