"""Robust optimization with uncertainty sets."""

import logging
from typing import Optional

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
        constraints: Optional[dict[str, any]] = None,
        uncertainty_radius_returns: float = 0.1,
        uncertainty_radius_cov: float = 0.1,
        objective: Optional[str] = None,
        robust_kappa: Optional[float] = None,
        robust_lambda: Optional[float] = None,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
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
            logger.warning("CVXPy not available, using mean-variance as fallback")
            from core.optimization_engine.mean_variance import (
                MeanVarianceOptimizer,
            )

            fallback = MeanVarianceOptimizer(
                self.returns,
                self.risk_free_rate,
                self.periods_per_year,
            )
            return fallback.optimize(constraints=constraints, objective=objective)

        if objective and objective != "maximize_sharpe":
            logger.warning(
                "Robust optimizer uses fixed robust utility objective; "
                "ignoring unsupported objective=%s",
                objective,
            )

        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        effective_cov = self._estimate_covariance_matrix(
            covariance_method=covariance_method,
            shrinkage_alpha=shrinkage_alpha,
        )

        n = len(self.tickers)

        try:
            # Decision variable
            w = cp.Variable(n)

            # Nominal annualized moments
            mu_nominal = self._mean_returns.values
            Sigma_nominal = effective_cov.values
            T_train = max(len(self.returns), 1)
            Sigma_mu = Sigma_nominal / T_train

            # Notebook-aligned robust parameters.
            # Keep backward compatibility with existing uncertainty_* params.
            kappa = (
                float(robust_kappa)
                if robust_kappa is not None
                else float(uncertainty_radius_returns)
            )
            lam = (
                float(robust_lambda)
                if robust_lambda is not None
                else max(float(uncertainty_radius_cov) * 10.0, 1e-8)
            )

            # Worst-case mean under ellipsoidal uncertainty:
            # mu_hat'w - kappa * sqrt(w' * Sigma_mu * w)
            Sigma_mu_chol = np.linalg.cholesky(Sigma_mu + 1e-12 * np.eye(n))
            worst_case_mean = mu_nominal @ w - kappa * cp.norm(Sigma_mu_chol @ w, 2)
            risk_penalty = cp.quad_form(w, Sigma_nominal)
            cp_objective = cp.Maximize(worst_case_mean - lam * risk_penalty)

            # Constraints
            constraints_list = [
                cp.sum(w) == 1.0,
                w >= min_bounds,
                w <= max_bounds,
            ]

            # Return constraint (if specified): enforce on worst-case mean
            if constraints_obj.min_return is not None:
                constraints_list.append(worst_case_mean >= constraints_obj.min_return)

            # Cash constraint (if specified)
            if constraints_obj.max_cash_weight is not None:
                cash_indices = [
                    i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
                ]
                if cash_indices:
                    # Constraint: sum of CASH weights <= max_cash_weight
                    constraints_list.append(
                        cp.sum([w[i] for i in cash_indices])
                        <= constraints_obj.max_cash_weight
                    )

            # Risk constraints
            if constraints_obj.max_volatility is not None:
                max_variance = constraints_obj.max_volatility**2
                constraints_list.append(risk_penalty <= max_variance)

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
                        ]
                        if constraints_obj.max_cash_weight is not None:
                            cash_indices = [
                                i
                                for i, ticker in enumerate(self.tickers)
                                if ticker == "CASH"
                            ]
                            if cash_indices:
                                relaxed_constraints.append(
                                    cp.sum([w[i] for i in cash_indices])
                                    <= constraints_obj.max_cash_weight
                                )
                        if constraints_obj.max_volatility is not None:
                            relaxed_constraints.append(
                                risk_penalty <= constraints_obj.max_volatility**2
                            )

                        relaxed_problem = cp.Problem(cp_objective, relaxed_constraints)
                        for solver in solvers_to_try:
                            try:
                                relaxed_problem.solve(solver=solver, verbose=False)
                                if relaxed_problem.status in [
                                    "optimal",
                                    "optimal_inaccurate",
                                ]:
                                    problem = relaxed_problem
                                    solved = True
                                    logger.info(
                                        "Successfully solved with relaxed constraints"
                                    )
                                    break
                            except Exception as e:
                                logger.debug(f"Relaxed solver {solver} failed: {e}")
                                continue

                if not solved:
                    status_msg = (
                        last_status if last_status else "unknown (solver error)"
                    )
                    error_msg = f"Robust optimization failed: {status_msg}."
                    if last_exception:
                        error_msg += f" Last error: {str(last_exception)}"
                    error_msg += " Tried solvers: ECOS, OSQP, SCS, CLARABEL. "
                    error_msg += (
                        "Try relaxing constraints (min_return, max_cash_weight)."
                    )
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
                    "robust_kappa": kappa,
                    "robust_lambda": lam,
                    # Legacy fields (for API compatibility)
                    "uncertainty_radius_returns": uncertainty_radius_returns,
                    "uncertainty_radius_cov": uncertainty_radius_cov,
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
                    "training_observations": T_train,
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
        self, constraints: Optional[dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)
