"""Risk Parity optimization."""

import logging
from typing import Optional

import numpy as np
import scipy.optimize as scipy_opt

from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class RiskParityOptimizer(BaseOptimizer):
    """
    Risk Parity optimizer.

    Allocates weights so that each asset contributes equally to
    portfolio risk. This typically results in better diversification
    than equal weights.

    Algorithm: Minimize sum of squared differences between risk contributions
    and their mean (target equal risk contribution).
    """

    def optimize(
        self,
        constraints: Optional[dict[str, any]] = None,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
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

        # Handle CASH: set minimum volatility to avoid division by zero
        # CASH typically has zero volatility, which breaks risk parity
        effective_cov = self._estimate_covariance_matrix(
            covariance_method=covariance_method,
            shrinkage_alpha=shrinkage_alpha,
        )
        cov_matrix = effective_cov.values.copy()
        cash_indices = [i for i, ticker in enumerate(self.tickers) if ticker == "CASH"]
        # Set minimum volatility for CASH to avoid numerical issues
        for cash_idx in cash_indices:
            if cov_matrix[cash_idx, cash_idx] < 1e-8:
                cov_matrix[cash_idx, cash_idx] = 1e-8
        non_cash_indices = [i for i in range(n) if i not in set(cash_indices)]

        # Risk parity: minimize sum of squared differences in risk
        # contributions from their mean
        def objective(weights: np.ndarray) -> float:
            # Calculate portfolio volatility
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_vol = np.sqrt(portfolio_variance)

            if portfolio_vol < 1e-8:
                return 1e10

            # Marginal contribution to risk (MCR)
            # MCR[i] = d(portfolio_vol) / d(weight[i])
            mcr = (cov_matrix @ weights) / portfolio_vol

            # Risk contribution per asset: RC[i] = weight[i] * MCR[i]
            risk_contrib = weights * mcr

            # Target: equal risk contribution for non-CASH assets.
            # CASH can have near-zero variance and should not distort ERC target.
            target_rc = (
                risk_contrib[non_cash_indices]
                if len(non_cash_indices) > 0
                else risk_contrib
            )
            # Minimize sum of squared differences from mean
            mean_rc = np.mean(target_rc)
            diff = target_rc - mean_rc

            return float(np.sum(diff**2))

        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]

        # Return constraint (if specified)
        if constraints_obj.min_return is not None:
            mean_returns = self._mean_returns.values
            constraints_list.append(
                {
                    "type": "ineq",
                    "fun": lambda w: np.dot(mean_returns, w)
                    - constraints_obj.min_return,
                }
            )

        # Add explicit cash constraint if specified
        if constraints_obj.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            if cash_indices:
                # Constraint: sum of CASH weights <= max_cash_weight
                constraints_list.append(
                    {
                        "type": "ineq",
                        "fun": lambda w: float(
                            constraints_obj.max_cash_weight
                            - sum(w[i] for i in cash_indices)
                        ),
                    }
                )

        # Initial guess: inverse volatility weights
        individual_vols = np.sqrt(np.diag(cov_matrix))
        inv_vols = 1.0 / (individual_vols + 1e-6)
        x0 = inv_vols / inv_vols.sum()

        try:
            result = scipy_opt.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(min_bounds, max_bounds)),
                constraints=constraints_list,
                options={"maxiter": 2000, "ftol": 1e-9},
            )

            if not result.success:
                logger.warning(
                    f"Risk parity optimization did not fully converge: "
                    f"{result.message}. Using best result."
                )

            weights = result.x

            # Normalize to ensure sum = 1.0
            weights = self._normalize_weights(weights, constraints_obj)

            # Apply bounds to full weights array
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

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
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
                    "erc_scope": (
                        "non_cash_assets" if len(non_cash_indices) > 0 else "all_assets"
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Risk parity optimization failed: {e}")
            # Fallback to inverse volatility weights
            weights = np.zeros(n)
            individual_vols = np.sqrt(np.diag(self._cov_matrix.values))
            inv_vols = 1.0 / (individual_vols + 1e-6)
            # Set CASH weight to zero
            for i, ticker in enumerate(self.tickers):
                if ticker == "CASH":
                    inv_vols[i] = 0.0
            if inv_vols.sum() > 0:
                weights = inv_vols / inv_vols.sum()
            else:
                weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Risk Parity",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )

    def _build_constraints(
        self, constraints: Optional[dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)
