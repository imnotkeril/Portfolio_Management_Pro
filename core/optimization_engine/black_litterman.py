"""Black-Litterman optimization."""

import logging
from typing import Any, Optional

import numpy as np

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints
from core.optimization_engine.mean_variance import MeanVarianceOptimizer

logger = logging.getLogger(__name__)


class BlackLittermanOptimizer(BaseOptimizer):
    """
    Black-Litterman optimizer.

    Combines market equilibrium (implied returns from market weights)
    with investor views using Bayesian updating.

    Algorithm:
    1. Reverse optimization: implied returns from market weights
    2. Bayesian update with user views
    3. Optimize using updated returns (Mean-Variance)

    Requires:
    - market_weights: Market cap weights (equilibrium)
    - views: List of views (e.g., "Tech outperforms by 5%")
    - view_confidences: Confidence in each view
    - tau: Scaling parameter (typically 0.05-0.1)
    """

    def optimize(
        self,
        constraints: Optional[dict[str, any]] = None,
        market_weights: Optional[np.ndarray] = None,
        views: Optional[list[dict[str, any]]] = None,
        view_confidences: Optional[list[float]] = None,
        tau: float = 0.05,
        objective: Optional[str] = None,
        covariance_method: str = "shrink",
        shrinkage_alpha: float = 0.25,
    ) -> OptimizationResult:
        """
        Optimize portfolio using Black-Litterman model.

        Args:
            constraints: Optional constraints dictionary
            market_weights: Market cap weights (equilibrium)
                          If None, uses equal weights
            views: List of view dictionaries, each with:
                  - "assets": List of ticker indices or names
                  - "returns": Expected return for the view
                  - "type": "absolute" or "relative"
            view_confidences: Confidence levels for each view (0-1)
            tau: Scaling parameter (default: 0.05)

        Returns:
            OptimizationResult with Black-Litterman optimal weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        effective_cov = self._estimate_covariance_matrix(
            covariance_method=covariance_method,
            shrinkage_alpha=shrinkage_alpha,
        )

        n = len(self.tickers)

        # Exclude CASH from Black-Litterman optimization
        # CASH has zero volatility and risk-free return,
        # which distorts optimization
        cash_indices = [i for i, ticker in enumerate(self.tickers) if ticker == "CASH"]
        non_cash_indices = [
            i for i, ticker in enumerate(self.tickers) if ticker != "CASH"
        ]

        if len(non_cash_indices) == 0:
            raise CalculationError(
                "Black-Litterman requires at least one non-CASH asset"
            )

        try:
            # Step 1: Reverse optimization to get implied returns
            # Work only with non-CASH assets
            n_non_cash = len(non_cash_indices)

            # Extract non-CASH covariance and returns
            cov_non_cash = effective_cov.iloc[non_cash_indices, non_cash_indices].values
            mean_returns_non_cash = self._mean_returns.iloc[non_cash_indices].values

            # If market_weights not provided, use equal weights for non-CASH
            if market_weights is None:
                market_weights_non_cash = np.ones(n_non_cash) / n_non_cash
            else:
                market_weights_array = np.array(market_weights)
                if len(market_weights_array) != n:
                    raise ValueError(
                        f"market_weights length {len(market_weights_array)} "
                        f"!= number of assets {n}"
                    )
                # Extract non-CASH weights
                market_weights_non_cash = market_weights_array[non_cash_indices]
                # Normalize to sum to 1 (excluding CASH)
                market_weights_non_cash = (
                    market_weights_non_cash / market_weights_non_cash.sum()
                )

            # Implied returns from market equilibrium (non-CASH only)
            # pi = lambda * Sigma * w_market
            # where lambda is risk aversion parameter
            # Estimate lambda from market portfolio Sharpe ratio
            market_portfolio_vol = np.sqrt(
                market_weights_non_cash.T @ cov_non_cash @ market_weights_non_cash
            )

            if market_portfolio_vol > 0:
                market_portfolio_return = np.dot(
                    market_weights_non_cash, mean_returns_non_cash
                )
                # Risk aversion: (return - rf) / variance
                lambda_market = (market_portfolio_return - self.risk_free_rate) / (
                    market_portfolio_vol**2
                )
            else:
                lambda_market = 1.0

            # Implied equilibrium returns (non-CASH only)
            pi = lambda_market * cov_non_cash @ market_weights_non_cash

            # Step 2: Bayesian update with views (non-CASH only)
            if views and len(views) > 0:
                # Build view matrix P and view returns Q
                num_views = len(views)
                P = np.zeros((num_views, n_non_cash))
                Q = np.zeros(num_views)

                # Default confidence if not provided
                if view_confidences is None:
                    view_confidences = [0.5] * num_views

                for i, view in enumerate(views):
                    view_assets = view.get("assets", [])
                    view_return = view.get("returns", 0.0)
                    view_type = view.get("type", "absolute")

                    if view_type == "absolute":
                        # Absolute view: specific return for assets
                        for asset in view_assets:
                            if isinstance(asset, str):
                                if asset == "CASH":
                                    logger.warning(
                                        "CASH views are ignored in "
                                        "Black-Litterman optimization"
                                    )
                                    continue
                                try:
                                    original_idx = self.tickers.index(asset)
                                    # Map to non-CASH index
                                    if original_idx in non_cash_indices:
                                        asset_idx = non_cash_indices.index(original_idx)
                                    else:
                                        continue
                                except ValueError:
                                    continue
                            else:
                                # Numeric index
                                if asset in non_cash_indices:
                                    asset_idx = non_cash_indices.index(asset)
                                else:
                                    continue
                            P[i, asset_idx] = 1.0
                        Q[i] = view_return
                    elif view_type == "relative":
                        # Relative view: asset 1 outperforms asset 2 by X%
                        if len(view_assets) >= 2:
                            asset1_name = view_assets[0]
                            asset2_name = view_assets[1]

                            # Skip if either is CASH
                            if asset1_name == "CASH" or asset2_name == "CASH":
                                logger.warning(
                                    "CASH views are ignored in "
                                    "Black-Litterman optimization"
                                )
                                continue

                            asset1 = (
                                self.tickers.index(asset1_name)
                                if isinstance(asset1_name, str)
                                else asset1_name
                            )
                            asset2 = (
                                self.tickers.index(asset2_name)
                                if isinstance(asset2_name, str)
                                else asset2_name
                            )

                            # Map to non-CASH indices
                            if (
                                asset1 in non_cash_indices
                                and asset2 in non_cash_indices
                            ):
                                asset1_idx = non_cash_indices.index(asset1)
                                asset2_idx = non_cash_indices.index(asset2)
                                P[i, asset1_idx] = 1.0
                                P[i, asset2_idx] = -1.0
                                Q[i] = view_return

                # Build view uncertainty matrix Omega (diagonal).
                # Notebook-consistent base: diag(P @ (tau * Sigma) @ P.T).
                # Confidence scales uncertainty: lower confidence => larger Omega.
                tau_Sigma = tau * cov_non_cash
                omega_base = np.diag(P @ tau_Sigma @ P.T)
                omega_base = np.maximum(omega_base, 1e-12)
                confidence_scale = np.array(
                    [1.0 / max(float(conf), 1e-6) for conf in view_confidences],
                    dtype=float,
                )
                Omega = np.diag(omega_base * confidence_scale)

                # Black-Litterman formula (non-CASH only):
                # mu_BL = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1
                #        * [(tau*Sigma)^-1 * pi + P^T * Omega^-1 * Q]
                tau_Sigma_inv = np.linalg.inv(tau_Sigma)
                Omega_inv = np.linalg.inv(Omega)

                # Calculate posterior mean returns (non-CASH only)
                A = tau_Sigma_inv + P.T @ Omega_inv @ P
                b = tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q
                mu_BL_non_cash = np.linalg.solve(A, b)
            else:
                # No views: use equilibrium returns (non-CASH only)
                mu_BL_non_cash = pi

            # Step 3: Optimize using updated returns
            # Include CASH in final optimization, but with correct parameters
            # CASH has risk-free return and zero volatility/correlation

            # Create MeanVarianceOptimizer with ALL assets (including CASH)
            mv_optimizer = MeanVarianceOptimizer(
                self.returns,
                self.risk_free_rate,
                self.periods_per_year,
            )
            mv_optimizer._cov_matrix = effective_cov.copy()

            # Temporarily update mean returns:
            # - Non-CASH assets: use Black-Litterman returns
            # - CASH: use risk-free rate (correct expected return)
            original_mean_returns = mv_optimizer._mean_returns.copy()
            updated_mean_returns = original_mean_returns.copy()

            # Update non-CASH returns with Black-Litterman returns
            for i, non_cash_idx in enumerate(non_cash_indices):
                ticker = self.tickers[non_cash_idx]
                updated_mean_returns[ticker] = mu_BL_non_cash[i]

            # CASH gets risk-free rate (if exists)
            for cash_idx in cash_indices:
                cash_ticker = self.tickers[cash_idx]
                updated_mean_returns[cash_ticker] = self.risk_free_rate

            mv_optimizer._mean_returns = updated_mean_returns

            # Optimize with BL posterior returns on the selected covariance.
            # Notebook 02 default is max Sharpe on posterior mu.
            optimize_kwargs: dict[str, Any] = {"constraints": constraints}
            optimize_kwargs["covariance_method"] = covariance_method
            optimize_kwargs["shrinkage_alpha"] = shrinkage_alpha
            optimize_kwargs["objective"] = objective or "maximize_sharpe"

            result = mv_optimizer.optimize(**optimize_kwargs)

            # Restore original mean returns
            mv_optimizer._mean_returns = original_mean_returns

            # Result already contains weights for all assets including CASH
            # Just update metadata and method name
            result.method = "Black-Litterman"
            result.message = "Black-Litterman optimization completed"
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "target_return": float(
                        np.dot(market_weights_non_cash, mu_BL_non_cash)
                    ),
                    "lambda_market": float(lambda_market),
                    "cash_excluded_from_implied_returns": True,
                    "tau": float(tau),
                    "covariance_method": covariance_method,
                    "shrinkage_alpha": (
                        float(shrinkage_alpha)
                        if covariance_method == "shrink"
                        else None
                    ),
                }
            )

            return result
        except Exception as e:
            logger.error(f"Black-Litterman optimization failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)

            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Black-Litterman",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )

    def _build_constraints(
        self, constraints: Optional[dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)
