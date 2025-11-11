"""Base optimizer class and result types."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import CalculationError
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: np.ndarray
    """Optimal weights (normalized to sum to 1.0)"""
    
    tickers: List[str]
    """List of tickers in order corresponding to weights"""
    
    expected_return: Optional[float] = None
    """Expected portfolio return (annualized)"""
    
    volatility: Optional[float] = None
    """Portfolio volatility (annualized)"""
    
    sharpe_ratio: Optional[float] = None
    """Sharpe ratio of optimized portfolio"""
    
    method: str = ""
    """Optimization method name"""
    
    success: bool = True
    """Whether optimization converged successfully"""
    
    message: str = ""
    """Optimization status message"""
    
    metadata: Dict[str, any] = None
    """Additional metadata (iterations, solve time, etc.)"""
    
    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, any]:
        """Convert result to dictionary."""
        return {
            "weights": self.weights.tolist(),
            "tickers": self.tickers,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "method": self.method,
            "success": self.success,
            "message": self.message,
            "metadata": self.metadata,
        }
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get weights as dictionary mapping ticker to weight."""
        return dict(zip(self.tickers, self.weights.tolist()))


class BaseOptimizer(ABC):
    """
    Abstract base class for all portfolio optimizers.
    
    All optimization methods must inherit from this class and implement
    the optimize() method.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize base optimizer.
        
        Args:
            returns: DataFrame with tickers as columns and dates as index
            risk_free_rate: Annual risk-free rate (default: 4.35%)
            periods_per_year: Trading periods per year (default: 252)
        """
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")
        
        if len(returns.columns) == 0:
            raise ValueError("Returns DataFrame must have at least one column")
        
        # Validate returns
        if returns.isna().all().any():
            raise ValueError("Some tickers have all NaN returns")
        
        self.returns = returns
        self.tickers = returns.columns.tolist()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # Calculate common statistics
        self._mean_returns = returns.mean() * periods_per_year  # Annualized
        self._cov_matrix = returns.cov() * periods_per_year  # Annualized
        
        logger.debug(
            f"Initialized optimizer: {len(self.tickers)} assets, "
            f"{len(returns)} periods"
        )
    
    @abstractmethod
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.
        
        Args:
            constraints: Dictionary of constraints (weights, risk, etc.)
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        pass
    
    def get_name(self) -> str:
        """Return optimizer name."""
        return self.__class__.__name__
    
    def _validate_weights(
        self,
        weights: np.ndarray,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        long_only: bool = True,
    ) -> bool:
        """
        Validate weights meet constraints.
        
        Args:
            weights: Weights array
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            long_only: If True, all weights must be >= 0
        
        Returns:
            True if weights are valid
        """
        if len(weights) != len(self.tickers):
            raise ValueError(
                f"Weights length {len(weights)} != "
                f"tickers length {len(self.tickers)}"
            )
        
        if long_only and (weights < 0).any():
            logger.warning("Negative weights found (short positions)")
            return False
        
        if min_weight is not None and (weights < min_weight).any():
            return False
        
        if max_weight is not None and (weights > max_weight).any():
            return False
        
        # Weights should sum to approximately 1.0
        weight_sum = weights.sum()
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {weight_sum:.4f}, expected 1.0"
            )
            return False
        
        return True
    
    def _normalize_weights(
        self,
        weights: np.ndarray,
        constraints_obj: Optional["OptimizationConstraints"] = None,
    ) -> np.ndarray:
        """
        Normalize weights to sum to 1.0, respecting max_cash_weight constraint.
        
        If CASH is at max_cash_weight, normalize only non-CASH assets.
        Otherwise, normalize all assets.
        
        Args:
            weights: Weights array
            constraints_obj: Optional constraints object with max_cash_weight
        
        Returns:
            Normalized weights array
        """
        weight_sum = weights.sum()
        if weight_sum == 0:
            raise CalculationError("Cannot normalize weights: sum is zero")
        
        # If no constraints or no max_cash_weight, normal normalization
        if constraints_obj is None or constraints_obj.max_cash_weight is None:
            return weights / weight_sum
        
        # Find CASH indices
        cash_indices = [
            i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
        ]
        
        if not cash_indices:
            # No CASH in portfolio, normal normalization
            return weights / weight_sum
        
        # Check if CASH is at or above max_cash_weight
        cash_weight = sum(weights[i] for i in cash_indices)
        max_cash = constraints_obj.max_cash_weight
        
        if cash_weight >= max_cash - 1e-6:  # Allow small numerical error
            # CASH is at maximum, fix it and normalize only non-CASH assets
            normalized_weights = weights.copy()
            
            # Set CASH to max_cash_weight
            for cash_idx in cash_indices:
                normalized_weights[cash_idx] = max_cash / len(cash_indices)
            
            # Normalize non-CASH assets to sum to (1 - max_cash)
            non_cash_mask = np.ones(len(weights), dtype=bool)
            for cash_idx in cash_indices:
                non_cash_mask[cash_idx] = False
            
            non_cash_weights = weights[non_cash_mask]
            non_cash_sum = non_cash_weights.sum()
            
            if non_cash_sum > 1e-8:
                # Normalize non-CASH to sum to (1 - max_cash)
                target_sum = 1.0 - max_cash
                normalized_non_cash = (non_cash_weights / non_cash_sum) * target_sum
                normalized_weights[non_cash_mask] = normalized_non_cash
            else:
                # All non-CASH weights are zero, distribute equally
                n_non_cash = non_cash_mask.sum()
                if n_non_cash > 0:
                    target_sum = 1.0 - max_cash
                    normalized_weights[non_cash_mask] = target_sum / n_non_cash
            
            return normalized_weights
        else:
            # CASH is below max, normal normalization
            return weights / weight_sum
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics for given weights.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Dictionary with expected_return, volatility, sharpe_ratio
        """
        # Portfolio expected return
        expected_return = float(np.dot(weights, self._mean_returns))
        
        # Portfolio variance
        portfolio_variance = float(
            weights.T @ self._cov_matrix @ weights
        )
        volatility = float(np.sqrt(portfolio_variance))
        
        # Sharpe ratio
        sharpe_ratio = None
        if volatility > 0:
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
        }
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """
        Build constraints object from dictionary.
        
        Supports all constraint types:
        - Weight constraints (min_weight, max_weight, long_only, weight_bounds)
        - Group constraints (group_constraints)
        - Risk constraints (max_volatility, max_var, max_cvar, max_beta)
        - Turnover constraints (max_turnover, min_trade_size)
        - Cardinality constraints (min_assets, max_assets)
        
        Args:
            constraints: Dictionary of constraints
        
        Returns:
            OptimizationConstraints object
        """
        constraints_obj = OptimizationConstraints(self.tickers)
        
        if not constraints:
            return constraints_obj
        
        # Weight constraints
        constraints_obj.set_weight_bounds(
            min_weight=constraints.get("min_weight"),
            max_weight=constraints.get("max_weight"),
            long_only=constraints.get("long_only", True),
        )
        
        # Asset-specific weight bounds
        weight_bounds = constraints.get("weight_bounds", {})
        for ticker, bounds in weight_bounds.items():
            if isinstance(bounds, dict):
                constraints_obj.set_asset_weight_bounds(
                    ticker=ticker,
                    min_weight=bounds.get("min"),
                    max_weight=bounds.get("max"),
                )
            elif isinstance(bounds, tuple):
                constraints_obj.set_asset_weight_bounds(
                    ticker=ticker,
                    min_weight=bounds[0],
                    max_weight=bounds[1],
                )
        
        # Group constraints
        group_constraints = constraints.get("group_constraints", {})
        for group_name, group_data in group_constraints.items():
            if isinstance(group_data, dict):
                tickers = group_data.get("tickers", [])
                max_weight = group_data.get("max_weight")
                if tickers and max_weight is not None:
                    constraints_obj.set_group_constraint(
                        group_name=group_name,
                        tickers=tickers,
                        max_weight=max_weight,
                    )
        
        # Risk constraints
        constraints_obj.set_risk_constraint(
            max_volatility=constraints.get("max_volatility"),
            max_var=constraints.get("max_var"),
            max_cvar=constraints.get("max_cvar"),
            max_beta=constraints.get("max_beta"),
        )
        
        # Turnover constraints
        constraints_obj.set_turnover_constraint(
            max_turnover=constraints.get("max_turnover"),
            min_trade_size=constraints.get("min_trade_size"),
        )
        
        # Cardinality constraints
        constraints_obj.set_cardinality_constraint(
            min_assets=constraints.get("min_assets"),
            max_assets=constraints.get("max_assets"),
        )
        
        # Cash constraints
        constraints_obj.set_cash_constraint(
            max_cash_weight=constraints.get("max_cash_weight"),
        )
        
        # Return constraints
        constraints_obj.set_return_constraint(
            min_return=constraints.get("min_return"),
            target_return=constraints.get("target_return"),
        )
        
        # Diversification regularization
        constraints_obj.set_diversification_regularization(
            diversification_lambda=constraints.get("diversification_lambda"),
        )
        
        return constraints_obj

