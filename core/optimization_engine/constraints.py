"""Constraint builders for portfolio optimization."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationConstraints:
    """Builder for optimization constraints."""
    
    def __init__(self, tickers: List[str]) -> None:
        """
        Initialize constraints builder.
        
        Args:
            tickers: List of ticker symbols
        """
        self.tickers = tickers
        self.n_assets = len(tickers)
        
        # Weight constraints
        self.min_weight: Optional[float] = None
        self.max_weight: Optional[float] = None
        self.long_only: bool = True
        self.weight_bounds: Dict[str, Tuple[float, float]] = {}
        
        # Group constraints (sector, asset class, etc.)
        self.group_constraints: Dict[str, Dict[str, float]] = {}
        
        # Risk constraints
        self.max_volatility: Optional[float] = None
        self.max_var: Optional[float] = None
        self.max_cvar: Optional[float] = None
        self.max_beta: Optional[float] = None
        
        # Turnover constraints
        self.max_turnover: Optional[float] = None
        self.min_trade_size: Optional[float] = None
        
        # Cardinality constraints
        self.min_assets: Optional[int] = None
        self.max_assets: Optional[int] = None
        
        # Cash constraints
        self.max_cash_weight: Optional[float] = None
        
        # Return constraints
        self.min_return: Optional[float] = None
        self.target_return: Optional[float] = None
        
        # Diversification regularization
        self.diversification_lambda: Optional[float] = None
    
    def set_weight_bounds(
        self,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        long_only: bool = True,
    ) -> "OptimizationConstraints":
        """
        Set global weight bounds.
        
        Args:
            min_weight: Minimum weight per asset (0.0 to 1.0)
            max_weight: Maximum weight per asset (0.0 to 1.0)
            long_only: If True, all weights >= 0 (no shorting)
        
        Returns:
            Self for method chaining
        """
        if min_weight is not None and (min_weight < 0 or min_weight > 1):
            raise ValueError("min_weight must be between 0.0 and 1.0")
        
        if max_weight is not None and (max_weight < 0 or max_weight > 1):
            raise ValueError("max_weight must be between 0.0 and 1.0")
        
        if (
            min_weight is not None
            and max_weight is not None
            and min_weight > max_weight
        ):
            raise ValueError("min_weight cannot be greater than max_weight")
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.long_only = long_only
        
        return self
    
    def set_asset_weight_bounds(
        self,
        ticker: str,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set weight bounds for specific asset.
        
        Args:
            ticker: Ticker symbol
            min_weight: Minimum weight for this asset
            max_weight: Maximum weight for this asset
        
        Returns:
            Self for method chaining
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker {ticker} not in portfolio")
        
        if min_weight is not None and (min_weight < 0 or min_weight > 1):
            raise ValueError("min_weight must be between 0.0 and 1.0")
        
        if max_weight is not None and (max_weight < 0 or max_weight > 1):
            raise ValueError("max_weight must be between 0.0 and 1.0")
        
        self.weight_bounds[ticker] = (min_weight, max_weight)
        
        return self
    
    def set_group_constraint(
        self,
        group_name: str,
        tickers: List[str],
        max_weight: float,
    ) -> "OptimizationConstraints":
        """
        Set maximum weight for a group of assets (e.g., sector limit).
        
        Args:
            group_name: Group identifier (e.g., "Technology")
            tickers: List of tickers in the group
            max_weight: Maximum combined weight for the group
        
        Returns:
            Self for method chaining
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        if max_weight < 0 or max_weight > 1:
            raise ValueError("max_weight must be between 0.0 and 1.0")
        
        # Validate all tickers exist
        for ticker in tickers:
            if ticker not in self.tickers:
                raise ValueError(f"Ticker {ticker} not in portfolio")
        
        self.group_constraints[group_name] = {
            "tickers": tickers,
            "max_weight": max_weight,
        }
        
        return self
    
    def set_risk_constraint(
        self,
        max_volatility: Optional[float] = None,
        max_var: Optional[float] = None,
        max_cvar: Optional[float] = None,
        max_beta: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set risk constraints.
        
        Args:
            max_volatility: Maximum portfolio volatility (annualized)
            max_var: Maximum Value at Risk (VaR)
            max_cvar: Maximum Conditional Value at Risk (CVaR)
            max_beta: Maximum beta to benchmark
        
        Returns:
            Self for method chaining
        """
        if max_volatility is not None and max_volatility <= 0:
            raise ValueError("max_volatility must be positive")
        
        if max_var is not None and max_var >= 0:
            raise ValueError("max_var must be negative (loss)")
        
        if max_cvar is not None and max_cvar >= 0:
            raise ValueError("max_cvar must be negative (loss)")
        
        if max_beta is not None and max_beta <= 0:
            raise ValueError("max_beta must be positive")
        
        self.max_volatility = max_volatility
        self.max_var = max_var
        self.max_cvar = max_cvar
        self.max_beta = max_beta
        
        return self
    
    def set_turnover_constraint(
        self,
        max_turnover: Optional[float] = None,
        min_trade_size: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set turnover constraints.
        
        Args:
            max_turnover: Maximum turnover (0.0 to 1.0)
            min_trade_size: Minimum trade size as fraction of portfolio
        
        Returns:
            Self for method chaining
        """
        if max_turnover is not None and (max_turnover < 0 or max_turnover > 1):
            raise ValueError("max_turnover must be between 0.0 and 1.0")
        
        if min_trade_size is not None and (
            min_trade_size < 0 or min_trade_size > 1
        ):
            raise ValueError("min_trade_size must be between 0.0 and 1.0")
        
        self.max_turnover = max_turnover
        self.min_trade_size = min_trade_size
        
        return self
    
    def set_cardinality_constraint(
        self,
        min_assets: Optional[int] = None,
        max_assets: Optional[int] = None,
    ) -> "OptimizationConstraints":
        """
        Set cardinality constraints (number of assets).
        
        Args:
            min_assets: Minimum number of assets to hold
            max_assets: Maximum number of assets to hold
        
        Returns:
            Self for method chaining
        """
        if min_assets is not None and (
            min_assets < 1 or min_assets > self.n_assets
        ):
            raise ValueError(
                f"min_assets must be between 1 and {self.n_assets}"
            )
        
        if max_assets is not None and (
            max_assets < 1 or max_assets > self.n_assets
        ):
            raise ValueError(
                f"max_assets must be between 1 and {self.n_assets}"
            )
        
        if (
            min_assets is not None
            and max_assets is not None
            and min_assets > max_assets
        ):
            raise ValueError("min_assets cannot be greater than max_assets")
        
        self.min_assets = min_assets
        self.max_assets = max_assets
        
        return self
    
    def set_cash_constraint(
        self,
        max_cash_weight: Optional[float] = None,
        min_cash_weight: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set cash constraints.
        
        Args:
            max_cash_weight: Maximum cash allocation (0.0 to 1.0)
            min_cash_weight: Minimum cash allocation (0.0 to 1.0)
        
        Returns:
            Self for method chaining
        """
        if max_cash_weight is not None and (max_cash_weight < 0 or max_cash_weight > 1):
            raise ValueError("max_cash_weight must be between 0.0 and 1.0")
        if min_cash_weight is not None and (min_cash_weight < 0 or min_cash_weight > 1):
            raise ValueError("min_cash_weight must be between 0.0 and 1.0")
        
        self.max_cash_weight = max_cash_weight
        # min_cash_weight not implemented yet
        return self
    
    def set_return_constraint(
        self,
        min_return: Optional[float] = None,
        target_return: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set return constraints.
        
        Args:
            min_return: Minimum expected return (annualized)
            target_return: Target expected return (annualized)
        
        Returns:
            Self for method chaining
        """
        self.min_return = min_return
        self.target_return = target_return
        return self
    
    def set_diversification_regularization(
        self,
        diversification_lambda: Optional[float] = None,
    ) -> "OptimizationConstraints":
        """
        Set diversification regularization parameter.
        
        Args:
            diversification_lambda: Regularization strength (>= 0)
                Higher values encourage more diversification
        
        Returns:
            Self for method chaining
        """
        if diversification_lambda is not None and diversification_lambda < 0:
            raise ValueError("diversification_lambda must be non-negative")
        
        self.diversification_lambda = diversification_lambda
        return self
    
    def to_dict(self) -> Dict[str, any]:
        """Convert constraints to dictionary."""
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "long_only": self.long_only,
            "weight_bounds": self.weight_bounds,
            "group_constraints": self.group_constraints,
            "max_volatility": self.max_volatility,
            "max_var": self.max_var,
            "max_cvar": self.max_cvar,
            "max_beta": self.max_beta,
            "max_turnover": self.max_turnover,
            "min_trade_size": self.min_trade_size,
            "min_assets": self.min_assets,
            "max_assets": self.max_assets,
            "max_cash_weight": self.max_cash_weight,
            "min_return": self.min_return,
            "target_return": self.target_return,
            "diversification_lambda": self.diversification_lambda,
        }
    
    def get_weight_bounds_array(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weight bounds as numpy arrays.
        
        Note: Group constraints (sector limits, asset class limits) are not
        directly applied here as they require optimization constraints rather
        than simple bounds. They should be handled in the optimizer's
        constraint functions.
        
        Returns:
            Tuple of (min_bounds, max_bounds) arrays
        """
        min_bounds = np.full(self.n_assets, 0.0 if self.long_only else -np.inf)
        max_bounds = np.full(self.n_assets, 1.0)
        
        # Apply global bounds
        if self.min_weight is not None:
            min_bounds = np.maximum(min_bounds, self.min_weight)
        
        if self.max_weight is not None:
            max_bounds = np.minimum(max_bounds, self.max_weight)
        
        # Apply asset-specific bounds
        for ticker, (min_w, max_w) in self.weight_bounds.items():
            idx = self.tickers.index(ticker)
            if min_w is not None:
                min_bounds[idx] = max(min_bounds[idx], min_w)
            if max_w is not None:
                max_bounds[idx] = min(max_bounds[idx], max_w)
        
        # Apply cash constraint
        if self.max_cash_weight is not None:
            cash_indices = [
                i for i, ticker in enumerate(self.tickers) if ticker == "CASH"
            ]
            for cash_idx in cash_indices:
                max_bounds[cash_idx] = min(
                    max_bounds[cash_idx], self.max_cash_weight
                )
        
        # Note: Group constraints (sector limits, asset class limits) require
        # linear inequality constraints in the optimizer, not simple bounds.
        # They are stored in self.group_constraints and should be applied
        # as constraints in the optimizer's constraint list.
        
        return min_bounds, max_bounds
    
    def get_group_constraints_list(
        self,
    ) -> List[Dict[str, any]]:
        """
        Get group constraints as list of constraint dictionaries for optimizer.
        
        Returns:
            List of constraint dictionaries with 'type' and 'fun' keys
        """
        constraints_list = []
        
        for group_name, group_data in self.group_constraints.items():
            tickers_in_group = group_data["tickers"]
            max_weight = group_data["max_weight"]
            
            # Create constraint: sum of weights for group <= max_weight
            def make_group_constraint(group_tickers, max_w):
                def constraint(weights: np.ndarray) -> float:
                    group_indices = [
                        self.tickers.index(t) for t in group_tickers
                    ]
                    group_weight_sum = float(np.sum(weights[group_indices]))
                    return max_w - group_weight_sum  # >= 0 means sum <= max_w
                
                return constraint
            
            constraints_list.append({
                "type": "ineq",
                "fun": make_group_constraint(tickers_in_group, max_weight),
            })
        
        return constraints_list

