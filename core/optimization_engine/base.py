"""Base optimizer class and result types."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.exceptions import CalculationError

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
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1.0."""
        weight_sum = weights.sum()
        if weight_sum == 0:
            raise CalculationError("Cannot normalize weights: sum is zero")
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

