"""Minimum tracking error optimization."""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.optimize as scipy_opt

from core.exceptions import CalculationError
from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MinTrackingErrorOptimizer(BaseOptimizer):
    """
    Minimum tracking error optimizer.
    
    Minimizes tracking error relative to a benchmark portfolio.
    Requires benchmark returns as input.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.0435,
        periods_per_year: int = 252,
    ) -> None:
        """
        Initialize optimizer with benchmark.
        
        Args:
            returns: Portfolio returns DataFrame
            benchmark_returns: Benchmark returns Series (aligned with returns)
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        super().__init__(returns, risk_free_rate, periods_per_year)
        
        if len(benchmark_returns) != len(returns):
            raise ValueError(
                "Benchmark returns must have same length as portfolio returns"
            )
        
        # Align benchmark with returns index
        benchmark_aligned = benchmark_returns.reindex(returns.index)
        if benchmark_aligned.isna().any():
            raise ValueError("Benchmark returns have missing values")
        
        self.benchmark_returns = benchmark_aligned * self.periods_per_year
        
        # Calculate benchmark statistics
        self._benchmark_mean = benchmark_aligned.mean() * periods_per_year
        self._benchmark_vol = benchmark_aligned.std() * np.sqrt(periods_per_year)
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio to minimize tracking error.
        
        Args:
            constraints: Optional constraints dictionary
        
        Returns:
            OptimizationResult with minimum tracking error weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        # Objective: minimize tracking error
        # Tracking error = std(portfolio_return - benchmark_return)
        def objective(weights: np.ndarray) -> float:
            # Portfolio returns
            portfolio_returns = (self.returns @ weights).values
            
            # Active returns (portfolio - benchmark)
            active_returns = (
                portfolio_returns - self.benchmark_returns.values
            )
            
            # Tracking error (annualized)
            tracking_error = np.std(active_returns) * np.sqrt(
                self.periods_per_year
            )
            
            return float(tracking_error)
        
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
            },
        ]
        
        x0 = np.ones(n) / n
        
        try:
            result = scipy_opt.minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=list(zip(min_bounds, max_bounds)),
                constraints=constraints_list,
                options={"maxiter": 1000},
            )
            
            if not result.success:
                raise CalculationError(
                    f"Optimization failed: {result.message}"
                )
            
            weights = self._normalize_weights(result.x, constraints_obj)
            metrics = self._calculate_portfolio_metrics(weights)
            
            # Calculate tracking error
            portfolio_returns = (self.returns @ weights).values
            active_returns = (
                portfolio_returns - self.benchmark_returns.values
            )
            tracking_error = (
                np.std(active_returns) * np.sqrt(self.periods_per_year)
            )
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Minimum Tracking Error",
                success=True,
                message="Minimized tracking error vs benchmark",
                metadata={
                    "iterations": result.nit,
                    "tracking_error": float(tracking_error),
                    "fun": float(result.fun),
                },
            )
        except Exception as e:
            logger.error(
                f"Min tracking error optimization failed: {e}"
            )
            return OptimizationResult(
                weights=np.ones(n) / n,
                tickers=self.tickers,
                method="Minimum Tracking Error",
                success=False,
                message=f"Optimization failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        # Call base class method to get all constraints including max_cash_weight, min_return, diversification_lambda
        return super()._build_constraints(constraints)

