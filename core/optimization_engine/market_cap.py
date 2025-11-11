"""Market Cap Weight optimization."""

import logging
from typing import Dict, Optional

import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from core.optimization_engine.base import BaseOptimizer, OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints

logger = logging.getLogger(__name__)


class MarketCapOptimizer(BaseOptimizer):
    """
    Market Cap Weight optimizer.
    
    Allocates weights proportional to market capitalization.
    This represents the market portfolio (CAPM equilibrium).
    
    Formula: Weight[i] = MarketCap[i] / Σ MarketCap
    """
    
    def optimize(
        self,
        constraints: Optional[Dict[str, any]] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using market cap weights.
        
        Args:
            constraints: Optional constraints dictionary
            (Note: Market cap weights may not satisfy all constraints)
        
        Returns:
            OptimizationResult with market cap weights
        """
        constraints_obj = self._build_constraints(constraints)
        min_bounds, max_bounds = constraints_obj.get_weight_bounds_array()
        
        n = len(self.tickers)
        
        try:
            # Fetch market cap for each ticker
            market_caps = {}
            
            if not YFINANCE_AVAILABLE:
                logger.warning(
                    "yfinance not available, using equal weights as fallback"
                )
                weights = np.ones(n) / n
            else:
                for ticker in self.tickers:
                    if ticker == "CASH":
                        # CASH has no market cap, skip it
                        continue
                    
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        # Try different market cap fields
                        market_cap = (
                            info.get("marketCap")
                            or info.get("totalAssets")
                            or info.get("enterpriseValue")
                        )
                        
                        if market_cap and market_cap > 0:
                            market_caps[ticker] = float(market_cap)
                        else:
                            logger.warning(
                                f"Could not get market cap for {ticker}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error fetching market cap for {ticker}: {e}"
                        )
                
                if not market_caps:
                    # Fallback to equal weights if no market caps available
                    logger.warning(
                        "No market cap data available, using equal weights"
                    )
                    weights = np.ones(n) / n
                else:
                    # Calculate weights proportional to market cap
                    weights = np.zeros(n)
                    total_market_cap = sum(market_caps.values())
                    
                    for i, ticker in enumerate(self.tickers):
                        if ticker in market_caps:
                            weights[i] = market_caps[ticker] / total_market_cap
                        elif ticker == "CASH":
                            # CASH gets zero weight (or remainder)
                            weights[i] = 0.0
                        else:
                            # Ticker without market cap gets zero weight
                            weights[i] = 0.0
                    
                    # Normalize (in case CASH or some tickers excluded)
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones(n) / n
            
            # Apply constraints (clip to bounds)
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            # Calculate metrics
            metrics = self._calculate_portfolio_metrics(weights)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                method="Market Cap Weight",
                success=True,
                message="Market cap weighting completed",
                metadata={
                    "market_caps": market_caps,
                    "total_market_cap": sum(market_caps.values())
                    if market_caps
                    else None,
                },
            )
        except Exception as e:
            logger.error(f"Market cap weighting failed: {e}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            weights = np.clip(weights, min_bounds, max_bounds)
            weights = self._normalize_weights(weights, constraints_obj)
            
            return OptimizationResult(
                weights=weights,
                tickers=self.tickers,
                method="Market Cap Weight",
                success=False,
                message=f"Calculation failed: {str(e)}",
            )
    
    def _build_constraints(
        self, constraints: Optional[Dict[str, any]]
    ) -> OptimizationConstraints:
        """Build constraints object from dictionary."""
        return super()._build_constraints(constraints)

