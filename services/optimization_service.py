"""Optimization service for orchestrating portfolio optimization."""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from core.exceptions import (
    CalculationError,
    InsufficientDataError,
)
from core.optimization_engine.base import OptimizationResult
from core.optimization_engine.efficient_frontier import EfficientFrontier
from core.optimization_engine.black_litterman import (
    BlackLittermanOptimizer,
)
from core.optimization_engine.cvar_optimization import CVaROptimizer
from core.optimization_engine.equal_weight import EqualWeightOptimizer
from core.optimization_engine.hrp import HRPOptimizer
from core.optimization_engine.inverse_correlation import (
    InverseCorrelationOptimizer,
)
from core.optimization_engine.kelly_criterion import KellyCriterionOptimizer
from core.optimization_engine.market_cap import MarketCapOptimizer
from core.optimization_engine.max_alpha import MaxAlphaOptimizer
from core.optimization_engine.max_diversification import (
    MaxDiversificationOptimizer,
)
from core.optimization_engine.max_return import MaxReturnOptimizer
from core.optimization_engine.max_sharpe import MaxSharpeOptimizer
from core.optimization_engine.mean_cvar import MeanCVaROptimizer
from core.optimization_engine.mean_variance import MeanVarianceOptimizer
from core.optimization_engine.min_correlation import MinCorrelationOptimizer
from core.optimization_engine.min_tracking_error import (
    MinTrackingErrorOptimizer,
)
from core.optimization_engine.min_variance import MinVarianceOptimizer
from core.optimization_engine.risk_parity import RiskParityOptimizer
from core.optimization_engine.robust import RobustOptimizer
from core.optimization_engine.sensitivity import SensitivityAnalyzer
from services.data_service import DataService
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

# Trading days per year for risk-free rate calculation
TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0435  # 4.35% annual

# Available optimization methods
OPTIMIZATION_METHODS = {
    "equal_weight": EqualWeightOptimizer,
    "mean_variance": MeanVarianceOptimizer,
    "min_variance": MinVarianceOptimizer,
    "max_sharpe": MaxSharpeOptimizer,
    "max_return": MaxReturnOptimizer,
    "risk_parity": RiskParityOptimizer,
    "kelly_criterion": KellyCriterionOptimizer,
    "min_tracking_error": MinTrackingErrorOptimizer,
    "max_alpha": MaxAlphaOptimizer,
    "hrp": HRPOptimizer,
    "cvar_optimization": CVaROptimizer,
    "mean_cvar": MeanCVaROptimizer,
    "robust": RobustOptimizer,
    "max_diversification": MaxDiversificationOptimizer,
    "min_correlation": MinCorrelationOptimizer,
    "inverse_correlation": InverseCorrelationOptimizer,
    "market_cap": MarketCapOptimizer,
    "black_litterman": BlackLittermanOptimizer,
}


class OptimizationService:
    """Service for orchestrating portfolio optimization."""
    
    def __init__(
        self,
        portfolio_service: Optional[PortfolioService] = None,
        data_service: Optional[DataService] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        """
        Initialize optimization service.
        
        Args:
            portfolio_service: Optional portfolio service instance
            data_service: Optional data service instance
            risk_free_rate: Annual risk-free rate (default: 4.35%)
        """
        self._portfolio_service = (
            portfolio_service or PortfolioService()
        )
        self._data_service = data_service or DataService()
        self._risk_free_rate = risk_free_rate
    
    def optimize_portfolio(
        self,
        portfolio_id: str,
        method: str,
        start_date: date,
        end_date: date,
        constraints: Optional[Dict[str, any]] = None,
        benchmark_ticker: Optional[str] = None,
        method_params: Optional[Dict[str, any]] = None,
        out_of_sample: bool = False,
        training_ratio: float = 0.3,
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            portfolio_id: Portfolio ID to optimize
            method: Optimization method name
            start_date: Start date for return calculation (validation start if out_of_sample)
            end_date: End date for return calculation (validation end)
            constraints: Optional constraints dictionary
            benchmark_ticker: Optional benchmark ticker (for methods that
                            require it)
            method_params: Optional method-specific parameters
            out_of_sample: If True, use period BEFORE start_date for training
            training_ratio: Ratio of analysis period to use for training
                          (e.g., 0.3 = 30% of analysis period before start_date)
        
        Returns:
            OptimizationResult with optimal weights
        
        Raises:
            ValueError: If method not found or invalid parameters
            InsufficientDataError: If insufficient price data
        """
        if method not in OPTIMIZATION_METHODS:
            raise ValueError(
                f"Unknown optimization method: {method}. "
                f"Available: {list(OPTIMIZATION_METHODS.keys())}"
            )
        
        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        positions = portfolio.get_all_positions()
        if len(positions) == 0:
            raise ValueError("Portfolio has no positions")
        
        # Check if CASH position exists
        has_cash = any(pos.ticker == "CASH" for pos in positions)
        
        # Get tickers (exclude CASH for price fetching, add it later)
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        
        if not tickers and not has_cash:
            raise ValueError("Portfolio has no valid tickers")
        
        # Determine optimization period (training or analysis)
        if out_of_sample:
            # Out-of-sample: use period BEFORE start_date for training
            analysis_days = (end_date - start_date).days
            training_days = int(analysis_days * training_ratio)
            optimization_start = start_date - timedelta(days=training_days)
            optimization_end = start_date  # Up to validation start
            logger.info(
                f"Out-of-sample mode: training period {optimization_start} to {optimization_end}, "
                f"validation period {start_date} to {end_date}"
            )
        else:
            # Regular mode: use specified period
            optimization_start = start_date
            optimization_end = end_date
        
        # Fetch price data using bulk method (if tickers exist)
        if tickers:
            logger.info(
                f"Fetching price data for {len(tickers)} tickers "
                f"from {optimization_start} to {optimization_end}"
            )
            
            price_data = self._data_service.fetch_bulk_prices(
                tickers=tickers,
                start_date=optimization_start,
                end_date=optimization_end,
                use_cache=True,
                save_to_db=False,
            )
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period "
                    f"{optimization_start} to {optimization_end}"
                )
            
            # Convert to pivot format (dates as index, tickers as columns)
            # fetch_bulk_prices returns DataFrame with Date, Adjusted_Close, Ticker columns
            if "Ticker" in price_data.columns and "Adjusted_Close" in price_data.columns:
                # Pivot to have dates as index, tickers as columns
                if "Date" in price_data.columns:
                    price_data["Date"] = pd.to_datetime(
                        price_data["Date"], errors="coerce"
                    )
                    price_data["Date"] = price_data["Date"].dt.tz_localize(None)
                    pivot_df = price_data.pivot_table(
                        index="Date",
                        columns="Ticker",
                        values="Adjusted_Close",
                        aggfunc="last",
                    )
                    price_data = pivot_df
            
            # Ensure index is datetime
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(
                    price_data.index, errors="coerce"
                )
            
            # Filter by date range
            start_ts = pd.Timestamp(optimization_start)
            end_ts = pd.Timestamp(optimization_end)
            price_data = price_data[
                (price_data.index >= start_ts) & (price_data.index <= end_ts)
            ]
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period "
                    f"{optimization_start} to {optimization_end}"
                )
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
        else:
            # Only CASH - create empty returns DataFrame with date index
            dr = pd.bdate_range(start=optimization_start, end=optimization_end, normalize=True)
            returns = pd.DataFrame(index=dr)
            returns.index = pd.to_datetime(returns.index, errors="coerce")
            returns.index = returns.index.tz_localize(None)
        
        # Add CASH returns if CASH position exists
        if has_cash:
            # Daily return = (1 + annual_rate)^(1/252) - 1
            daily_return = (1 + self._risk_free_rate) ** (
                1.0 / TRADING_DAYS_PER_YEAR
            ) - 1
            
            # Create CASH returns series with same index as returns
            cash_returns = pd.Series(
                data=daily_return,
                index=returns.index,
                name="CASH",
            )
            
            # Add CASH column to returns DataFrame
            returns["CASH"] = cash_returns
        
        if returns.empty or len(returns) < 30:
            raise InsufficientDataError(
                "Insufficient data points for optimization "
                f"(need at least 30, got {len(returns)})"
            )
        
        # Get optimizer class
        optimizer_class = OPTIMIZATION_METHODS[method]
        
        # If using mean_variance with maximize_sharpe objective, use MaxSharpeOptimizer
        # for consistency with Efficient Frontier calculation
        if (
            method == "mean_variance"
            and method_params
            and method_params.get("objective") == "maximize_sharpe"
        ):
            logger.info(
                "Using MaxSharpeOptimizer for consistency with Efficient Frontier"
            )
            optimizer_class = MaxSharpeOptimizer
            # Remove objective from method_params since MaxSharpeOptimizer doesn't need it
            method_params = {
                k: v for k, v in method_params.items() if k != "objective"
            }
        
        # Initialize optimizer
        if method in ["min_tracking_error", "max_alpha"]:
            # Methods that require benchmark
            if not benchmark_ticker:
                raise ValueError(
                    f"Method {method} requires benchmark_ticker"
                )
            
            benchmark_prices = self._data_service.fetch_historical_prices(
                ticker=benchmark_ticker,
                start_date=optimization_start,
                end_date=optimization_end,
                use_cache=True,
                save_to_db=False,
            )
            
            if benchmark_prices.empty:
                raise InsufficientDataError(
                    f"No price data for benchmark {benchmark_ticker}"
                )
            
            # Set Date as index if available
            if "Date" in benchmark_prices.columns:
                benchmark_prices.set_index("Date", inplace=True)
                benchmark_prices.index = pd.to_datetime(
                    benchmark_prices.index, errors="coerce"
                )
                benchmark_prices.index = benchmark_prices.index.tz_localize(None)
            
            # Extract Adjusted_Close and calculate returns
            benchmark_returns = (
                benchmark_prices["Adjusted_Close"].pct_change().dropna()
            )
            
            # Align returns - ensure both have DatetimeIndex
            returns.index = pd.to_datetime(returns.index, errors="coerce")
            returns.index = returns.index.tz_localize(None)
            benchmark_returns.index = pd.to_datetime(
                benchmark_returns.index, errors="coerce"
            )
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)
            
            # Find common index
            common_index = returns.index.intersection(benchmark_returns.index)
            
            if len(common_index) < 30:
                raise InsufficientDataError(
                    "Insufficient aligned data points"
                )
            
            # Align both to common index
            returns_aligned = returns.reindex(common_index).dropna()
            benchmark_aligned = benchmark_returns.reindex(common_index).dropna()
            
            # Final intersection after dropna
            final_index = returns_aligned.index.intersection(
                benchmark_aligned.index
            )
            
            if len(final_index) < 30:
                raise InsufficientDataError(
                    "Insufficient aligned data points after filtering"
                )
            
            optimizer = optimizer_class(
                returns_aligned.reindex(final_index),
                benchmark_aligned.reindex(final_index),
            )
        else:
            optimizer = optimizer_class(returns)
        
        # Run optimization with method-specific parameters
        logger.info(f"Running {method} optimization...")
        
        # Prepare optimize() arguments
        optimize_kwargs = {"constraints": constraints}
        
        # Add method-specific parameters (including objective if present)
        if method_params:
            # Create a copy to avoid modifying original
            params_to_add = method_params.copy()
            
            # Remove 'objective' for methods that don't support it
            # These methods have fixed objectives and don't accept objective parameter
            methods_without_objective = [
                "risk_parity",
                "hrp",
                "cvar_optimization",
                "mean_cvar",
                "max_diversification",
                "min_correlation",
                "inverse_correlation",
                "min_variance",
                "max_return",
                "max_sharpe",
                "equal_weight",
                "market_cap",
                "kelly_criterion",
                "min_tracking_error",
                "max_alpha",
            ]
            
            if method in methods_without_objective:
                params_to_add.pop("objective", None)
            
            optimize_kwargs.update(params_to_add)
        
        result = optimizer.optimize(**optimize_kwargs)
        
        if not result.success:
            logger.warning(
                f"Optimization {method} failed: {result.message}"
            )
        
        return result
    
    def generate_efficient_frontier(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
        n_points: int = 50,
        constraints: Optional[Dict[str, any]] = None,
    ) -> Dict[str, any]:
        """
        Generate efficient frontier for portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            start_date: Start date for return calculation
            end_date: End date for return calculation
            n_points: Number of points on frontier
            constraints: Optional constraints dictionary
        
        Returns:
            Dictionary with frontier data
        """
        # Get portfolio and calculate returns (same as optimize_portfolio)
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        positions = portfolio.get_all_positions()
        # Check if CASH position exists
        has_cash = any(pos.ticker == "CASH" for pos in positions)
        
        # Get tickers (exclude CASH for price fetching, add it later)
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        
        if not tickers and not has_cash:
            raise ValueError("Portfolio has no valid tickers")
        
        # Fetch price data using bulk method (if tickers exist)
        if tickers:
            price_data = self._data_service.fetch_bulk_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                save_to_db=False,
            )
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period "
                    f"{start_date} to {end_date}"
                )
            
            # Convert to pivot format if needed (same as optimize_portfolio)
            if "Ticker" in price_data.columns and "Adjusted_Close" in price_data.columns:
                if "Date" in price_data.columns:
                    price_data["Date"] = pd.to_datetime(
                        price_data["Date"], errors="coerce"
                    )
                    price_data["Date"] = price_data["Date"].dt.tz_localize(None)
                    pivot_df = price_data.pivot_table(
                        index="Date",
                        columns="Ticker",
                        values="Adjusted_Close",
                        aggfunc="last",
                    )
                    price_data = pivot_df
            
            # Ensure index is datetime
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(
                    price_data.index, errors="coerce"
                )
            
            # Filter by date range
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            price_data = price_data[
                (price_data.index >= start_ts) & (price_data.index <= end_ts)
            ]
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period "
                    f"{start_date} to {end_date}"
                )
            
            returns = price_data.pct_change().dropna()
        else:
            # Only CASH - create empty returns DataFrame with date index
            dr = pd.bdate_range(start=start_date, end=end_date, normalize=True)
            returns = pd.DataFrame(index=dr)
            returns.index = pd.to_datetime(returns.index, errors="coerce")
            returns.index = returns.index.tz_localize(None)
        
        # Add CASH returns if CASH position exists
        if has_cash:
            # Daily return = (1 + annual_rate)^(1/252) - 1
            daily_return = (1 + self._risk_free_rate) ** (
                1.0 / TRADING_DAYS_PER_YEAR
            ) - 1
            
            # Create CASH returns series with same index as returns
            cash_returns = pd.Series(
                data=daily_return,
                index=returns.index,
                name="CASH",
            )
            
            # Add CASH column to returns DataFrame
            returns["CASH"] = cash_returns
        
        if returns.empty or len(returns) < 30:
            raise InsufficientDataError(
                "Insufficient data points for efficient frontier"
            )
        
        # Generate frontier
        # Pass risk_free_rate and periods_per_year to ensure consistency
        frontier = EfficientFrontier(
            returns,
            risk_free_rate=self._risk_free_rate,
            periods_per_year=TRADING_DAYS_PER_YEAR,
        )
        returns_array, volatilities_array, portfolios = (
            frontier.generate_frontier(n_points, constraints)
        )
        
        # Get tangency and min variance portfolios
        # These must use the SAME constraints as the frontier line
        tangency = frontier.get_tangency_portfolio(constraints)
        min_var = frontier.get_min_variance_portfolio(constraints)
        
        return {
            "returns": returns_array.tolist(),
            "volatilities": volatilities_array.tolist(),
            "portfolios": portfolios,
            "tangency_portfolio": tangency,
            "min_variance_portfolio": min_var,
        }
    
    def generate_trade_list(
        self,
        portfolio_id: str,
        optimal_weights: OptimizationResult,
    ) -> List[Dict[str, any]]:
        """
        Generate trade list to rebalance portfolio to optimal weights.
        
        Args:
            portfolio_id: Portfolio ID
            optimal_weights: OptimizationResult with optimal weights
        
        Returns:
            List of trade dictionaries (ticker, action, shares, value)
        """
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        # Get current portfolio value
        positions = portfolio.get_all_positions()
        tickers = [pos.ticker for pos in positions if pos.ticker != "CASH"]
        prices = {}
        for ticker in tickers:
            price = self._data_service.fetch_current_price(ticker)
            if price:
                prices[ticker] = price
        
        # Add CASH price (always 1.0)
        if portfolio.get_position("CASH"):
            prices["CASH"] = 1.0
        
        current_value = portfolio.calculate_current_value(prices)
        
        # Get current weights
        current_weights_dict = {}
        for pos in portfolio.get_all_positions():
            if pos.ticker in optimal_weights.tickers:
                if pos.ticker == "CASH":
                    pos_value = pos.shares  # CASH shares = dollar amount
                else:
                    current_price = prices.get(pos.ticker)
                    if not current_price:
                        continue
                    pos_value = current_price * pos.shares
                
                current_weights_dict[pos.ticker] = (
                    pos_value / current_value
                    if current_value > 0
                    else 0.0
                )
        
        # Calculate optimal weights dict
        optimal_weights_dict = optimal_weights.get_weights_dict()
        
        # Generate trades
        trades = []
        all_tickers = set(current_weights_dict.keys()) | set(
            optimal_weights_dict.keys()
        )
        
        for ticker in all_tickers:
            current_weight = current_weights_dict.get(ticker, 0.0)
            optimal_weight = optimal_weights_dict.get(ticker, 0.0)
            
            weight_diff = optimal_weight - current_weight
            
            if abs(weight_diff) < 0.001:  # Ignore tiny differences
                continue
            
            if ticker == "CASH":
                current_price = 1.0  # CASH is always 1.0
            else:
                current_price = prices.get(ticker)
                if current_price is None or current_price <= 0:
                    logger.warning(
                        f"Invalid price for {ticker}, skipping trade"
                    )
                    continue
            
            # Calculate target value and shares
            target_value = optimal_weight * current_value
            target_shares = target_value / current_price
            
            # Get current shares
            current_pos = portfolio.get_position(ticker)
            current_shares = current_pos.shares if current_pos else 0.0
            
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) < 0.01:  # Ignore tiny differences
                continue
            
            action = "BUY" if shares_diff > 0 else "SELL"
            
            trades.append({
                "ticker": ticker,
                "action": action,
                "shares": abs(shares_diff),
                "value": abs(weight_diff * current_value),
                "current_weight": current_weight,
                "optimal_weight": optimal_weight,
                "current_shares": current_shares,
                "target_shares": target_shares,
            })
        
        # Sort by absolute value (largest trades first)
        trades.sort(key=lambda x: x["value"], reverse=True)
        
        return trades
    
    def get_available_methods(self) -> List[str]:
        """Get list of available optimization methods."""
        return list(OPTIMIZATION_METHODS.keys())
    
    def perform_sensitivity_analysis(
        self,
        portfolio_id: str,
        method: str,
        start_date: date,
        end_date: date,
        base_constraints: Optional[Dict[str, any]] = None,
        analysis_type: str = "returns",
        variation_range: float = 0.1,
        num_points: int = 10,
    ) -> Dict[str, any]:
        """
        Perform sensitivity analysis on optimization results.
        
        Args:
            portfolio_id: Portfolio ID
            method: Optimization method name
            start_date: Start date for return calculation
            end_date: End date for return calculation
            base_constraints: Base constraints dictionary
            analysis_type: Type of analysis:
                          "returns", "covariance", "constraints"
            variation_range: Range of variation for returns/covariance
            num_points: Number of points to test
        
        Returns:
            Dictionary with sensitivity analysis results
        """
        # First, run base optimization
        base_result = self.optimize_portfolio(
            portfolio_id=portfolio_id,
            method=method,
            start_date=start_date,
            end_date=end_date,
            constraints=base_constraints,
        )
        
        if not base_result.success:
            raise CalculationError(
                f"Base optimization failed: {base_result.message}"
            )
        
        # Get portfolio and prepare optimizer
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)
        positions = portfolio.get_all_positions()
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        
        # Fetch returns (same as optimize_portfolio)
        if tickers:
            price_data = self._data_service.fetch_bulk_prices(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                save_to_db=False,
            )
            
            if price_data.empty:
                raise InsufficientDataError(
                    f"No price data available for period "
                    f"{start_date} to {end_date}"
                )
            
            # Convert to pivot format
            if "Ticker" in price_data.columns:
                if "Date" in price_data.columns:
                    price_data["Date"] = pd.to_datetime(
                        price_data["Date"], errors="coerce"
                    )
                    price_data["Date"] = price_data["Date"].dt.tz_localize(
                        None
                    )
                    pivot_df = price_data.pivot_table(
                        index="Date",
                        columns="Ticker",
                        values="Adjusted_Close",
                        aggfunc="last",
                    )
                    price_data = pivot_df
            
            returns = price_data.pct_change().dropna()
        else:
            dr = pd.bdate_range(
                start=start_date, end=end_date, normalize=True
            )
            returns = pd.DataFrame(index=dr)
        
        # Add CASH if exists
        has_cash = any(pos.ticker == "CASH" for pos in positions)
        if has_cash:
            daily_return = (1 + self._risk_free_rate) ** (
                1.0 / TRADING_DAYS_PER_YEAR
            ) - 1
            cash_returns = pd.Series(
                data=daily_return,
                index=returns.index,
                name="CASH",
            )
            returns["CASH"] = cash_returns
        
        if returns.empty or len(returns) < 30:
            raise InsufficientDataError(
                "Insufficient data for sensitivity analysis"
            )
        
        # Get optimizer class and create instance
        if method not in OPTIMIZATION_METHODS:
            raise ValueError(f"Unknown method: {method}")
        
        optimizer_class = OPTIMIZATION_METHODS[method]
        
        # Handle methods that require benchmark
        if method in ["min_tracking_error", "max_alpha"]:
            # These require benchmark, skip sensitivity for now
            raise ValueError(
                f"Sensitivity analysis not yet supported for {method}"
            )
        
        optimizer = optimizer_class(returns)
        
        # Create sensitivity analyzer
        analyzer = SensitivityAnalyzer(optimizer, base_result)
        
        # Perform analysis based on type
        if analysis_type == "returns":
            results_df = analyzer.analyze_return_sensitivity(
                variation_range=variation_range,
                num_points=num_points,
                constraints=base_constraints,
            )
        elif analysis_type == "covariance":
            results_df = analyzer.analyze_covariance_sensitivity(
                variation_range=variation_range,
                num_points=num_points,
                constraints=base_constraints,
            )
        else:
            raise ValueError(
                f"Unknown analysis type: {analysis_type}. "
                "Use 'returns' or 'covariance'"
            )
        
        return {
            "analysis_type": analysis_type,
            "results": results_df.to_dict("records"),
            "base_weights": base_result.get_weights_dict(),
            "variation_range": variation_range,
            "num_points": num_points,
        }

