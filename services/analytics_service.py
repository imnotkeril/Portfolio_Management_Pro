"""Analytics service for orchestrating portfolio analytics."""

import logging
from datetime import date
from typing import Dict, Optional

import pandas as pd

from core.analytics_engine.engine import AnalyticsEngine
from core.exceptions import InsufficientDataError, ValidationError
from services.data_service import DataService
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for orchestrating analytics calculations."""

    def __init__(
        self,
        analytics_engine: Optional[AnalyticsEngine] = None,
        portfolio_service: Optional[PortfolioService] = None,
        data_service: Optional[DataService] = None,
    ) -> None:
        """
        Initialize analytics service.

        Args:
            analytics_engine: Optional analytics engine instance
            portfolio_service: Optional portfolio service instance
            data_service: Optional data service instance
        """
        self._engine = analytics_engine or AnalyticsEngine()
        self._portfolio_service = portfolio_service or PortfolioService()
        self._data_service = data_service or DataService()

    def calculate_portfolio_metrics(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
        benchmark_ticker: Optional[str] = None,
        comparison_type: Optional[str] = None,  # 'ticker' | 'portfolio'
        comparison_value: Optional[str] = None,  # ticker or portfolio_id
    ) -> Dict[str, any]:
        """
        Calculate all metrics for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            start_date: Start date of analysis period
            end_date: End date of analysis period
            benchmark_ticker: Optional benchmark ticker (e.g., "SPY")

        Returns:
            Dictionary with all metrics and metadata

        Raises:
            ValidationError: If date range is invalid
            InsufficientDataError: If insufficient data available
        """
        # Validate date range
        if start_date >= end_date:
            raise ValidationError(
                "Start date must be before end date"
            )

        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Get tickers
        positions = portfolio.get_all_positions()
        tickers = [pos.ticker for pos in positions]

        if not tickers:
            raise InsufficientDataError(
                "Portfolio has no positions"
            )

        logger.info(
            f"Calculating metrics for portfolio {portfolio_id} "
            f"({len(tickers)} positions) from {start_date} to {end_date}"
        )

        # Fetch portfolio price data
        try:
            portfolio_prices = self._fetch_portfolio_prices(
                tickers, start_date, end_date
            )
        except Exception as e:
            logger.error(f"Error fetching portfolio prices: {e}")
            raise InsufficientDataError(
                f"Failed to fetch price data: {e}"
            ) from e

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(
            portfolio_prices, positions
        )

        if portfolio_returns.empty:
            raise InsufficientDataError(
                "Unable to calculate portfolio returns from price data"
            )

        # Calculate portfolio values
        portfolio_values = self._calculate_portfolio_values(
            portfolio_prices, positions, portfolio.starting_capital
        )

        # Fetch benchmark data if provided (legacy support). Will be shown in comparison too.
        benchmark_returns: Optional[pd.Series] = None
        if benchmark_ticker:
            try:
                bm_prices = self._fetch_portfolio_prices([benchmark_ticker], start_date, end_date)
                if not bm_prices.empty and benchmark_ticker in bm_prices.columns:
                    bm_series = bm_prices[benchmark_ticker].sort_index().ffill().bfill()
                    bm_ret = bm_series.pct_change().dropna()
                    # Align to portfolio dates
                    benchmark_returns = bm_ret.reindex(portfolio_returns.index, method="ffill").dropna()
                else:
                    benchmark_returns = pd.Series(dtype=float)
                    logger.warning(f"Empty benchmark prices for {benchmark_ticker}")
            except Exception as e:
                logger.warning(f"Failed to fetch benchmark data for {benchmark_ticker}: {e}")
                # Continue without benchmark

        # === Comparison support (one series) ===
        comparison_label: Optional[str] = None
        comparison_returns: Optional[pd.Series] = None
        comparison_metrics: Optional[Dict[str, float]] = None
        try:
            if comparison_type == "ticker" and comparison_value:
                comparison_label = comparison_value.upper()
                series = self._get_single_ticker_returns(comparison_label, start_date, end_date)
                if not series.empty:
                    # Normalize tz and align strictly by intersection
                    try:
                        pr_index = portfolio_returns.index.tz_localize(None)
                    except Exception:
                        pr_index = portfolio_returns.index
                    try:
                        sr_index = series.index.tz_localize(None)
                    except Exception:
                        sr_index = series.index
                    common_idx = pr_index.intersection(sr_index)
                    series = series.loc[common_idx]
                    comparison_returns = series.copy()
            elif comparison_type == "portfolio" and comparison_value:
                comparison_label = f"PORT:{comparison_value}"
                series = self._get_portfolio_returns_by_id(comparison_value, start_date, end_date)
                if not series.empty:
                    try:
                        pr_index = portfolio_returns.index.tz_localize(None)
                    except Exception:
                        pr_index = portfolio_returns.index
                    try:
                        sr_index = series.index.tz_localize(None)
                    except Exception:
                        sr_index = series.index
                    common_idx = pr_index.intersection(sr_index)
                    series = series.loc[common_idx]
                    comparison_returns = series.copy()
            if comparison_returns is not None and not comparison_returns.empty:
                comparison_metrics = self._compute_basic_metrics_from_returns(comparison_returns)
                # Use comparison as benchmark for engine metrics (beta, alpha, etc.)
                benchmark_returns = comparison_returns
        except Exception as e:
            logger.warning(f"Comparison fetch failed: type={comparison_type}, value={comparison_value}, error={e}")

        # Calculate all metrics
        metrics = self._engine.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            start_date=start_date,
            end_date=end_date,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
        )

        logger.info(
            f"Metrics calculation completed for portfolio {portfolio_id}"
        )

        # Return metrics with returns data for charts
        return {
            **metrics,
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": benchmark_returns,
            "portfolio_values": portfolio_values,
            "comparison_label": comparison_label,
            "comparison_returns": comparison_returns,
            "comparison_metrics": comparison_metrics,
        }

    def _get_single_ticker_returns(self, ticker: str, start_date: date, end_date: date) -> pd.Series:
        """Return ETF returns using the SAME path as portfolio (virtual 100% position)."""
        prices = self._fetch_portfolio_prices([ticker], start_date, end_date)
        if prices.empty or ticker not in prices.columns:
            return pd.Series(dtype=float)
        # Build virtual position list with 100% in the ticker
        class _TmpPos:
            def __init__(self, t: str, s: float) -> None:
                self.ticker = t
                self.shares = s
        positions = [_TmpPos(ticker, 1.0)]
        ret = self._calculate_portfolio_returns(prices, positions)
        if not ret.empty:
            try:
                ret.index = ret.index.tz_localize(None)
            except Exception:
                pass
        return ret

    def _get_portfolio_returns_by_id(self, portfolio_id: str, start_date: date, end_date: date) -> pd.Series:
        other = self._portfolio_service.get_portfolio(portfolio_id)
        if other is None:
            return pd.Series(dtype=float)
        positions = other.get_all_positions()
        tickers = [p.ticker for p in positions]
        prices = self._fetch_portfolio_prices(tickers, start_date, end_date)
        series = self._calculate_portfolio_returns(prices, positions)
        if not series.empty:
            try:
                series.index = series.index.tz_localize(None)
            except Exception:
                pass
        return series

    def _compute_basic_metrics_from_returns(self, returns: pd.Series) -> Dict[str, float]:
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.risk_metrics import (
            calculate_volatility,
            calculate_max_drawdown,
        )
        from core.analytics_engine.ratios import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )
        metrics: Dict[str, float] = {}
        try:
            metrics["total_return"] = float((1 + returns).prod() - 1)
            metrics["annualized_return"] = float(calculate_annualized_return(returns))
            vol = calculate_volatility(returns)
            metrics["volatility"] = float(vol.get("annual", 0.0) if isinstance(vol, dict) else vol)
            dd = calculate_max_drawdown(returns)
            metrics["max_drawdown"] = float(dd[0] if isinstance(dd, tuple) else dd)
            metrics["sharpe_ratio"] = float(calculate_sharpe_ratio(returns) or 0)
            metrics["sortino_ratio"] = float(calculate_sortino_ratio(returns) or 0)
        except Exception:
            pass
        return metrics

    def _fetch_portfolio_prices(
        self, tickers: list[str], start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Fetch price data for all portfolio tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data for all tickers
        """
        all_prices = []

        for ticker in tickers:
            try:
                # Handle CASH locally to avoid external fetches and TZ issues
                if ticker == "CASH":
                    # Business-day date range
                    dr = pd.bdate_range(start=start_date, end=end_date, normalize=True)
                    prices = pd.DataFrame({
                        "Date": dr,
                        "Adjusted_Close": 1.0,
                    })
                else:
                    prices = self._data_service.fetch_historical_prices(
                        ticker,
                        start_date,
                        end_date,
                        use_cache=True,
                        save_to_db=True,
                    )

                if not prices.empty:
                    prices["Ticker"] = ticker
                    all_prices.append(prices)

            except Exception as e:
                logger.warning(
                    f"Failed to fetch prices for {ticker}: {e}"
                )
                continue

        if not all_prices:
            return pd.DataFrame()

        # Combine all price data
        combined = pd.concat(all_prices, ignore_index=True)

        # Ensure Date column exists and is tz-naive before pivot
        if "Date" in combined.columns:
            combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
            try:
                combined["Date"] = combined["Date"].dt.tz_localize(None)
            except Exception:
                pass

        # Pivot to have dates as index, tickers as columns
        if "Adjusted_Close" in combined.columns:
            pivot_df = combined.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="last",
            )

            # Ensure index is tz-naive pandas Timestamps
            pivot_df.index = pd.to_datetime(pivot_df.index, errors="coerce")
            try:
                pivot_df.index = pivot_df.index.tz_localize(None)
            except Exception:
                pass

            # Filter by date range (inclusive)
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            try:
                start_ts = start_ts.tz_localize(None)
                end_ts = end_ts.tz_localize(None)
            except Exception:
                pass
            pivot_df = pivot_df[(pivot_df.index >= start_ts) & (pivot_df.index <= end_ts)]

            return pivot_df
        else:
            return pd.DataFrame()

    def _calculate_portfolio_returns(
        self, prices: pd.DataFrame, positions: list
    ) -> pd.Series:
        """
        Calculate portfolio returns from individual asset prices.

        Args:
            prices: DataFrame with prices (dates × tickers)
            positions: List of Position objects

        Returns:
            Series of portfolio returns
        """
        if prices.empty:
            return pd.Series(dtype=float)

        # Map tickers to shares
        ticker_to_shares = {pos.ticker: pos.shares for pos in positions}

        # Fill missing prices forward/backward to avoid artificial jumps
        filled_prices = prices.sort_index().ffill().bfill()

        # For each date, calculate portfolio value
        portfolio_values = pd.Series(dtype=float, index=filled_prices.index)

        for date_idx in filled_prices.index:
            total_value = 0.0
            for ticker, shares in ticker_to_shares.items():
                if ticker in filled_prices.columns:
                    # Treat CASH as price 1.0
                    if ticker == "CASH":
                        price = 1.0
                    else:
                        price = float(filled_prices.loc[date_idx, ticker])
                    if pd.notna(price):
                        total_value += shares * price

            if total_value > 0:
                portfolio_values.loc[date_idx] = total_value

        # Calculate returns from values
        portfolio_values = portfolio_values.dropna()
        if len(portfolio_values) < 2:
            return pd.Series(dtype=float)

        returns = portfolio_values.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)
        try:
            returns.index = returns.index.tz_localize(None)
        except Exception:
            pass

        return returns

    def _calculate_portfolio_values(
        self,
        prices: pd.DataFrame,
        positions: list,
        starting_capital: float,
    ) -> pd.Series:
        """
        Calculate portfolio values over time.

        Args:
            prices: DataFrame with prices (dates × tickers)
            positions: List of Position objects
            starting_capital: Starting capital

        Returns:
            Series of portfolio values indexed by date
        """
        if prices.empty:
            return pd.Series(dtype=float)

        ticker_to_shares = {pos.ticker: pos.shares for pos in positions}

        filled_prices = prices.sort_index().ffill().bfill()

        portfolio_values = pd.Series(dtype=float, index=filled_prices.index)

        for date_idx in filled_prices.index:
            total_value = 0.0
            for ticker, shares in ticker_to_shares.items():
                if ticker in filled_prices.columns:
                    if ticker == "CASH":
                        price = 1.0
                    else:
                        price = float(filled_prices.loc[date_idx, ticker])
                    if pd.notna(price):
                        total_value += shares * price

            if total_value > 0:
                portfolio_values.loc[date_idx] = total_value

        portfolio_values = portfolio_values.dropna()
        portfolio_values.index = pd.to_datetime(portfolio_values.index)
        try:
            portfolio_values.index = portfolio_values.index.tz_localize(None)
        except Exception:
            pass

        return portfolio_values

