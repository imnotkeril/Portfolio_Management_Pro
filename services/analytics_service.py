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

        # Fetch benchmark data if provided
        benchmark_returns: Optional[pd.Series] = None
        if benchmark_ticker:
            try:
                benchmark_prices = self._data_service.fetch_historical_prices(
                    benchmark_ticker,
                    start_date,
                    end_date,
                    use_cache=True,
                    save_to_db=True,
                )

                if (
                    not benchmark_prices.empty
                    and "Adjusted_Close" in benchmark_prices.columns
                ):
                    benchmark_close = benchmark_prices["Adjusted_Close"]
                    benchmark_returns = (
                        benchmark_close.pct_change().dropna()
                    )

                    logger.info(
                        f"Benchmark {benchmark_ticker} data loaded "
                        f"({len(benchmark_returns)} data points)"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch benchmark data "
                    f"for {benchmark_ticker}: {e}"
                )
                # Continue without benchmark

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

        # Pivot to have dates as index, tickers as columns
        if "Adjusted_Close" in combined.columns:
            pivot_df = combined.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="last",
            )

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

        # Calculate weights (initial weights based on purchase or equal)
        ticker_to_shares = {pos.ticker: pos.shares for pos in positions}

        # For each date, calculate portfolio value
        portfolio_values = pd.Series(dtype=float, index=prices.index)

        for date_idx in prices.index:
            total_value = 0.0
            for ticker, shares in ticker_to_shares.items():
                if ticker in prices.columns:
                    price = float(prices.loc[date_idx, ticker])
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

        portfolio_values = pd.Series(dtype=float, index=prices.index)

        for date_idx in prices.index:
            total_value = 0.0
            for ticker, shares in ticker_to_shares.items():
                if ticker in prices.columns:
                    price = float(prices.loc[date_idx, ticker])
                    if pd.notna(price):
                        total_value += shares * price

            if total_value > 0:
                portfolio_values.loc[date_idx] = total_value

        portfolio_values = portfolio_values.dropna()
        portfolio_values.index = pd.to_datetime(portfolio_values.index)

        return portfolio_values

