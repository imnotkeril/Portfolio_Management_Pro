"""Analytics service for orchestrating portfolio analytics."""

import logging
from datetime import date
from typing import Optional

import numpy as np
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
    ) -> dict[str, any]:
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
            raise ValidationError("Start date must be before end date")

        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Get tickers
        positions = portfolio.get_all_positions()
        tickers = [pos.ticker for pos in positions]

        if not tickers:
            raise InsufficientDataError("Portfolio has no positions")

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
            raise InsufficientDataError(f"Failed to fetch price data: {e}") from e

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
                bm_prices = self._fetch_portfolio_prices(
                    [benchmark_ticker], start_date, end_date
                )
                if not bm_prices.empty and benchmark_ticker in bm_prices.columns:
                    bm_series = bm_prices[benchmark_ticker].sort_index().ffill().bfill()
                    bm_ret = bm_series.pct_change().dropna()
                    # Align to portfolio dates
                    benchmark_returns = bm_ret.reindex(
                        portfolio_returns.index, method="ffill"
                    ).dropna()
                else:
                    benchmark_returns = pd.Series(dtype=float)
                    logger.warning(f"Empty benchmark prices for {benchmark_ticker}")
            except Exception as e:
                logger.warning(
                    f"Failed to fetch benchmark data for {benchmark_ticker}: {e}"
                )
                # Continue without benchmark

        # === Comparison support (one series) ===
        comparison_label: Optional[str] = None
        comparison_returns: Optional[pd.Series] = None
        comparison_metrics: Optional[dict[str, float]] = None
        try:
            if comparison_type == "ticker" and comparison_value:
                comparison_label = comparison_value.upper()
                series = self._get_single_ticker_returns(
                    comparison_label, start_date, end_date
                )
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
                    common_idx = pr_index.intersection(sr_index).sort_values()
                    series = series.loc[common_idx]
                    comparison_returns = series.copy()
            elif comparison_type == "portfolio" and comparison_value:
                comparison_label = f"PORT:{comparison_value}"
                series = self._get_portfolio_returns_by_id(
                    comparison_value, start_date, end_date
                )
                if not series.empty:
                    try:
                        pr_index = portfolio_returns.index.tz_localize(None)
                    except Exception:
                        pr_index = portfolio_returns.index
                    try:
                        sr_index = series.index.tz_localize(None)
                    except Exception:
                        sr_index = series.index
                    common_idx = pr_index.intersection(sr_index).sort_values()
                    series = series.loc[common_idx]
                    comparison_returns = series.copy()
            if comparison_returns is not None and not comparison_returns.empty:
                benchmark_returns = comparison_returns
                portfolio_returns, benchmark_returns, portfolio_values = (
                    self._align_portfolio_with_benchmark(
                        portfolio_returns, benchmark_returns, portfolio_values
                    )
                )
                comparison_returns = benchmark_returns
                comparison_metrics = self._compute_basic_metrics_from_returns(
                    comparison_returns
                )
        except Exception as e:
            logger.warning(
                f"Comparison fetch failed: type={comparison_type}, value={comparison_value}, error={e}"
            )

        # Calculate all metrics
        metrics = self._engine.calculate_all_metrics(
            portfolio_returns=portfolio_returns,
            start_date=start_date,
            end_date=end_date,
            benchmark_returns=benchmark_returns,
            portfolio_values=portfolio_values,
        )

        logger.info(f"Metrics calculation completed for portfolio {portfolio_id}")

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

    def _get_single_ticker_returns(
        self, ticker: str, start_date: date, end_date: date
    ) -> pd.Series:
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

    def _get_portfolio_returns_by_id(
        self, portfolio_id: str, start_date: date, end_date: date
    ) -> pd.Series:
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

    def _compute_basic_metrics_from_returns(
        self, returns: pd.Series
    ) -> dict[str, float]:
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.ratios import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )
        from core.analytics_engine.risk_metrics import (
            calculate_max_drawdown,
            calculate_volatility,
        )

        metrics: dict[str, float] = {}
        try:
            metrics["total_return"] = float((1 + returns).prod() - 1)
            metrics["annualized_return"] = float(calculate_annualized_return(returns))
            vol = calculate_volatility(returns)
            metrics["volatility"] = float(
                vol.get("annual", 0.0) if isinstance(vol, dict) else vol
            )
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

                    # Calculate CASH prices with risk-free rate compound interest
                    # Daily return = (1 + annual_rate)^(1/252) - 1
                    TRADING_DAYS_PER_YEAR = 252
                    daily_return = (1 + self._engine.risk_free_rate) ** (
                        1.0 / TRADING_DAYS_PER_YEAR
                    ) - 1

                    # Create prices starting from 1.0, compounding daily
                    prices_list = []
                    for i, date_val in enumerate(dr):
                        # Compound interest: price = 1.0 * (1 + daily_return)^days
                        price = (1.0 + daily_return) ** i
                        prices_list.append(price)

                    prices = pd.DataFrame(
                        {
                            "Date": dr,
                            "Adjusted_Close": prices_list,
                        }
                    )
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
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")
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
            pivot_df = pivot_df[
                (pivot_df.index >= start_ts) & (pivot_df.index <= end_ts)
            ]

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
                    # CASH prices now include risk-free rate growth
                    # (already calculated in _fetch_portfolio_prices)
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
        returns = returns.sort_index()
        if returns.index.has_duplicates:
            returns = returns[~returns.index.duplicated(keep="last")]

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
                    # CASH prices now include risk-free rate growth
                    # (already calculated in _fetch_portfolio_prices)
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
        portfolio_values = portfolio_values.sort_index()
        if portfolio_values.index.has_duplicates:
            portfolio_values = portfolio_values[
                ~portfolio_values.index.duplicated(keep="last")
            ]

        return portfolio_values

    def _align_portfolio_with_benchmark(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_values: Optional[pd.Series],
    ) -> tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """Align portfolio returns/values to benchmark on identical sorted dates.

        Without this, the benchmark series is clipped to the intersection while the
        portfolio series stays longer—frontend cumulative charts then disagree with
        summary metrics computed from portfolio values / benchmark-only windows.
        """
        port = portfolio_returns.sort_index()
        bench = benchmark_returns.sort_index()
        if port.index.has_duplicates:
            port = port[~port.index.duplicated(keep="last")]
        if bench.index.has_duplicates:
            bench = bench[~bench.index.duplicated(keep="last")]
        common = port.index.intersection(bench.index).sort_values()
        if len(common) == 0:
            logger.warning(
                "Portfolio and benchmark have no overlapping dates; keeping unaligned series"
            )
            return portfolio_returns, benchmark_returns, portfolio_values
        port = port.loc[common]
        bench = bench.loc[common]
        valid = port.notna() & bench.notna()
        port = port.loc[valid]
        bench = bench.loc[valid]
        if port.empty:
            logger.warning(
                "Portfolio/benchmark overlap has no valid paired returns; keeping originals"
            )
            return portfolio_returns, benchmark_returns, portfolio_values

        pv_out: Optional[pd.Series] = portfolio_values
        if portfolio_values is not None and not portfolio_values.empty:
            pv = portfolio_values.sort_index()
            if pv.index.has_duplicates:
                pv = pv[~pv.index.duplicated(keep="last")]
            pv_aligned = pv.reindex(port.index).ffill().bfill()
            pv_out = pv_aligned

        return port, bench, pv_out

    def simulate_buy_and_hold_returns_from_weights(
        self,
        weights: dict[str, float],
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """
        Buy-and-hold daily simple returns with fixed share counts.

        Share counts are chosen so that dollar weights match the given ``weights``
        on the first date where all tickers have valid prices. This matches
        ``_calculate_portfolio_returns`` for the live portfolio (fixed shares, weights
        drift over time) — unlike a constant daily sum(w_i * r_i), which implies
        daily rebalancing to fixed weights.
        """
        tickers = [t for t, w in weights.items() if abs(float(w)) > 1e-12]
        if not tickers:
            return pd.Series(dtype=float)

        prices = self._fetch_portfolio_prices(tickers, start_date, end_date)
        if prices.empty:
            return pd.Series(dtype=float)

        filled = prices.sort_index().ffill().bfill()
        cols = [t for t in tickers if t in filled.columns]
        if not cols:
            return pd.Series(dtype=float)

        sub = filled[cols]
        valid = sub.dropna(how="any")
        if valid.empty:
            return pd.Series(dtype=float)

        t0 = valid.index[0]
        p0 = valid.loc[t0]
        wvec = np.array([float(weights[t]) for t in cols], dtype=float)
        wsum = float(wvec.sum())
        if wsum <= 1e-14:
            return pd.Series(dtype=float)
        wvec = wvec / wsum

        pv = p0.values.astype(float)
        if np.any(pv <= 0) or not np.all(np.isfinite(pv)):
            return pd.Series(dtype=float)

        # Initial value = 1; n_i * P_i0 = w_i  =>  n_i = w_i / P_i0
        n_shares = wvec / pv
        tail = filled.loc[filled.index >= t0, cols]
        values = (tail * n_shares).sum(axis=1).dropna()
        if len(values) < 2:
            return pd.Series(dtype=float)

        rets = values.pct_change().dropna()
        rets.index = pd.to_datetime(rets.index)
        try:
            rets.index = rets.index.tz_localize(None)
        except Exception:
            pass
        return rets
