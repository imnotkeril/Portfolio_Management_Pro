"""Price data fetching from multiple sources with fallback."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from config.constants import (
    CACHE_TTL_CURRENT_PRICE,
    CACHE_TTL_HISTORICAL_PRICES,
    INITIAL_RETRY_DELAY,
    MAX_RETRIES_API,
    MAX_WORKERS_PARALLEL_FETCH,
    RETRY_BACKOFF_MULTIPLIER,
)
from core.data_manager.cache import Cache
from core.exceptions import DataFetchError, TickerNotFoundError

logger = logging.getLogger(__name__)


class PriceManager:
    """Fetch and manage price data from multiple sources."""

    def __init__(self, cache: Optional[Cache] = None) -> None:
        """
        Initialize price manager.

        Args:
            cache: Optional cache instance for price data
        """
        self._cache = cache or Cache()
        self._yahoo_enabled = True  # Will be configurable via settings

    def fetch_historical_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "AAPL")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            use_cache: Whether to use cached data (default: True)

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adjusted_Close, Volume

        Raises:
            TickerNotFoundError: If ticker is not found
            DataFetchError: If data cannot be fetched
        """
        ticker = ticker.strip().upper()

        # Check cache first
        if use_cache:
            cache_key = f"prices:{ticker}:{start_date}:{end_date}"
            cached_data = self._cache.get(cache_key)
            if cached_data is not None:
                # Verify cached data is a DataFrame
                if not isinstance(cached_data, pd.DataFrame):
                    logger.warning(
                        f"Cache returned non-DataFrame for {ticker}: "
                        f"{type(cached_data)}. Clearing cache entry."
                    )
                    # Clear corrupted cache entry
                    try:
                        self._cache.delete(cache_key)
                    except Exception:
                        pass
                    # Continue to fetch from API
                else:
                    logger.debug(f"Cache hit for historical prices: {ticker}")
                    return cached_data

        # Fetch from Yahoo Finance
        try:
            df = self._fetch_from_yahoo(ticker, start_date, end_date)
        except DataFetchError:
            # Re-raise DataFetchError as-is (already has context)
            raise
        except TickerNotFoundError:
            # Re-raise TickerNotFoundError as-is
            raise
        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance for {ticker}: {e}", exc_info=True)
            # TODO: Add fallback to other sources (Alpha Vantage, IEX Cloud)
            raise DataFetchError(f"Failed to fetch price data for {ticker}") from e

        # Standardize columns
        df = self._standardize_dataframe(df)

        # Cache the result
        if use_cache:
            self._cache.set(cache_key, df, ttl=CACHE_TTL_HISTORICAL_PRICES)

        logger.info(
            f"Fetched {len(df)} days of price data for {ticker} "
            f"({start_date} to {end_date})"
        )

        return df

    def fetch_current_price(self, ticker: str, use_cache: bool = True) -> float:
        """
        Fetch current price for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "AAPL")
            use_cache: Whether to use cached data (default: True, 5min TTL)

        Returns:
            Current price as float

        Raises:
            TickerNotFoundError: If ticker is not found
            DataFetchError: If price cannot be fetched
        """
        ticker = ticker.strip().upper()

        # Check cache first (5 minute TTL for current prices)
        if use_cache:
            cache_key = f"current_price:{ticker}"
            cached_price = self._cache.get(cache_key)
            if cached_price is not None:
                logger.debug(f"Cache hit for current price: {ticker}")
                return float(cached_price)

        # Fetch latest price
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="1d")

            if data.empty:
                raise TickerNotFoundError(f"No price data available for ticker: {ticker}")

            current_price = float(data["Close"].iloc[-1])

            # Cache for 5 minutes
            if use_cache:
                self._cache.set(cache_key, current_price, ttl=CACHE_TTL_CURRENT_PRICE)

            logger.debug(f"Fetched current price for {ticker}: ${current_price:.2f}")

            return current_price

        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}", exc_info=True)
            raise DataFetchError(f"Failed to fetch current price for {ticker}") from e

    def fetch_bulk_prices(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for multiple tickers with parallel fetching.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            use_cache: Whether to use cached data (default: True)

        Returns:
            DataFrame with columns: Date, Ticker, Adjusted_Close, etc.

        Raises:
            DataFetchError: If bulk fetch fails
        """
        tickers = [t.strip().upper() for t in tickers]

        if not tickers:
            return pd.DataFrame()

        logger.info(f"Fetching bulk prices for {len(tickers)} tickers (parallel: {len(tickers) > 1})")

        # For single ticker, use regular fetch
        if len(tickers) == 1:
            df = self.fetch_historical_prices(tickers[0], start_date, end_date, use_cache=use_cache)
            if not df.empty:
                df["Ticker"] = tickers[0]
            return df

        # Check cache first - if all are cached, fetch sequentially (fast)
        all_cached = True
        for ticker in tickers:
            cache_key = f"prices:{ticker}:{start_date}:{end_date}"
            cached_data = self._cache.get(cache_key)
            if cached_data is None:
                all_cached = False
                break

        # If all cached, fetch sequentially (very fast)
        if all_cached and use_cache:
            dfs = []
            for ticker in tickers:
                try:
                    df = self.fetch_historical_prices(ticker, start_date, end_date, use_cache=True)
                    if not df.empty:
                        df["Ticker"] = ticker
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to fetch cached data for {ticker}: {e}")
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                logger.info(f"Successfully fetched {len(dfs)} tickers from cache")
                return result

        # For uncached data, use parallel fetching
        try:
            # Try yfinance bulk download first (fastest for multiple tickers)
            try:
                data = yf.download(
                    " ".join(tickers),
                    start=start_date,
                    end=end_date + timedelta(days=1),  # yfinance end is exclusive
                    progress=False,
                    group_by="ticker",
                    threads=True,  # Enable threading in yfinance
                )

                if not data.empty:
                    # Standardize the format
                    dfs = []
                    for ticker in tickers:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if ticker in data.columns.levels[0]:
                                    ticker_data = data[ticker].copy()
                                else:
                                    continue
                            else:
                                # Single ticker case
                                ticker_data = data.copy()
                            
                            ticker_data = self._standardize_dataframe(ticker_data)
                            ticker_data["Ticker"] = ticker
                            dfs.append(ticker_data)
                        except Exception as e:
                            logger.warning(f"Error processing {ticker} from bulk download: {e}")

                    if dfs:
                        result = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Successfully fetched bulk prices for {len(dfs)} tickers using yfinance bulk")
                        return result
            except Exception as e:
                logger.warning(f"yfinance bulk download failed, falling back to parallel fetching: {e}")

            # Fallback: Parallel fetching using ThreadPoolExecutor
            dfs = []
            failed_tickers = []

            def fetch_single(ticker: str) -> tuple[str, Optional[pd.DataFrame]]:
                """Fetch single ticker, return (ticker, df)."""
                try:
                    df = self.fetch_historical_prices(ticker, start_date, end_date, use_cache=use_cache)
                    if not df.empty:
                        df["Ticker"] = ticker
                    return (ticker, df)
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {e}")
                    return (ticker, None)

            # Use ThreadPoolExecutor for parallel fetching
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS_PARALLEL_FETCH, len(tickers))) as executor:
                future_to_ticker = {
                    executor.submit(fetch_single, ticker): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker, df = future.result()
                    if df is not None and not df.empty:
                        dfs.append(df)
                    else:
                        failed_tickers.append(ticker)

            if not dfs:
                raise DataFetchError(f"No data returned for any ticker. Failed: {failed_tickers}")

            result = pd.concat(dfs, ignore_index=True)
            
            if failed_tickers:
                logger.warning(f"Failed to fetch {len(failed_tickers)} tickers: {failed_tickers}")

            logger.info(f"Successfully fetched bulk prices for {len(dfs)}/{len(tickers)} tickers (parallel)")

            return result

        except Exception as e:
            logger.error(f"Error in bulk price fetch: {e}", exc_info=True)
            raise DataFetchError(f"Failed to fetch bulk prices: {e}") from e

    def _fetch_from_yahoo(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance with retry logic.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data

        Raises:
            DataFetchError: If fetch fails after retries
        """
        if not self._yahoo_enabled:
            raise DataFetchError("Yahoo Finance is disabled")

        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(MAX_RETRIES_API):
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date + timedelta(days=1),  # yfinance end is exclusive
                )

                if data.empty:
                    # Check if ticker exists at all
                    info = ticker_obj.info
                    if not info or "symbol" not in info:
                        raise TickerNotFoundError(f"Ticker not found: {ticker}")

                    logger.warning(
                        f"No price data for {ticker} in date range {start_date} to {end_date}"
                    )
                    # Return empty DataFrame with correct columns
                    return pd.DataFrame(
                        columns=["Open", "High", "Low", "Close", "Volume"]
                    )

                return data

            except TickerNotFoundError:
                raise
            except Exception as e:
                if attempt < MAX_RETRIES_API - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{MAX_RETRIES_API} failed for {ticker}: {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= RETRY_BACKOFF_MULTIPLIER
                else:
                    logger.error(f"All {MAX_RETRIES_API} attempts failed for {ticker}")
                    raise DataFetchError(f"Failed to fetch data for {ticker} after {MAX_RETRIES_API} attempts") from e

        raise DataFetchError(f"Failed to fetch data for {ticker}")  # Should never reach here

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format.

        Args:
            df: Input DataFrame from yfinance

        Returns:
            Standardized DataFrame with columns: Date, Open, High, Low, Close, Adjusted_Close, Volume
        """
        if df.empty:
            return df

        # Reset index to make Date a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if "Date" not in df.columns:
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        # Standardize column names
        column_mapping = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Adj Close": "Adjusted_Close",
            "Adj_Close": "Adjusted_Close",
            "Volume": "Volume",
        }

        result = pd.DataFrame()

        for col, std_name in column_mapping.items():
            if col in df.columns:
                result[std_name] = df[col]
            elif std_name == "Adjusted_Close" and "Close" in df.columns:
                # Use Close as Adjusted_Close if not available
                result["Adjusted_Close"] = df["Close"]

        # Ensure Date column exists and is tz-naive
        if "Date" not in result.columns and "Date" in df.columns:
            result["Date"] = df["Date"]

        if "Date" in result.columns:
            result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
            try:
                # Drop timezone info to avoid comparison issues
                result["Date"] = result["Date"].dt.tz_localize(None)
            except Exception:
                # If already tz-naive or conversion fails, continue
                pass

        # Sort by date
        if "Date" in result.columns:
            result = result.sort_values("Date").reset_index(drop=True)

        return result

