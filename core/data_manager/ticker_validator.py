"""Ticker validation and info retrieval."""

import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import yfinance as yf

from config.constants import CACHE_TTL_TICKER_VALIDATION, MAX_TICKER_LENGTH, MIN_TICKER_LENGTH
from core.data_manager.cache import Cache
from core.exceptions import TickerNotFoundError, ValidationError

logger = logging.getLogger(__name__)

# Ticker format: 1-10 alphanumeric characters or hyphens, uppercase
TICKER_PATTERN = re.compile(rf"^[A-Z0-9-]{{{MIN_TICKER_LENGTH},{MAX_TICKER_LENGTH}}}$")


class TickerInfo:
    """Information about a ticker symbol."""

    def __init__(
        self,
        ticker: str,
        name: str,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap: Optional[float] = None,
        currency: str = "USD",
    ) -> None:
        """
        Create ticker metadata container.

        Args:
            ticker: Canonical ticker symbol.
            name: Display name for the security.
            sector: Optional sector classification.
            industry: Optional industry classification.
            market_cap: Latest reported market capitalization.
            currency: Trading currency code for the security.
        """
        self.ticker = ticker
        self.name = name
        self.sector = sector
        self.industry = industry
        self.market_cap = market_cap
        self.currency = currency


class TickerValidator:
    """Validate ticker symbols and retrieve ticker information."""

    def __init__(self, cache: Optional[Cache] = None) -> None:
        """
        Initialize ticker validator.

        Args:
            cache: Optional cache instance for validation results
        """
        self._cache = cache or Cache()
        self._validation_cache: Dict[str, tuple[bool, Optional[datetime]]] = {}

    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol exists and is valid.

        Args:
            ticker: Ticker symbol to validate (e.g., "AAPL")

        Returns:
            True if ticker is valid and exists, False otherwise

        Raises:
            ValidationError: If ticker format is invalid
        """
        # Normalize ticker: strip whitespace, uppercase
        ticker = ticker.strip().upper()

        # Validate format
        if not TICKER_PATTERN.match(ticker):
            raise ValidationError(
                f"Invalid ticker format: {ticker}. Must be 1-10 alphanumeric characters."
            )

        # Check cache first
        cache_key = f"ticker_validation:{ticker}"
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for ticker validation: {ticker}")
            return bool(cached_result)

        # Check in-memory cache with TTL
        if ticker in self._validation_cache:
            is_valid, cached_time = self._validation_cache[ticker]
            if cached_time and datetime.now() - cached_time < timedelta(
                seconds=CACHE_TTL_TICKER_VALIDATION
            ):
                logger.debug(f"In-memory cache hit for ticker validation: {ticker}")
                return is_valid

        # Validate via API with retry logic
        max_retries = 2
        retry_delay = 0.5
        is_valid = False  # Default to False
        
        for attempt in range(max_retries):
            try:
                ticker_obj = yf.Ticker(ticker)
                
                # Try multiple methods to validate ticker
                # Method 1: Check info dict (fastest, but may fail)
                try:
                    info = ticker_obj.info
                    if info and isinstance(info, dict) and len(info) > 0:
                        # Check for symbol or other indicators of valid ticker
                        if "symbol" in info or "longName" in info or "shortName" in info:
                            is_valid = True
                            break  # Success, exit retry loop
                        else:
                            # If info exists but no symbol, try method 2
                            is_valid = None
                    else:
                        is_valid = None
                except Exception as e:
                    error_str = str(e).lower()
                    # If rate limited or unauthorized, wait and retry
                    if "401" in error_str or "rate limit" in error_str or "429" in error_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit for {ticker}, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                    is_valid = None
                
                # Method 2: Check historical data (more reliable, but slower)
                if is_valid is None:
                    try:
                        # Try to get recent history (last 30 days for better reliability)
                        end_date = date.today()
                        start_date = end_date - timedelta(days=30)
                        history = ticker_obj.history(
                            start=start_date, end=end_date, period="1mo"
                        )
                        is_valid = not history.empty and len(history) > 0
                        if is_valid:
                            break  # Success, exit retry loop
                    except Exception as e:
                        error_str = str(e).lower()
                        if "401" in error_str or "rate limit" in error_str or "429" in error_str:
                            if attempt < max_retries - 1:
                                logger.warning(f"Rate limit hit for {ticker} (history), retrying...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                        is_valid = False
                
                # Method 3: If still not determined, check if ticker has any data
                if is_valid is None or is_valid is False:
                    try:
                        # Try getting fast_info as last resort (lightweight)
                        fast_info = ticker_obj.fast_info
                        if fast_info and hasattr(fast_info, 'lastPrice'):
                            is_valid = True
                            break  # Success, exit retry loop
                        else:
                            is_valid = False
                    except Exception:
                        is_valid = False
                
                # If we got here and is_valid is still None, set to False
                if is_valid is None:
                    is_valid = False
                    
                break  # Exit retry loop if we got here
                
            except Exception as e:
                error_str = str(e).lower()
                # Handle rate limiting and retry
                if ("401" in error_str or "rate limit" in error_str or "429" in error_str) and attempt < max_retries - 1:
                    logger.warning(f"API error for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(f"Error validating ticker {ticker}: {e}", exc_info=True)
                    is_valid = False
                    break

        # Cache results (after retry loop completes)
        cache_until = datetime.now() + timedelta(seconds=CACHE_TTL_TICKER_VALIDATION)
        self._validation_cache[ticker] = (is_valid, cache_until)
        self._cache.set(cache_key, is_valid, ttl=CACHE_TTL_TICKER_VALIDATION)

        if is_valid:
            logger.info(f"Ticker validated: {ticker}")
        else:
            logger.warning(f"Ticker not found or invalid: {ticker}")

        return is_valid

        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}", exc_info=True)
            return False

    def validate_tickers(self, tickers: list[str]) -> Dict[str, bool]:
        """
        Validate multiple tickers with rate limiting protection.

        Args:
            tickers: List of ticker symbols to validate

        Returns:
            Dictionary mapping ticker to validation result (True/False)
        """
        results: Dict[str, bool] = {}
        
        # Add delay between requests to avoid rate limiting
        # yfinance can be sensitive to rapid requests
        delay_between_requests = 0.2  # 200ms delay between requests

        for i, ticker in enumerate(tickers):
            try:
                # Add delay before each request (except the first one)
                if i > 0:
                    time.sleep(delay_between_requests)
                
                results[ticker] = self.validate_ticker(ticker)
            except ValidationError as e:
                logger.warning(f"Validation error for {ticker}: {e.message}")
                results[ticker] = False
            except Exception as e:
                logger.error(f"Unexpected error validating {ticker}: {e}", exc_info=True)
                results[ticker] = False
                
                # If we hit rate limiting, add extra delay before next request
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str or "401" in error_str:
                    logger.warning(f"Rate limiting detected, adding extra delay...")
                    time.sleep(1.0)  # Wait 1 second before continuing

        return results

    def get_ticker_info(self, ticker: str) -> TickerInfo:
        """
        Get detailed information about a ticker.

        Args:
            ticker: Ticker symbol (e.g., "AAPL")

        Returns:
            TickerInfo object with ticker details

        Raises:
            TickerNotFoundError: If ticker is not found or invalid
            ValidationError: If ticker format is invalid
        """
        # Validate format first
        ticker = ticker.strip().upper()
        if not TICKER_PATTERN.match(ticker):
            raise ValidationError(
                f"Invalid ticker format: {ticker}. Must be 1-10 alphanumeric characters."
            )

        # Check if ticker is valid
        if not self.validate_ticker(ticker):
            raise TickerNotFoundError(f"Ticker not found: {ticker}")

        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info

            if not info or "symbol" not in info:
                raise TickerNotFoundError(f"Ticker not found: {ticker}")

            return TickerInfo(
                ticker=ticker,
                name=info.get("longName") or info.get("shortName") or ticker,
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("marketCap"),
                currency=info.get("currency", "USD"),
            )

        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {e}", exc_info=True)
            raise TickerNotFoundError(f"Failed to fetch info for ticker: {ticker}") from e

