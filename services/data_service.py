"""Data service for orchestrating data operations."""

import logging
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import and_

from core.data_manager.cache import Cache
from core.data_manager.price_manager import PriceManager
from core.data_manager.ticker_validator import TickerInfo, TickerValidator
from core.exceptions import ValidationError
from database.session import get_db_session
from models.price_history import PriceHistory

logger = logging.getLogger(__name__)


class DataService:
    """Service for orchestrating data operations."""

    def __init__(
        self,
        cache: Optional[Cache] = None,
        price_manager: Optional[PriceManager] = None,
        ticker_validator: Optional[TickerValidator] = None,
    ) -> None:
        """
        Initialize data service.

        Args:
            cache: Optional cache instance
            price_manager: Optional price manager instance
            ticker_validator: Optional ticker validator instance
        """
        self._cache = cache or Cache()
        self._price_manager = price_manager or PriceManager(cache=self._cache)
        self._ticker_validator = ticker_validator or TickerValidator(cache=self._cache)

    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate a ticker symbol.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            True if ticker is valid, False otherwise

        Raises:
            ValidationError: If ticker format is invalid
        """
        try:
            return self._ticker_validator.validate_ticker(ticker)
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating ticker {ticker}: {e}", exc_info=True)
            return False

    def validate_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Validate multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to validation result
        """
        return self._ticker_validator.validate_tickers(tickers)

    def get_ticker_info(self, ticker: str) -> TickerInfo:
        """
        Get ticker information.

        Args:
            ticker: Ticker symbol

        Returns:
            TickerInfo object

        Raises:
            TickerNotFoundError: If ticker not found
        """
        return self._ticker_validator.get_ticker_info(ticker)

    def fetch_historical_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical prices with database caching.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache (default: True)
            save_to_db: Whether to save fetched data to database (default: True)

        Returns:
            DataFrame with price data

        Raises:
            TickerNotFoundError: If ticker not found
            DataFetchError: If data cannot be fetched
        """
        ticker = ticker.strip().upper()

        # Check database first
        if use_cache:
            db_data = self._get_prices_from_db(ticker, start_date, end_date)
            if db_data is not None and not db_data.empty:
                logger.info(f"Loaded {len(db_data)} records from database for {ticker}")
                return db_data

        # Fetch from API
        logger.info(f"Fetching prices from API for {ticker} ({start_date} to {end_date})")
        df = self._price_manager.fetch_historical_prices(
            ticker, start_date, end_date, use_cache=use_cache
        )

        # Save to database
        if save_to_db and not df.empty:
            self._save_prices_to_db(ticker, df)

        return df

    def fetch_current_price(self, ticker: str, use_cache: bool = True) -> float:
        """
        Fetch current price for a ticker.

        Args:
            ticker: Ticker symbol
            use_cache: Whether to use cache (default: True)

        Returns:
            Current price

        Raises:
            TickerNotFoundError: If ticker not found
            DataFetchError: If price cannot be fetched
        """
        return self._price_manager.fetch_current_price(ticker, use_cache=use_cache)

    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to latest price

        Raises:
            DataFetchError: If prices cannot be fetched
        """
        prices: Dict[str, float] = {}
        for ticker in tickers:
            try:
                ticker = ticker.strip().upper()
                price = self.fetch_current_price(ticker, use_cache=True)
                prices[ticker] = price
            except Exception as e:
                logger.warning(f"Error fetching price for {ticker}: {e}")
                # Continue with other tickers
        return prices

    def fetch_bulk_prices(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache (default: True)
            save_to_db: Whether to save to database (default: True)

        Returns:
            DataFrame with price data for all tickers

        Raises:
            DataFetchError: If bulk fetch fails
        """
        return self._price_manager.fetch_bulk_prices(tickers, start_date, end_date, use_cache=use_cache)

    def _get_prices_from_db(
        self, ticker: str, start_date: date, end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Get prices from database.

        Args:
            ticker: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data or None if not found
        """
        try:
            with get_db_session() as session:
                records = (
                    session.query(PriceHistory)
                    .filter(
                        and_(
                            PriceHistory.ticker == ticker,
                            PriceHistory.date >= start_date,
                            PriceHistory.date <= end_date,
                        )
                    )
                    .order_by(PriceHistory.date)
                    .all()
                )

                if not records:
                    return None

                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
                        "Date": record.date,
                        "Open": record.open,
                        "High": record.high,
                        "Low": record.low,
                        "Close": record.close,
                        "Adjusted_Close": record.adjusted_close,
                        "Volume": int(record.volume) if record.volume else None,
                    })

                df = pd.DataFrame(data)
                return df

        except Exception as e:
            logger.warning(f"Error reading prices from database for {ticker}: {e}")
            return None

    def _save_prices_to_db(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Save prices to database.

        Args:
            ticker: Ticker symbol
            df: DataFrame with price data
        """
        if df.empty:
            return

        try:
            with get_db_session() as session:
                # Check which records already exist
                existing_dates = {
                    row.date
                    for row in session.query(PriceHistory.date)
                    .filter(PriceHistory.ticker == ticker)
                    .all()
                }

                # Prepare records to insert
                records = []
                for _, row in df.iterrows():
                    price_date = row.get("Date")
                    if price_date and price_date not in existing_dates:
                        records.append(
                            PriceHistory(
                                ticker=ticker,
                                date=price_date if isinstance(price_date, date) else price_date.date(),
                                open=row.get("Open"),
                                high=row.get("High"),
                                low=row.get("Low"),
                                close=row.get("Close"),
                                adjusted_close=row.get("Adjusted_Close", row.get("Close")),
                                volume=row.get("Volume"),
                            )
                        )

                if records:
                    session.add_all(records)
                    logger.info(f"Saved {len(records)} price records to database for {ticker}")

        except Exception as e:
            logger.error(f"Error saving prices to database for {ticker}: {e}", exc_info=True)
            # Don't raise - database save failure shouldn't break API response
