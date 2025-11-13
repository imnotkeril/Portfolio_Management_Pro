"""Market data utilities for dashboard."""

from datetime import date, timedelta
from typing import Dict, Optional

import yfinance as yf

from services.data_service import DataService


def get_index_data(
    symbol: str,
    data_service: Optional[DataService] = None,
) -> Dict[str, Optional[float]]:
    """
    Get current index data including price and daily change.

    Args:
        symbol: Index symbol (e.g., "^GSPC")
        data_service: Optional DataService instance

    Returns:
        Dict with 'price', 'change', 'change_pct'
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price
        current_price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )

        # Get previous close for change calculation
        previous_close = info.get("previousClose")

        if current_price and previous_close:
            change = current_price - previous_close
            change_pct = (change / previous_close) * 100 if previous_close > 0 else 0.0

            return {
                "price": float(current_price),
                "change": float(change),
                "change_pct": float(change_pct),
            }
        else:
            # Fallback: try to get from history
            hist = ticker.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                current_price = float(hist["Close"].iloc[-1])
                previous_close = float(hist["Close"].iloc[-2])
                change = current_price - previous_close
                change_pct = (change / previous_close) * 100 if previous_close > 0 else 0.0

                return {
                    "price": current_price,
                    "change": change,
                    "change_pct": change_pct,
                }

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to fetch index data for {symbol}: {e}")

    return {"price": None, "change": None, "change_pct": None}

