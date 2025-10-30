"""Formatting utilities for UI display."""

from datetime import date, datetime
from typing import Optional


def format_currency(
    value: float,
    currency: str = "USD",
    show_cents: bool = True,
) -> str:
    """
    Format currency value.

    Args:
        value: Currency value
        currency: Currency symbol (default: USD)
        show_cents: Whether to show cents (default: True)

    Returns:
        Formatted currency string (e.g., "$100,000.00")
    """
    if value is None:
        return "-"

    if currency == "USD":
        symbol = "$"
    else:
        symbol = currency

    if show_cents:
        return f"{symbol}{value:,.2f}"
    else:
        return f"{symbol}{value:,.0f}"


def format_percentage(
    value: Optional[float],
    decimals: int = 2,
    show_sign: bool = True,
) -> str:
    """
    Format percentage value.

    Args:
        value: Percentage as decimal (e.g., 0.2534 for 25.34%)
        decimals: Number of decimal places (default: 2)
        show_sign: Whether to show + sign for positive (default: True)

    Returns:
        Formatted percentage string (e.g., "25.34%")
    """
    if value is None:
        return "-"

    # Handle tuple values (some metrics return tuples)
    if isinstance(value, (tuple, list)):
        if len(value) > 0:
            value = value[0]
        else:
            return "-"

    # Convert to float if needed
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"

    percentage = value * 100

    if show_sign and percentage > 0:
        sign = "+"
    elif show_sign and percentage < 0:
        sign = ""
    else:
        sign = ""

    return f"{sign}{percentage:.{decimals}f}%"


def format_date(date_value: Optional[date]) -> str:
    """
    Format date value.

    Args:
        date_value: Date object

    Returns:
        Formatted date string (e.g., "2025-01-15")
    """
    if date_value is None:
        return "-"

    if isinstance(date_value, datetime):
        return date_value.date().isoformat()

    return date_value.isoformat()


def format_large_number(value: Optional[float]) -> str:
    """
    Format large numbers with abbreviations.

    Args:
        value: Number to format

    Returns:
        Formatted string (e.g., "1.5M", "2.3B")
    """
    if value is None:
        return "-"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000_000:.2f}T"
    elif abs_value >= 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.2f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.2f}M"
    elif abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.2f}K"
    else:
        return f"{sign}{abs_value:.2f}"
