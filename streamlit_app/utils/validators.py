"""Validation utilities for UI input."""

import re
from typing import List, Optional, Tuple


def validate_ticker_format(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Validate ticker format.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker cannot be empty"

    ticker = ticker.strip().upper()

    if len(ticker) < 1 or len(ticker) > 10:
        return False, "Ticker must be 1-10 characters"

    if not re.match(r"^[A-Z0-9]+$", ticker):
        return False, "Ticker must contain only uppercase letters and numbers"

    return True, None


def validate_shares(shares: Optional[float]) -> Tuple[bool, Optional[str]]:
    """
    Validate shares value.

    Args:
        shares: Number of shares

    Returns:
        Tuple of (is_valid, error_message)
    """
    if shares is None:
        return False, "Shares cannot be empty"

    if shares <= 0:
        return False, "Shares must be greater than 0"

    if shares > 1_000_000_000:
        return False, "Shares value is too large"

    return True, None


def validate_weights_sum(
    weights: List[float], tolerance: float = 0.0001
) -> Tuple[bool, Optional[str]]:
    """
    Validate that weights sum to approximately 1.0.

    Args:
        weights: List of weight values
        tolerance: Tolerance for sum check (default: 0.0001)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not weights:
        return False, "No weights provided"

    total = sum(weights)

    if abs(total - 1.0) > tolerance:
        return (
            False,
            f"Weights must sum to 1.0, got {total:.4f}",
        )

    return True, None


def validate_portfolio_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate portfolio name.

    Args:
        name: Portfolio name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Portfolio name cannot be empty"

    name = name.strip()

    if len(name) < 1:
        return False, "Portfolio name must be at least 1 character"

    if len(name) > 100:
        return False, "Portfolio name must be at most 100 characters"

    return True, None


def validate_starting_capital(
    capital: Optional[float],
) -> Tuple[bool, Optional[str]]:
    """
    Validate starting capital.

    Args:
        capital: Starting capital amount

    Returns:
        Tuple of (is_valid, error_message)
    """
    if capital is None:
        return False, "Starting capital cannot be empty"

    if capital <= 0:
        return False, "Starting capital must be greater than 0"

    if capital > 1_000_000_000_000:
        return False, "Starting capital value is too large"

    return True, None
