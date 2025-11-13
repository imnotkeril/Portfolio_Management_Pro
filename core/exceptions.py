"""Custom exception hierarchy for WMC application."""

from typing import Dict, Optional


class WMCBaseException(Exception):
    """Base exception for all WMC exceptions."""

    def __init__(self, message: str, details: Optional[Dict] = None) -> None:
        """
        Initialize base exception.

        Args:
            message: Human-readable description of the error.
            details: Optional structured metadata providing additional context.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# Data-related errors
class DataError(WMCBaseException):
    """Base class for data-related errors."""

    pass


class TickerNotFoundError(DataError):
    """Ticker symbol not found in data source."""

    pass


class DataFetchError(DataError):
    """Error fetching data from external source."""

    pass


class InsufficientDataError(DataError):
    """Not enough data points for calculation."""

    pass


# Validation errors
class ValidationError(WMCBaseException):
    """Input validation failed."""

    pass


class ConflictError(WMCBaseException):
    """Resource conflict (e.g., duplicate name)."""

    pass


# Calculation errors
class CalculationError(WMCBaseException):
    """Base class for calculation errors."""

    pass


class OptimizationError(CalculationError):
    """Optimization failed to converge."""

    pass


class NumericalError(CalculationError):
    """Numerical computation error (overflow, NaN, etc.)."""

    pass


# Portfolio errors
class PortfolioError(WMCBaseException):
    """Base class for portfolio-related errors."""

    pass


class PortfolioNotFoundError(PortfolioError):
    """Portfolio not found in database."""

    pass


class PositionNotFoundError(PortfolioError):
    """Position not found in portfolio."""

    pass

