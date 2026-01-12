"""Transaction domain model."""

import logging
from datetime import date
from typing import Optional

from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class Transaction:
    """Transaction domain model (pure Python, no ORM dependencies)."""

    TRANSACTION_TYPES = ["BUY", "SELL", "DEPOSIT", "WITHDRAWAL"]

    def __init__(
        self,
        transaction_date: date,
        transaction_type: str,
        ticker: str,
        shares: float,
        price: float,
        amount: Optional[float] = None,
        fees: float = 0.0,
        notes: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> None:
        """
        Initialize transaction.

        Args:
            transaction_date: Date of transaction
            transaction_type: Type (BUY, SELL, DEPOSIT, WITHDRAWAL)
            ticker: Ticker symbol (or 'CASH' for DEPOSIT/WITHDRAWAL)
            shares: Number of shares (or amount for CASH)
            price: Price per share (or 1.0 for CASH)
            amount: Total amount (shares * price, calculated if not provided)
            fees: Transaction fees
            notes: Optional notes
            transaction_id: Optional transaction ID (for existing transactions)
        """
        # Normalize transaction type and ticker first
        normalized_type = transaction_type.upper()
        normalized_ticker = ticker.strip().upper()

        if normalized_type not in self.TRANSACTION_TYPES:
            raise ValidationError(
                f"Invalid transaction type: {transaction_type}. "
                f"Must be one of {self.TRANSACTION_TYPES}"
            )

        if shares <= 0:
            raise ValidationError("Shares must be greater than 0")

        if price <= 0:
            raise ValidationError("Price must be greater than 0")

        if normalized_type in ["DEPOSIT", "WITHDRAWAL"] and normalized_ticker != "CASH":
            raise ValidationError(
                "DEPOSIT/WITHDRAWAL transactions must have ticker='CASH'"
            )

        self.id = transaction_id
        self.transaction_date = transaction_date
        self.transaction_type = normalized_type
        self.ticker = normalized_ticker
        self.shares = shares
        self.price = price
        self.amount = amount if amount is not None else shares * price
        self.fees = fees
        self.notes = notes

    def __repr__(self) -> str:
        return (
            f"<Transaction(type={self.transaction_type}, "
            f"ticker={self.ticker}, date={self.transaction_date}, "
            f"shares={self.shares}, price={self.price})>"
        )

    def __eq__(self, other: object) -> bool:
        """Compare transactions by ID if available, otherwise by attributes."""
        if not isinstance(other, Transaction):
            return False
        if self.id and other.id:
            return self.id == other.id
        return (
            self.transaction_date == other.transaction_date
            and self.transaction_type == other.transaction_type
            and self.ticker == other.ticker
            and abs(self.shares - other.shares) < 0.0001
            and abs(self.price - other.price) < 0.01
        )

