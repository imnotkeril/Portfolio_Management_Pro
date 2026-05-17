"""Transaction domain model."""

import logging
from datetime import date
from typing import Optional

from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class Transaction:
    """Transaction domain model (pure Python, no ORM dependencies)."""

    TRANSACTION_TYPES = [
        "BUY",
        "SELL",
        "DEPOSIT",
        "WITHDRAWAL",
        "DIVIDEND",
        "SPLIT",
    ]
    CASH_TYPES = {"DEPOSIT", "WITHDRAWAL"}
    EQUITY_TYPES = {"BUY", "SELL", "DIVIDEND", "SPLIT"}

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
        reinvest: Optional[bool] = None,
        split_ratio: Optional[float] = None,
        currency: str = "USD",
    ) -> None:
        normalized_type = transaction_type.upper()
        normalized_ticker = ticker.strip().upper()
        normalized_currency = currency.strip().upper()

        if normalized_type not in self.TRANSACTION_TYPES:
            raise ValidationError(
                f"Invalid transaction type: {transaction_type}. "
                f"Must be one of {self.TRANSACTION_TYPES}"
            )

        if normalized_type in self.CASH_TYPES:
            if normalized_ticker != "CASH":
                raise ValidationError(
                    "DEPOSIT/WITHDRAWAL transactions must have ticker='CASH'"
                )
        elif normalized_ticker == "CASH":
            raise ValidationError(
                f"{normalized_type} transactions cannot use ticker='CASH'"
            )

        if normalized_type == "DIVIDEND":
            if reinvest is None:
                raise ValidationError("DIVIDEND transactions require reinvest (bool)")
        elif normalized_type == "SPLIT":
            if split_ratio is None or split_ratio <= 0:
                raise ValidationError("SPLIT transactions require split_ratio > 0")
        elif reinvest is not None:
            raise ValidationError("reinvest is only valid for DIVIDEND transactions")

        if normalized_type != "SPLIT":
            if shares <= 0:
                raise ValidationError("Shares must be greater than 0")
            if price <= 0:
                raise ValidationError("Price must be greater than 0")
        else:
            if shares <= 0:
                shares = 1.0
            if price <= 0:
                price = 1.0

        self.id = transaction_id
        self.transaction_date = transaction_date
        self.transaction_type = normalized_type
        self.ticker = normalized_ticker
        self.shares = shares
        self.price = price
        self.amount = amount if amount is not None else shares * price
        self.fees = fees
        self.notes = notes
        self.reinvest = reinvest
        self.split_ratio = split_ratio
        self.currency = normalized_currency

    def __repr__(self) -> str:
        return (
            f"<Transaction(type={self.transaction_type}, "
            f"ticker={self.ticker}, date={self.transaction_date}, "
            f"shares={self.shares}, price={self.price})>"
        )

    def __eq__(self, other: object) -> bool:
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
