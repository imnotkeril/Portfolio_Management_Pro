"""Transaction ORM model."""

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from models.portfolio import Portfolio
else:
    Portfolio = "Portfolio"

from database.session import Base


class Transaction(Base):
    """Transaction ORM model for portfolio operations."""

    __tablename__ = "transactions"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Foreign key
    portfolio_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Attributes
    transaction_date: Mapped[date] = mapped_column(
        Date, nullable=False, index=True
    )
    transaction_type: Mapped[str] = mapped_column(
        String(20), nullable=False  # BUY, SELL, DEPOSIT, WITHDRAWAL
    )
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(
        Float, nullable=False
    )  # shares * price
    fees: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio",
        back_populates="transactions",
    )

    # Indexes
    __table_args__ = (
        Index("idx_transaction_portfolio", "portfolio_id"),
        Index("idx_transaction_date", "transaction_date"),
        Index(
            "idx_transaction_portfolio_date",
            "portfolio_id",
            "transaction_date",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction(id={self.id}, type={self.transaction_type}, "
            f"ticker={self.ticker}, date={self.transaction_date})>"
        )

