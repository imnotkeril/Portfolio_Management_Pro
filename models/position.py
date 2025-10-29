"""Position ORM model."""

import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, Float, ForeignKey, Index, String, UniqueConstraint

if TYPE_CHECKING:
    from models.portfolio import Portfolio
else:
    Portfolio = "Portfolio"

from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.session import Base


class Position(Base):
    """Position ORM model."""

    __tablename__ = "positions"

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
    )

    # Attributes
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    weight_target: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    purchase_price: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    purchase_date: Mapped[date | None] = mapped_column(
        Date, nullable=True
    )

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio",
        back_populates="positions",
    )

    # Indexes and constraints
    __table_args__ = (
        Index("idx_position_portfolio", "portfolio_id"),
        Index("idx_position_ticker", "ticker"),
        UniqueConstraint(
            "portfolio_id",
            "ticker",
            name="uq_position_portfolio_ticker",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Position(ticker={self.ticker}, "
            f"shares={self.shares}, portfolio_id={self.portfolio_id})>"
        )

