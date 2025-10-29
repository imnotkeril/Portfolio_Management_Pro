"""Price history ORM model."""

from datetime import date

from sqlalchemy import Date, Float, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from database.session import Base


class PriceHistory(Base):
    """Price history model for storing OHLCV data."""

    __tablename__ = "price_history"

    # Composite primary key
    ticker: Mapped[str] = mapped_column(
        String(10), primary_key=True, nullable=False
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True, nullable=False)

    # OHLCV data
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float] = mapped_column(
        Float, nullable=False
    )
    # Use Float for large numbers
    volume: Mapped[int | None] = mapped_column(Float, nullable=True)

    # Indexes for fast lookups
    __table_args__ = (
        Index("idx_price_history_ticker", "ticker"),
        Index("idx_price_history_date", "date"),
        Index(
            "idx_price_history_ticker_date", "ticker", "date"
        ),
        UniqueConstraint(
            "ticker",
            "date",
            name="uq_price_history_ticker_date",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<PriceHistory(ticker={self.ticker}, "
            f"date={self.date}, close={self.close})>"
        )
