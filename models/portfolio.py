"""Portfolio ORM model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from models.position import Position
    from models.transaction import Transaction
    from models.user import User
else:
    Transaction = "Transaction"
    User = "User"

from database.session import Base


class Portfolio(Base):
    """Portfolio ORM model."""

    __tablename__ = "portfolios"

    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Attributes
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    starting_capital: Mapped[float] = mapped_column(Float, nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    cost_basis_method: Mapped[str] = mapped_column(
        String(10), default="fifo", nullable=False
    )
    rebalance_interval_months: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    user: Mapped["User"] = relationship(  # noqa: F821
        "User",
        back_populates="portfolios",
        lazy="selectin",
    )
    positions: Mapped[list["Position"]] = relationship(  # noqa: F821
        "Position",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin",  # Eager load positions
    )
    transactions: Mapped[list["Transaction"]] = relationship(  # noqa: F821
        "Transaction",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="select",  # Changed from selectin to avoid initialization issues
        # Note: order_by removed - transactions should be sorted in queries if needed
        # Using order_by with string reference causes issues with TYPE_CHECKING
    )

    # Indexes
    __table_args__ = (
        Index("idx_portfolio_user_name", "user_id", "name"),
        Index("idx_portfolio_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Portfolio(id={self.id}, name={self.name})>"
