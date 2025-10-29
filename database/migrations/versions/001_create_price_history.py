"""Create price_history table.

Revision ID: 001_create_price_history
Revises:
Create Date: 2025-01-XX XX:XX:XX.XXXXXX
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "001_create_price_history"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create price_history table."""
    op.create_table(
        "price_history",
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=True),
        sa.Column("high", sa.Float(), nullable=True),
        sa.Column("low", sa.Float(), nullable=True),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("adjusted_close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("ticker", "date"),
        sa.UniqueConstraint("ticker", "date", name="uq_price_history_ticker_date"),
    )

    # Create indexes
    op.create_index("idx_price_history_ticker", "price_history", ["ticker"])
    op.create_index("idx_price_history_date", "price_history", ["date"])
    op.create_index(
        "idx_price_history_ticker_date",
        "price_history",
        ["ticker", "date"],
    )


def downgrade() -> None:
    """Drop price_history table."""
    op.drop_index("idx_price_history_ticker_date", "price_history")
    op.drop_index("idx_price_history_date", "price_history")
    op.drop_index("idx_price_history_ticker", "price_history")
    op.drop_table("price_history")

