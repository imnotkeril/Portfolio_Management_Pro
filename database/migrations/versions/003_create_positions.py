"""Create positions table.

Revision ID: 003_create_positions
Revises: 002_create_portfolios
Create Date: 2025-01-XX XX:XX:XX.XXXXXX
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "003_create_positions"
down_revision = "002_create_portfolios"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create positions table."""
    op.create_table(
        "positions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("portfolio_id", sa.String(length=36), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("shares", sa.Float(), nullable=False),
        sa.Column("weight_target", sa.Float(), nullable=True),
        sa.Column("purchase_price", sa.Float(), nullable=True),
        sa.Column("purchase_date", sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "portfolio_id",
            "ticker",
            name="uq_position_portfolio_ticker",
        ),
    )

    # Create indexes
    op.create_index("idx_position_portfolio", "positions", ["portfolio_id"])
    op.create_index("idx_position_ticker", "positions", ["ticker"])


def downgrade() -> None:
    """Drop positions table."""
    op.drop_index("idx_position_ticker", "positions")
    op.drop_index("idx_position_portfolio", "positions")
    op.drop_table("positions")

