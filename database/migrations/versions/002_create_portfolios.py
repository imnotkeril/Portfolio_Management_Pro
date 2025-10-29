"""Create portfolios table.

Revision ID: 002_create_portfolios
Revises: 001_create_price_history
Create Date: 2025-01-XX XX:XX:XX.XXXXXX
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import func

# revision identifiers, used by Alembic.
revision = "002_create_portfolios"
down_revision = "001_create_price_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create portfolios table."""
    op.create_table(
        "portfolios",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("starting_capital", sa.Float(), nullable=False),
        sa.Column("base_currency", sa.String(length=3), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index("idx_portfolio_name", "portfolios", ["name"])
    op.create_index("idx_portfolio_created", "portfolios", ["created_at"])


def downgrade() -> None:
    """Drop portfolios table."""
    op.drop_index("idx_portfolio_created", "portfolios")
    op.drop_index("idx_portfolio_name", "portfolios")
    op.drop_table("portfolios")

