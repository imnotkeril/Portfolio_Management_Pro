"""Create transactions table.

Revision ID: 004_create_transactions
Revises: 003_create_positions
Create Date: 2025-01-XX XX:XX:XX.XXXXXX
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "004_create_transactions"
down_revision = "003_create_positions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create transactions table."""
    op.create_table(
        "transactions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("portfolio_id", sa.String(length=36), nullable=False),
        sa.Column("transaction_date", sa.Date(), nullable=False),
        sa.Column("transaction_type", sa.String(length=20), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("shares", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("fees", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["portfolio_id"],
            ["portfolios.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes
    op.create_index(
        "idx_transaction_portfolio", "transactions", ["portfolio_id"]
    )
    op.create_index("idx_transaction_date", "transactions", ["transaction_date"])
    op.create_index(
        "idx_transaction_portfolio_date",
        "transactions",
        ["portfolio_id", "transaction_date"],
    )


def downgrade() -> None:
    """Drop transactions table."""
    op.drop_index("idx_transaction_portfolio_date", "transactions")
    op.drop_index("idx_transaction_date", "transactions")
    op.drop_index("idx_transaction_portfolio", "transactions")
    op.drop_table("transactions")

