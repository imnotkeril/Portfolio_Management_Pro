"""Add cost_basis_method to portfolios.

Revision ID: 008_portfolio_cost_basis_method
Revises: 007_extend_transactions
Create Date: 2026-05-17
"""

import sqlalchemy as sa
from alembic import op

revision = "008_portfolio_cost_basis_method"
down_revision = "007_extend_transactions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "portfolios",
        sa.Column(
            "cost_basis_method",
            sa.String(length=10),
            nullable=False,
            server_default="fifo",
        ),
    )


def downgrade() -> None:
    op.drop_column("portfolios", "cost_basis_method")
