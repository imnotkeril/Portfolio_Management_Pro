"""Add rebalance_interval_months to portfolios.

Revision ID: 009_portfolio_rebalance_interval
Revises: 008_portfolio_cost_basis_method
Create Date: 2026-05-19
"""

import sqlalchemy as sa
from alembic import op

revision = "009_portfolio_rebalance_interval"
down_revision = "008_portfolio_cost_basis_method"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "portfolios",
        sa.Column("rebalance_interval_months", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("portfolios", "rebalance_interval_months")
