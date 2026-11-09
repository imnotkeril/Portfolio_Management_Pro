"""Add ledger_mode to portfolios (buy_hold vs transactions).

Revision ID: 010_portfolio_ledger_mode
Revises: 009_portfolio_rebalance_interval
Create Date: 2026-05-19
"""

import sqlalchemy as sa
from alembic import op

revision = "010_portfolio_ledger_mode"
down_revision = "009_portfolio_rebalance_interval"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "portfolios",
        sa.Column(
            "ledger_mode",
            sa.String(20),
            nullable=False,
            server_default="buy_hold",
        ),
    )
    op.execute("""
        UPDATE portfolios
        SET ledger_mode = 'transactions'
        WHERE id IN (SELECT DISTINCT portfolio_id FROM transactions)
        """)


def downgrade() -> None:
    op.drop_column("portfolios", "ledger_mode")
