"""Extend transactions with dividend/split fields.

Revision ID: 007_extend_transactions
Revises: 006_portfolios_user_id
Create Date: 2026-05-17
"""

import sqlalchemy as sa
from alembic import op

revision = "007_extend_transactions"
down_revision = "006_portfolios_user_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transactions",
        sa.Column("reinvest", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "transactions",
        sa.Column("split_ratio", sa.Float(), nullable=True),
    )
    op.add_column(
        "transactions",
        sa.Column(
            "currency",
            sa.String(length=3),
            nullable=False,
            server_default="USD",
        ),
    )


def downgrade() -> None:
    op.drop_column("transactions", "currency")
    op.drop_column("transactions", "split_ratio")
    op.drop_column("transactions", "reinvest")
