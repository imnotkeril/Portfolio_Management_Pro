"""Add subscriptions and stripe_events tables.

Revision ID: 011_subscriptions_stripe
Revises: 010_portfolio_ledger_mode
Create Date: 2026-05-22
"""

import sqlalchemy as sa
from alembic import op

revision = "011_subscriptions_stripe"
down_revision = "010_portfolio_ledger_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("stripe_customer_id", sa.String(length=255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(length=255), nullable=True),
        sa.Column("plan", sa.String(length=20), nullable=False, server_default="free"),
        sa.Column(
            "status", sa.String(length=30), nullable=False, server_default="active"
        ),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])

    op.create_table(
        "stripe_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("stripe_event_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stripe_event_id"),
    )
    op.create_index(
        "ix_stripe_events_stripe_event_id", "stripe_events", ["stripe_event_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_stripe_events_stripe_event_id", table_name="stripe_events")
    op.drop_table("stripe_events")
    op.drop_index("ix_subscriptions_user_id", table_name="subscriptions")
    op.drop_table("subscriptions")
