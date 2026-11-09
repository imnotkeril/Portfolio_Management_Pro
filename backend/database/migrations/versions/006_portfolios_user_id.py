"""Add user_id to portfolios and backfill with system user.

Revision ID: 006_portfolios_user_id
Revises: 005_create_users
Create Date: 2026-05-17
"""

import sqlalchemy as sa
from alembic import op

from core.auth.constants import SYSTEM_USER_EMAIL, SYSTEM_USER_ID
from core.auth.password import hash_password

revision = "006_portfolios_user_id"
down_revision = "005_create_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Seed system user, add portfolios.user_id, backfill existing rows."""
    users = sa.table(
        "users",
        sa.column("id", sa.String),
        sa.column("email", sa.String),
        sa.column("password_hash", sa.String),
        sa.column("is_active", sa.Boolean),
    )
    op.bulk_insert(
        users,
        [
            {
                "id": SYSTEM_USER_ID,
                "email": SYSTEM_USER_EMAIL,
                "password_hash": hash_password("not-used-migration-user"),
                "is_active": True,
            }
        ],
    )

    op.add_column(
        "portfolios",
        sa.Column("user_id", sa.String(length=36), nullable=True),
    )
    op.execute(
        sa.text("UPDATE portfolios SET user_id = :uid").bindparams(uid=SYSTEM_USER_ID)
    )
    op.alter_column("portfolios", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_portfolios_user_id_users",
        "portfolios",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("idx_portfolio_user_name", "portfolios", ["user_id", "name"])

    # Legacy global name index replaced by per-user composite index
    op.drop_index("idx_portfolio_name", table_name="portfolios")


def downgrade() -> None:
    """Remove user_id from portfolios and system user."""
    op.create_index("idx_portfolio_name", "portfolios", ["name"])
    op.drop_index("idx_portfolio_user_name", table_name="portfolios")
    op.drop_constraint("fk_portfolios_user_id_users", "portfolios", type_="foreignkey")
    op.drop_column("portfolios", "user_id")
    op.execute(
        sa.text("DELETE FROM users WHERE id = :uid").bindparams(uid=SYSTEM_USER_ID)
    )
