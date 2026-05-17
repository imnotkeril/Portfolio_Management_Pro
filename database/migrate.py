"""Programmatic Alembic upgrades (Docker entrypoint and app startup)."""

from alembic import command
from alembic.config import Config


def upgrade_head() -> None:
    """Apply all pending migrations."""
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
