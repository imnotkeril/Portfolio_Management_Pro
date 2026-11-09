"""Database session management."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from config.settings import settings


# Base class for models
class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite:")


# Engine
def _create_engine():
    url = settings.database_url
    if _is_sqlite_url(url) and ":memory:" in url:
        return create_engine(
            url,
            echo=settings.database_echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(
        url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


engine = _create_engine()

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Usage:
        with get_db_session() as session:
            portfolio = session.query(Portfolio).first()
            portfolio.name = "Updated"
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables via SQLAlchemy metadata (SQLite / local dev / tests only)."""
    if not _is_sqlite_url(settings.database_url):
        raise RuntimeError(
            "init_db() is for SQLite only. Use ensure_database_schema() with PostgreSQL."
        )

    db_path = settings.database_url.replace("sqlite:///", "")
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    from models import (  # noqa: F401
        Portfolio,
        Position,
        PriceHistory,
        Transaction,
        User,
    )

    Base.metadata.create_all(bind=engine)

    from core.data_manager.user_repository import UserRepository

    UserRepository().ensure_system_user()


def ensure_database_schema() -> None:
    """
    Prepare the database schema.

    - SQLite: create tables if missing (tests, Streamlit local).
    - PostgreSQL: run Alembic migrations (Docker / production).
    """
    if _is_sqlite_url(settings.database_url):
        init_db()
        return

    from database.migrate import upgrade_head

    upgrade_head()
