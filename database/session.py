"""Database session management."""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings


# Base class for models
class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""

    pass


# Engine
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

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
    """Create all database tables."""
    # Ensure database directory exists for SQLite
    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.replace("sqlite:///", "")
        if db_path != ":memory:":
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Import all models to ensure they are registered
    from models import Portfolio, Position, PriceHistory, Transaction  # noqa: F401
    
    # Create all tables
    Base.metadata.create_all(bind=engine)

