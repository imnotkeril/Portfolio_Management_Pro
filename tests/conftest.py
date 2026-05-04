"""Pytest fixtures and configuration."""

from pathlib import Path

import pytest

from core.data_manager.cache import Cache
from core.data_manager.price_manager import PriceManager
from core.data_manager.ticker_validator import TickerValidator
from database.session import Base
from services.data_service import DataService


@pytest.fixture(scope="session", autouse=True)
def _init_database_schema():
    """Ensure SQLite tables exist for tests using get_db_session().

    CI and fresh clones have no migrated DB file; repository/service tests
    otherwise hit sqlite with an empty schema (no such table: portfolios).
    """
    from database.session import init_db

    init_db()
    yield  # session scoped teardown not needed


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir: Path) -> Cache:
    """Create cache instance with temporary directory."""
    return Cache(cache_dir=temp_cache_dir)


@pytest.fixture
def ticker_validator(cache: Cache) -> TickerValidator:
    """Create ticker validator instance."""
    return TickerValidator(cache=cache)


@pytest.fixture
def price_manager(cache: Cache) -> PriceManager:
    """Create price manager instance."""
    return PriceManager(cache=cache)


@pytest.fixture
def data_service(cache: Cache) -> DataService:
    """Create data service instance."""
    return DataService(cache=cache)


@pytest.fixture(scope="function")
def test_db():
    """Create test database and cleanup after test."""
    # Use in-memory SQLite for tests
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    SessionLocal = sessionmaker(bind=test_engine)

    yield SessionLocal

    Base.metadata.drop_all(test_engine)
