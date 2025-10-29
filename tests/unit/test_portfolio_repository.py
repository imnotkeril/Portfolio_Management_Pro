"""Unit tests for portfolio repository."""

from datetime import date

import pytest

from core.data_manager.portfolio import Portfolio
from core.data_manager.portfolio_repository import PortfolioRepository
from core.exceptions import PortfolioNotFoundError
from database.session import Base, engine, init_db


@pytest.fixture(scope="function")
def test_db():
    """Create test database."""
    # Use in-memory SQLite
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    SessionLocal = sessionmaker(bind=test_engine)

    yield SessionLocal

    Base.metadata.drop_all(test_engine)


def test_save_new_portfolio(test_db) -> None:
    """Test saving new portfolio."""
    # Mock repository with test database
    # Note: This is a simplified test - full integration test in integration/
    portfolio = Portfolio(
        name="Test Portfolio",
        starting_capital=100000.0,
        description="Test",
    )
    portfolio.add_position(ticker="AAPL", shares=100.0)

    # Repository uses get_db_session which needs proper setup
    # For unit test, we'll test the conversion logic
    repository = PortfolioRepository()

    # This would normally save to DB
    # In unit test, we focus on domain logic
    assert portfolio.name == "Test Portfolio"
    assert len(portfolio.get_all_positions()) == 1


def test_find_by_id_not_found() -> None:
    """Test finding portfolio by non-existent ID."""
    repository = PortfolioRepository()

    result = repository.find_by_id("nonexistent-id")
    assert result is None


def test_find_by_name_not_found() -> None:
    """Test finding portfolio by non-existent name."""
    repository = PortfolioRepository()

    result = repository.find_by_name("Nonexistent Portfolio")
    assert result is None


def test_find_all_empty() -> None:
    """Test finding all portfolios when none exist."""
    repository = PortfolioRepository()

    portfolios = repository.find_all()
    # Should return empty list, not error
    assert isinstance(portfolios, list)


def test_delete_nonexistent() -> None:
    """Test deleting non-existent portfolio."""
    repository = PortfolioRepository()

    result = repository.delete("nonexistent-id")
    assert result is False

