"""Unit tests for portfolio service."""

from unittest.mock import MagicMock, patch

import pytest

from core.data_manager.portfolio import Portfolio
from core.data_manager.portfolio_repository import PortfolioRepository
from core.exceptions import ConflictError, PortfolioNotFoundError, ValidationError
from services.data_service import DataService
from services.portfolio_service import PortfolioService
from services.schemas import (
    AddPositionRequest,
    CreatePortfolioRequest,
    PositionSchema,
    UpdatePortfolioRequest,
    UpdatePositionRequest,
)


@pytest.fixture
def mock_repository() -> PortfolioRepository:
    """Create mock portfolio repository."""
    return MagicMock(spec=PortfolioRepository)


@pytest.fixture
def mock_data_service() -> DataService:
    """Create mock data service."""
    return MagicMock(spec=DataService)


@pytest.fixture
def portfolio_service(
    mock_repository: PortfolioRepository, mock_data_service: DataService
) -> PortfolioService:
    """Create portfolio service with mocked dependencies."""
    return PortfolioService(repository=mock_repository, data_service=mock_data_service)


def test_create_portfolio_success(portfolio_service: PortfolioService) -> None:
    """Test successfully creating a portfolio."""
    request = CreatePortfolioRequest(
        name="Test Portfolio",
        description="Test description",
        starting_capital=100000.0,
        positions=[
            PositionSchema(ticker="AAPL", shares=100.0, weight_target=0.5),
            PositionSchema(ticker="MSFT", shares=50.0, weight_target=0.5),
        ],
    )

    # Mock repository
    portfolio_service._repository.find_by_name.return_value = None
    
    # Create mock portfolio with required attributes
    saved_portfolio = MagicMock(spec=Portfolio)
    saved_portfolio.id = "test-id"
    saved_portfolio.name = "Test Portfolio"
    saved_portfolio.get_all_positions.return_value = []
    portfolio_service._repository.save.return_value = saved_portfolio

    # Mock data service
    portfolio_service._data_service.validate_tickers.return_value = {
        "AAPL": True,
        "MSFT": True,
    }

    result = portfolio_service.create_portfolio(request)

    assert result == saved_portfolio
    portfolio_service._repository.find_by_name.assert_called_once_with("Test Portfolio")
    portfolio_service._data_service.validate_tickers.assert_called_once_with(
        ["AAPL", "MSFT"]
    )
    portfolio_service._repository.save.assert_called_once()


def test_create_portfolio_duplicate_name(portfolio_service: PortfolioService) -> None:
    """Test creating portfolio with duplicate name."""
    request = CreatePortfolioRequest(
        name="Existing Portfolio",
        starting_capital=100000.0,
        positions=[PositionSchema(ticker="AAPL", shares=100.0)],
    )

    # Mock repository to return existing portfolio
    existing_portfolio = MagicMock()
    portfolio_service._repository.find_by_name.return_value = existing_portfolio

    with pytest.raises(ConflictError, match="already exists"):
        portfolio_service.create_portfolio(request)


def test_create_portfolio_invalid_ticker(portfolio_service: PortfolioService) -> None:
    """Test creating portfolio with invalid ticker."""
    request = CreatePortfolioRequest(
        name="Test Portfolio",
        starting_capital=100000.0,
        positions=[PositionSchema(ticker="INVALID", shares=100.0)],
    )

    portfolio_service._repository.find_by_name.return_value = None
    portfolio_service._data_service.validate_tickers.return_value = {
        "INVALID": False
    }

    with pytest.raises(ValidationError, match="Invalid tickers"):
        portfolio_service.create_portfolio(request)


def test_get_portfolio_success(portfolio_service: PortfolioService) -> None:
    """Test successfully getting a portfolio."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.id = "test-id"
    portfolio_service._repository.find_by_id.return_value = portfolio

    result = portfolio_service.get_portfolio("test-id")

    assert result == portfolio
    portfolio_service._repository.find_by_id.assert_called_once_with("test-id")


def test_get_portfolio_not_found(portfolio_service: PortfolioService) -> None:
    """Test getting non-existent portfolio."""
    portfolio_service._repository.find_by_id.return_value = None

    with pytest.raises(PortfolioNotFoundError):
        portfolio_service.get_portfolio("nonexistent-id")


def test_list_portfolios(portfolio_service: PortfolioService) -> None:
    """Test listing portfolios."""
    portfolios = [MagicMock(spec=Portfolio), MagicMock(spec=Portfolio)]
    portfolio_service._repository.find_all.return_value = portfolios

    result = portfolio_service.list_portfolios()

    assert result == portfolios
    portfolio_service._repository.find_all.assert_called_once()


def test_update_portfolio_success(portfolio_service: PortfolioService) -> None:
    """Test successfully updating a portfolio."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.id = "test-id"
    portfolio.name = "Original Name"
    portfolio.description = "Original description"

    # update_portfolio calls get_portfolio which calls find_by_id
    portfolio_service._repository.find_by_id.return_value = portfolio
    # Check name conflict - new name doesn't exist
    portfolio_service._repository.find_by_name.return_value = None
    portfolio_service._repository.save.return_value = portfolio

    request = UpdatePortfolioRequest(name="Updated Name", description="New description")
    result = portfolio_service.update_portfolio("test-id", request)

    assert result == portfolio
    # get_portfolio is called, which calls find_by_id
    assert portfolio_service._repository.find_by_id.called
    # Should check for name conflict
    portfolio_service._repository.find_by_name.assert_called_once_with("Updated Name")
    portfolio_service._repository.save.assert_called_once()


def test_update_portfolio_not_found(portfolio_service: PortfolioService) -> None:
    """Test updating non-existent portfolio."""
    portfolio_service._repository.find_by_id.return_value = None

    request = UpdatePortfolioRequest(name="Updated Name")
    with pytest.raises(PortfolioNotFoundError):
        portfolio_service.update_portfolio("nonexistent-id", request)


def test_delete_portfolio_success(portfolio_service: PortfolioService) -> None:
    """Test successfully deleting a portfolio."""
    # delete_portfolio calls repository.delete directly
    portfolio_service._repository.delete.return_value = True

    result = portfolio_service.delete_portfolio("test-id")

    assert result is True
    portfolio_service._repository.delete.assert_called_once_with("test-id")


def test_delete_portfolio_not_found(portfolio_service: PortfolioService) -> None:
    """Test deleting non-existent portfolio."""
    # delete_portfolio calls repository.delete directly, not get_portfolio
    portfolio_service._repository.delete.return_value = False

    result = portfolio_service.delete_portfolio("nonexistent-id")

    # Returns False if not found, doesn't raise exception
    assert result is False


def test_add_position_success(portfolio_service: PortfolioService) -> None:
    """Test successfully adding a position."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.id = "test-id"
    portfolio_service._repository.find_by_id.return_value = portfolio
    portfolio_service._repository.save.return_value = portfolio
    portfolio_service._data_service.validate_ticker.return_value = True

    request = AddPositionRequest(ticker="AAPL", shares=100.0, weight_target=0.3)
    result = portfolio_service.add_position("test-id", request)

    assert result == portfolio
    portfolio.add_position.assert_called_once()
    portfolio_service._data_service.validate_ticker.assert_called_once_with("AAPL")
    portfolio_service._repository.save.assert_called_once()


def test_add_position_invalid_ticker(portfolio_service: PortfolioService) -> None:
    """Test adding position with invalid ticker."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio_service._repository.find_by_id.return_value = portfolio
    portfolio_service._data_service.validate_ticker.return_value = False

    request = AddPositionRequest(ticker="INVALID", shares=100.0)

    with pytest.raises(ValidationError, match="Invalid ticker"):
        portfolio_service.add_position("test-id", request)


def test_remove_position_success(portfolio_service: PortfolioService) -> None:
    """Test successfully removing a position."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.id = "test-id"
    portfolio_service._repository.find_by_id.return_value = portfolio
    portfolio_service._repository.save.return_value = portfolio

    result = portfolio_service.remove_position("test-id", "AAPL")

    assert result == portfolio
    portfolio.remove_position.assert_called_once_with("AAPL")
    portfolio_service._repository.save.assert_called_once()


def test_update_position_success(portfolio_service: PortfolioService) -> None:
    """Test successfully updating a position."""
    portfolio = MagicMock(spec=Portfolio)
    portfolio.id = "test-id"
    portfolio_service._repository.find_by_id.return_value = portfolio
    portfolio_service._repository.save.return_value = portfolio

    request = UpdatePositionRequest(shares=150.0, weight_target=0.4)
    result = portfolio_service.update_position("test-id", "AAPL", request)

    assert result == portfolio
    portfolio.update_position.assert_called_once()
    portfolio_service._repository.save.assert_called_once()


def test_clone_portfolio_success(portfolio_service: PortfolioService) -> None:
    """Test successfully cloning a portfolio."""
    source_portfolio = MagicMock(spec=Portfolio)
    source_portfolio.id = "source-id"
    source_portfolio.name = "Original"
    source_portfolio.starting_capital = 100000.0
    source_portfolio.description = "Original description"
    source_portfolio.base_currency = "USD"
    source_portfolio.get_all_positions.return_value = []

    portfolio_service._repository.find_by_id.return_value = source_portfolio
    portfolio_service._repository.find_by_name.return_value = None  # New name available
    portfolio_service._repository.save.return_value = source_portfolio

    result = portfolio_service.clone_portfolio("source-id", "Cloned Portfolio")

    assert result is not None
    portfolio_service._repository.find_by_name.assert_called_once_with(
        "Cloned Portfolio"
    )
    portfolio_service._repository.save.assert_called_once()


def test_clone_portfolio_duplicate_name(portfolio_service: PortfolioService) -> None:
    """Test cloning portfolio with duplicate name."""
    source_portfolio = MagicMock(spec=Portfolio)
    portfolio_service._repository.find_by_id.return_value = source_portfolio
    portfolio_service._repository.find_by_name.return_value = MagicMock()  # Name exists

    with pytest.raises(ConflictError, match="already exists"):
        portfolio_service.clone_portfolio("source-id", "Existing Name")

