"""Integration tests for portfolio service."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from core.data_manager.cache import Cache
from core.exceptions import ConflictError, PortfolioNotFoundError
from services.data_service import DataService
from services.portfolio_service import PortfolioService
from services.schemas import (
    AddPositionRequest,
    CreatePortfolioRequest,
    PositionSchema,
    UpdatePortfolioRequest,
)


@pytest.fixture
def mock_data_service() -> DataService:
    """Create mock data service."""
    service = DataService()
    # Mock ticker validation
    with patch.object(service._ticker_validator, "validate_ticker") as mock_val:
        mock_val.return_value = True
        yield service


@pytest.fixture
def portfolio_service(mock_data_service: DataService) -> PortfolioService:
    """Create portfolio service with mocked data service."""
    return PortfolioService(data_service=mock_data_service)


def test_create_portfolio(portfolio_service: PortfolioService) -> None:
    """Test creating a portfolio."""
    request = CreatePortfolioRequest(
        name="Test Portfolio",
        description="Test description",
        starting_capital=100000.0,
        positions=[
            PositionSchema(ticker="AAPL", shares=100.0, weight_target=0.5),
            PositionSchema(ticker="MSFT", shares=50.0, weight_target=0.5),
        ],
    )

    with patch.object(
        portfolio_service._data_service, "validate_tickers"
    ) as mock_validate:
        mock_validate.return_value = {"AAPL": True, "MSFT": True}

        portfolio = portfolio_service.create_portfolio(request)

        assert portfolio.name == "Test Portfolio"
        assert portfolio.starting_capital == 100000.0
        assert len(portfolio.get_all_positions()) == 2


def test_create_portfolio_duplicate_name(
    portfolio_service: PortfolioService,
) -> None:
    """Test creating portfolio with duplicate name."""
    request = CreatePortfolioRequest(
        name="Existing Portfolio",
        starting_capital=100000.0,
        positions=[PositionSchema(ticker="AAPL", shares=100.0)],
    )

    # Mock repository to return existing portfolio
    with patch.object(
        portfolio_service._repository, "find_by_name"
    ) as mock_find:
        mock_portfolio = MagicMock()
        mock_portfolio.id = "existing-id"
        mock_find.return_value = mock_portfolio

        with pytest.raises(ConflictError, match="already exists"):
            portfolio_service.create_portfolio(request)


def test_create_portfolio_invalid_ticker(
    portfolio_service: PortfolioService,
) -> None:
    """Test creating portfolio with invalid ticker."""
    request = CreatePortfolioRequest(
        name="Test Portfolio",
        starting_capital=100000.0,
        positions=[PositionSchema(ticker="INVALID", shares=100.0)],
    )

    with patch.object(
        portfolio_service._data_service, "validate_tickers"
    ) as mock_validate:
        mock_validate.return_value = {"INVALID": False}

        with pytest.raises(Exception):  # Should raise ValidationError
            portfolio_service.create_portfolio(request)


def test_get_portfolio_not_found(
    portfolio_service: PortfolioService,
) -> None:
    """Test getting non-existent portfolio."""
    with patch.object(
        portfolio_service._repository, "find_by_id"
    ) as mock_find:
        mock_find.return_value = None

        with pytest.raises(PortfolioNotFoundError):
            portfolio_service.get_portfolio("nonexistent-id")


def test_update_portfolio(portfolio_service: PortfolioService) -> None:
    """Test updating portfolio."""
    # Create portfolio first
    portfolio = MagicMock()
    portfolio.id = "test-id"
    portfolio.name = "Original Name"

    with patch.object(
        portfolio_service._repository, "find_by_id"
    ) as mock_find, patch.object(
        portfolio_service._repository, "save"
    ) as mock_save:
        mock_find.return_value = portfolio
        mock_save.return_value = portfolio

        request = UpdatePortfolioRequest(name="Updated Name")
        result = portfolio_service.update_portfolio("test-id", request)

        assert result is not None


def test_add_position(portfolio_service: PortfolioService) -> None:
    """Test adding position to portfolio."""
    portfolio = MagicMock()
    portfolio.id = "test-id"

    with patch.object(
        portfolio_service._repository, "find_by_id"
    ) as mock_find, patch.object(
        portfolio_service._data_service, "validate_ticker"
    ) as mock_validate, patch.object(
        portfolio_service._repository, "save"
    ) as mock_save:
        mock_find.return_value = portfolio
        mock_validate.return_value = True
        mock_save.return_value = portfolio

        request = AddPositionRequest(ticker="AAPL", shares=100.0)
        result = portfolio_service.add_position("test-id", request)

        portfolio.add_position.assert_called_once()
        assert result is not None


def test_remove_position(portfolio_service: PortfolioService) -> None:
    """Test removing position from portfolio."""
    portfolio = MagicMock()
    portfolio.id = "test-id"

    with patch.object(
        portfolio_service._repository, "find_by_id"
    ) as mock_find, patch.object(
        portfolio_service._repository, "save"
    ) as mock_save:
        mock_find.return_value = portfolio
        mock_save.return_value = portfolio

        result = portfolio_service.remove_position("test-id", "AAPL")

        portfolio.remove_position.assert_called_once_with("AAPL")
        assert result is not None


def test_clone_portfolio(portfolio_service: PortfolioService) -> None:
    """Test cloning portfolio."""
    source_portfolio = MagicMock()
    source_portfolio.id = "source-id"
    source_portfolio.name = "Original"
    source_portfolio.starting_capital = 100000.0
    source_portfolio.description = "Original description"
    source_portfolio.base_currency = "USD"
    source_portfolio.get_all_positions.return_value = []

    with patch.object(
        portfolio_service._repository, "find_by_name"
    ) as mock_find_name, patch.object(
        portfolio_service._repository, "save"
    ) as mock_save:
        mock_find_name.return_value = None  # New name doesn't exist
        mock_save.return_value = source_portfolio

        # Mock get_portfolio to return source
        with patch.object(
            portfolio_service, "get_portfolio"
        ) as mock_get:
            mock_get.return_value = source_portfolio

            result = portfolio_service.clone_portfolio(
                "source-id", "Cloned Portfolio"
            )

            assert result is not None
            mock_save.assert_called_once()

