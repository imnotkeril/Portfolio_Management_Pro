"""
Portfolio service — orchestrates persistence and validation for portfolios.

Coordinates ``PortfolioRepository``, ``DataService`` (tickers/prices), and
domain ``Portfolio`` logic for create/update/delete and position management.
"""

import logging
from typing import Optional

from config.settings import settings
from core.auth.constants import SYSTEM_USER_ID
from core.data_manager.portfolio import Portfolio
from core.data_manager.portfolio_repository import PortfolioRepository
from core.exceptions import (
    ConflictError,
    DataFetchError,
    PortfolioNotFoundError,
    TickerNotFoundError,
    ValidationError,
)
from services.data_service import DataService
from services.schemas import (
    AddPositionRequest,
    CreatePortfolioRequest,
    UpdatePortfolioRequest,
    UpdatePositionRequest,
)

logger = logging.getLogger(__name__)


def _resolve_user_id(user_id: str | None) -> str:
    """Use explicit user_id, or system user when AUTH_DISABLED (Streamlit dev)."""
    if user_id is not None:
        return user_id
    if settings.auth_disabled:
        return SYSTEM_USER_ID
    raise ValidationError("user_id is required when authentication is enabled")


class PortfolioService:
    """Facade over portfolio CRUD, ticker validation, and position updates."""

    def __init__(
        self,
        repository: Optional[PortfolioRepository] = None,
        data_service: Optional[DataService] = None,
    ) -> None:
        """
        Initialize portfolio service.

        Args:
            repository: Optional portfolio repository
            data_service: Optional data service for price/ticker operations
        """
        self._repository = repository or PortfolioRepository()
        self._data_service = data_service or DataService()

    def create_portfolio(
        self, request: CreatePortfolioRequest, user_id: str | None = None
    ) -> Portfolio:
        """
        Create a new portfolio.

        Args:
            request: Create portfolio request with validation

        Returns:
            Created portfolio

        Raises:
            ConflictError: If portfolio name already exists
            ValidationError: If tickers are invalid
        """
        uid = _resolve_user_id(user_id)

        # Check for duplicate name
        existing = self._repository.find_by_name(request.name, uid)
        if existing:
            raise ConflictError(f"Portfolio name '{request.name}' already exists")

        # Validate tickers (skip CASH - it's a special position)
        tickers = [pos.ticker for pos in request.positions if pos.ticker != "CASH"]
        if tickers:
            validation_results = self._data_service.validate_tickers(tickers)

            invalid_tickers = [
                ticker for ticker, valid in validation_results.items() if not valid
            ]
            if invalid_tickers:
                raise ValidationError(f"Invalid tickers: {', '.join(invalid_tickers)}")

        # Create domain model
        portfolio = Portfolio(
            name=request.name,
            starting_capital=request.starting_capital,
            description=request.description,
            base_currency=request.base_currency,
            rebalance_interval_months=request.rebalance_interval_months,
            ledger_mode=request.ledger_mode,
        )

        # Add positions
        logger.info(f"Adding {len(request.positions)} positions to portfolio")
        for pos_schema in request.positions:
            # If shares not provided, use minimum value
            # (will be calculated later)
            if pos_schema.shares is not None:
                shares = pos_schema.shares
            else:
                shares = 0.01
            logger.debug(
                f"Adding position: {pos_schema.ticker}, "
                f"shares={shares}, weight={pos_schema.weight_target}"
            )
            portfolio.add_position(
                ticker=pos_schema.ticker,
                shares=shares,
                weight_target=pos_schema.weight_target,
                purchase_price=pos_schema.purchase_price,
                purchase_date=pos_schema.purchase_date,
            )

        logger.info(
            f"Portfolio has {len(portfolio.get_all_positions())} positions "
            f"before save"
        )

        # Save to database
        saved_portfolio = self._repository.save(portfolio, uid)

        logger.info(
            f"Portfolio saved with {len(saved_portfolio.get_all_positions())} "
            f"positions after save"
        )

        logger.info(
            f"Created portfolio: {saved_portfolio.id} " f"({saved_portfolio.name})"
        )

        return saved_portfolio

    def get_portfolio(self, portfolio_id: str, user_id: str | None = None) -> Portfolio:
        """
        Get portfolio by ID.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Portfolio

        Raises:
            PortfolioNotFoundError: If portfolio not found
        """
        uid = _resolve_user_id(user_id)
        portfolio = self._repository.find_by_id(portfolio_id, uid)
        if not portfolio:
            raise PortfolioNotFoundError(f"Portfolio not found: {portfolio_id}")
        return portfolio

    def list_portfolios(
        self, user_id: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Portfolio]:
        """
        List all portfolios with pagination.

        Args:
            limit: Maximum number of portfolios
            offset: Number of portfolios to skip

        Returns:
            List of portfolios
        """
        uid = _resolve_user_id(user_id)
        return self._repository.find_all(uid, limit=limit, offset=offset)

    def update_portfolio(
        self,
        portfolio_id: str,
        request: UpdatePortfolioRequest,
        user_id: str | None = None,
    ) -> Portfolio:
        """
        Update portfolio attributes.

        Args:
            portfolio_id: Portfolio ID
            request: Update request with optional fields

        Returns:
            Updated portfolio

        Raises:
            PortfolioNotFoundError: If portfolio not found
            ConflictError: If new name conflicts with existing portfolio
        """
        uid = _resolve_user_id(user_id)
        portfolio = self.get_portfolio(portfolio_id, uid)

        # Check name conflict if name is being updated
        if request.name and request.name != portfolio.name:
            existing = self._repository.find_by_name(request.name, uid)
            if existing and existing.id != portfolio_id:
                raise ConflictError(f"Portfolio name '{request.name}' already exists")

        # Update attributes
        if request.name is not None:
            portfolio.name = request.name
        if request.description is not None:
            portfolio.description = request.description
        if request.starting_capital is not None:
            portfolio.starting_capital = request.starting_capital
        if request.base_currency is not None:
            portfolio.base_currency = request.base_currency
        if "rebalance_interval_months" in request.model_fields_set:
            portfolio.rebalance_interval_months = request.rebalance_interval_months

        # Save changes
        updated = self._repository.save(portfolio, uid)

        logger.info(f"Updated portfolio: {portfolio_id}")
        return updated

    def delete_portfolio(self, portfolio_id: str, user_id: str | None = None) -> bool:
        """
        Delete portfolio.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            True if deleted, False if not found
        """
        uid = _resolve_user_id(user_id)
        result = self._repository.delete(portfolio_id, uid)
        if result:
            logger.info(f"Deleted portfolio: {portfolio_id}")
        return result

    def add_position(
        self,
        portfolio_id: str,
        request: AddPositionRequest,
        user_id: str | None = None,
    ) -> Portfolio:
        """
        Add position to portfolio.

        Args:
            portfolio_id: Portfolio ID
            request: Add position request

        Returns:
            Updated portfolio

        Raises:
            PortfolioNotFoundError: If portfolio not found
            ValidationError: If ticker is invalid or already exists
        """
        uid = _resolve_user_id(user_id)
        portfolio = self.get_portfolio(portfolio_id, uid)

        # Validate ticker
        if not self._data_service.validate_ticker(request.ticker):
            raise ValidationError(f"Invalid ticker: {request.ticker}")

        # Add position
        portfolio.add_position(
            ticker=request.ticker,
            shares=request.shares,
            weight_target=request.weight_target,
            purchase_price=request.purchase_price,
            purchase_date=request.purchase_date,
        )

        # Save changes
        updated = self._repository.save(portfolio, uid)

        logger.info(f"Added position {request.ticker} to portfolio {portfolio_id}")
        return updated

    def remove_position(
        self, portfolio_id: str, ticker: str, user_id: str | None = None
    ) -> Portfolio:
        """
        Remove position from portfolio.

        Args:
            portfolio_id: Portfolio ID
            ticker: Ticker symbol to remove

        Returns:
            Updated portfolio

        Raises:
            PortfolioNotFoundError: If portfolio not found
            ValidationError: If position not found
        """
        uid = _resolve_user_id(user_id)
        portfolio = self.get_portfolio(portfolio_id, uid)

        portfolio.remove_position(ticker)

        # Save changes
        updated = self._repository.save(portfolio, uid)

        logger.info(f"Removed position {ticker} from portfolio {portfolio_id}")
        return updated

    def update_position(
        self,
        portfolio_id: str,
        ticker: str,
        request: UpdatePositionRequest,
        user_id: str | None = None,
    ) -> Portfolio:
        """
        Update position in portfolio.

        Args:
            portfolio_id: Portfolio ID
            ticker: Ticker symbol
            request: Update position request

        Returns:
            Updated portfolio

        Raises:
            PortfolioNotFoundError: If portfolio not found
            ValidationError: If position not found
        """
        uid = _resolve_user_id(user_id)
        portfolio = self.get_portfolio(portfolio_id, uid)

        portfolio.update_position(
            ticker=ticker,
            shares=request.shares,
            weight_target=request.weight_target,
            purchase_price=request.purchase_price,
            purchase_date=request.purchase_date,
        )

        # Save changes
        updated = self._repository.save(portfolio, uid)

        logger.info(f"Updated position {ticker} in portfolio {portfolio_id}")
        return updated

    def clone_portfolio(
        self, portfolio_id: str, new_name: str, user_id: str | None = None
    ) -> Portfolio:
        """
        Clone portfolio with new name.

        Args:
            portfolio_id: Source portfolio ID
            new_name: Name for cloned portfolio

        Returns:
            Cloned portfolio

        Raises:
            PortfolioNotFoundError: If source portfolio not found
            ConflictError: If new name already exists
        """
        # Get source portfolio
        uid = _resolve_user_id(user_id)
        source_portfolio = self.get_portfolio(portfolio_id, uid)

        # Check name conflict
        existing = self._repository.find_by_name(new_name, uid)
        if existing:
            raise ConflictError(f"Portfolio name '{new_name}' already exists")

        # Create new portfolio with same attributes
        cloned = Portfolio(
            name=new_name,
            starting_capital=source_portfolio.starting_capital,
            description=source_portfolio.description,
            base_currency=source_portfolio.base_currency,
        )

        # Copy positions
        for position in source_portfolio.get_all_positions():
            cloned.add_position(
                ticker=position.ticker,
                shares=position.shares,
                weight_target=position.weight_target,
                purchase_price=position.purchase_price,
                purchase_date=position.purchase_date,
            )

        # Save cloned portfolio
        saved = self._repository.save(cloned, uid)

        logger.info(f"Cloned portfolio {portfolio_id} as {saved.id} ({new_name})")

        return saved

    def calculate_portfolio_metrics(
        self, portfolio_id: str, user_id: str | None = None
    ) -> dict[str, float]:
        """
        Calculate current portfolio value and weights.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Dictionary with current_value and weights

        Raises:
            PortfolioNotFoundError: If portfolio not found
            DataFetchError: If prices cannot be fetched
        """
        portfolio = self.get_portfolio(portfolio_id, user_id)

        # Get current prices for all tickers
        tickers = [pos.ticker for pos in portfolio.get_all_positions()]
        prices: dict[str, float] = {}

        for ticker in tickers:
            try:
                price = self._data_service.fetch_current_price(ticker)
                prices[ticker] = price
            except (DataFetchError, TickerNotFoundError) as e:
                logger.warning(
                    f"Failed to fetch price for {ticker}: {e.message if hasattr(e, 'message') else str(e)}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error fetching price for {ticker}: {e}", exc_info=True
                )
                # Continue with other tickers
                continue

        # Calculate metrics
        current_value = portfolio.calculate_current_value(prices)
        weights = portfolio.calculate_current_weights(prices)

        return {
            "current_value": current_value,
            "weights": weights,
            "positions_count": len(portfolio.get_all_positions()),
        }
