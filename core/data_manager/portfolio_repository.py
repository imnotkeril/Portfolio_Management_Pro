"""Repository for portfolio persistence."""

import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from core.data_manager.portfolio import Portfolio
from core.exceptions import PortfolioNotFoundError
from database.session import get_db_session
from models.portfolio import Portfolio as PortfolioORM
from models.position import Position as PositionORM

logger = logging.getLogger(__name__)


class PortfolioRepository:
    """Repository for portfolio persistence operations."""

    def save(self, portfolio: Portfolio) -> Portfolio:
        """
        Save portfolio to database (create or update).

        Args:
            portfolio: Domain portfolio object

        Returns:
            Saved portfolio with ID set

        Raises:
            ValidationError: If validation fails
        """
        with get_db_session() as session:
            if portfolio.id:
                # Update existing portfolio
                return self._update_portfolio(session, portfolio)
            else:
                # Create new portfolio
                return self._create_portfolio(session, portfolio)

    def _create_portfolio(
        self, session: Session, portfolio: Portfolio
    ) -> Portfolio:
        """Create new portfolio in database."""
        # Convert domain model to ORM
        portfolio_orm = PortfolioORM(
            name=portfolio.name,
            description=portfolio.description,
            starting_capital=portfolio.starting_capital,
            base_currency=portfolio.base_currency,
        )

        session.add(portfolio_orm)
        session.flush()  # Get ID before commit

        # Convert positions
        positions_count = 0
        for pos in portfolio.get_all_positions():
            position_orm = PositionORM(
                portfolio_id=portfolio_orm.id,
                ticker=pos.ticker,
                shares=pos.shares,
                weight_target=pos.weight_target,
                purchase_price=pos.purchase_price,
                purchase_date=pos.purchase_date,
            )
            session.add(position_orm)
            positions_count += 1

        session.flush()  # Flush positions to DB
        session.refresh(portfolio_orm)  # Refresh to load relationships

        logger.info(
            f"Created portfolio: {portfolio_orm.id} ({portfolio.name}) "
            f"with {positions_count} positions (DB has {len(portfolio_orm.positions)})"
        )

        # Convert back to domain model (commit happens automatically in get_db_session)
        return self._orm_to_domain(portfolio_orm)

    def _update_portfolio(
        self, session: Session, portfolio: Portfolio
    ) -> Portfolio:
        """Update existing portfolio in database."""
        portfolio_orm = (
            session.query(PortfolioORM)
            .filter(
                PortfolioORM.id == portfolio.id
            )
            .first()
        )

        if not portfolio_orm:
            raise PortfolioNotFoundError(
                f"Portfolio not found: {portfolio.id}"
            )

        # Update attributes
        portfolio_orm.name = portfolio.name
        portfolio_orm.description = portfolio.description
        portfolio_orm.starting_capital = portfolio.starting_capital
        portfolio_orm.base_currency = portfolio.base_currency

        # Update positions: delete existing and add new
        session.query(PositionORM).filter(
            PositionORM.portfolio_id == portfolio.id
        ).delete()

        for pos in portfolio.get_all_positions():
            position_orm = PositionORM(
                portfolio_id=portfolio_orm.id,
                ticker=pos.ticker,
                shares=pos.shares,
                weight_target=pos.weight_target,
                purchase_price=pos.purchase_price,
                purchase_date=pos.purchase_date,
            )
            session.add(position_orm)

        logger.info(f"Updated portfolio: {portfolio.id} ({portfolio.name})")

        # Refresh and convert back to domain model
        session.refresh(portfolio_orm)
        return self._orm_to_domain(portfolio_orm)

    def find_by_id(self, portfolio_id: str) -> Optional[Portfolio]:
        """
        Find portfolio by ID.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            Portfolio domain object or None if not found
        """
        with get_db_session() as session:
            portfolio_orm = (
                session.query(PortfolioORM)
                .filter(PortfolioORM.id == portfolio_id)
                .first()
            )

            if not portfolio_orm:
                return None

            return self._orm_to_domain(portfolio_orm)

    def find_by_name(self, name: str) -> Optional[Portfolio]:
        """
        Find portfolio by name.

        Args:
            name: Portfolio name

        Returns:
            Portfolio domain object or None if not found
        """
        with get_db_session() as session:
            portfolio_orm = (
                session.query(PortfolioORM)
                .filter(PortfolioORM.name == name)
                .first()
            )

            if not portfolio_orm:
                return None

            return self._orm_to_domain(portfolio_orm)

    def find_all(
        self, limit: int = 100, offset: int = 0
    ) -> List[Portfolio]:
        """
        Find all portfolios with pagination.

        Args:
            limit: Maximum number of portfolios to return
            offset: Number of portfolios to skip

        Returns:
            List of Portfolio domain objects
        """
        with get_db_session() as session:
            portfolios_orm = (
                session.query(PortfolioORM)
                .limit(limit)
                .offset(offset)
                .all()
            )

            return [self._orm_to_domain(p) for p in portfolios_orm]

    def delete(self, portfolio_id: str) -> bool:
        """
        Delete portfolio by ID.

        Args:
            portfolio_id: Portfolio ID

        Returns:
            True if deleted, False if not found
        """
        with get_db_session() as session:
            # Use query with explicit options to avoid loading relationships
            from sqlalchemy.orm import noload
            
            portfolio_orm = (
                session.query(PortfolioORM)
                .options(
                    noload(PortfolioORM.positions),
                    noload(PortfolioORM.transactions),  # Don't load transactions
                )
                .filter(PortfolioORM.id == portfolio_id)
                .first()
            )

            if not portfolio_orm:
                return False

            session.delete(portfolio_orm)
            logger.info(f"Deleted portfolio: {portfolio_id}")
            return True

    def _orm_to_domain(self, portfolio_orm: PortfolioORM) -> Portfolio:
        """
        Convert ORM model to domain model.

        Args:
            portfolio_orm: ORM portfolio object

        Returns:
            Domain portfolio object
        """
        portfolio = Portfolio(
            name=portfolio_orm.name,
            starting_capital=portfolio_orm.starting_capital,
            description=portfolio_orm.description,
            base_currency=portfolio_orm.base_currency,
            portfolio_id=portfolio_orm.id,
        )

        # Add positions
        for pos_orm in portfolio_orm.positions:
            portfolio.add_position(
                ticker=pos_orm.ticker,
                shares=pos_orm.shares,
                weight_target=pos_orm.weight_target,
                purchase_price=pos_orm.purchase_price,
                purchase_date=pos_orm.purchase_date,
            )

        return portfolio

