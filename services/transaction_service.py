"""Service for orchestrating transaction operations."""

import logging
from datetime import date
from typing import List, Optional

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from core.exceptions import ValidationError
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class TransactionService:
    """Service for orchestrating transaction operations."""

    def __init__(
        self,
        repository: Optional[TransactionRepository] = None,
        portfolio_service: Optional[PortfolioService] = None,
    ) -> None:
        """
        Initialize transaction service.

        Args:
            repository: Optional transaction repository
            portfolio_service: Optional portfolio service
        """
        self._repository = repository or TransactionRepository()
        self._portfolio_service = portfolio_service or PortfolioService()

    def add_transaction(
        self,
        portfolio_id: str,
        transaction_date: date,
        transaction_type: str,
        ticker: str,
        shares: float,
        price: float,
        fees: float = 0.0,
        notes: Optional[str] = None,
    ) -> Transaction:
        """
        Add transaction to portfolio.

        Args:
            portfolio_id: Portfolio ID
            transaction_date: Transaction date
            transaction_type: BUY, SELL, DEPOSIT, WITHDRAWAL
            ticker: Ticker symbol (or 'CASH' for DEPOSIT/WITHDRAWAL)
            shares: Number of shares (or amount for CASH)
            price: Price per share (or 1.0 for CASH)
            fees: Transaction fees
            notes: Optional notes

        Returns:
            Created transaction

        Raises:
            PortfolioNotFoundError: If portfolio doesn't exist
            ValidationError: If transaction data is invalid
        """
        # Validate portfolio exists
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Create transaction
        transaction = Transaction(
            transaction_date=transaction_date,
            transaction_type=transaction_type,
            ticker=ticker,
            shares=shares,
            price=price,
            fees=fees,
            notes=notes,
        )

        # Save to database
        saved_transaction = self._repository.save(transaction, portfolio_id)

        logger.info(
            f"Added transaction {saved_transaction.id} to portfolio {portfolio_id}"
        )

        return saved_transaction

    def get_transactions(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Transaction]:
        """
        Get all transactions for a portfolio.

        Args:
            portfolio_id: Portfolio ID
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of transactions
        """
        return self._repository.find_by_portfolio(
            portfolio_id, start_date, end_date
        )

    def delete_transaction(self, transaction_id: str) -> bool:
        """
        Delete transaction by ID.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if deleted, False if not found
        """
        return self._repository.delete(transaction_id)

