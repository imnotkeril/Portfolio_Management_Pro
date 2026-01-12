"""Repository for transaction persistence."""

import logging
from datetime import date
from typing import List, Optional

from sqlalchemy.orm import Session

from core.data_manager.transaction import Transaction
from core.exceptions import ValidationError
from database.session import get_db_session
from models.transaction import Transaction as TransactionORM

logger = logging.getLogger(__name__)


class TransactionRepository:
    """Repository for transaction persistence operations."""

    def save(
        self, transaction: Transaction, portfolio_id: str
    ) -> Transaction:
        """
        Save transaction to database.

        Args:
            transaction: Transaction domain model
            portfolio_id: Portfolio ID

        Returns:
            Saved transaction with ID
        """
        with get_db_session() as session:
            if transaction.id:
                return self._update_transaction(
                    session, transaction, portfolio_id
                )
            else:
                return self._create_transaction(
                    session, transaction, portfolio_id
                )

    def find_by_portfolio(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Transaction]:
        """
        Find all transactions for a portfolio, optionally filtered by date range.

        Args:
            portfolio_id: Portfolio ID
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of Transaction domain models
        """
        with get_db_session() as session:
            query = session.query(TransactionORM).filter(
                TransactionORM.portfolio_id == portfolio_id
            )

            if start_date:
                query = query.filter(
                    TransactionORM.transaction_date >= start_date
                )
            if end_date:
                query = query.filter(
                    TransactionORM.transaction_date <= end_date
                )

            query = query.order_by(TransactionORM.transaction_date)

            transactions_orm = query.all()
            return [self._orm_to_domain(txn) for txn in transactions_orm]

    def delete(self, transaction_id: str) -> bool:
        """
        Delete transaction by ID.

        Args:
            transaction_id: Transaction ID

        Returns:
            True if deleted, False if not found
        """
        with get_db_session() as session:
            transaction_orm = (
                session.query(TransactionORM)
                .filter(TransactionORM.id == transaction_id)
                .first()
            )

            if not transaction_orm:
                return False

            session.delete(transaction_orm)
            return True

    def _create_transaction(
        self, session: Session, transaction: Transaction, portfolio_id: str
    ) -> Transaction:
        """Create new transaction in database."""
        transaction_orm = TransactionORM(
            portfolio_id=portfolio_id,
            transaction_date=transaction.transaction_date,
            transaction_type=transaction.transaction_type,
            ticker=transaction.ticker,
            shares=transaction.shares,
            price=transaction.price,
            amount=transaction.amount,
            fees=transaction.fees or 0.0,
            notes=transaction.notes,
        )

        session.add(transaction_orm)
        session.flush()

        logger.info(f"Created transaction: {transaction_orm.id}")
        return self._orm_to_domain(transaction_orm)

    def _update_transaction(
        self, session: Session, transaction: Transaction, portfolio_id: str
    ) -> Transaction:
        """Update existing transaction in database."""
        transaction_orm = (
            session.query(TransactionORM)
            .filter(TransactionORM.id == transaction.id)
            .first()
        )

        if not transaction_orm:
            raise ValidationError(
                f"Transaction not found: {transaction.id}"
            )

        # Update fields
        transaction_orm.transaction_date = transaction.transaction_date
        transaction_orm.transaction_type = transaction.transaction_type
        transaction_orm.ticker = transaction.ticker
        transaction_orm.shares = transaction.shares
        transaction_orm.price = transaction.price
        transaction_orm.amount = transaction.amount
        transaction_orm.fees = transaction.fees or 0.0
        transaction_orm.notes = transaction.notes

        session.flush()
        return self._orm_to_domain(transaction_orm)

    def _orm_to_domain(
        self, transaction_orm: TransactionORM
    ) -> Transaction:
        """Convert ORM model to domain model."""
        return Transaction(
            transaction_date=transaction_orm.transaction_date,
            transaction_type=transaction_orm.transaction_type,
            ticker=transaction_orm.ticker,
            shares=transaction_orm.shares,
            price=transaction_orm.price,
            amount=transaction_orm.amount,
            fees=transaction_orm.fees or 0.0,
            notes=transaction_orm.notes,
            transaction_id=transaction_orm.id,
        )

