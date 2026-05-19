"""Repository for transaction persistence."""

import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from core.data_manager.transaction import Transaction
from core.exceptions import ValidationError
from database.session import get_db_session
from models.transaction import Transaction as TransactionORM

logger = logging.getLogger(__name__)


class TransactionRepository:
    """Repository for transaction persistence operations."""

    def save(self, transaction: Transaction, portfolio_id: str) -> Transaction:
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

                return self._update_transaction(session, transaction, portfolio_id)

            return self._create_transaction(session, transaction, portfolio_id)

    def find_by_portfolio(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_type: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> list[Transaction]:
        """

        Find all transactions for a portfolio, optionally filtered.



        Args:

            portfolio_id: Portfolio ID

            start_date: Optional start date filter

            end_date: Optional end date filter

            transaction_type: Optional type filter (BUY, SELL, ...)

            ticker: Optional ticker filter



        Returns:

            List of Transaction domain models

        """

        with get_db_session() as session:

            query = session.query(TransactionORM).filter(
                TransactionORM.portfolio_id == portfolio_id
            )

            if start_date:

                query = query.filter(TransactionORM.transaction_date >= start_date)

            if end_date:

                query = query.filter(TransactionORM.transaction_date <= end_date)

            if transaction_type:

                query = query.filter(
                    TransactionORM.transaction_type == transaction_type.upper()
                )

            if ticker:

                query = query.filter(TransactionORM.ticker == ticker.strip().upper())

            query = query.order_by(
                TransactionORM.transaction_date,
                TransactionORM.transaction_type,
                TransactionORM.ticker,
                TransactionORM.created_at,
            )

            transactions_orm = query.all()

            return [self._orm_to_domain(txn) for txn in transactions_orm]

    def find_portfolio_id(self, transaction_id: str) -> Optional[str]:
        """Return portfolio_id for a transaction, or None if not found."""

        with get_db_session() as session:

            row = (
                session.query(TransactionORM.portfolio_id)
                .filter(TransactionORM.id == transaction_id)
                .first()
            )

            return row[0] if row else None

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
            reinvest=transaction.reinvest,
            split_ratio=transaction.split_ratio,
            currency=transaction.currency,
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

            raise ValidationError(f"Transaction not found: {transaction.id}")

        transaction_orm.transaction_date = transaction.transaction_date

        transaction_orm.transaction_type = transaction.transaction_type

        transaction_orm.ticker = transaction.ticker

        transaction_orm.shares = transaction.shares

        transaction_orm.price = transaction.price

        transaction_orm.amount = transaction.amount

        transaction_orm.fees = transaction.fees or 0.0

        transaction_orm.notes = transaction.notes

        transaction_orm.reinvest = transaction.reinvest

        transaction_orm.split_ratio = transaction.split_ratio

        transaction_orm.currency = transaction.currency

        session.flush()

        return self._orm_to_domain(transaction_orm)

    def _orm_to_domain(self, transaction_orm: TransactionORM) -> Transaction:
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
            reinvest=transaction_orm.reinvest,
            split_ratio=transaction_orm.split_ratio,
            currency=getattr(transaction_orm, "currency", None) or "USD",
        )
