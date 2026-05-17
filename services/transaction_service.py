"""Service for orchestrating transaction operations."""

import logging
from datetime import date
from typing import Optional

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from core.exceptions import PortfolioNotFoundError
from services.dividend_processor import DividendProcessor
from services.holdings import HoldingRow, HoldingsBuilder
from services.performance_attribution import (
    PerformanceAttributionService,
    PerformanceSummary,
)
from services.portfolio_service import PortfolioService
from services.position_sync import sync_positions_from_transactions

logger = logging.getLogger(__name__)


class TransactionService:
    """Service for orchestrating transaction operations."""

    def __init__(
        self,
        repository: Optional[TransactionRepository] = None,
        portfolio_service: Optional[PortfolioService] = None,
        holdings_builder: Optional[HoldingsBuilder] = None,
        performance_service: Optional[PerformanceAttributionService] = None,
        dividend_processor: Optional[DividendProcessor] = None,
    ) -> None:
        self._repository = repository or TransactionRepository()
        self._portfolio_service = portfolio_service or PortfolioService()
        self._holdings_builder = holdings_builder or HoldingsBuilder()
        self._performance = performance_service or PerformanceAttributionService()
        self._dividend_processor = dividend_processor or DividendProcessor()

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
        user_id: str | None = None,
        reinvest: Optional[bool] = None,
        split_ratio: Optional[float] = None,
        currency: str = "USD",
    ) -> Transaction:
        """Add transaction to portfolio and sync positions cache."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)

        transaction = Transaction(
            transaction_date=transaction_date,
            transaction_type=transaction_type,
            ticker=ticker,
            shares=shares,
            price=price,
            fees=fees,
            notes=notes,
            reinvest=reinvest,
            split_ratio=split_ratio,
            currency=currency,
        )

        saved = self._repository.save(transaction, portfolio_id)
        self._sync_positions(portfolio_id, user_id, portfolio.cost_basis_method)
        logger.info(
            "Added transaction %s to portfolio %s",
            saved.id,
            portfolio_id,
        )
        return saved

    def get_transactions(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        transaction_type: Optional[str] = None,
        ticker: Optional[str] = None,
        user_id: str | None = None,
    ) -> list[Transaction]:
        """Get transactions with optional filters."""
        self._portfolio_service.get_portfolio(portfolio_id, user_id)
        return self._repository.find_by_portfolio(
            portfolio_id,
            start_date=start_date,
            end_date=end_date,
            transaction_type=transaction_type,
            ticker=ticker,
        )

    def get_dividends(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        user_id: str | None = None,
    ) -> list[Transaction]:
        """Return DIVIDEND transactions only."""
        return self.get_transactions(
            portfolio_id,
            start_date=start_date,
            end_date=end_date,
            transaction_type="DIVIDEND",
            user_id=user_id,
        )

    def delete_transaction(
        self,
        transaction_id: str,
        user_id: str | None = None,
        portfolio_id: str | None = None,
    ) -> bool:
        """Delete transaction and resync positions."""
        pid = portfolio_id or self._repository.find_portfolio_id(transaction_id)
        if not pid:
            return False
        actual_pid = self._repository.find_portfolio_id(transaction_id)
        if portfolio_id and actual_pid and actual_pid != portfolio_id:
            raise PortfolioNotFoundError(
                f"Transaction {transaction_id} not in portfolio {portfolio_id}"
            )
        portfolio = self._portfolio_service.get_portfolio(pid, user_id)
        deleted = self._repository.delete(transaction_id)
        if deleted:
            self._sync_positions(pid, user_id, portfolio.cost_basis_method)
        return deleted

    def get_holdings(
        self, portfolio_id: str, user_id: str | None = None
    ) -> list[HoldingRow]:
        """Current holdings from transaction ledger."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._repository.find_by_portfolio(portfolio_id)
        return self._holdings_builder.build(txs, portfolio.cost_basis_method)

    def get_pnl(
        self, portfolio_id: str, user_id: str | None = None
    ) -> PerformanceSummary:
        """Realized/unrealized PnL and TWR/MWR from ledger."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._repository.find_by_portfolio(portfolio_id)
        return self._performance.summarize(
            txs,
            portfolio.starting_capital,
            portfolio.cost_basis_method,
        )

    def sync_dividends(
        self,
        portfolio_id: str,
        tickers: list[str],
        start_date: date,
        end_date: date,
        user_id: str | None = None,
        reinvest: bool = False,
    ) -> list[Transaction]:
        """Import dividends from market data (idempotent)."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        created = self._dividend_processor.sync_dividends(
            portfolio_id,
            tickers,
            start_date,
            end_date,
            user_id,
            reinvest,
        )
        if created:
            self._sync_positions(portfolio_id, user_id, portfolio.cost_basis_method)
        return created

    def _sync_positions(
        self, portfolio_id: str, user_id: str | None, _cost_basis_method: str
    ) -> None:
        from services.portfolio_service import _resolve_user_id

        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._repository.find_by_portfolio(portfolio_id)
        sync_positions_from_transactions(
            portfolio,
            txs,
            user_id=_resolve_user_id(user_id),
        )
