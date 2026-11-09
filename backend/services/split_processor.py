"""Stock split sync (disabled: see SplitProcessor.sync_splits)."""

from __future__ import annotations

import logging
from datetime import date

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class SplitProcessor:
    """Former SPLIT-row sync from yfinance; now a deliberate no-op."""

    def __init__(
        self,
        transaction_repository: TransactionRepository | None = None,
        portfolio_service: PortfolioService | None = None,
    ) -> None:
        self._repository = transaction_repository or TransactionRepository()
        self._portfolio_service = portfolio_service or PortfolioService()

    def sync_splits(
        self,
        portfolio_id: str,
        tickers: list[str],
        start_date: date,
        end_date: date,
        user_id: str | None = None,
    ) -> list[Transaction]:
        """Auto-sync of SPLIT rows is disabled; always returns no transactions.

        Market prices are split-adjusted back in time and every share count the app
        records (initial positions, rebalances, price autofill) is on that adjusted
        basis, so an extra SPLIT row multiplies holdings a second time. That made a
        10:1 split inflate one position tenfold and trigger rebalances many times
        the portfolio size. Existing auto-synced rows are ignored by the ledger.
        """
        return []
