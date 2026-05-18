"""Sync stock split transactions from market data."""

from __future__ import annotations

import logging
from datetime import date

import yfinance as yf

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class SplitProcessor:
    """Create SPLIT rows from yfinance split history (idempotent)."""

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
        """Fetch splits and insert SPLIT transactions (skip duplicates)."""
        self._portfolio_service.get_portfolio(portfolio_id, user_id)
        existing = self._repository.find_by_portfolio(
            portfolio_id,
            start_date=start_date,
            end_date=end_date,
            transaction_type="SPLIT",
        )
        existing_keys = {
            (tx.transaction_date, tx.ticker, round(tx.split_ratio or 0.0, 6))
            for tx in existing
        }

        created: list[Transaction] = []
        for ticker in tickers:
            sym = ticker.strip().upper()
            if sym == "CASH":
                continue
            try:
                splits = yf.Ticker(sym).splits
            except Exception as exc:
                logger.warning("Split fetch failed for %s: %s", sym, exc)
                continue

            if splits is None or splits.empty:
                continue

            for idx, ratio in splits.items():
                split_date = idx.date() if hasattr(idx, "date") else idx
                if split_date < start_date or split_date > end_date:
                    continue
                ratio_f = float(ratio)
                if ratio_f <= 0:
                    continue
                key = (split_date, sym, round(ratio_f, 6))
                if key in existing_keys:
                    continue
                txn = Transaction(
                    transaction_date=split_date,
                    transaction_type="SPLIT",
                    ticker=sym,
                    shares=1.0,
                    price=1.0,
                    split_ratio=ratio_f,
                    notes="Auto-synced split",
                )
                saved = self._repository.save(txn, portfolio_id)
                created.append(saved)
                existing_keys.add(key)

        return created
