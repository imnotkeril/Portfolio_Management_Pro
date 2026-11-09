"""Sync dividend transactions from market data."""

from __future__ import annotations

import logging
from datetime import date

import yfinance as yf

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_repository import TransactionRepository
from services.cost_basis import CostBasisCalculator
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class DividendProcessor:
    """Create DIVIDEND rows from yfinance dividend history (idempotent)."""

    def __init__(
        self,
        transaction_repository: TransactionRepository | None = None,
        portfolio_service: PortfolioService | None = None,
    ) -> None:
        self._repository = transaction_repository or TransactionRepository()
        self._portfolio_service = portfolio_service or PortfolioService()

    def sync_dividends(
        self,
        portfolio_id: str,
        tickers: list[str],
        start_date: date,
        end_date: date,
        user_id: str | None = None,
        reinvest: bool = False,
    ) -> list[Transaction]:
        """Fetch dividends and insert DIVIDEND transactions (skip duplicates)."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        existing = self._repository.find_by_portfolio(
            portfolio_id,
            start_date=start_date,
            end_date=end_date,
            transaction_type="DIVIDEND",
        )
        existing_keys = {
            (tx.transaction_date, tx.ticker, round(tx.amount, 4)) for tx in existing
        }
        existing_ticker_days = {
            (tx.transaction_date, tx.ticker.upper()) for tx in existing
        }

        all_txs = self._repository.find_by_portfolio(portfolio_id)
        created: list[Transaction] = []
        for ticker in tickers:
            sym = ticker.strip().upper()
            if sym == "CASH":
                continue
            try:
                divs = yf.Ticker(sym).dividends
            except Exception as exc:
                logger.warning("Dividend fetch failed for %s: %s", sym, exc)
                continue

            if divs is None or divs.empty:
                continue

            for idx, amount in divs.items():
                div_date = idx.date() if hasattr(idx, "date") else idx
                if div_date < start_date or div_date > end_date:
                    continue
                if float(amount) <= 0:
                    continue
                key = (div_date, sym, round(float(amount), 4))
                if key in existing_keys or (div_date, sym) in existing_ticker_days:
                    continue
                dps = float(amount)
                shares_held = self._shares_held_on(
                    all_txs + created, sym, div_date, portfolio.cost_basis_method
                )
                if shares_held <= 1e-9:
                    continue
                total_cash = round(shares_held * dps, 4)
                if total_cash <= 0:
                    continue
                txn = Transaction(
                    transaction_date=div_date,
                    transaction_type="DIVIDEND",
                    ticker=sym,
                    shares=shares_held,
                    price=dps,
                    amount=total_cash,
                    reinvest=reinvest,
                    notes="Auto-synced dividend",
                )
                saved = self._repository.save(txn, portfolio_id)
                created.append(saved)
                existing_keys.add(key)
                existing_ticker_days.add((div_date, sym))

        return created

    @staticmethod
    def _shares_held_on(
        transactions: list[Transaction],
        ticker: str,
        as_of: date,
        cost_basis_method: str,
    ) -> float:
        """Shares held before ex-date (same-day BUY does not qualify)."""
        through = [t for t in transactions if t.transaction_date < as_of]
        summary = CostBasisCalculator(method=cost_basis_method).summarize(through)
        leg = summary.holdings.get(ticker.upper())
        return leg.quantity if leg else 0.0
