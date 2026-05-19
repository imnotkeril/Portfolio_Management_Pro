"""Keep transaction ledger current: splits, dividends, scheduled rebalancing."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from core.data_manager.portfolio import Portfolio
from core.data_manager.transaction import Transaction
from services.portfolio_lock import portfolio_maintenance_lock
from services.portfolio_service import PortfolioService
from services.rebalance_service import (
    MAX_REBALANCE_SHIFT_DAYS,
    RebalanceService,
    _resolve_execution_date,
    has_rebalance_for_scheduled,
)
from services.transaction_service import TransactionService

logger = logging.getLogger(__name__)


class PortfolioLedgerService:
    """Idempotent maintenance from inception through today."""

    def __init__(
        self,
        portfolio_service: PortfolioService | None = None,
        transaction_service: TransactionService | None = None,
        rebalance_service: RebalanceService | None = None,
    ) -> None:
        self._portfolio_service = portfolio_service or PortfolioService()
        self._transaction_service = transaction_service or TransactionService()
        self._rebalance_service = rebalance_service or RebalanceService()

    def maintain(self, portfolio_id: str, user_id: str | None = None) -> dict[str, int]:
        """
        Sync splits and dividends up to each rebalance date, then rebalance.

        Safe to call repeatedly; skips duplicates already in the ledger.
        """
        with portfolio_maintenance_lock(portfolio_id):
            return self._maintain_unlocked(portfolio_id, user_id)

    def _maintain_unlocked(
        self, portfolio_id: str, user_id: str | None = None
    ) -> dict[str, int]:
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._transaction_service.get_transactions(portfolio_id, user_id=user_id)
        if not txs:
            return {"splits": 0, "dividends": 0, "rebalance": 0}

        portfolio = self._ensure_target_weights(portfolio, txs, user_id)
        first_date = min(t.transaction_date for t in txs)
        through = date.today()
        tickers = self._ledger_tickers(txs)

        rebalance_dates: list[date] = []
        if portfolio.rebalance_interval_months:
            rebalance_dates = self._rebalance_service._scheduled_dates(
                first_date, portfolio.rebalance_interval_months, through
            )

        splits = 0
        dividends = 0
        rebalances = 0
        catchup_error: str | None = None

        checkpoints = sorted(set(rebalance_dates + [through]))
        cursor = first_date

        for checkpoint in checkpoints:
            if tickers and cursor <= checkpoint:
                try:
                    splits += len(
                        self._transaction_service.sync_splits(
                            portfolio_id,
                            tickers,
                            cursor,
                            checkpoint,
                            user_id,
                            sync_positions=False,
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "Split sync failed %s..%s for %s: %s",
                        cursor,
                        checkpoint,
                        portfolio_id,
                        exc,
                    )
                try:
                    dividends += len(
                        self._transaction_service.sync_dividends(
                            portfolio_id,
                            tickers,
                            cursor,
                            checkpoint,
                            user_id,
                            sync_positions=False,
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "Dividend sync failed %s..%s for %s: %s",
                        cursor,
                        checkpoint,
                        portfolio_id,
                        exc,
                    )

            if checkpoint in rebalance_dates and not has_rebalance_for_scheduled(
                txs, checkpoint
            ):
                try:
                    batch = self._rebalance_service.execute(
                        portfolio_id, checkpoint, user_id
                    )
                    rebalances += len(batch)
                    txs = self._transaction_service.get_transactions(
                        portfolio_id, user_id=user_id
                    )
                except Exception as exc:
                    logger.exception(
                        "Scheduled rebalance failed on %s for %s: %s",
                        checkpoint,
                        portfolio_id,
                        exc,
                    )

            cursor = checkpoint + timedelta(days=1)

        catchup_error = self._cash_catchup_rebalance_if_needed(
            portfolio, portfolio_id, user_id
        )

        self._transaction_service.sync_positions_for_portfolio(portfolio_id, user_id)

        logger.info(
            "Ledger maintain %s: splits=%d dividends=%d rebalance=%d",
            portfolio_id,
            splits,
            dividends,
            rebalances,
        )
        result: dict[str, int | str] = {
            "splits": splits,
            "dividends": dividends,
            "rebalance": rebalances,
        }
        if catchup_error:
            result["catchup_rebalance_error"] = catchup_error
        return result

    def _cash_catchup_rebalance_if_needed(
        self, portfolio: Portfolio, portfolio_id: str, user_id: str | None
    ) -> str | None:
        """If cash drifted well above target, run an extra rebalance today."""
        if not portfolio.rebalance_interval_months:
            return None
        cash_target = 0.0
        for pos in portfolio.get_all_positions():
            if pos.ticker == "CASH" and pos.weight_target and pos.weight_target > 0:
                cash_target = float(pos.weight_target)
                break
        if cash_target <= 0:
            return None

        txs = self._transaction_service.get_transactions(portfolio_id, user_id=user_id)
        if not txs:
            return None
        from services.cost_basis import CostBasisCalculator

        summary = CostBasisCalculator(method=portfolio.cost_basis_method).summarize(txs)
        cash = summary.cash_balance

        stock_weights = {
            p.ticker: float(p.weight_target)
            for p in portfolio.get_all_positions()
            if p.ticker != "CASH"
            and p.weight_target is not None
            and p.weight_target > 0
        }
        tickers = list(stock_weights.keys())
        today = date.today()
        prices: dict[str, float] = {}
        if tickers:
            resolved = _resolve_execution_date(
                self._rebalance_service._data_service,
                tickers,
                today,
                max_shift_days=MAX_REBALANCE_SHIFT_DAYS,
            )
            _, prices = resolved

        stock_val = sum(
            leg.quantity * prices[ticker]
            for ticker, leg in summary.holdings.items()
            if ticker in prices
        )
        total = cash + stock_val
        if total <= 0:
            return None
        cash_pct = cash / total
        if cash_pct <= cash_target + 0.03:
            return None

        if has_rebalance_for_scheduled(txs, today):
            return None
        try:
            self._rebalance_service.execute(portfolio_id, today, user_id)
            logger.info(
                "Cash catch-up rebalance on %s (cash %.1f%% vs target %.1f%%)",
                portfolio_id,
                cash_pct * 100,
                cash_target * 100,
            )
            return None
        except Exception as exc:
            msg = f"Cash catch-up rebalance failed: {exc}"
            logger.warning("%s for %s", msg, portfolio_id)
            return msg

    def _ledger_tickers(self, transactions: list[Transaction]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for tx in transactions:
            sym = tx.ticker.strip().upper()
            if sym == "CASH" or sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
        return out

    def _ensure_target_weights(
        self,
        portfolio: Portfolio,
        transactions: list[Transaction],
        user_id: str | None,
    ) -> Portfolio:
        """Restore target weights when position sync wiped them (infer from initial BUYs)."""
        has_stock_targets = any(
            p.weight_target is not None and p.weight_target > 0
            for p in portfolio.get_all_positions()
            if p.ticker != "CASH"
        )
        has_cash_target = any(
            p.ticker == "CASH" and p.weight_target is not None and p.weight_target > 0
            for p in portfolio.get_all_positions()
        )
        if has_stock_targets and has_cash_target:
            return portfolio

        first_date = min(t.transaction_date for t in transactions)
        buys = [
            t
            for t in transactions
            if t.transaction_date == first_date
            and t.transaction_type == "BUY"
            and t.ticker.upper() != "CASH"
        ]

        if not has_cash_target:
            stock_target_sum = sum(
                p.weight_target or 0.0
                for p in portfolio.get_all_positions()
                if p.ticker != "CASH"
            )
            cash_target: float | None = None
            if 0 < stock_target_sum < 1.0 - 1e-4:
                cash_target = 1.0 - stock_target_sum
            elif buys:
                deposit_total = sum(
                    t.shares * t.price
                    for t in transactions
                    if t.transaction_date == first_date
                    and t.transaction_type == "DEPOSIT"
                    and t.ticker.upper() == "CASH"
                )
                buy_total = sum(t.shares * t.price for t in buys)
                portfolio_total = deposit_total if deposit_total > 0 else buy_total
                if portfolio_total > 0:
                    cash_target = max(
                        0.0, (portfolio_total - buy_total) / portfolio_total
                    )

            if cash_target and cash_target > 0:
                try:
                    portfolio.update_position(ticker="CASH", weight_target=cash_target)
                except Exception:
                    portfolio.add_position(
                        ticker="CASH",
                        shares=0,
                        weight_target=cash_target,
                        purchase_price=1.0,
                    )

        if not has_stock_targets and buys:
            deposit_total = sum(
                t.shares * t.price
                for t in transactions
                if t.transaction_date == first_date
                and t.transaction_type == "DEPOSIT"
                and t.ticker.upper() == "CASH"
            )
            buy_total = sum(t.shares * t.price for t in buys)
            portfolio_total = deposit_total if deposit_total > 0 else buy_total
            if portfolio_total <= 0:
                return portfolio

            for tx in buys:
                weight = (tx.shares * tx.price) / portfolio_total
                ticker = tx.ticker.upper()
                try:
                    portfolio.update_position(ticker=ticker, weight_target=weight)
                except Exception:
                    portfolio.add_position(
                        ticker=ticker,
                        shares=tx.shares,
                        weight_target=weight,
                    )

        if user_id is None:
            return portfolio
        saved = self._portfolio_service._repository.save(portfolio, user_id)
        logger.info(
            "Restored %d target weights on portfolio %s from initial BUYs",
            len(buys),
            portfolio.id,
        )
        return saved
