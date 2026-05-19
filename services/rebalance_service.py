"""Rebalance portfolio to target weights via transaction ledger."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from core.data_manager.transaction import Transaction
from core.exceptions import ValidationError
from services.cost_basis import CostBasisCalculator
from services.data_service import DataService
from services.ib_commission import estimate_ib_commission
from services.portfolio_service import PortfolioService
from services.transaction_service import TransactionService

logger = logging.getLogger(__name__)

REBALANCE_NOTE = "Rebalance to target weights"


@dataclass
class PlannedTrade:
    ticker: str
    action: str
    shares: int
    price: float
    fees: float
    current_shares: float
    target_shares: int
    current_weight: float
    target_weight: float


class RebalanceService:
    def __init__(
        self,
        portfolio_service: PortfolioService | None = None,
        transaction_service: TransactionService | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self._portfolio_service = portfolio_service or PortfolioService()
        self._transaction_service = transaction_service or TransactionService()
        self._data_service = data_service or DataService()

    def preview(
        self,
        portfolio_id: str,
        as_of_date: date,
        user_id: str | None = None,
    ) -> list[PlannedTrade]:
        return self._plan_trades(portfolio_id, as_of_date, user_id)

    def execute(
        self,
        portfolio_id: str,
        as_of_date: date,
        user_id: str | None = None,
    ) -> list[Transaction]:
        trades = self._plan_trades(portfolio_id, as_of_date, user_id)
        if not trades:
            return []

        created: list[Transaction] = []
        sells = [t for t in trades if t.action == "SELL"]
        buys = [t for t in trades if t.action == "BUY"]

        for trade in sells + buys:
            note = (
                f"{REBALANCE_NOTE} | "
                f"weight {trade.current_weight * 100:.1f}% -> "
                f"target {trade.target_weight * 100:.1f}%"
            )
            tx = self._transaction_service.add_transaction(
                portfolio_id=portfolio_id,
                transaction_date=as_of_date,
                transaction_type=trade.action,
                ticker=trade.ticker,
                shares=float(trade.shares),
                price=trade.price,
                fees=trade.fees,
                notes=note,
                user_id=user_id,
                sync_positions=False,
            )
            created.append(tx)

        if created:
            self._transaction_service.sync_positions_for_portfolio(
                portfolio_id, user_id
            )

        logger.info(
            "Rebalance on %s for portfolio %s: %d trades",
            as_of_date,
            portfolio_id,
            len(created),
        )
        return created

    def execute_scheduled(
        self,
        portfolio_id: str,
        user_id: str | None = None,
        through_date: date | None = None,
    ) -> list[Transaction]:
        """Run rebalance on each scheduled date from first tx to through_date."""
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        interval = portfolio.rebalance_interval_months
        if not interval:
            raise ValidationError("Rebalancing is not enabled for this portfolio")

        through = through_date or date.today()
        txs = self._transaction_service.get_transactions(portfolio_id, user_id=user_id)
        if not txs:
            raise ValidationError("No transactions; create initial positions first")

        first_date = min(t.transaction_date for t in txs)
        rebalance_dates = self._scheduled_dates(first_date, interval, through)

        all_created: list[Transaction] = []
        for rb_date in rebalance_dates:
            if self._has_rebalance_on_date(txs, rb_date):
                continue
            batch = self.execute(portfolio_id, rb_date, user_id)
            all_created.extend(batch)
            txs = self._transaction_service.get_transactions(
                portfolio_id, user_id=user_id
            )

        return all_created

    def _scheduled_dates(
        self, first_date: date, interval_months: int, through: date
    ) -> list[date]:
        """Dates to rebalance (first event is interval after inception)."""
        d = first_date + relativedelta(months=interval_months)
        out: list[date] = []
        while d <= through:
            out.append(d)
            d = d + relativedelta(months=interval_months)
        return out

    def _has_rebalance_on_date(
        self, transactions: list[Transaction], on_date: date
    ) -> bool:
        return any(
            t.transaction_date == on_date and (t.notes or "").startswith(REBALANCE_NOTE)
            for t in transactions
        )

    def _plan_trades(
        self,
        portfolio_id: str,
        as_of_date: date,
        user_id: str | None = None,
    ) -> list[PlannedTrade]:
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._transaction_service.get_transactions(portfolio_id, user_id=user_id)
        txs_as_of = [t for t in txs if t.transaction_date <= as_of_date]
        if not txs_as_of:
            raise ValidationError("No transaction history on or before rebalance date")

        target_weights = self._target_weights(portfolio)
        if not target_weights:
            raise ValidationError("No target weights on positions")

        summary = CostBasisCalculator(method=portfolio.cost_basis_method).summarize(
            txs_as_of
        )

        stock_tickers = [
            t for t in target_weights if t != "CASH" and target_weights.get(t, 0) > 0
        ]
        prices = self._prices_on_date(stock_tickers, as_of_date)

        cash = summary.cash_balance
        stock_value = 0.0
        current_shares: dict[str, float] = {}
        for ticker, leg in summary.holdings.items():
            if ticker == "CASH":
                continue
            current_shares[ticker] = leg.quantity
            px = prices.get(ticker)
            if px and px > 0:
                stock_value += leg.quantity * px

        total_value = cash + stock_value
        if total_value <= 0:
            raise ValidationError("Portfolio value is zero on rebalance date")

        cash_target = min(1.0, max(0.0, target_weights.get("CASH", 0.0)))
        stock_budget = total_value * (1.0 - cash_target)

        stock_weights = {
            t: w for t, w in target_weights.items() if t != "CASH" and w > 0
        }
        stock_weight_sum = sum(stock_weights.values())
        if stock_weight_sum <= 0:
            raise ValidationError("No stock target weights on positions")

        target_shares: dict[str, int] = {}
        for ticker, weight in stock_weights.items():
            px = prices.get(ticker)
            if not px or px <= 0:
                continue
            norm_w = weight / stock_weight_sum
            target_shares[ticker] = max(0, math.floor(norm_w * stock_budget / px))

        trades: list[PlannedTrade] = []
        all_tickers = set(current_shares) | set(target_shares)

        for ticker in sorted(all_tickers):
            if ticker == "CASH":
                continue
            px = prices.get(ticker)
            if not px or px <= 0:
                continue

            cur = int(math.floor(current_shares.get(ticker, 0)))
            tgt = target_shares.get(ticker, 0)
            if cur == tgt:
                continue

            diff = tgt - cur
            action = "BUY" if diff > 0 else "SELL"
            shares = abs(diff)
            fees = estimate_ib_commission(shares, px)

            cur_w = (cur * px / total_value) if total_value else 0.0
            tgt_w = target_weights.get(ticker, 0.0)

            trades.append(
                PlannedTrade(
                    ticker=ticker,
                    action=action,
                    shares=shares,
                    price=px,
                    fees=fees,
                    current_shares=float(cur),
                    target_shares=tgt,
                    current_weight=cur_w,
                    target_weight=tgt_w,
                )
            )

        sells = sorted(
            [t for t in trades if t.action == "SELL"],
            key=lambda t: t.shares * t.price,
            reverse=True,
        )
        buys = sorted(
            [t for t in trades if t.action == "BUY"],
            key=lambda t: t.shares * t.price,
            reverse=True,
        )
        return sells + buys

    def _target_weights(self, portfolio) -> dict[str, float]:
        weights: dict[str, float] = {}
        for pos in portfolio.get_all_positions():
            if pos.weight_target is not None and pos.weight_target > 0:
                weights[pos.ticker] = pos.weight_target
        if not weights:
            return {}
        total = sum(weights.values())
        if total <= 0:
            return {}
        cash_w = weights.get("CASH", 0.0)
        if cash_w > 0 and abs(total - 1.0) > 1e-4:
            stock_sum = sum(w for t, w in weights.items() if t != "CASH")
            if stock_sum > 0:
                scale = (1.0 - cash_w) / stock_sum
                return {
                    t: (cash_w if t == "CASH" else w * scale)
                    for t, w in weights.items()
                }
        if abs(total - 1.0) > 1e-4:
            return {t: w / total for t, w in weights.items()}
        return weights

    def _prices_on_date(self, tickers: list[str], on_date: date) -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            resolved = self._data_service.fetch_close_on_nearest_trading_day(
                ticker, on_date
            )
            if resolved:
                prices[ticker], _ = resolved
        return prices
