"""Rebalance portfolio to target weights via transaction ledger.

Production docs: docs/production/phases/phase-3-transaction-ledger.md,
docs/production/phases/phase-4-optimization-ledger-parity.md (synthetic planner).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

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
MAX_REBALANCE_SHIFT_DAYS = 10
CASH_WEIGHT_TOLERANCE = 0.012  # 1.2 pp — residual after whole-share sweep


def _cash_tolerance_dollars(total_value: float) -> float:
    """Min dollars of cash allowed above target (whole-share residual)."""
    return max(500.0, total_value * 0.002)


def scheduled_rebalance_tag(scheduled: date) -> str:
    return f"scheduled={scheduled.isoformat()}"


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


@dataclass
class RebalancePlan:
    scheduled_date: date
    execution_date: date
    trades: list[PlannedTrade]
    cash_target_pct: float
    projected_cash_pct: float
    complete: bool
    message: str


def scheduled_rebalance_dates(
    first_date: date, interval_months: int, through: date
) -> list[date]:
    """Scheduled rebalance calendar dates (first event is ``interval`` after inception)."""
    d = first_date + relativedelta(months=interval_months)
    out: list[date] = []
    while d <= through:
        out.append(d)
        d = d + relativedelta(months=interval_months)
    return out


def normalize_target_weights(user_weights: dict[str, float]) -> dict[str, float]:
    """Normalize arbitrary positive weights (incl. CASH) to portfolio-style targets."""
    weights = {
        str(k).strip().upper(): float(v)
        for k, v in user_weights.items()
        if float(v) > 1e-15
    }
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
                t: (cash_w if t == "CASH" else w * scale) for t, w in weights.items()
            }
    if abs(total - 1.0) > 1e-4:
        return {t: w / total for t, w in weights.items()}
    return weights


def plan_rebalance_to_target_weights(
    txs: list[Transaction],
    target_weights: dict[str, float],
    scheduled_date: date,
    cost_basis_method: str,
    data_service: DataService,
) -> RebalancePlan:
    """Plan trades to move holdings toward ``target_weights`` (same math as live rebalance)."""
    stock_weights = {t: w for t, w in target_weights.items() if t != "CASH" and w > 0}
    stock_weight_sum = sum(stock_weights.values())
    if stock_weight_sum <= 0:
        raise ValidationError("No stock target weights")

    txs_as_of = [t for t in txs if t.transaction_date <= scheduled_date]
    if not txs_as_of:
        raise ValidationError(
            "No transaction history on or before rebalance schedule date"
        )

    summary_preview = CostBasisCalculator(method=cost_basis_method).summarize(txs_as_of)
    tickers_needed = set(stock_weights) | {
        t
        for t, leg in summary_preview.holdings.items()
        if t != "CASH" and leg.quantity > 1e-9
    }

    execution_date, prices = _resolve_execution_date(
        data_service, list(tickers_needed), scheduled_date
    )

    txs_as_of = [t for t in txs if t.transaction_date <= execution_date]
    summary = CostBasisCalculator(method=cost_basis_method).summarize(txs_as_of)

    cash = summary.cash_balance
    current_shares: dict[str, float] = {}
    stock_value = 0.0
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
    cash_target_amt = total_value * cash_target
    investable_stocks = total_value * (1.0 - cash_target)

    target_shares = _allocate_target_shares(
        investable_stocks, stock_weights, stock_weight_sum, prices
    )
    if not target_shares:
        raise ValidationError("Could not compute stock targets for rebalance")

    logger.info(
        "Rebalance %s: total $%.0f, cash $%.0f (target $%.0f / %.1f%%), "
        "stocks target $%.0f",
        execution_date.isoformat(),
        total_value,
        cash,
        cash_target_amt,
        cash_target * 100,
        investable_stocks,
    )

    trades: list[PlannedTrade] = []
    all_tickers = set(current_shares) | set(target_shares)

    for ticker in sorted(all_tickers):
        if ticker not in stock_weights:
            continue
        px = prices.get(ticker)
        if not px or px <= 0:
            continue
        cur = int(math.floor(current_shares.get(ticker, 0)))
        tgt = target_shares.get(ticker, 0)
        if cur == tgt:
            continue

        cur_mv = cur * px
        diff = tgt - cur
        action = "BUY" if diff > 0 else "SELL"
        shares_n = abs(diff)
        trades.append(
            PlannedTrade(
                ticker=ticker,
                action=action,
                shares=shares_n,
                price=px,
                fees=estimate_ib_commission(shares_n, px),
                current_shares=float(cur),
                target_shares=tgt,
                current_weight=cur_mv / total_value if total_value else 0.0,
                target_weight=target_weights.get(ticker, 0.0),
            )
        )

    for ticker, qty in list(current_shares.items()):
        if ticker in stock_weights or qty <= 1e-9:
            continue
        px = prices.get(ticker)
        if not px or px <= 0:
            continue
        cur = int(math.floor(qty))
        if cur <= 0:
            continue
        trades.append(
            PlannedTrade(
                ticker=ticker,
                action="SELL",
                shares=cur,
                price=px,
                fees=estimate_ib_commission(cur, px),
                current_shares=float(cur),
                target_shares=0,
                current_weight=(cur * px / total_value) if total_value else 0.0,
                target_weight=0.0,
            )
        )

    trades = _finalize_rebalance_trades(
        trades,
        cash=cash,
        shares=current_shares,
        prices=prices,
        target_shares=target_shares,
        target_weights=target_weights,
        cash_target=cash_target,
        total_value=total_value,
    )

    projected = _projected_cash_weight(cash, current_shares, trades, prices)
    sim_cash_end, sim_shares_end = _apply_trades_simulation(
        cash, current_shares, trades
    )
    post_total = _portfolio_market_value(sim_cash_end, sim_shares_end, prices)
    excess_dollars = max(0.0, sim_cash_end - post_total * cash_target)
    complete = abs(
        projected - cash_target
    ) <= CASH_WEIGHT_TOLERANCE and excess_dollars <= _cash_tolerance_dollars(post_total)
    message = (
        f"Projected cash {projected * 100:.2f}% vs target {cash_target * 100:.2f}% "
        f"(~${excess_dollars:,.0f} above cash target)"
    )
    if not complete:
        message = (
            f"Rebalance incomplete: {message}. "
            "Cash sweep could not reach target — check prices or run again."
        )

    return RebalancePlan(
        scheduled_date=scheduled_date,
        execution_date=execution_date,
        trades=trades,
        cash_target_pct=cash_target,
        projected_cash_pct=projected,
        complete=complete,
        message=message,
    )


def rebalance_plan_to_synthetic_transactions(
    plan: RebalancePlan,
    *,
    notes_suffix: str = "pmpro:synthetic-optimized-ledger",
) -> list[Transaction]:
    """Turn a :class:`RebalancePlan` into ledger rows (not persisted)."""
    tag = scheduled_rebalance_tag(plan.scheduled_date)
    out: list[Transaction] = []
    ordered_trades = sorted(
        plan.trades,
        key=lambda t: (0 if t.action == "SELL" else 1, t.ticker),
    )
    for trade in ordered_trades:
        note = (
            f"{REBALANCE_NOTE} | {tag} | "
            f"weight {trade.current_weight * 100:.1f}% -> "
            f"target {trade.target_weight * 100:.1f}% | {notes_suffix}"
        )
        out.append(
            Transaction(
                transaction_date=plan.execution_date,
                transaction_type=trade.action,
                ticker=trade.ticker,
                shares=float(trade.shares),
                price=trade.price,
                fees=trade.fees,
                notes=note,
            )
        )
    return out


def has_rebalance_for_scheduled(
    transactions: list[Transaction], scheduled: date
) -> bool:
    """True if this schedule checkpoint already produced rebalance trades."""
    tag = scheduled_rebalance_tag(scheduled)
    for tx in transactions:
        notes = tx.notes or ""
        if REBALANCE_NOTE in notes and tag in notes:
            return True
    return any(
        tx.transaction_date == scheduled and (tx.notes or "").startswith(REBALANCE_NOTE)
        for tx in transactions
    )


def _allocate_target_shares(
    stock_budget: float,
    stock_weights: dict[str, float],
    stock_weight_sum: float,
    prices: dict[str, float],
) -> dict[str, int]:
    """Whole-share targets that spend stock_budget (largest-remainder)."""
    if stock_budget <= 0 or stock_weight_sum <= 0:
        return {}

    exact: dict[str, float] = {}
    for ticker, weight in stock_weights.items():
        px = prices.get(ticker)
        if not px or px <= 0:
            continue
        exact[ticker] = (weight / stock_weight_sum) * stock_budget / px

    if not exact:
        return {}

    shares = {t: int(math.floor(v)) for t, v in exact.items()}
    invested = sum(shares[t] * prices[t] for t in shares)
    remainder = stock_budget - invested
    guard = 0
    while remainder > 1e-6 and guard < 100_000:
        order = sorted(exact.keys(), key=lambda t: exact[t] - shares[t], reverse=True)
        progressed = False
        for ticker in order:
            px = prices[ticker]
            if px <= remainder + 1e-9:
                shares[ticker] += 1
                remainder -= px
                progressed = True
                break
        if not progressed:
            break
        guard += 1
    return shares


def _prices_on_trading_day(
    data_service: DataService, tickers: list[str], on_date: date
) -> dict[str, float] | None:
    """Close on exactly on_date (on_date must be a trading day with a bar)."""
    prices: dict[str, float] = {}
    for ticker in tickers:
        resolved = data_service.fetch_close_on_nearest_trading_day(
            ticker, on_date, lookforward_days=0
        )
        if not resolved:
            return None
        px, trade_date = resolved
        if trade_date != on_date or px <= 0:
            return None
        prices[ticker] = px
    return prices


def _resolve_execution_date(
    data_service: DataService,
    tickers: list[str],
    scheduled: date,
    max_shift_days: int = MAX_REBALANCE_SHIFT_DAYS,
) -> tuple[date, dict[str, float]]:
    """Shift forward day-by-day until every ticker has a close on the same session."""
    today = date.today()
    unique = sorted({t for t in tickers if t and t != "CASH"})
    if not unique:
        raise ValidationError("No stock tickers to rebalance")

    for offset in range(max_shift_days + 1):
        candidate = scheduled + timedelta(days=offset)
        if candidate > today:
            break
        prices = _prices_on_trading_day(data_service, unique, candidate)
        if prices is not None:
            if offset > 0:
                logger.info(
                    "Rebalance shifted %s -> %s (%d day(s)) — all prices available",
                    scheduled.isoformat(),
                    candidate.isoformat(),
                    offset,
                )
            return candidate, prices

    # Same-day catch-up: today's close is often not in history until after the
    # US session ends; walk backward to the latest common trading session.
    as_of = min(scheduled, today)
    for offset in range(1, max_shift_days + 1):
        candidate = as_of - timedelta(days=offset)
        prices = _prices_on_trading_day(data_service, unique, candidate)
        if prices is not None:
            logger.info(
                "Rebalance using prior session %s (no close on %s yet)",
                candidate.isoformat(),
                scheduled.isoformat(),
            )
            return candidate, prices

    raise ValidationError(
        f"Cannot rebalance: no closing prices for all tickers near "
        f"{scheduled.isoformat()} (checked {max_shift_days} day(s) forward and "
        f"backward; if the date is today, market close may not be published yet)"
    )


def _apply_trades_simulation(
    cash: float,
    shares: dict[str, float],
    trades: list[PlannedTrade],
) -> tuple[float, dict[str, float]]:
    sim_cash = cash
    sim_shares = {k: float(v) for k, v in shares.items()}
    ordered = sorted(trades, key=lambda t: (0 if t.action == "SELL" else 1, t.ticker))
    for t in ordered:
        if t.action == "SELL":
            sim_cash += t.shares * t.price - t.fees
            sim_shares[t.ticker] = sim_shares.get(t.ticker, 0.0) - t.shares
        else:
            sim_cash -= t.shares * t.price + t.fees
            sim_shares[t.ticker] = sim_shares.get(t.ticker, 0.0) + t.shares
    return sim_cash, sim_shares


def _portfolio_market_value(
    cash: float, shares: dict[str, float], prices: dict[str, float]
) -> float:
    stock_val = sum(
        max(0.0, shares.get(t, 0.0)) * prices[t] for t in prices if t != "CASH"
    )
    return cash + stock_val


def _merge_buy(trades: list[PlannedTrade], extra: PlannedTrade) -> None:
    for t in trades:
        if t.action == "BUY" and t.ticker == extra.ticker:
            t.shares += extra.shares
            t.fees = estimate_ib_commission(t.shares, t.price)
            t.target_shares = extra.target_shares
            return
    trades.append(extra)


def _finalize_rebalance_trades(
    trades: list[PlannedTrade],
    *,
    cash: float,
    shares: dict[str, float],
    prices: dict[str, float],
    target_shares: dict[str, int],
    target_weights: dict[str, float],
    cash_target: float,
    total_value: float,
) -> list[PlannedTrade]:
    """
    Sells first, then buys toward targets, then deploy excess cash above target %.
    """
    sells = [t for t in trades if t.action == "SELL"]
    buys = [t for t in trades if t.action == "BUY"]
    other = [t for t in trades if t.action not in ("SELL", "BUY")]

    sim_cash, sim_shares = _apply_trades_simulation(cash, shares, sells)
    finalized: list[PlannedTrade] = list(sells)

    for t in sorted(buys, key=lambda x: x.shares * x.price, reverse=True):
        need = t.shares
        px = t.price
        while need > 0:
            fees = estimate_ib_commission(need, px)
            cost = need * px + fees
            if cost <= sim_cash + 1e-6:
                finalized.append(
                    PlannedTrade(
                        ticker=t.ticker,
                        action="BUY",
                        shares=need,
                        price=px,
                        fees=fees,
                        current_shares=t.current_shares,
                        target_shares=t.target_shares,
                        current_weight=t.current_weight,
                        target_weight=t.target_weight,
                    )
                )
                sim_cash -= cost
                sim_shares[t.ticker] = sim_shares.get(t.ticker, 0.0) + need
                break
            need -= 1
        if need > 0 and t.shares > 0:
            logger.warning(
                "Rebalance partial BUY %s: %d of %d shares (cash %.2f)",
                t.ticker,
                t.shares - need,
                t.shares,
                sim_cash,
            )

    sim_cash, sim_shares = _sweep_excess_cash(
        finalized,
        sim_cash,
        sim_shares,
        prices,
        target_weights,
        cash_target,
    )

    return other + sorted(
        finalized, key=lambda t: (0 if t.action == "SELL" else 1, t.ticker)
    )


def _sweep_excess_cash(
    finalized: list[PlannedTrade],
    sim_cash: float,
    sim_shares: dict[str, float],
    prices: dict[str, float],
    target_weights: dict[str, float],
    cash_target: float,
) -> tuple[float, dict[str, float]]:
    """
    Deploy cash above target % into underweight names (by portfolio weight).

    Continues past lot targets — excess cash often remains when cur >= target_shares
    but every line is still underweight vs weight_target (e.g. after a bad rebalance).
    """
    guard = 0
    while guard < 100_000:
        guard += 1
        total = _portfolio_market_value(sim_cash, sim_shares, prices)
        if total <= 0:
            break
        target_cash = total * cash_target
        excess = sim_cash - target_cash
        if excess <= _cash_tolerance_dollars(total):
            break

        best_ticker: str | None = None
        best_deficit = 0.0
        for ticker, tgt_w in target_weights.items():
            if ticker == "CASH":
                continue
            px = prices.get(ticker)
            if not px or px <= 0:
                continue
            cur_w = (max(0.0, sim_shares.get(ticker, 0.0)) * px) / total
            deficit = tgt_w - cur_w
            if deficit > best_deficit:
                best_deficit = deficit
                best_ticker = ticker

        if best_ticker is None or best_deficit <= 1e-6:
            break

        px = prices[best_ticker]
        fee1 = estimate_ib_commission(1, px)
        if excess < px + fee1:
            break

        max_by_cash = 1
        while max_by_cash < 500:
            fee_n = estimate_ib_commission(max_by_cash + 1, px)
            cost_n = (max_by_cash + 1) * px + fee_n
            if cost_n > excess + 1e-6:
                break
            max_by_cash += 1

        cur = int(sim_shares.get(best_ticker, 0))
        cur_w = (cur * px / total) if total else 0.0
        fees = estimate_ib_commission(max_by_cash, px)
        _merge_buy(
            finalized,
            PlannedTrade(
                ticker=best_ticker,
                action="BUY",
                shares=max_by_cash,
                price=px,
                fees=fees,
                current_shares=float(cur),
                target_shares=cur + max_by_cash,
                current_weight=cur_w,
                target_weight=target_weights.get(best_ticker, 0.0),
            ),
        )
        sim_cash -= max_by_cash * px + fees
        sim_shares[best_ticker] = cur + max_by_cash

    return sim_cash, sim_shares


def _projected_cash_weight(
    cash: float,
    shares: dict[str, float],
    trades: list[PlannedTrade],
    prices: dict[str, float],
) -> float:
    sim_cash, sim_shares = _apply_trades_simulation(cash, shares, trades)
    total = _portfolio_market_value(sim_cash, sim_shares, prices)
    if total <= 0:
        return 0.0
    return sim_cash / total


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
        scheduled_date: date,
        user_id: str | None = None,
    ) -> RebalancePlan:
        return self._plan_trades(portfolio_id, scheduled_date, user_id)

    def execute(
        self,
        portfolio_id: str,
        scheduled_date: date,
        user_id: str | None = None,
    ) -> list[Transaction]:
        plan = self._plan_trades(portfolio_id, scheduled_date, user_id)
        if not plan.trades:
            if not plan.complete:
                logger.warning(
                    "Rebalance on %s produced no trades: %s",
                    scheduled_date,
                    plan.message,
                )
            return []
        if not plan.complete:
            logger.warning(
                "Rebalance on %s may leave cash off target: %s",
                scheduled_date,
                plan.message,
            )

        created: list[Transaction] = []
        tag = scheduled_rebalance_tag(plan.scheduled_date)
        ordered_trades = sorted(
            plan.trades,
            key=lambda t: (0 if t.action == "SELL" else 1, t.ticker),
        )

        for trade in ordered_trades:
            note = (
                f"{REBALANCE_NOTE} | {tag} | "
                f"weight {trade.current_weight * 100:.1f}% -> "
                f"target {trade.target_weight * 100:.1f}%"
            )
            tx = self._transaction_service.add_transaction(
                portfolio_id=portfolio_id,
                transaction_date=plan.execution_date,
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
            "Rebalance scheduled %s executed %s for portfolio %s: %d trades, "
            "projected cash %.2f%% (target %.2f%%)",
            plan.scheduled_date,
            plan.execution_date,
            portfolio_id,
            len(created),
            plan.projected_cash_pct * 100,
            plan.cash_target_pct * 100,
        )
        if not plan.complete:
            logger.error("Rebalance incomplete: %s", plan.message)

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
            if has_rebalance_for_scheduled(txs, rb_date):
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
        return scheduled_rebalance_dates(first_date, interval_months, through)

    def _plan_trades(
        self,
        portfolio_id: str,
        scheduled_date: date,
        user_id: str | None = None,
    ) -> RebalancePlan:
        portfolio = self._portfolio_service.get_portfolio(portfolio_id, user_id)
        txs = self._transaction_service.get_transactions(portfolio_id, user_id=user_id)

        target_weights = self._target_weights(portfolio)
        if not target_weights:
            raise ValidationError("No target weights on positions")

        return plan_rebalance_to_target_weights(
            txs,
            target_weights,
            scheduled_date,
            portfolio.cost_basis_method,
            self._data_service,
        )

    def _target_weights(self, portfolio) -> dict[str, float]:
        from services.strategy_service import extract_target_weights

        return extract_target_weights(portfolio)
