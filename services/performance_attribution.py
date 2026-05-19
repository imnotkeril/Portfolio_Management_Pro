"""Time-weighted and money-weighted return from cash flows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_sort import sort_transactions
from services.cost_basis import CostBasisCalculator
from services.holdings import HoldingsBuilder


@dataclass
class PerformanceSummary:
    """Portfolio performance metrics from ledger."""

    realized_pnl: float
    unrealized_pnl: float | None
    dividend_income: float
    cost_basis: float
    market_value: float | None
    total_return_twr: float | None
    total_return_mwr: float | None
    cash_balance: float


def _net_external_contributions(transactions: list[Transaction]) -> float:
    """Net deposits minus withdrawals from the ledger."""
    net = 0.0
    for tx in sort_transactions(transactions):
        if tx.transaction_type == "DEPOSIT":
            net += tx.amount
        elif tx.transaction_type == "WITHDRAWAL":
            net -= tx.amount
    return net


def _capital_base(transactions: list[Transaction], starting_capital: float) -> float:
    """
    Invested capital for return denominators.

    With-transactions portfolios record inception as DEPOSIT; starting_capital
    is the same number for display — using both double-counts (~-87% bogus TWR).
    """
    from_ledger = _net_external_contributions(transactions)
    if from_ledger > 1e-9:
        return from_ledger
    return starting_capital


def _ledger_has_external_flows(transactions: list[Transaction]) -> bool:
    return any(tx.transaction_type in ("DEPOSIT", "WITHDRAWAL") for tx in transactions)


def calculate_twr(
    transactions: list[Transaction],
    starting_capital: float,
    end_value: float,
) -> float | None:
    """
    Simple period return: end value vs net external contributions.

    Uses ledger deposits/withdrawals when present; otherwise starting_capital.
    """
    base = _capital_base(transactions, starting_capital)
    if base <= 0 or end_value <= 0:
        return None
    return (end_value / base) - 1.0


def calculate_mwr(
    transactions: list[Transaction],
    starting_capital: float,
    end_value: float,
    end_date: date,
) -> float | None:
    """
    Money-weighted return (IRR) on cash flows.

    Returns None when IRR does not converge.
    """
    base = _capital_base(transactions, starting_capital)
    if base <= 0:
        return None

    flows: list[tuple[float, float]] = []
    if _ledger_has_external_flows(transactions):
        for tx in sort_transactions(transactions):
            days = (tx.transaction_date - end_date).days
            years = days / 365.25
            if tx.transaction_type == "DEPOSIT":
                flows.append((years, -tx.amount))
            elif tx.transaction_type == "WITHDRAWAL":
                flows.append((years, tx.amount))
            elif tx.transaction_type == "DIVIDEND" and not tx.reinvest:
                flows.append((years, tx.amount))
    else:
        flows.append((0.0, -starting_capital))

    last_days = 0.0
    flows.append((last_days, end_value))

    if len(flows) < 2:
        return None

    def npv(rate: float) -> float:
        return sum(amount / (1.0 + rate) ** t for t, amount in flows)

    try:
        from scipy.optimize import brentq

        rate = brentq(npv, -0.999, 10.0, maxiter=200)
        return float(rate)
    except (ValueError, RuntimeError):
        return None


class PerformanceAttributionService:
    """Ledger-based PnL and return metrics."""

    def __init__(
        self,
        holdings_builder: HoldingsBuilder | None = None,
    ) -> None:
        self._holdings_builder = holdings_builder or HoldingsBuilder()

    def summarize(
        self,
        transactions: list[Transaction],
        starting_capital: float,
        cost_basis_method: str = "fifo",
        as_of: date | None = None,
    ) -> PerformanceSummary:
        as_of = as_of or date.today()
        basis = CostBasisCalculator(method=cost_basis_method).summarize(transactions)
        holdings = self._holdings_builder.build(transactions, cost_basis_method)

        market_value = sum(h.market_value or 0.0 for h in holdings)
        if not holdings:
            market_value = None
        unrealized = sum(
            h.unrealized_pnl or 0.0 for h in holdings if h.unrealized_pnl is not None
        )
        if not any(h.unrealized_pnl is not None for h in holdings):
            unrealized = None

        total_cost = sum(h.cost_basis for h in holdings)
        end_val = (market_value or 0.0) + basis.cash_balance

        twr = calculate_twr(transactions, starting_capital, end_val)
        mwr = calculate_mwr(transactions, starting_capital, end_val, as_of)

        return PerformanceSummary(
            realized_pnl=basis.realized_pnl,
            unrealized_pnl=unrealized,
            dividend_income=basis.dividend_income,
            cost_basis=total_cost,
            market_value=market_value,
            total_return_twr=twr,
            total_return_mwr=mwr,
            cash_balance=basis.cash_balance,
        )
