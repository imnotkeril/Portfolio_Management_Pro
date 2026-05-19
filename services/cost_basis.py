"""Cost basis and realized PnL from transaction ledger."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_sort import sort_transactions
from core.exceptions import ValidationError


@dataclass
class Lot:
    """FIFO lot: quantity and per-share cost."""

    quantity: float
    cost_per_share: float


@dataclass
class TickerLedger:
    """Per-ticker position state while processing transactions."""

    quantity: float = 0.0
    total_cost: float = 0.0
    lots: list[Lot] = field(default_factory=list)


@dataclass
class CostBasisSummary:
    """Aggregated ledger metrics."""

    holdings: dict[str, TickerLedger]
    realized_pnl: float
    dividend_income: float
    cash_balance: float


class CostBasisCalculator:
    """Compute holdings and realized PnL from transactions (FIFO or average)."""

    def __init__(self, method: str = "fifo") -> None:
        method = method.lower()
        if method not in ("fifo", "average"):
            raise ValueError("method must be 'fifo' or 'average'")
        self._method = method
        self.reset()

    def reset(self) -> None:
        """Clear ledger state for incremental replay."""
        self._holdings: dict[str, TickerLedger] = {}
        self._cash_balance = 0.0
        self._realized_pnl = 0.0
        self._dividend_income = 0.0

    def snapshot_quantities(self) -> tuple[dict[str, float], float]:
        """Current share quantities and cash (USD face value)."""
        quantities = {
            ticker: leg.quantity
            for ticker, leg in self._holdings.items()
            if leg.quantity > 1e-9
        }
        return quantities, self._cash_balance

    def snapshot_equity_rows(self) -> list[tuple[str, float, float | None]]:
        """(ticker, shares, avg_cost) for each equity line."""
        rows: list[tuple[str, float, float | None]] = []
        for ticker, leg in self._holdings.items():
            if leg.quantity <= 1e-9:
                continue
            avg = leg.total_cost / leg.quantity if leg.quantity > 0 else None
            rows.append((ticker, leg.quantity, avg))
        return rows

    def apply(self, tx: Transaction) -> None:
        """Apply one transaction to running ledger state."""
        if tx.transaction_type in ("DEPOSIT", "WITHDRAWAL"):
            if tx.transaction_type == "DEPOSIT":
                self._cash_balance += tx.amount
            else:
                self._cash_balance -= tx.amount
            return

        if tx.transaction_type == "DIVIDEND":
            self._dividend_income += tx.amount
            if tx.reinvest:
                self._apply_buy(self._holdings, tx.ticker, tx.shares, tx.price, tx.fees)
                self._cash_balance -= tx.amount + (tx.fees or 0.0)
            else:
                self._cash_balance += tx.amount
            return

        if tx.transaction_type == "SPLIT":
            self._apply_split(self._holdings, tx.ticker, tx.split_ratio or 1.0)
            return

        if tx.transaction_type == "BUY":
            cost = tx.amount + (tx.fees or 0.0)
            self._cash_balance -= cost
            self._apply_buy(
                self._holdings,
                tx.ticker,
                tx.shares,
                (tx.amount + (tx.fees or 0.0)) / tx.shares,
                0.0,
            )
            return

        if tx.transaction_type == "SELL":
            proceeds = tx.amount - (tx.fees or 0.0)
            self._cash_balance += proceeds
            self._realized_pnl += self._apply_sell(
                self._holdings, tx.ticker, tx.shares, tx.price, tx.fees or 0.0
            )

    def summarize(self, transactions: list[Transaction]) -> CostBasisSummary:
        """Process transactions in chronological order."""
        self.reset()
        for tx in sort_transactions(transactions):
            self.apply(tx)
        return CostBasisSummary(
            holdings=self._holdings,
            realized_pnl=self._realized_pnl,
            dividend_income=self._dividend_income,
            cash_balance=self._cash_balance,
        )

    def _ledger(self, holdings: dict[str, TickerLedger], ticker: str) -> TickerLedger:
        if ticker not in holdings:
            holdings[ticker] = TickerLedger()
        return holdings[ticker]

    def _apply_buy(
        self,
        holdings: dict[str, TickerLedger],
        ticker: str,
        shares: float,
        cost_per_share: float,
        fees: float,
    ) -> None:
        leg = self._ledger(holdings, ticker)
        total_cost = shares * cost_per_share + fees
        leg.quantity += shares
        leg.total_cost += total_cost
        if self._method == "fifo":
            leg.lots.append(Lot(quantity=shares, cost_per_share=total_cost / shares))

    def _apply_sell(
        self,
        holdings: dict[str, TickerLedger],
        ticker: str,
        shares: float,
        price: float,
        fees: float,
    ) -> float:
        leg = self._ledger(holdings, ticker)
        if shares > leg.quantity + 1e-9:
            raise ValidationError(
                f"Cannot sell {shares} shares of {ticker}; only {leg.quantity}"
            )

        proceeds = shares * price - fees
        if self._method == "average":
            avg = leg.total_cost / leg.quantity if leg.quantity > 0 else 0.0
            cost_removed = avg * shares
            leg.quantity -= shares
            leg.total_cost -= cost_removed
            return proceeds - cost_removed

        remaining = shares
        cost_removed = 0.0
        new_lots: list[Lot] = []
        for lot in leg.lots:
            if remaining <= 0:
                new_lots.append(lot)
                continue
            if lot.quantity <= remaining:
                cost_removed += lot.quantity * lot.cost_per_share
                remaining -= lot.quantity
            else:
                cost_removed += remaining * lot.cost_per_share
                lot.quantity -= remaining
                new_lots.append(lot)
                remaining = 0.0
        leg.lots = new_lots
        leg.quantity -= shares
        leg.total_cost -= cost_removed
        return proceeds - cost_removed

    def _apply_split(
        self, holdings: dict[str, TickerLedger], ticker: str, ratio: float
    ) -> None:
        if ticker not in holdings:
            return
        leg = holdings[ticker]
        if leg.quantity <= 0:
            return
        leg.quantity *= ratio
        if self._method == "fifo":
            for lot in leg.lots:
                lot.quantity *= ratio
                lot.cost_per_share /= ratio
        # total_cost unchanged for average and fifo
