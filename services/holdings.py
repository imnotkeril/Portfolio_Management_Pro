"""Build portfolio holdings from transaction ledger."""

from __future__ import annotations

from dataclasses import dataclass

from core.data_manager.transaction import Transaction
from services.cost_basis import CostBasisCalculator
from services.data_service import DataService


@dataclass
class HoldingRow:
    """Single holding with market data."""

    ticker: str
    quantity: float
    avg_cost: float
    market_price: float | None
    market_value: float | None
    cost_basis: float
    unrealized_pnl: float | None


class HoldingsBuilder:
    """Aggregate transactions into current holdings with live prices."""

    def __init__(
        self,
        data_service: DataService | None = None,
    ) -> None:
        self._data_service = data_service or DataService()

    def build(
        self,
        transactions: list[Transaction],
        cost_basis_method: str = "fifo",
    ) -> list[HoldingRow]:
        summary = CostBasisCalculator(method=cost_basis_method).summarize(transactions)
        tickers = [
            t
            for t, leg in summary.holdings.items()
            if leg.quantity > 1e-9 and t != "CASH"
        ]
        prices = self._data_service.get_latest_prices(tickers) if tickers else {}

        rows: list[HoldingRow] = []
        for ticker, leg in summary.holdings.items():
            if leg.quantity <= 1e-9 or ticker == "CASH":
                continue
            avg_cost = leg.total_cost / leg.quantity if leg.quantity > 0 else 0.0
            market_price = prices.get(ticker)
            market_value = (
                leg.quantity * market_price if market_price is not None else None
            )
            unrealized = (
                market_value - leg.total_cost if market_value is not None else None
            )
            rows.append(
                HoldingRow(
                    ticker=ticker,
                    quantity=leg.quantity,
                    avg_cost=avg_cost,
                    market_price=market_price,
                    market_value=market_value,
                    cost_basis=leg.total_cost,
                    unrealized_pnl=unrealized,
                )
            )
        return sorted(rows, key=lambda r: r.ticker)
