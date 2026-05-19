"""Build daily portfolio value/return series by replaying the transaction ledger."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

import pandas as pd

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_sort import sort_transactions
from services.cost_basis import CostBasisCalculator

logger = logging.getLogger(__name__)

PriceFetcher = Callable[[list[str], date, date], pd.DataFrame]


def equity_tickers_from_transactions(transactions: list[Transaction]) -> list[str]:
    """Tickers that can appear in holdings (excludes CASH flows)."""
    tickers: set[str] = set()
    for tx in transactions:
        if tx.ticker == "CASH":
            continue
        if tx.transaction_type in ("BUY", "SELL", "DIVIDEND", "SPLIT"):
            tickers.add(tx.ticker.upper())
    return sorted(tickers)


def first_transaction_date(transactions: list[Transaction]) -> date | None:
    if not transactions:
        return None
    return min(tx.transaction_date for tx in transactions)


class LedgerPortfolioSeriesBuilder:
    """Replay transactions on each business day and mark-to-market holdings + cash."""

    def build_values(
        self,
        transactions: list[Transaction],
        start_date: date,
        end_date: date,
        cost_basis_method: str,
        price_fetcher: PriceFetcher,
    ) -> tuple[pd.Series, date]:
        """
        Daily portfolio market value (cash at face value + equities).

        Returns:
            (values indexed by date, effective_start_date)
        """
        sorted_txs = sort_transactions(
            [t for t in transactions if t.transaction_date <= end_date]
        )
        if not sorted_txs:
            return pd.Series(dtype=float), start_date

        inception = first_transaction_date(sorted_txs)
        if inception is None:
            return pd.Series(dtype=float), start_date

        effective_start = max(start_date, inception)
        if effective_start > end_date:
            return pd.Series(dtype=float), effective_start

        tickers = equity_tickers_from_transactions(sorted_txs)
        if not tickers:
            return pd.Series(dtype=float), effective_start

        prices = price_fetcher(tickers, effective_start, end_date)
        if prices.empty:
            return pd.Series(dtype=float), effective_start

        prices = prices.sort_index().ffill().bfill()
        try:
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
        except Exception:
            prices.index = pd.to_datetime(prices.index)

        calc = CostBasisCalculator(method=cost_basis_method)
        tx_idx = 0
        n_tx = len(sorted_txs)

        values: dict[pd.Timestamp, float] = {}
        for day in prices.index:
            day_date = day.date() if hasattr(day, "date") else day
            while tx_idx < n_tx and sorted_txs[tx_idx].transaction_date <= day_date:
                calc.apply(sorted_txs[tx_idx])
                tx_idx += 1

            quantities, cash = calc.snapshot_quantities()
            total = float(cash)
            for ticker, shares in quantities.items():
                if ticker not in prices.columns:
                    continue
                px = float(prices.loc[day, ticker])
                if pd.notna(px):
                    total += shares * px

            if total > 0:
                values[pd.Timestamp(day)] = total

        series = pd.Series(values, dtype=float).sort_index()
        if series.index.has_duplicates:
            series = series[~series.index.duplicated(keep="last")]
        return series, effective_start

    def build_returns(
        self,
        transactions: list[Transaction],
        start_date: date,
        end_date: date,
        cost_basis_method: str,
        price_fetcher: PriceFetcher,
    ) -> tuple[pd.Series, pd.Series, date]:
        """Daily simple returns and values from ledger replay."""
        values, effective_start = self.build_values(
            transactions, start_date, end_date, cost_basis_method, price_fetcher
        )
        if len(values) < 2:
            return pd.Series(dtype=float), values, effective_start

        returns = values.pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)
        try:
            returns.index = returns.index.tz_localize(None)
        except Exception:
            pass
        return returns, values, effective_start


class SnapshotPosition:
    """Minimal position row for analytics (ledger as-of date)."""

    __slots__ = ("ticker", "shares", "purchase_price", "weight_target")

    def __init__(
        self,
        ticker: str,
        shares: float,
        purchase_price: float | None = None,
    ) -> None:
        self.ticker = ticker
        self.shares = shares
        self.purchase_price = purchase_price
        self.weight_target = None


def positions_snapshot_at(
    transactions: list[Transaction],
    as_of: date,
    cost_basis_method: str,
) -> list[SnapshotPosition]:
    """Build position-like rows for analytics as of a calendar date."""
    calc = CostBasisCalculator(method=cost_basis_method)
    for tx in sort_transactions(transactions):
        if tx.transaction_date > as_of:
            break
        calc.apply(tx)

    rows: list[SnapshotPosition] = [
        SnapshotPosition(ticker, shares, avg)
        for ticker, shares, avg in calc.snapshot_equity_rows()
    ]
    _, cash = calc.snapshot_quantities()
    if cash > 1e-9:
        rows.append(SnapshotPosition("CASH", cash, 1.0))
    return rows
