"""Unit tests for ledger-based daily portfolio series."""

from datetime import date

import pandas as pd

from core.data_manager.transaction import Transaction
from services.ledger_portfolio_series import (
    LedgerPortfolioSeriesBuilder,
    first_transaction_date,
)


def _deposit(d: date, amount: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="DEPOSIT",
        ticker="CASH",
        shares=amount,
        price=1.0,
    )


def _buy(d: date, ticker: str, shares: float, price: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="BUY",
        ticker=ticker,
        shares=shares,
        price=price,
    )


def test_first_transaction_date() -> None:
    txs = [
        _buy(date(2024, 2, 1), "AAPL", 10, 100),
        _deposit(date(2024, 1, 1), 1000),
    ]
    assert first_transaction_date(txs) == date(2024, 1, 1)


def test_build_values_grows_with_price() -> None:
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 10, 100),
    ]
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0]},
        index=dates,
    )

    def fetch(_tickers, _start, _end):
        return prices

    builder = LedgerPortfolioSeriesBuilder()
    values, eff_start = builder.build_values(
        txs, date(2024, 1, 2), date(2024, 1, 8), "fifo", fetch
    )
    assert eff_start == date(2024, 1, 2)
    assert len(values) >= 2
    assert float(values.iloc[-1]) > float(values.iloc[0])
