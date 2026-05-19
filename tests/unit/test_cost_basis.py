"""Unit tests for cost basis calculator."""

from datetime import date

import pytest

from core.data_manager.transaction import Transaction
from core.exceptions import ValidationError
from services.cost_basis import CostBasisCalculator


def _buy(d: date, ticker: str, shares: float, price: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="BUY",
        ticker=ticker,
        shares=shares,
        price=price,
    )


def _sell(d: date, ticker: str, shares: float, price: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="SELL",
        ticker=ticker,
        shares=shares,
        price=price,
    )


def test_fifo_partial_sell_realized_pnl() -> None:
    txs = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0),
        _sell(date(2024, 2, 1), "AAPL", 5, 120.0),
    ]
    summary = CostBasisCalculator(method="fifo").summarize(txs)
    assert summary.realized_pnl == pytest.approx(100.0)
    assert summary.holdings["AAPL"].quantity == pytest.approx(5.0)


def test_average_cost_sell() -> None:
    txs = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0),
        _buy(date(2024, 1, 15), "AAPL", 10, 110.0),
        _sell(date(2024, 2, 1), "AAPL", 10, 120.0),
    ]
    summary = CostBasisCalculator(method="average").summarize(txs)
    assert summary.realized_pnl == pytest.approx(150.0)


def test_dividend_reinvest_increases_quantity() -> None:
    txs = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0),
        Transaction(
            transaction_date=date(2024, 3, 1),
            transaction_type="DIVIDEND",
            ticker="AAPL",
            shares=2.0,
            price=100.0,
            reinvest=True,
        ),
    ]
    summary = CostBasisCalculator(method="fifo").summarize(txs)
    assert summary.holdings["AAPL"].quantity == pytest.approx(12.0)


def test_split_doubles_quantity_halves_cost() -> None:
    txs = [
        _buy(date(2024, 1, 1), "AAPL", 10, 100.0),
        Transaction(
            transaction_date=date(2024, 6, 1),
            transaction_type="SPLIT",
            ticker="AAPL",
            shares=1.0,
            price=1.0,
            split_ratio=2.0,
        ),
    ]
    summary = CostBasisCalculator(method="fifo").summarize(txs)
    leg = summary.holdings["AAPL"]
    assert leg.quantity == pytest.approx(20.0)
    assert leg.lots[0].cost_per_share == pytest.approx(50.0)


def test_deposit_and_withdrawal_cash_balance() -> None:
    txs = [
        Transaction(
            transaction_date=date(2024, 1, 1),
            transaction_type="DEPOSIT",
            ticker="CASH",
            shares=10_000.0,
            price=1.0,
        ),
        Transaction(
            transaction_date=date(2024, 1, 5),
            transaction_type="WITHDRAWAL",
            ticker="CASH",
            shares=3_000.0,
            price=1.0,
        ),
    ]
    summary = CostBasisCalculator(method="fifo").summarize(txs)
    assert summary.cash_balance == pytest.approx(7_000.0)


def test_sell_more_than_held_raises() -> None:
    txs = [_buy(date(2024, 1, 1), "AAPL", 5, 100.0)]
    with pytest.raises(ValidationError):
        CostBasisCalculator(method="fifo").summarize(
            txs + [_sell(date(2024, 2, 1), "AAPL", 10, 100.0)]
        )
