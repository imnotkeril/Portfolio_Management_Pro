"""Unit tests for ledger TWR/MWR (no double-count of starting_capital + DEPOSIT)."""

from datetime import date

import pytest

from core.data_manager.transaction import Transaction
from services.performance_attribution import calculate_mwr, calculate_twr


def _deposit(d: date, amount: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="DEPOSIT",
        ticker="CASH",
        shares=amount,
        price=1.0,
    )


def test_twr_not_double_counted_with_deposit_and_starting_capital() -> None:
    """Inception DEPOSIT + starting_capital must not halve return (~-87% bug)."""
    txs = [_deposit(date(2023, 5, 19), 100_000.0)]
    twr = calculate_twr(txs, starting_capital=100_000.0, end_value=125_000.0)
    assert twr == pytest.approx(0.25, rel=1e-6)


def test_twr_uses_starting_capital_when_no_deposits() -> None:
    twr = calculate_twr([], starting_capital=100_000.0, end_value=125_000.0)
    assert twr == pytest.approx(0.25, rel=1e-6)


def test_mwr_positive_when_portfolio_grew() -> None:
    txs = [_deposit(date(2023, 5, 19), 100_000.0)]
    mwr = calculate_mwr(
        txs,
        starting_capital=100_000.0,
        end_value=125_000.0,
        end_date=date(2026, 5, 19),
    )
    assert mwr is not None
    assert mwr > 0
