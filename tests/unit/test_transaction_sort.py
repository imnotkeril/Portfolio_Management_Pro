"""Transaction ordering for ledger replay and UI."""

from datetime import date

from core.data_manager.transaction import Transaction
from core.data_manager.transaction_sort import sort_transactions


def test_same_day_buy_before_dividend() -> None:
    txs = [
        Transaction(
            transaction_date=date(2025, 5, 19),
            transaction_type="DIVIDEND",
            ticker="CVX",
            shares=49,
            price=1.71,
            amount=83.79,
            reinvest=False,
        ),
        Transaction(
            transaction_date=date(2025, 5, 19),
            transaction_type="BUY",
            ticker="CVX",
            shares=49,
            price=134.14,
            amount=6572.86,
        ),
    ]
    ordered = sort_transactions(txs)
    assert ordered[0].transaction_type == "BUY"
    assert ordered[1].transaction_type == "DIVIDEND"
