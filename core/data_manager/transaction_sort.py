"""Chronological ordering for ledger processing and API responses."""

from __future__ import annotations

from core.data_manager.transaction import Transaction

# Stable tie-break when several rows share the same calendar date.
_TYPE_ORDER = {
    "DEPOSIT": 0,
    "WITHDRAWAL": 1,
    "DIVIDEND": 2,
    "SPLIT": 3,
    "BUY": 4,
    "SELL": 5,
}


def transaction_sort_key(tx: Transaction) -> tuple:
    """Sort key: date, then deposit before buys on the same day."""
    return (
        tx.transaction_date,
        _TYPE_ORDER.get(tx.transaction_type, 99),
        tx.ticker,
        tx.id or "",
    )


def sort_transactions(transactions: list[Transaction]) -> list[Transaction]:
    """Return a new list sorted for ledger replay and UI."""
    return sorted(transactions, key=transaction_sort_key)
