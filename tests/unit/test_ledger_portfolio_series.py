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


def _withdrawal(d: date, amount: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="WITHDRAWAL",
        ticker="CASH",
        shares=amount,
        price=1.0,
    )


def _flat_prices(days: int, price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    return pd.DataFrame({"AAPL": [price] * days}, index=dates)


def _build_returns(txs, prices):
    builder = LedgerPortfolioSeriesBuilder()
    returns, values, _ = builder.build_returns(
        txs,
        date(2024, 1, 2),
        prices.index[-1].date(),
        "fifo",
        lambda _t, _s, _e: prices,
    )
    return returns, values


def test_deposit_is_not_counted_as_return() -> None:
    """A large deposit on flat prices must produce ~0% return, not +900%."""
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _deposit(date(2024, 1, 5), 90_000),
    ]
    returns, values = _build_returns(txs, _flat_prices(6))
    assert float(values.iloc[-1]) == 100_000.0
    assert returns.abs().max() < 1e-9
    assert float((1 + returns).prod() - 1) < 1e-9


def test_withdrawal_is_not_counted_as_loss() -> None:
    txs = [
        _deposit(date(2024, 1, 2), 20_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _withdrawal(date(2024, 1, 5), 5_000),
    ]
    returns, _ = _build_returns(txs, _flat_prices(6))
    assert returns.abs().max() < 1e-9


def test_return_after_deposit_uses_start_of_day_capital() -> None:
    """Deposit on the day the stock rises 10%: half of capital is idle cash."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"AAPL": [100.0, 100.0, 110.0]}, index=dates)
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _deposit(dates[2].date(), 10_000),
    ]
    returns, _ = _build_returns(txs, prices)
    # (11_000 + 10_000) / (10_000 + 10_000) - 1 = 5%
    assert abs(float(returns.iloc[-1]) - 0.05) < 1e-9


def test_flow_on_non_trading_day_applies_to_next_trading_day() -> None:
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _deposit(date(2024, 1, 6), 50_000),  # Saturday
    ]
    returns, _ = _build_returns(txs, _flat_prices(6))
    assert returns.abs().max() < 1e-9


def test_time_weighted_nav_ignores_deposit() -> None:
    from services.ledger_portfolio_series import time_weighted_nav

    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _deposit(date(2024, 1, 5), 90_000),
    ]
    returns, values = _build_returns(txs, _flat_prices(6))
    nav = time_weighted_nav(values, returns)
    assert len(nav) == len(values)
    assert float(nav.iloc[0]) == 10_000.0
    assert abs(float(nav.iloc[-1]) - 10_000.0) < 1e-6
    # engine formula: total return from the NAV path is 0%, not +900%
    assert abs(float(nav.iloc[-1]) / float(nav.iloc[0]) - 1) < 1e-9


def _split(d: date, ticker: str, ratio: float) -> Transaction:
    return Transaction(
        transaction_date=d,
        transaction_type="SPLIT",
        ticker=ticker,
        shares=1.0,
        price=1.0,
        split_ratio=ratio,
    )


def test_split_does_not_create_a_jump_with_split_adjusted_prices() -> None:
    """Prices are split-adjusted back in time, so the ledger must not multiply
    shares a second time on the split date (that read as an N-fold gain)."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    # Adjusted history: the 2:1 split on day 4 makes the whole series 50.
    prices = pd.DataFrame({"AAPL": [50.0] * 6}, index=dates)
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _split(dates[3].date(), "AAPL", 2.0),
    ]
    returns, values = _build_returns(txs, prices)
    assert float(values.iloc[0]) == 10_000.0
    assert returns.abs().max() < 1e-9
    assert abs(float(values.iloc[-1]) - 10_000.0) < 1e-6


def test_split_after_window_end_still_adjusts_earlier_holdings() -> None:
    """A split dated after end_date is already baked into adjusted prices."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"AAPL": [50.0] * 4}, index=dates)
    txs = [
        _deposit(date(2024, 1, 2), 10_000),
        _buy(date(2024, 1, 2), "AAPL", 100, 100),
        _split(date(2024, 6, 3), "AAPL", 2.0),
    ]
    returns, values = _build_returns(txs, prices)
    assert abs(float(values.iloc[-1]) - 10_000.0) < 1e-6
    assert returns.abs().max() < 1e-9
