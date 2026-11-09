"""Rebalance planning with high cash must deploy excess, not only rotate."""

from datetime import date
from unittest.mock import MagicMock

from services.rebalance_service import (
    PlannedTrade,
    _allocate_target_shares,
    _finalize_rebalance_trades,
    _projected_cash_weight,
    _resolve_execution_date,
    _sweep_excess_cash,
)


def test_allocate_target_shares_spends_most_of_stock_budget() -> None:
    prices = {"AAA": 100.0, "BBB": 200.0}
    weights = {"AAA": 0.5, "BBB": 0.5}
    stock_budget = 95_000.0
    shares = _allocate_target_shares(stock_budget, weights, 1.0, prices)
    invested = shares["AAA"] * 100.0 + shares["BBB"] * 200.0
    assert invested >= stock_budget * 0.98
    assert invested <= stock_budget


def test_resolve_execution_date_falls_back_before_today() -> None:
    """Catch-up on calendar today: use last session when today's bar is missing."""
    import services.rebalance_service as mod

    today = date.today()
    original = mod._prices_on_trading_day

    def fake(_ds: object, tickers: list[str], on_date: date) -> dict[str, float] | None:
        if on_date == today:
            return None
        if on_date.weekday() < 5:
            return {t: 42.0 for t in tickers}
        return None

    mod._prices_on_trading_day = fake
    try:
        exec_d, prices = _resolve_execution_date(
            MagicMock(), ["AAA"], today, max_shift_days=10
        )
        assert exec_d < today
        assert exec_d.weekday() < 5
        assert prices["AAA"] == 42.0
    finally:
        mod._prices_on_trading_day = original


def test_resolve_execution_date_shifts_forward() -> None:
    ds = MagicMock()
    calls: list = []

    def fake_prices(tickers: list[str], on_date) -> dict[str, float] | None:
        calls.append(on_date)
        if on_date.weekday() >= 5:
            return None
        return {t: 100.0 for t in tickers}

    import services.rebalance_service as mod

    original = mod._prices_on_trading_day
    mod._prices_on_trading_day = lambda _ds, tickers, d: fake_prices(tickers, d)
    try:
        scheduled = __import__("datetime").date(2024, 6, 1)
        exec_d, prices = _resolve_execution_date(
            ds, ["AAA"], scheduled, max_shift_days=5
        )
        assert exec_d.weekday() < 5
        assert prices["AAA"] == 100.0
        assert exec_d > scheduled
    finally:
        mod._prices_on_trading_day = original


def test_holistic_high_cash_rebalance_sells_then_buys() -> None:
    """One snapshot: overweight cash -> net BUYs after finalize; SELLs listed first."""
    prices = {"AAA": 100.0, "BBB": 100.0}
    stock_weights = {"AAA": 0.5, "BBB": 0.45}
    stock_sum = 0.95
    cash = 20_000.0
    shares = {"AAA": 400, "BBB": 400}
    total = cash + sum(shares[t] * prices[t] for t in shares)
    cash_target = 0.05
    investable = total * (1.0 - cash_target)
    target_shares = _allocate_target_shares(
        investable, stock_weights, stock_sum, prices
    )

    raw: list[PlannedTrade] = []
    for ticker in stock_weights:
        px = prices[ticker]
        cur = shares[ticker]
        tgt = target_shares[ticker]
        if cur == tgt:
            continue
        diff = tgt - cur
        raw.append(
            PlannedTrade(
                ticker=ticker,
                action="BUY" if diff > 0 else "SELL",
                shares=abs(diff),
                price=px,
                fees=1.0,
                current_shares=float(cur),
                target_shares=tgt,
                current_weight=cur * px / total,
                target_weight=stock_weights[ticker],
            )
        )

    target_weights = {**stock_weights, "CASH": cash_target}
    finalized = _finalize_rebalance_trades(
        raw,
        cash=cash,
        shares={k: float(v) for k, v in shares.items()},
        prices=prices,
        target_shares=target_shares,
        target_weights=target_weights,
        cash_target=cash_target,
        total_value=total,
    )

    sell_idx = [i for i, t in enumerate(finalized) if t.action == "SELL"]
    buy_idx = [i for i, t in enumerate(finalized) if t.action == "BUY"]
    if sell_idx and buy_idx:
        assert max(sell_idx) < min(buy_idx)

    buy_value = sum(t.shares * t.price for t in finalized if t.action == "BUY")
    assert buy_value >= (cash - total * cash_target) * 0.5

    projected = _projected_cash_weight(cash, shares, finalized, prices)
    assert projected <= cash_target + 0.03


def test_sweep_buys_past_lot_target_when_underweight() -> None:
    prices = {"AAA": 100.0}
    finalized: list[PlannedTrade] = []
    sim_cash = 50_000.0
    sim_shares = {"AAA": 500}
    target_weights = {"AAA": 0.95, "CASH": 0.05}
    sim_cash, _ = _sweep_excess_cash(
        finalized,
        sim_cash,
        sim_shares,
        prices,
        target_weights,
        0.05,
    )
    assert sim_cash < 10_000.0
    assert sum(t.shares for t in finalized if t.action == "BUY") > 0


def test_finalize_deploys_excess_cash_toward_target() -> None:
    prices = {"AAA": 50.0, "BBB": 50.0}
    cash = 10_000.0
    shares = {"AAA": 900, "BBB": 900}
    target_shares = {"AAA": 950, "BBB": 950}
    target_weights = {"AAA": 0.475, "BBB": 0.475, "CASH": 0.05}
    total = cash + 900 * 50 + 900 * 50

    trades = _finalize_rebalance_trades(
        [
            PlannedTrade(
                ticker="AAA",
                action="BUY",
                shares=50,
                price=50.0,
                fees=1.0,
                current_shares=900,
                target_shares=950,
                current_weight=0.45,
                target_weight=0.475,
            ),
        ],
        cash=cash,
        shares=shares,
        prices=prices,
        target_shares=target_shares,
        target_weights=target_weights,
        cash_target=0.05,
        total_value=total,
    )

    projected = _projected_cash_weight(cash, shares, trades, prices)
    assert projected <= 0.05 + 0.02
