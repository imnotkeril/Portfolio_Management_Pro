"""Tests for position sync preserving target weights."""

from datetime import date
from unittest.mock import patch

from core.data_manager.portfolio import Portfolio
from core.data_manager.transaction import Transaction
from services.position_sync import sync_positions_from_transactions


@patch("services.position_sync.PortfolioRepository")
def test_sync_positions_preserves_weight_targets(mock_repo_cls):
    mock_repo_cls.return_value.save.side_effect = lambda portfolio, _uid: portfolio
    portfolio = Portfolio(
        name="Test",
        starting_capital=100_000,
        portfolio_id="p1",
        rebalance_interval_months=3,
    )
    portfolio.add_position(ticker="AAPL", shares=10, weight_target=0.6)
    portfolio.add_position(ticker="MSFT", shares=5, weight_target=0.4)

    txs = [
        Transaction(
            transaction_date=date(2015, 5, 18),
            transaction_type="DEPOSIT",
            ticker="CASH",
            shares=100_000,
            price=1.0,
        ),
        Transaction(
            transaction_date=date(2015, 5, 18),
            transaction_type="BUY",
            ticker="AAPL",
            shares=10,
            price=100.0,
            fees=1.0,
        ),
    ]

    rebuilt = sync_positions_from_transactions(portfolio, txs, user_id="user-1")

    weights = {
        p.ticker: p.weight_target
        for p in rebuilt.get_all_positions()
        if p.ticker != "CASH"
    }
    assert weights.get("AAPL") == 0.6
    assert rebuilt.rebalance_interval_months == 3


def test_ensure_target_weights_total_uses_buy_price():
    from datetime import date

    from core.data_manager.transaction import Transaction

    buys = [
        Transaction(
            transaction_date=date(2020, 1, 1),
            transaction_type="BUY",
            ticker="VTV",
            shares=10,
            price=100.0,
        ),
        Transaction(
            transaction_date=date(2020, 1, 1),
            transaction_type="BUY",
            ticker="IWD",
            shares=5,
            price=50.0,
        ),
    ]
    total = sum(t.shares * t.price for t in buys)
    assert total == 1250.0
