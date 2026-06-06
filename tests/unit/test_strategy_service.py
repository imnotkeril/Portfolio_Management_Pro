"""Unit tests for Phase 5 strategy service."""

from unittest.mock import MagicMock

import pytest

from core.data_manager.portfolio import Portfolio
from core.exceptions import ValidationError
from services.rebalance_service import RebalanceService
from services.schemas import UpdateStrategyRequest
from services.strategy_service import StrategyService, extract_target_weights


def _portfolio_with_targets() -> Portfolio:
    p = Portfolio("Test", 100_000.0, portfolio_id="pid-1")
    p.add_position("AAPL", 10, weight_target=0.6)
    p.add_position("MSFT", 5, weight_target=0.3)
    p.add_position("CASH", 1000, weight_target=0.1)
    p.rebalance_interval_months = 3
    return p


def test_extract_target_weights_matches_sum_one() -> None:
    p = _portfolio_with_targets()
    weights = extract_target_weights(p)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["AAPL"] == pytest.approx(0.6, abs=1e-4)
    assert weights["CASH"] == pytest.approx(0.1, abs=1e-4)


def test_rebalance_service_target_weights_delegates() -> None:
    p = _portfolio_with_targets()
    svc = RebalanceService()
    assert svc._target_weights(p) == extract_target_weights(p)


def test_update_strategy_validates_sum() -> None:
    mock_ps = MagicMock()
    mock_ps.get_portfolio.return_value = _portfolio_with_targets()
    svc = StrategyService(portfolio_service=mock_ps)
    req = UpdateStrategyRequest(
        targets={"AAPL": 0.5, "MSFT": 0.3},
        replace_targets=True,
    )
    with pytest.raises(ValidationError, match="sum to 1.0"):
        svc.update_strategy("pid-1", req, user_id="u1")


def test_get_strategy_snapshot() -> None:
    mock_ps = MagicMock()
    mock_ps.get_portfolio.return_value = _portfolio_with_targets()
    snap = StrategyService(portfolio_service=mock_ps).get_strategy("pid-1", "u1")
    assert snap.rebalance_interval_months == 3
    assert snap.is_active is True
    assert "AAPL" in snap.targets_normalized
