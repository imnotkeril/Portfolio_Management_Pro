"""Unit tests for OptimizationConstraints and BaseOptimizer helpers."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import CalculationError
from core.optimization_engine.base import OptimizationResult
from core.optimization_engine.constraints import OptimizationConstraints
from core.optimization_engine.equal_weight import EqualWeightOptimizer

TICKERS = ["AAA", "BBB", "CASH"]


@pytest.fixture
def cons() -> OptimizationConstraints:
    return OptimizationConstraints(TICKERS)


@pytest.fixture
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0005, 0.01, size=(250, 3))
    data[:, 2] = 0.0001
    return pd.DataFrame(data, columns=TICKERS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_weight": -0.1},
        {"min_weight": 1.5},
        {"max_weight": -0.1},
        {"max_weight": 2.0},
        {"min_weight": 0.6, "max_weight": 0.4},
    ],
)
def test_weight_bounds_validation(cons, kwargs) -> None:
    with pytest.raises(ValueError):
        cons.set_weight_bounds(**kwargs)


def test_asset_bounds_validation(cons) -> None:
    with pytest.raises(ValueError):
        cons.set_asset_weight_bounds("ZZZ", 0.1, 0.2)
    with pytest.raises(ValueError):
        cons.set_asset_weight_bounds("AAA", min_weight=-1)
    with pytest.raises(ValueError):
        cons.set_asset_weight_bounds("AAA", max_weight=5)


def test_group_constraint_validation_and_callable(cons) -> None:
    with pytest.raises(ValueError):
        cons.set_group_constraint("g", [], 0.5)
    with pytest.raises(ValueError):
        cons.set_group_constraint("g", ["AAA"], 1.5)
    with pytest.raises(ValueError):
        cons.set_group_constraint("g", ["ZZZ"], 0.5)
    cons.set_group_constraint("tech", ["AAA", "BBB"], 0.6)
    (item,) = cons.get_group_constraints_list()
    assert item["type"] == "ineq"
    assert item["fun"](np.array([0.2, 0.2, 0.6])) == pytest.approx(0.2)
    assert item["fun"](np.array([0.5, 0.5, 0.0])) < 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_volatility": 0},
        {"max_var": 0.1},
        {"max_cvar": 0.1},
        {"max_beta": 0},
    ],
)
def test_risk_constraint_validation(cons, kwargs) -> None:
    with pytest.raises(ValueError):
        cons.set_risk_constraint(**kwargs)


def test_risk_constraint_ok(cons) -> None:
    cons.set_risk_constraint(
        max_volatility=0.2, max_var=-0.05, max_cvar=-0.08, max_beta=1.2
    )
    assert cons.max_volatility == 0.2 and cons.max_beta == 1.2


def test_turnover_and_cardinality_validation(cons) -> None:
    with pytest.raises(ValueError):
        cons.set_turnover_constraint(max_turnover=2)
    with pytest.raises(ValueError):
        cons.set_turnover_constraint(min_trade_size=-1)
    cons.set_turnover_constraint(0.5, 0.01)
    assert cons.max_turnover == 0.5
    with pytest.raises(ValueError):
        cons.set_cardinality_constraint(min_assets=0)
    with pytest.raises(ValueError):
        cons.set_cardinality_constraint(max_assets=9)
    with pytest.raises(ValueError):
        cons.set_cardinality_constraint(min_assets=3, max_assets=2)
    cons.set_cardinality_constraint(1, 2)
    assert (cons.min_assets, cons.max_assets) == (1, 2)


def test_cash_return_diversification(cons) -> None:
    with pytest.raises(ValueError):
        cons.set_cash_constraint(max_cash_weight=2)
    with pytest.raises(ValueError):
        cons.set_cash_constraint(min_cash_weight=-1)
    cons.set_cash_constraint(max_cash_weight=0.1)
    cons.set_return_constraint(min_return=0.05, target_return=0.08)
    with pytest.raises(ValueError):
        cons.set_diversification_regularization(-1)
    cons.set_diversification_regularization(0.5)
    d = cons.to_dict()
    assert d["max_cash_weight"] == 0.1
    assert d["target_return"] == 0.08
    assert d["diversification_lambda"] == 0.5


def test_weight_bounds_array(cons) -> None:
    cons.set_weight_bounds(min_weight=0.05, max_weight=0.7)
    cons.set_asset_weight_bounds("AAA", min_weight=0.2, max_weight=0.5)
    cons.set_cash_constraint(max_cash_weight=0.1)
    lo, hi = cons.get_weight_bounds_array()
    assert lo.tolist() == [0.2, 0.05, 0.05]
    assert hi.tolist() == [0.5, 0.7, 0.1]


def test_weight_bounds_array_allows_short_when_not_long_only(cons) -> None:
    cons.set_weight_bounds(long_only=False)
    lo, _ = cons.get_weight_bounds_array()
    assert np.isneginf(lo).all()


def test_optimization_result_helpers() -> None:
    res = OptimizationResult(weights=np.array([0.5, 0.5]), tickers=["A", "B"])
    assert res.metadata == {}
    assert res.get_weights_dict() == {"A": 0.5, "B": 0.5}
    assert res.to_dict()["weights"] == [0.5, 0.5]


def test_base_optimizer_init_validation() -> None:
    with pytest.raises(ValueError):
        EqualWeightOptimizer(pd.DataFrame())
    with pytest.raises(ValueError):
        EqualWeightOptimizer(pd.DataFrame({"A": [np.nan, np.nan]}))


def test_validate_weights(returns_df) -> None:
    opt = EqualWeightOptimizer(returns_df)
    assert opt.get_name() == "EqualWeightOptimizer"
    with pytest.raises(ValueError):
        opt._validate_weights(np.array([1.0]))
    assert opt._validate_weights(np.array([0.4, 0.3, 0.3]))
    assert not opt._validate_weights(np.array([1.2, -0.2, 0.0]))
    assert not opt._validate_weights(np.array([0.1, 0.1, 0.8]), min_weight=0.2)
    assert not opt._validate_weights(np.array([0.1, 0.1, 0.8]), max_weight=0.5)
    assert not opt._validate_weights(np.array([0.1, 0.1, 0.1]))


def test_normalize_weights(returns_df) -> None:
    opt = EqualWeightOptimizer(returns_df)
    with pytest.raises(CalculationError):
        opt._normalize_weights(np.zeros(3))
    plain = opt._normalize_weights(np.array([1.0, 1.0, 2.0]))
    assert plain.sum() == pytest.approx(1.0)

    c = OptimizationConstraints(TICKERS)
    c.set_cash_constraint(max_cash_weight=0.2)
    # cash at/above cap -> pinned to cap, others rescaled
    out = opt._normalize_weights(np.array([0.1, 0.3, 0.6]), c)
    assert out[2] == pytest.approx(0.2)
    assert out.sum() == pytest.approx(1.0)
    # all non-cash zero -> distributed equally
    out = opt._normalize_weights(np.array([0.0, 0.0, 1.0]), c)
    assert out[0] == pytest.approx(0.4) and out.sum() == pytest.approx(1.0)
    # cash below cap -> plain normalisation
    out = opt._normalize_weights(np.array([0.5, 0.45, 0.05]), c)
    assert out.sum() == pytest.approx(1.0) and out[2] == pytest.approx(0.05)

    # CASH not in portfolio
    opt2 = EqualWeightOptimizer(returns_df[["AAA", "BBB"]])
    out = opt2._normalize_weights(np.array([1.0, 3.0]), c)
    np.testing.assert_allclose(out, [0.25, 0.75])


def test_covariance_methods(returns_df) -> None:
    opt = EqualWeightOptimizer(returns_df)
    sample = opt._estimate_covariance_matrix("sample")
    shrink = opt._estimate_covariance_matrix("shrink", 0.5)
    lw = opt._estimate_covariance_matrix("ledoit_wolf")
    assert sample.shape == shrink.shape == lw.shape == (3, 3)
    # shrinkage reduces off-diagonals, keeps diagonal
    assert abs(shrink.iloc[0, 1]) <= abs(sample.iloc[0, 1])
    np.testing.assert_allclose(np.diag(shrink), np.diag(sample))
    with pytest.raises(ValueError):
        opt._estimate_covariance_matrix("bogus")


def test_covariance_ledoit_wolf_fallback(returns_df, monkeypatch) -> None:
    import core.optimization_engine.base as base

    monkeypatch.setattr(base, "SKLEARN_AVAILABLE", False)
    opt = EqualWeightOptimizer(returns_df)
    assert opt._estimate_covariance_matrix("ledoit_wolf").shape == (3, 3)


def test_build_constraints_from_dict(returns_df) -> None:
    opt = EqualWeightOptimizer(returns_df)
    assert opt._build_constraints(None).max_weight is None
    c = opt._build_constraints(
        {
            "min_weight": 0.0,
            "max_weight": 0.8,
            "weight_bounds": {"AAA": {"min": 0.1, "max": 0.5}, "BBB": (0.0, 0.4)},
            "group_constraints": {"g": {"tickers": ["AAA", "BBB"], "max_weight": 0.7}},
            "max_volatility": 0.3,
            "max_turnover": 0.5,
            "min_assets": 1,
            "max_assets": 3,
            "max_cash_weight": 0.2,
            "min_return": 0.01,
            "diversification_lambda": 0.1,
        }
    )
    assert c.weight_bounds["AAA"] == (0.1, 0.5)
    assert c.weight_bounds["BBB"] == (0.0, 0.4)
    assert "g" in c.group_constraints
    assert c.max_cash_weight == 0.2


def test_calculate_portfolio_metrics(returns_df) -> None:
    opt = EqualWeightOptimizer(returns_df)
    m = opt._calculate_portfolio_metrics(np.array([1 / 3] * 3))
    assert m["volatility"] > 0 and m["sharpe_ratio"] is not None
    res = opt.optimize()
    assert res.weights.sum() == pytest.approx(1.0)
