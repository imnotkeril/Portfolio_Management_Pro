"""Unit tests for risk engine, scenario engine, and light optimization paths."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.exceptions import InsufficientDataError
from core.optimization_engine.constraints import OptimizationConstraints
from core.optimization_engine.equal_weight import EqualWeightOptimizer
from core.optimization_engine.max_return import MaxReturnOptimizer
from core.optimization_engine.min_variance import MinVarianceOptimizer
from core.risk_engine.monte_carlo import MonteCarloSimulator, simulate_portfolio_paths
from core.risk_engine.stress_testing import (
    HISTORICAL_SCENARIOS,
    StressScenario,
    StressTester,
    apply_stress_scenario,
    get_historical_scenarios,
    get_scenario_by_name,
)
from core.risk_engine.var_calculator import (
    calculate_portfolio_var_covariance,
    calculate_var,
    calculate_var_all_methods,
    calculate_var_monte_carlo,
)
from core.scenario_engine.custom_scenarios import (
    CustomScenario,
    create_custom_scenario,
    validate_scenario,
)
from core.scenario_engine.historical_scenarios import (
    HISTORICAL_SCENARIOS as PORTFOLIO_SCENARIOS,
)
from core.scenario_engine.historical_scenarios import (
    get_all_scenarios,
)
from core.scenario_engine.historical_scenarios import (
    get_scenario_by_name as get_portfolio_scenario,
)
from core.scenario_engine.historical_scenarios import (
    get_scenarios_by_date_range,
)
from core.scenario_engine.scenario_chain import (
    apply_scenario_chain,
    create_scenario_chain,
)


@pytest.fixture
def daily_returns() -> pd.Series:
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0005, 0.01, 300))


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    np.random.seed(1)
    n = 120
    a = np.random.normal(0.0004, 0.012, n)
    b = np.random.normal(0.0003, 0.014, n)
    return pd.DataFrame({"AAA": a, "BBB": b})


class TestVarCalculator:
    def test_historical_var(self, daily_returns: pd.Series) -> None:
        v = calculate_var(daily_returns, 0.95, "historical")
        assert v < 0

    def test_parametric_var(self, daily_returns: pd.Series) -> None:
        v = calculate_var(daily_returns, 0.99, "parametric")
        assert v < 0

    def test_cornish_fisher(self, daily_returns: pd.Series) -> None:
        v = calculate_var(daily_returns, 0.95, "cornish_fisher")
        assert v < 0

    def test_monte_carlo_wrapper_points_to_dedicated_func(
        self, daily_returns: pd.Series
    ) -> None:
        with pytest.raises(ValueError, match="calculate_var_monte_carlo"):
            calculate_var(daily_returns, 0.95, "monte_carlo")

    def test_monte_carlo_var(self, daily_returns: pd.Series) -> None:
        v = calculate_var_monte_carlo(
            daily_returns, 0.95, num_simulations=2000, time_horizon=1, random_seed=3
        )
        assert v < 0

    def test_all_methods(self, daily_returns: pd.Series) -> None:
        out = calculate_var_all_methods(
            daily_returns, 0.95, num_simulations=2000, time_horizon=1
        )
        assert set(out.keys()) == {
            "historical",
            "parametric",
            "cornish_fisher",
            "monte_carlo",
        }

    def test_empty_returns(self) -> None:
        s = pd.Series([], dtype=float)
        with pytest.raises(InsufficientDataError):
            calculate_var(s, 0.95, "historical")

    def test_bad_confidence(self, daily_returns: pd.Series) -> None:
        with pytest.raises(ValueError):
            calculate_var(daily_returns, 0.5, "historical")

    def test_bad_method(self, daily_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_var(daily_returns, 0.95, "nope")

    def test_monte_carlo_validation(self, daily_returns: pd.Series) -> None:
        with pytest.raises(ValueError, match="at least 1,000"):
            calculate_var_monte_carlo(daily_returns, 0.95, num_simulations=100)
        with pytest.raises(ValueError, match="Time horizon"):
            calculate_var_monte_carlo(
                daily_returns, 0.95, num_simulations=2000, time_horizon=0
            )

    def test_portfolio_var_covariance(self, two_asset_returns: pd.DataFrame) -> None:
        w = np.array([0.6, 0.4])
        out = calculate_portfolio_var_covariance(
            two_asset_returns, w, confidence_level=0.95, time_horizon=5
        )
        assert "portfolio_var" in out
        assert "marginal_var" in out
        assert len(out["marginal_var"]) == 2

    def test_portfolio_var_empty(self) -> None:
        with pytest.raises(InsufficientDataError):
            calculate_portfolio_var_covariance(
                pd.DataFrame(), np.array([1.0]), confidence_level=0.95
            )


class TestMonteCarlo:
    def test_gbm_simulation(self, daily_returns: pd.Series) -> None:
        sim = MonteCarloSimulator(model="gbm", random_seed=4)
        res = sim.simulate(daily_returns, time_horizon=5, num_simulations=500)
        assert res.simulated_paths.shape == (500, 5)
        assert res.final_values.shape == (500,)
        assert 50.0 in res.percentiles

    def test_jump_diffusion(self, daily_returns: pd.Series) -> None:
        sim = MonteCarloSimulator(model="jump_diffusion", random_seed=5)
        res = sim.simulate(daily_returns, time_horizon=3, num_simulations=400)
        assert res.statistics["mean"] > 0

    def test_unknown_model(self, daily_returns: pd.Series) -> None:
        sim = MonteCarloSimulator(model="invalid")
        with pytest.raises(ValueError, match="Unknown model"):
            sim.simulate(daily_returns, time_horizon=2, num_simulations=100)

    def test_empty_returns(self) -> None:
        sim = MonteCarloSimulator(model="gbm")
        with pytest.raises(InsufficientDataError):
            sim.simulate(pd.Series([], dtype=float), time_horizon=2, num_simulations=10)

    def test_convenience_function(self, daily_returns: pd.Series) -> None:
        res = simulate_portfolio_paths(
            daily_returns,
            time_horizon=4,
            num_simulations=300,
            model="gbm",
            random_seed=6,
        )
        assert len(res.final_values) == 300


class TestStressTesting:
    def test_apply_scenario(self) -> None:
        scen = StressScenario(
            name="t",
            description="d",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 6, 1),
            market_shock=-0.1,
            sector_shocks={},
            asset_shocks={"X": -0.2},
        )
        res = apply_stress_scenario({"X": 0.5, "Y": 0.5}, 100_000.0, scen)
        assert res.scenario_name == "t"
        assert res.portfolio_impact_value != 0

    def test_tester_multiple(self) -> None:
        tester = StressTester()
        scen = list(HISTORICAL_SCENARIOS.values())[:2]
        out = tester.apply_multiple_scenarios({"AAPL": 1.0}, 50_000.0, scen)
        assert len(out) == 2

    def test_get_historical_and_lookup(self) -> None:
        alls = get_historical_scenarios()
        assert "2008_financial_crisis" in alls
        g = get_scenario_by_name("2008_financial_crisis")
        assert g is not None
        assert get_scenario_by_name("unknown_xyz") is None


class TestScenarioEnginePortfolio:
    def test_custom_scenario_roundtrip(self) -> None:
        cs = create_custom_scenario(
            "n",
            "d",
            -0.15,
            sector_impacts={"Tech": -0.2},
            asset_impacts={"AAPL": -0.1},
        )
        hs = cs.to_historical_scenario(date(2019, 1, 1), date(2019, 12, 31))
        assert hs.market_impact_pct == -0.15

    def test_validate_scenario(self) -> None:
        ok, err = validate_scenario(
            CustomScenario(
                name="x",
                description="y",
                market_impact_pct=-0.1,
                sector_impacts={},
                asset_impacts={},
            )
        )
        assert ok is True
        assert err is None

        bad = CustomScenario(
            name="",
            description="y",
            market_impact_pct=-0.1,
            sector_impacts={},
            asset_impacts={},
        )
        ok2, err2 = validate_scenario(bad)
        assert ok2 is False

    def test_portfolio_historical_registry(self) -> None:
        assert len(get_all_scenarios()) >= 1
        s = get_portfolio_scenario("2008_financial_crisis")
        assert s is not None
        dr = get_scenarios_by_date_range(date(2008, 1, 1), date(2009, 12, 31))
        assert isinstance(dr, list)
        assert len(dr) >= 1

    def test_scenario_chain(self) -> None:
        a = PORTFOLIO_SCENARIOS["2008_financial_crisis"]
        b = PORTFOLIO_SCENARIOS["2020_covid_crash"]
        chain = create_scenario_chain("c", "d", [a, b])
        out = apply_scenario_chain({"MSFT": 1.0}, 10_000.0, chain)
        assert out["num_scenarios"] == 2
        assert "cumulative_impact_pct" in out


class TestEqualWeightOptimizer:
    def test_optimize(self, two_asset_returns: pd.DataFrame) -> None:
        opt = EqualWeightOptimizer(two_asset_returns)
        res = opt.optimize()
        assert res.success
        assert len(res.weights) == 2
        assert pytest.approx(res.weights.sum(), abs=1e-6) == 1.0


class TestScipyOptimizers:
    def test_min_variance(self, two_asset_returns: pd.DataFrame) -> None:
        opt = MinVarianceOptimizer(two_asset_returns)
        res = opt.optimize(covariance_method="sample")
        assert res.success
        assert len(res.weights) == 2

    def test_max_return(self, two_asset_returns: pd.DataFrame) -> None:
        opt = MaxReturnOptimizer(two_asset_returns)
        res = opt.optimize()
        assert res.success


class TestOptimizationConstraints:
    def test_build_default_bounds(self, two_asset_returns: pd.DataFrame) -> None:
        oc = OptimizationConstraints(tickers=list(two_asset_returns.columns))
        oc.set_weight_bounds(min_weight=0.0, max_weight=1.0)
        lo, hi = oc.get_weight_bounds_array()
        assert len(lo) == 2
        assert all(h <= 1.0 for h in hi)
