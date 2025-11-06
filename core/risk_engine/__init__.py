"""Risk Engine module for advanced risk analysis."""

from core.risk_engine.var_calculator import (
    calculate_var,
    calculate_var_monte_carlo,
)
from core.risk_engine.monte_carlo import (
    MonteCarloSimulator,
    simulate_portfolio_paths,
)
from core.risk_engine.stress_testing import (
    StressTester,
    apply_stress_scenario,
)

__all__ = [
    "calculate_var",
    "calculate_var_monte_carlo",
    "MonteCarloSimulator",
    "simulate_portfolio_paths",
    "StressTester",
    "apply_stress_scenario",
]

