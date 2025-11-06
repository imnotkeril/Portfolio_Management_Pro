"""Scenario Engine module for scenario analysis."""

from core.scenario_engine.historical_scenarios import (
    get_all_scenarios,
    get_scenario_by_name,
    HistoricalScenario,
)
from core.scenario_engine.custom_scenarios import (
    CustomScenario,
    create_custom_scenario,
)
from core.scenario_engine.scenario_chain import (
    ScenarioChain,
    create_scenario_chain,
    apply_scenario_chain,
)

__all__ = [
    "get_all_scenarios",
    "get_scenario_by_name",
    "HistoricalScenario",
    "CustomScenario",
    "create_custom_scenario",
    "ScenarioChain",
    "create_scenario_chain",
    "apply_scenario_chain",
]

