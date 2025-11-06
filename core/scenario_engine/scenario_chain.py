"""
Scenario chain builder for sequential scenario analysis.

This module allows users to chain multiple scenarios together to
analyze cumulative portfolio impacts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from core.scenario_engine.historical_scenarios import HistoricalScenario


@dataclass
class ScenarioChain:
    """
    Chain of scenarios to apply sequentially.

    Allows analysis of cumulative portfolio impacts from multiple
    sequential market events.
    """

    name: str
    description: str
    scenarios: List[HistoricalScenario]  # Ordered list of scenarios
    cumulative_impact: Optional[float] = None  # Calculated cumulative impact


def create_scenario_chain(
    name: str,
    description: str,
    scenarios: List[HistoricalScenario],
) -> ScenarioChain:
    """
    Create a scenario chain from a list of scenarios.

    Args:
        name: Chain name
        description: Chain description
        scenarios: Ordered list of scenarios to apply

    Returns:
        ScenarioChain object

    Example:
        >>> crisis_2008 = get_scenario_by_name("2008_financial_crisis")
        >>> covid_2020 = get_scenario_by_name("2020_covid_crash")
        >>> chain = create_scenario_chain(
        ...     name="Double Crisis",
        ...     description="2008 + 2020 crashes",
        ...     scenarios=[crisis_2008, covid_2020]
        ... )
    """
    return ScenarioChain(
        name=name,
        description=description,
        scenarios=scenarios,
    )


def apply_scenario_chain(
    portfolio_positions: Dict[str, float],  # ticker -> weight
    current_portfolio_value: float,
    chain: ScenarioChain,
) -> Dict[str, any]:
    """
    Apply scenario chain to portfolio and calculate cumulative impact.

    Args:
        portfolio_positions: Dictionary of ticker -> weight
        current_portfolio_value: Current portfolio value
        chain: ScenarioChain to apply

    Returns:
        Dictionary with:
        - cumulative_impact_pct: Total portfolio impact %
        - cumulative_impact_value: Total portfolio impact in currency
        - scenario_results: List of individual scenario results
        - worst_scenario: Name of worst scenario
        - recovery_time_days: Total estimated recovery time
    """
    cumulative_impact = 0.0
    scenario_results = []
    portfolio_value = current_portfolio_value
    worst_scenario = None
    worst_impact = 0.0
    total_recovery_days = 0

    for scenario in chain.scenarios:
        # Calculate impact for this scenario
        scenario_impact = 0.0
        position_impacts = {}

        for ticker, weight in portfolio_positions.items():
            # Check for asset-specific impact
            if ticker in scenario.asset_impacts:
                shock = scenario.asset_impacts[ticker]
            # Check for sector-specific impact (simplified - would need sector mapping)
            # For now, use market impact as default
            else:
                shock = scenario.market_impact_pct

            # Calculate position impact
            position_impact = weight * shock
            position_impacts[ticker] = position_impact
            scenario_impact += position_impact

        # Update cumulative impact
        cumulative_impact += scenario_impact
        portfolio_value *= 1 + scenario_impact

        # Track worst scenario
        if scenario_impact < worst_impact:
            worst_impact = scenario_impact
            worst_scenario = scenario.name

        # Add recovery time
        if scenario.recovery_period_days:
            total_recovery_days += scenario.recovery_period_days

        # Store scenario result
        scenario_results.append({
            "scenario_name": scenario.name,
            "impact_pct": scenario_impact,
            "impact_value": portfolio_value * scenario_impact,
            "cumulative_impact_pct": cumulative_impact,
            "position_impacts": position_impacts,
        })

    return {
        "cumulative_impact_pct": cumulative_impact,
        "cumulative_impact_value": current_portfolio_value * cumulative_impact,
        "final_portfolio_value": portfolio_value,
        "scenario_results": scenario_results,
        "worst_scenario": worst_scenario,
        "worst_impact_pct": worst_impact,
        "total_recovery_days": total_recovery_days,
        "num_scenarios": len(chain.scenarios),
    }

