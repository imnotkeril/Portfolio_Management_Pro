"""
Stress testing module for portfolio risk analysis.

This module provides historical and custom stress scenario testing
capabilities to assess portfolio performance under adverse conditions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date


@dataclass
class StressScenario:
    """Represents a stress scenario for portfolio testing."""

    name: str
    description: str
    start_date: date
    end_date: date
    market_shock: float  # Overall market shock percentage
    sector_shocks: Dict[str, float]  # Sector-specific shocks
    asset_shocks: Dict[str, float]  # Asset-specific shocks (ticker -> shock %)
    correlation_shift: Optional[float] = None  # Correlation increase factor


@dataclass
class StressTestResult:
    """Result of stress test application."""

    scenario_name: str
    portfolio_impact_pct: float  # Portfolio value change %
    portfolio_impact_value: float  # Portfolio value change in currency
    worst_position_impact: Tuple[str, float]  # (ticker, impact %)
    best_position_impact: Tuple[str, float]  # (ticker, impact %)
    recovery_time_days: Optional[int] = None  # Estimated recovery time
    details: Dict = None  # Additional details


class StressTester:
    """
    Stress testing engine for portfolio analysis.

    Applies historical or custom scenarios to portfolios and calculates
    the impact on portfolio value and positions.
    """

    def __init__(self):
        """Initialize stress tester."""
        pass

    def apply_scenario(
        self,
        portfolio_positions: Dict[str, float],  # ticker -> weight
        current_portfolio_value: float,
        scenario: StressScenario,
    ) -> StressTestResult:
        """
        Apply stress scenario to portfolio.

        Args:
            portfolio_positions: Dictionary of ticker -> weight
            current_portfolio_value: Current portfolio value
            scenario: StressScenario to apply

        Returns:
            StressTestResult with impact analysis
        """
        total_impact = 0.0
        position_impacts = {}

        # Calculate impact for each position
        for ticker, weight in portfolio_positions.items():
            # Check for asset-specific shock
            if ticker in scenario.asset_shocks:
                shock = scenario.asset_shocks[ticker]
            # Check for sector-specific shock (would need sector mapping)
            # For now, use market shock as default
            else:
                shock = scenario.market_shock

            # Calculate position impact
            position_impact = weight * shock
            position_impacts[ticker] = position_impact
            total_impact += position_impact

        # Find worst and best positions
        worst_ticker = min(
            position_impacts.items(), key=lambda x: x[1]
        )[0]
        worst_impact = position_impacts[worst_ticker]

        best_ticker = max(
            position_impacts.items(), key=lambda x: x[1]
        )[0]
        best_impact = position_impacts[best_ticker]

        # Calculate portfolio impact
        portfolio_impact_pct = total_impact
        portfolio_impact_value = current_portfolio_value * total_impact

        # Estimate recovery time (simplified - would need historical data)
        # Assume recovery time proportional to shock magnitude
        recovery_time_days = None
        if abs(shock) > 0.05:  # 5% threshold
            # Rough estimate: 1 day per 1% shock
            recovery_time_days = int(abs(shock) * 100)

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_impact_pct=portfolio_impact_pct,
            portfolio_impact_value=portfolio_impact_value,
            worst_position_impact=(worst_ticker, worst_impact),
            best_position_impact=(best_ticker, best_impact),
            recovery_time_days=recovery_time_days,
            details={
                "position_impacts": position_impacts,
                "market_shock": scenario.market_shock,
            },
        )

    def apply_multiple_scenarios(
        self,
        portfolio_positions: Dict[str, float],
        current_portfolio_value: float,
        scenarios: List[StressScenario],
    ) -> List[StressTestResult]:
        """
        Apply multiple stress scenarios to portfolio.

        Args:
            portfolio_positions: Dictionary of ticker -> weight
            current_portfolio_value: Current portfolio value
            scenarios: List of StressScenario objects

        Returns:
            List of StressTestResult objects
        """
        results = []
        for scenario in scenarios:
            result = self.apply_scenario(
                portfolio_positions, current_portfolio_value, scenario
            )
            results.append(result)
        return results


def apply_stress_scenario(
    portfolio_positions: Dict[str, float],
    current_portfolio_value: float,
    scenario: StressScenario,
) -> StressTestResult:
    """
    Convenience function to apply a stress scenario.

    Args:
        portfolio_positions: Dictionary of ticker -> weight
        current_portfolio_value: Current portfolio value
        scenario: StressScenario to apply

    Returns:
        StressTestResult with impact analysis
    """
    tester = StressTester()
    return tester.apply_scenario(
        portfolio_positions, current_portfolio_value, scenario
    )


# Historical stress scenarios definitions
# These are simplified representations - full implementation would
# use actual historical market data

HISTORICAL_SCENARIOS: Dict[str, StressScenario] = {
    "2008_financial_crisis": StressScenario(
        name="2008 Financial Crisis",
        description="Global financial crisis following Lehman Brothers collapse",
        start_date=date(2008, 9, 15),
        end_date=date(2009, 3, 9),
        market_shock=-0.37,  # S&P 500 dropped ~37%
        sector_shocks={
            "Financials": -0.55,
            "Real Estate": -0.45,
            "Consumer Discretionary": -0.40,
            "Industrials": -0.35,
            "Technology": -0.30,
            "Healthcare": -0.25,
            "Consumer Staples": -0.20,
            "Utilities": -0.15,
        },
        asset_shocks={},
    ),
    "2020_covid_crash": StressScenario(
        name="2020 COVID-19 Crash",
        description="Market crash due to COVID-19 pandemic",
        start_date=date(2020, 2, 19),
        end_date=date(2020, 3, 23),
        market_shock=-0.34,  # S&P 500 dropped ~34%
        sector_shocks={
            "Energy": -0.50,
            "Financials": -0.35,
            "Industrials": -0.40,
            "Consumer Discretionary": -0.35,
            "Technology": -0.25,
            "Healthcare": -0.20,
            "Consumer Staples": -0.10,
        },
        asset_shocks={},
    ),
    "2000_dotcom_bust": StressScenario(
        name="2000 Dot-com Bust",
        description="Technology bubble burst",
        start_date=date(2000, 3, 24),
        end_date=date(2002, 10, 9),
        market_shock=-0.49,  # NASDAQ dropped ~78%, S&P 500 ~49%
        sector_shocks={
            "Technology": -0.78,
            "Telecommunications": -0.60,
            "Consumer Discretionary": -0.35,
            "Financials": -0.25,
            "Healthcare": -0.20,
        },
        asset_shocks={},
    ),
    "1987_black_monday": StressScenario(
        name="1987 Black Monday",
        description="Largest one-day market crash in history",
        start_date=date(1987, 10, 19),
        end_date=date(1987, 10, 19),
        market_shock=-0.20,  # One-day drop ~20%
        sector_shocks={},
        asset_shocks={},
    ),
    "2011_european_debt_crisis": StressScenario(
        name="2011 European Debt Crisis",
        description="European sovereign debt crisis",
        start_date=date(2011, 7, 1),
        end_date=date(2011, 10, 3),
        market_shock=-0.19,
        sector_shocks={
            "Financials": -0.30,
            "Industrials": -0.25,
            "Energy": -0.20,
        },
        asset_shocks={},
    ),
    "2022_rate_hikes": StressScenario(
        name="2022 Rate Hike Cycle",
        description="Aggressive Fed rate hikes",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 12, 31),
        market_shock=-0.19,
        sector_shocks={
            "Technology": -0.33,
            "Growth": -0.30,
            "Consumer Discretionary": -0.25,
            "Financials": -0.10,
            "Energy": 0.50,
        },
        asset_shocks={},
    ),
    "2023_banking_crisis": StressScenario(
        name="2023 Banking Crisis (SVB)",
        description="Regional banking crisis following SVB collapse",
        start_date=date(2023, 3, 10),
        end_date=date(2023, 3, 20),
        market_shock=-0.05,
        sector_shocks={
            "Financials": -0.15,
            "Regional Banks": -0.25,
        },
        asset_shocks={},
    ),
}


def get_historical_scenarios() -> Dict[str, StressScenario]:
    """Get all available historical stress scenarios."""
    return HISTORICAL_SCENARIOS.copy()


def get_scenario_by_name(name: str) -> Optional[StressScenario]:
    """Get historical scenario by name."""
    # Try exact match first
    if name in HISTORICAL_SCENARIOS:
        return HISTORICAL_SCENARIOS[name]

    # Try case-insensitive match
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    for key, scenario in HISTORICAL_SCENARIOS.items():
        if key.lower() == name_lower:
            return scenario

    return None

