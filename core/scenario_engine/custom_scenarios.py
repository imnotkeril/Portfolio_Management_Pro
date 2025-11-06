"""
Custom scenario builder for portfolio analysis.

This module allows users to create custom stress scenarios for testing
portfolio performance under hypothetical conditions.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

from core.scenario_engine.historical_scenarios import HistoricalScenario


@dataclass
class CustomScenario:
    """
    User-defined custom scenario for portfolio testing.

    Allows users to specify market shocks, sector impacts, and
    asset-specific impacts for hypothetical scenarios.
    """

    name: str
    description: str
    market_impact_pct: float  # Overall market impact (%)
    sector_impacts: Dict[str, float]  # Sector -> impact %
    asset_impacts: Dict[str, float]  # Ticker -> impact %
    volatility_spike: Optional[float] = None
    recovery_period_days: Optional[int] = None

    def to_historical_scenario(
        self, start_date: date, end_date: date
    ) -> HistoricalScenario:
        """
        Convert custom scenario to HistoricalScenario format.

        Args:
            start_date: Scenario start date
            end_date: Scenario end date

        Returns:
            HistoricalScenario object
        """
        return HistoricalScenario(
            name=self.name,
            description=self.description,
            start_date=start_date,
            end_date=end_date,
            market_impact_pct=self.market_impact_pct,
            sector_impacts=self.sector_impacts,
            asset_impacts=self.asset_impacts,
            volatility_spike=self.volatility_spike,
            recovery_period_days=self.recovery_period_days,
        )


def create_custom_scenario(
    name: str,
    description: str,
    market_impact_pct: float,
    sector_impacts: Optional[Dict[str, float]] = None,
    asset_impacts: Optional[Dict[str, float]] = None,
    volatility_spike: Optional[float] = None,
    recovery_period_days: Optional[int] = None,
) -> CustomScenario:
    """
    Create a custom scenario for portfolio testing.

    Args:
        name: Scenario name
        description: Scenario description
        market_impact_pct: Overall market impact percentage
        sector_impacts: Dictionary of sector -> impact %
        asset_impacts: Dictionary of ticker -> impact %
        volatility_spike: Volatility spike (VIX-like)
        recovery_period_days: Estimated recovery period

    Returns:
        CustomScenario object

    Example:
        >>> scenario = create_custom_scenario(
        ...     name="Tech Recession",
        ...     description="Hypothetical tech sector crash",
        ...     market_impact_pct=-0.20,
        ...     sector_impacts={"Technology": -0.40},
        ...     asset_impacts={"AAPL": -0.50, "MSFT": -0.45}
        ... )
    """
    return CustomScenario(
        name=name,
        description=description,
        market_impact_pct=market_impact_pct,
        sector_impacts=sector_impacts or {},
        asset_impacts=asset_impacts or {},
        volatility_spike=volatility_spike,
        recovery_period_days=recovery_period_days,
    )


def validate_scenario(scenario: CustomScenario) -> tuple[bool, Optional[str]]:
    """
    Validate custom scenario parameters.

    Args:
        scenario: CustomScenario to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not scenario.name or len(scenario.name.strip()) == 0:
        return False, "Scenario name cannot be empty"

    if abs(scenario.market_impact_pct) > 1.0:
        return False, "Market impact must be between -100% and +100%"

    for sector, impact in scenario.sector_impacts.items():
        if abs(impact) > 1.0:
            return (
                False,
                f"Sector impact for {sector} must be "
                "between -100% and +100%",
            )

    for ticker, impact in scenario.asset_impacts.items():
        if abs(impact) > 2.0:  # Allow larger asset-specific impacts
            return (
                False,
                f"Asset impact for {ticker} must be "
                "between -200% and +200%",
            )

    if scenario.volatility_spike is not None and scenario.volatility_spike < 0:
        return False, "Volatility spike must be non-negative"

    if (
        scenario.recovery_period_days is not None
        and scenario.recovery_period_days < 0
    ):
        return False, "Recovery period must be non-negative"

    return True, None

