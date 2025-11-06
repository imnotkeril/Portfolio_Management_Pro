"""
Historical scenario definitions for portfolio analysis.

This module defines 25+ historical market scenarios that can be applied
to portfolios for stress testing and scenario analysis.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional


@dataclass
class HistoricalScenario:
    """
    Historical market scenario definition.

    Represents a specific historical market event with date ranges,
    market impacts, and sector-specific effects.
    """

    name: str
    description: str
    start_date: date
    end_date: date
    market_impact_pct: float  # Overall market impact (%)
    sector_impacts: Dict[str, float]  # Sector -> impact %
    asset_impacts: Dict[str, float]  # Ticker -> impact %
    volatility_spike: Optional[float] = None  # VIX-like spike
    recovery_period_days: Optional[int] = None  # Estimated recovery time


# Historical scenarios database
HISTORICAL_SCENARIOS: Dict[str, HistoricalScenario] = {
    "2008_financial_crisis": HistoricalScenario(
        name="2008 Financial Crisis",
        description=(
            "Global financial crisis following Lehman Brothers collapse. "
            "S&P 500 dropped ~37% from peak to trough."
        ),
        start_date=date(2008, 9, 15),
        end_date=date(2009, 3, 9),
        market_impact_pct=-0.37,
        sector_impacts={
            "Financials": -0.55,
            "Real Estate": -0.45,
            "Consumer Discretionary": -0.40,
            "Industrials": -0.35,
            "Technology": -0.30,
            "Healthcare": -0.25,
            "Consumer Staples": -0.20,
            "Utilities": -0.15,
        },
        asset_impacts={},
        volatility_spike=80.0,
        recovery_period_days=180,
    ),
    "2020_covid_crash": HistoricalScenario(
        name="2020 COVID-19 Crash",
        description=(
            "Rapid market crash due to COVID-19 pandemic. "
            "Fastest 30% decline in history."
        ),
        start_date=date(2020, 2, 19),
        end_date=date(2020, 3, 23),
        market_impact_pct=-0.34,
        sector_impacts={
            "Energy": -0.50,
            "Financials": -0.35,
            "Industrials": -0.40,
            "Consumer Discretionary": -0.35,
            "Technology": -0.25,
            "Healthcare": -0.20,
            "Consumer Staples": -0.10,
        },
        asset_impacts={},
        volatility_spike=82.0,
        recovery_period_days=120,
    ),
    "2000_dotcom_bust": HistoricalScenario(
        name="2000 Dot-com Bust",
        description="Technology bubble burst. NASDAQ fell ~78%, "
        "S&P 500 fell ~49% over 2.5 years.",
        start_date=date(2000, 3, 24),
        end_date=date(2002, 10, 9),
        market_impact_pct=-0.49,
        sector_impacts={
            "Technology": -0.78,
            "Telecommunications": -0.60,
            "Consumer Discretionary": -0.35,
            "Financials": -0.25,
            "Healthcare": -0.20,
        },
        asset_impacts={},
        volatility_spike=45.0,
        recovery_period_days=900,
    ),
    "1987_black_monday": HistoricalScenario(
        name="1987 Black Monday",
        description="Largest one-day market crash in history. "
        "S&P 500 dropped ~20% in one day.",
        start_date=date(1987, 10, 19),
        end_date=date(1987, 10, 19),
        market_impact_pct=-0.20,
        sector_impacts={},
        asset_impacts={},
        volatility_spike=150.0,
        recovery_period_days=2,
    ),
    "1997_asian_crisis": HistoricalScenario(
        name="1997 Asian Financial Crisis",
        description="Currency crisis in Southeast Asia spreading globally.",
        start_date=date(1997, 7, 2),
        end_date=date(1998, 8, 31),
        market_impact_pct=-0.19,
        sector_impacts={
            "Financials": -0.30,
            "Emerging Markets": -0.50,
        },
        asset_impacts={},
        volatility_spike=35.0,
        recovery_period_days=180,
    ),
    "1998_russian_default": HistoricalScenario(
        name="1998 Russian Default / LTCM",
        description="Russian debt default and Long-Term Capital Management collapse.",
        start_date=date(1998, 8, 17),
        end_date=date(1998, 10, 8),
        market_impact_pct=-0.19,
        sector_impacts={
            "Financials": -0.35,
            "Emerging Markets": -0.40,
        },
        asset_impacts={},
        volatility_spike=45.0,
        recovery_period_days=60,
    ),
    "2010_flash_crash": HistoricalScenario(
        name="2010 Flash Crash",
        description="Rapid market decline and recovery within minutes "
        "due to algorithmic trading.",
        start_date=date(2010, 5, 6),
        end_date=date(2010, 5, 6),
        market_impact_pct=-0.09,
        sector_impacts={},
        asset_impacts={},
        volatility_spike=25.0,
        recovery_period_days=1,
    ),
    "2011_european_debt_crisis": HistoricalScenario(
        name="2011 European Debt Crisis",
        description="European sovereign debt crisis.",
        start_date=date(2011, 7, 1),
        end_date=date(2011, 10, 3),
        market_impact_pct=-0.19,
        sector_impacts={
            "Financials": -0.30,
            "Industrials": -0.25,
            "Energy": -0.20,
        },
        asset_impacts={},
        volatility_spike=48.0,
        recovery_period_days=90,
    ),
    "2013_taper_tantrum": HistoricalScenario(
        name="2013 Taper Tantrum",
        description="Market volatility following Fed taper announcement.",
        start_date=date(2013, 5, 22),
        end_date=date(2013, 6, 24),
        market_impact_pct=-0.06,
        sector_impacts={
            "Emerging Markets": -0.15,
            "Bonds": -0.05,
        },
        asset_impacts={},
        volatility_spike=20.0,
        recovery_period_days=30,
    ),
    "2015_china_devaluation": HistoricalScenario(
        name="2015 China Devaluation",
        description="Chinese yuan devaluation causing global market volatility.",
        start_date=date(2015, 8, 11),
        end_date=date(2015, 8, 25),
        market_impact_pct=-0.11,
        sector_impacts={
            "Emerging Markets": -0.20,
            "Commodities": -0.15,
        },
        asset_impacts={},
        volatility_spike=40.0,
        recovery_period_days=60,
    ),
    "2015_oil_collapse": HistoricalScenario(
        name="2015 Oil Price Collapse",
        description="Oil prices collapse from $100+ to $30 per barrel.",
        start_date=date(2014, 6, 20),
        end_date=date(2016, 2, 11),
        market_impact_pct=-0.13,
        sector_impacts={
            "Energy": -0.50,
            "Oil & Gas": -0.55,
        },
        asset_impacts={},
        volatility_spike=30.0,
        recovery_period_days=180,
    ),
    "2016_brexit": HistoricalScenario(
        name="2016 Brexit",
        description="UK votes to leave EU, causing market uncertainty.",
        start_date=date(2016, 6, 23),
        end_date=date(2016, 6, 27),
        market_impact_pct=-0.05,
        sector_impacts={
            "Financials": -0.12,
            "UK Stocks": -0.15,
        },
        asset_impacts={},
        volatility_spike=26.0,
        recovery_period_days=5,
    ),
    "2018_q4_volatility": HistoricalScenario(
        name="2018 Q4 Volatility Spike",
        description="Trade war concerns and Fed policy uncertainty.",
        start_date=date(2018, 10, 1),
        end_date=date(2018, 12, 24),
        market_impact_pct=-0.20,
        sector_impacts={
            "Technology": -0.25,
            "Consumer Discretionary": -0.22,
        },
        asset_impacts={},
        volatility_spike=36.0,
        recovery_period_days=90,
    ),
    "2018_trade_war": HistoricalScenario(
        name="2018 Trade War",
        description="US-China trade tensions and tariffs.",
        start_date=date(2018, 3, 1),
        end_date=date(2018, 12, 31),
        market_impact_pct=-0.06,
        sector_impacts={
            "Technology": -0.10,
            "Industrials": -0.08,
            "Consumer Discretionary": -0.08,
        },
        asset_impacts={},
        volatility_spike=25.0,
        recovery_period_days=180,
    ),
    "2020_march_volatility": HistoricalScenario(
        name="2020 March Volatility Spike",
        description="Extreme volatility during COVID-19 uncertainty.",
        start_date=date(2020, 3, 9),
        end_date=date(2020, 3, 18),
        market_impact_pct=-0.15,
        sector_impacts={},
        asset_impacts={},
        volatility_spike=82.0,
        recovery_period_days=10,
    ),
    "2021_gamestop": HistoricalScenario(
        name="2021 GameStop / Meme Stock",
        description="Retail trading frenzy and short squeeze.",
        start_date=date(2021, 1, 13),
        end_date=date(2021, 1, 29),
        market_impact_pct=0.02,
        sector_impacts={
            "Retail": 0.15,
        },
        asset_impacts={
            "GME": 15.0,  # GameStop
            "AMC": 8.0,
        },
        volatility_spike=30.0,
        recovery_period_days=30,
    ),
    "2021_archegos": HistoricalScenario(
        name="2021 Archegos Collapse",
        description="Family office collapse causing massive stock sales.",
        start_date=date(2021, 3, 26),
        end_date=date(2021, 3, 29),
        market_impact_pct=-0.02,
        sector_impacts={
            "Media": -0.15,
            "Chinese Tech": -0.20,
        },
        asset_impacts={},
        volatility_spike=20.0,
        recovery_period_days=5,
    ),
    "2022_rate_hikes": HistoricalScenario(
        name="2022 Rate Hike Cycle",
        description="Aggressive Fed rate hikes to combat inflation.",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 12, 31),
        market_impact_pct=-0.19,
        sector_impacts={
            "Technology": -0.33,
            "Growth": -0.30,
            "Consumer Discretionary": -0.25,
            "Financials": -0.10,
            "Energy": 0.50,
        },
        asset_impacts={},
        volatility_spike=32.0,
        recovery_period_days=365,
    ),
    "2022_russia_ukraine": HistoricalScenario(
        name="2022 Russia-Ukraine War",
        description="Geopolitical conflict causing market volatility.",
        start_date=date(2022, 2, 24),
        end_date=date(2022, 3, 8),
        market_impact_pct=-0.08,
        sector_impacts={
            "Energy": 0.20,
            "Defense": 0.15,
            "European Stocks": -0.15,
        },
        asset_impacts={},
        volatility_spike=38.0,
        recovery_period_days=30,
    ),
    "2022_energy_crisis": HistoricalScenario(
        name="2022 Energy Crisis",
        description="Energy price spikes due to supply constraints.",
        start_date=date(2022, 3, 1),
        end_date=date(2022, 6, 30),
        market_impact_pct=-0.05,
        sector_impacts={
            "Energy": 0.40,
            "Utilities": -0.10,
            "Consumer Discretionary": -0.08,
        },
        asset_impacts={},
        volatility_spike=28.0,
        recovery_period_days=90,
    ),
    "2023_banking_crisis": HistoricalScenario(
        name="2023 Banking Crisis (SVB)",
        description="Regional banking crisis following SVB collapse.",
        start_date=date(2023, 3, 10),
        end_date=date(2023, 3, 20),
        market_impact_pct=-0.05,
        sector_impacts={
            "Financials": -0.15,
            "Regional Banks": -0.25,
        },
        asset_impacts={},
        volatility_spike=26.0,
        recovery_period_days=30,
    ),
    "2023_japanese_intervention": HistoricalScenario(
        name="2023 Japanese Intervention",
        description="Bank of Japan intervention in currency markets.",
        start_date=date(2022, 9, 22),
        end_date=date(2022, 10, 24),
        market_impact_pct=-0.03,
        sector_impacts={
            "Japanese Stocks": -0.10,
        },
        asset_impacts={},
        volatility_spike=20.0,
        recovery_period_days=15,
    ),
    "ftx_collapse": HistoricalScenario(
        name="FTX Collapse (Crypto Contagion)",
        description="FTX exchange collapse causing crypto market crash.",
        start_date=date(2022, 11, 8),
        end_date=date(2022, 11, 11),
        market_impact_pct=-0.02,
        sector_impacts={
            "Crypto": -0.70,
            "Crypto-Related": -0.30,
        },
        asset_impacts={},
        volatility_spike=25.0,
        recovery_period_days=60,
    ),
    "tech_selloff_2022": HistoricalScenario(
        name="Tech Selloff 2022",
        description="Technology sector selloff due to rate hikes.",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 12, 31),
        market_impact_pct=-0.15,
        sector_impacts={
            "Technology": -0.33,
            "Growth": -0.30,
        },
        asset_impacts={},
        volatility_spike=30.0,
        recovery_period_days=365,
    ),
    "bond_rout_2022": HistoricalScenario(
        name="Bond Rout 2022",
        description="Worst bond market performance in decades.",
        start_date=date(2022, 1, 1),
        end_date=date(2022, 10, 31),
        market_impact_pct=-0.15,
        sector_impacts={
            "Bonds": -0.20,
            "Fixed Income": -0.18,
        },
        asset_impacts={},
        volatility_spike=25.0,
        recovery_period_days=180,
    ),
}


def get_all_scenarios() -> Dict[str, HistoricalScenario]:
    """
    Get all available historical scenarios.

    Returns:
        Dictionary mapping scenario keys to HistoricalScenario objects
    """
    return HISTORICAL_SCENARIOS.copy()


def get_scenario_by_name(name: str) -> Optional[HistoricalScenario]:
    """
    Get historical scenario by name (case-insensitive).

    Args:
        name: Scenario name or key

    Returns:
        HistoricalScenario if found, None otherwise
    """
    # Try exact key match first
    if name in HISTORICAL_SCENARIOS:
        return HISTORICAL_SCENARIOS[name]

    # Try case-insensitive name match
    name_normalized = name.lower().replace(" ", "_").replace("-", "_")
    for key, scenario in HISTORICAL_SCENARIOS.items():
        if (
            key.lower() == name_normalized
            or scenario.name.lower() == name.lower()
        ):
            return scenario

    return None


def get_scenarios_by_date_range(
    start_date: date, end_date: date
) -> List[HistoricalScenario]:
    """
    Get scenarios that occurred within a date range.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of HistoricalScenario objects
    """
    scenarios = []
    for scenario in HISTORICAL_SCENARIOS.values():
        if scenario.start_date <= end_date and scenario.end_date >= start_date:
            scenarios.append(scenario)
    return scenarios

