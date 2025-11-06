"""
Risk service for orchestrating advanced risk analysis.

This service coordinates risk calculations, Monte Carlo simulations,
and stress testing across the risk engine and scenario engine modules.
"""

import logging
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from core.risk_engine.var_calculator import calculate_var_all_methods
from core.risk_engine.monte_carlo import simulate_portfolio_paths
from core.risk_engine.stress_testing import (
    StressTester,
    apply_stress_scenario,
)
from core.scenario_engine.historical_scenarios import (
    get_all_scenarios as get_all_historical_scenarios,
    get_scenario_by_name,
    HistoricalScenario,
)
from core.scenario_engine.custom_scenarios import (
    CustomScenario,
    validate_scenario,
)
from core.scenario_engine.scenario_chain import (
    ScenarioChain,
    apply_scenario_chain,
)
from core.exceptions import InsufficientDataError, ValidationError
from services.portfolio_service import PortfolioService
from services.data_service import DataService
from services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class RiskService:
    """
    Service for orchestrating risk analysis operations.

    Coordinates VaR calculations, Monte Carlo simulations, stress testing,
    and scenario analysis.
    """

    def __init__(
        self,
        portfolio_service: Optional[PortfolioService] = None,
        data_service: Optional[DataService] = None,
        analytics_service: Optional[AnalyticsService] = None,
    ) -> None:
        """
        Initialize risk service.

        Args:
            portfolio_service: Optional portfolio service instance
            data_service: Optional data service instance
            analytics_service: Optional analytics service instance
        """
        self._portfolio_service = portfolio_service or PortfolioService()
        self._data_service = data_service or DataService()
        self._analytics_service = analytics_service or AnalyticsService()

    def calculate_var_analysis(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
        confidence_level: float = 0.95,
        include_monte_carlo: bool = True,
        num_simulations: int = 10000,
        time_horizon: int = 1,
    ) -> Dict[str, any]:
        """
        Calculate comprehensive VaR analysis for portfolio.

        Args:
            portfolio_id: Portfolio ID
            start_date: Start date for returns calculation
            end_date: End date for returns calculation
            confidence_level: Confidence level (0.90, 0.95, or 0.99)
            include_monte_carlo: Whether to include Monte Carlo VaR
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days

        Returns:
            Dictionary with VaR results for all methods
        """
        # Get portfolio returns
        returns = self._get_portfolio_returns(
            portfolio_id, start_date, end_date
        )

        # Calculate VaR using all methods
        var_results = calculate_var_all_methods(
            returns,
            confidence_level,
            num_simulations if include_monte_carlo else 1000,
            time_horizon,
        )

        return {
            "portfolio_id": portfolio_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "confidence_level": confidence_level,
            "var_results": var_results,
            "time_horizon": time_horizon,
        }

    def run_monte_carlo_simulation(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
        time_horizon: int,
        num_simulations: int = 10000,
        initial_value: float = 1.0,
        model: str = "gbm",
    ) -> Dict[str, any]:
        """
        Run Monte Carlo simulation for portfolio.

        Args:
            portfolio_id: Portfolio ID
            start_date: Start date for historical returns
            end_date: End date for historical returns
            time_horizon: Number of days to simulate
            num_simulations: Number of simulation paths
            initial_value: Starting portfolio value
            model: Model type ('gbm' or 'jump_diffusion')

        Returns:
            Dictionary with simulation results
        """
        # Get portfolio returns
        returns = self._get_portfolio_returns(
            portfolio_id, start_date, end_date
        )

        # Run simulation
        result = simulate_portfolio_paths(
            returns,
            time_horizon,
            num_simulations,
            initial_value,
            model,
        )

        return {
            "portfolio_id": portfolio_id,
            "time_horizon": time_horizon,
            "num_simulations": num_simulations,
            "model": model,
            "percentiles": result.percentiles,
            "statistics": result.statistics,
            "final_values": result.final_values.tolist(),
            "simulated_paths": (
                result.simulated_paths.tolist()
                if hasattr(result, "simulated_paths")
                else None
            ),
        }

    def run_stress_test(
        self,
        portfolio_id: str,
        scenario_names: List[str],
    ) -> List[Dict[str, any]]:
        """
        Run stress tests using historical scenarios.

        Args:
            portfolio_id: Portfolio ID
            scenario_names: List of scenario names to test

        Returns:
            List of stress test results
        """
        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Get positions
        positions = portfolio.get_all_positions()
        portfolio_positions = {
            pos.ticker: pos.weight_target or 0.0 for pos in positions
        }

        # Normalize weights
        total_weight = sum(portfolio_positions.values())
        if total_weight > 0:
            portfolio_positions = {
                k: v / total_weight
                for k, v in portfolio_positions.items()
            }

        # Get current portfolio value
        # Fetch current prices for all positions
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        prices = {}
        for ticker in tickers:
            try:
                current_price = self._data_service.fetch_current_price(ticker)
                if current_price:
                    prices[ticker] = current_price
            except Exception:
                logger.warning(f"Could not fetch price for {ticker}")

        # Add CASH price (always 1.0) if CASH position exists
        if any(pos.ticker == "CASH" for pos in positions):
            prices["CASH"] = 1.0

        current_value = portfolio.calculate_current_value(prices)

        # Get scenarios and convert HistoricalScenario to StressScenario
        from core.risk_engine.stress_testing import StressScenario

        stress_scenarios = []
        for name in scenario_names:
            historical_scenario = get_scenario_by_name(name)
            if historical_scenario:
                # Convert HistoricalScenario to StressScenario
                stress_scenario = StressScenario(
                    name=historical_scenario.name,
                    description=historical_scenario.description,
                    start_date=historical_scenario.start_date,
                    end_date=historical_scenario.end_date,
                    market_shock=historical_scenario.market_impact_pct,
                    sector_shocks=historical_scenario.sector_impacts,
                    asset_shocks=historical_scenario.asset_impacts,
                )
                stress_scenarios.append(stress_scenario)

        if not stress_scenarios:
            raise ValidationError("No valid scenarios found")

        # Apply stress tests
        tester = StressTester()
        results = tester.apply_multiple_scenarios(
            portfolio_positions, current_value, stress_scenarios
        )

        # Convert to dictionaries
        return [
            {
                "scenario_name": r.scenario_name,
                "portfolio_impact_pct": r.portfolio_impact_pct,
                "portfolio_impact_value": r.portfolio_impact_value,
                "worst_position": {
                    "ticker": r.worst_position_impact[0],
                    "impact_pct": r.worst_position_impact[1],
                },
                "best_position": {
                    "ticker": r.best_position_impact[0],
                    "impact_pct": r.best_position_impact[1],
                },
                "recovery_time_days": r.recovery_time_days,
                "details": r.details,
            }
            for r in results
        ]

    def run_custom_scenario(
        self,
        portfolio_id: str,
        scenario: CustomScenario,
    ) -> Dict[str, any]:
        """
        Run custom scenario test.

        Args:
            portfolio_id: Portfolio ID
            scenario: CustomScenario to apply

        Returns:
            Stress test result dictionary
        """
        # Validate scenario
        is_valid, error_msg = validate_scenario(scenario)
        if not is_valid:
            raise ValidationError(f"Invalid scenario: {error_msg}")

        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Get positions
        positions = portfolio.get_all_positions()
        portfolio_positions = {
            pos.ticker: pos.weight_target or 0.0 for pos in positions
        }

        # Normalize weights
        total_weight = sum(portfolio_positions.values())
        if total_weight > 0:
            portfolio_positions = {
                k: v / total_weight
                for k, v in portfolio_positions.items()
            }

        # Get current portfolio value
        # Fetch current prices for all positions
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        prices = {}
        for ticker in tickers:
            try:
                current_price = self._data_service.fetch_current_price(ticker)
                if current_price:
                    prices[ticker] = current_price
            except Exception:
                logger.warning(f"Could not fetch price for {ticker}")

        # Add CASH price (always 1.0) if CASH position exists
        if any(pos.ticker == "CASH" for pos in positions):
            prices["CASH"] = 1.0

        current_value = portfolio.calculate_current_value(prices)

        # Convert to stress scenario format
        from core.risk_engine.stress_testing import StressScenario
        from datetime import date

        stress_scenario = StressScenario(
            name=scenario.name,
            description=scenario.description,
            start_date=date.today(),
            end_date=date.today(),
            market_shock=scenario.market_impact_pct,
            sector_shocks=scenario.sector_impacts,
            # asset_impacts -> asset_shocks
            asset_shocks=scenario.asset_impacts,
        )

        # Apply scenario
        result = apply_stress_scenario(
            portfolio_positions, current_value, stress_scenario
        )

        return {
            "scenario_name": result.scenario_name,
            "portfolio_impact_pct": result.portfolio_impact_pct,
            "portfolio_impact_value": result.portfolio_impact_value,
            "worst_position": {
                "ticker": result.worst_position_impact[0],
                "impact_pct": result.worst_position_impact[1],
            },
            "best_position": {
                "ticker": result.best_position_impact[0],
                "impact_pct": result.best_position_impact[1],
            },
            "recovery_time_days": result.recovery_time_days,
            "details": result.details,
        }

    def run_scenario_chain(
        self,
        portfolio_id: str,
        chain: ScenarioChain,
    ) -> Dict[str, any]:
        """
        Run scenario chain analysis.

        Args:
            portfolio_id: Portfolio ID
            chain: ScenarioChain to apply

        Returns:
            Dictionary with chain analysis results
        """
        # Get portfolio
        portfolio = self._portfolio_service.get_portfolio(portfolio_id)

        # Get positions
        positions = portfolio.get_all_positions()
        portfolio_positions = {
            pos.ticker: pos.weight_target or 0.0 for pos in positions
        }

        # Normalize weights
        total_weight = sum(portfolio_positions.values())
        if total_weight > 0:
            portfolio_positions = {
                k: v / total_weight
                for k, v in portfolio_positions.items()
            }

        # Get current portfolio value
        # Fetch current prices for all positions
        tickers = [
            pos.ticker for pos in positions if pos.ticker != "CASH"
        ]
        prices = {}
        for ticker in tickers:
            try:
                current_price = self._data_service.fetch_current_price(ticker)
                if current_price:
                    prices[ticker] = current_price
            except Exception:
                logger.warning(f"Could not fetch price for {ticker}")

        # Add CASH price (always 1.0) if CASH position exists
        if any(pos.ticker == "CASH" for pos in positions):
            prices["CASH"] = 1.0

        current_value = portfolio.calculate_current_value(prices)

        # Apply chain
        results = apply_scenario_chain(
            portfolio_positions, current_value, chain
        )

        return results

    def get_available_scenarios(self) -> Dict[str, HistoricalScenario]:
        """Get all available historical scenarios."""
        return get_all_historical_scenarios()

    def _get_portfolio_returns(
        self, portfolio_id: str, start_date: date, end_date: date
    ) -> pd.Series:
        """Get portfolio returns for analysis period."""
        # Use analytics service to get returns
        metrics = self._analytics_service.calculate_portfolio_metrics(
            portfolio_id, start_date, end_date
        )

        # Extract returns directly from result
        returns = metrics.get("portfolio_returns")
        if returns is None or returns.empty:
            raise InsufficientDataError(
                "No returns data available for portfolio"
            )

        return returns
