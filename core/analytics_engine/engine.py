"""Main analytics engine orchestrator."""

import logging
import time
from datetime import date
from typing import Dict, Optional

import pandas as pd

from core.analytics_engine import market_metrics, performance, ratios
from core.analytics_engine.risk_metrics import (
    calculate_average_drawdown,
    calculate_current_drawdown,
    calculate_drawdown_duration,
    calculate_kurtosis,
    calculate_max_drawdown,
    calculate_pain_index,
    calculate_recovery_time,
    calculate_skewness,
    calculate_ulcer_index,
    calculate_var,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_semi_deviation,
    calculate_volatility,
)
from core.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.0435


class AnalyticsEngine:
    """
    Main analytics engine that orchestrates all metric calculations.

    This class coordinates the calculation of 70+ portfolio metrics
    across 4 categories: Performance, Risk, Ratios, Market Metrics.
    """

    def __init__(
        self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ) -> None:
        """
        Initialize analytics engine.

        Args:
            risk_free_rate: Annual risk-free rate (default: 4.35%)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        portfolio_returns: pd.Series,
        start_date: date,
        end_date: date,
        benchmark_returns: Optional[pd.Series] = None,
        portfolio_values: Optional[pd.Series] = None,
    ) -> Dict[str, any]:
        """
        Calculate all 70+ metrics for a portfolio.

        Args:
            portfolio_returns: Series of portfolio returns indexed by date
            start_date: Start date of analysis period
            end_date: End date of analysis period
            benchmark_returns: Optional benchmark returns (for market metrics)
            portfolio_values: Optional series of portfolio values for
                            period returns calculation

        Returns:
            Dictionary containing all metrics organized by category:
            - performance: 18 metrics
            - risk: 22 metrics
            - ratios: 15 ratios
            - market: 15 metrics (if benchmark provided)
            - metadata: calculation metadata

        Raises:
            InsufficientDataError: If insufficient data for calculations
        """
        start_time = time.time()

        if portfolio_returns.empty:
            raise InsufficientDataError(
                "Portfolio returns series is empty"
            )

        logger.info(
            f"Calculating metrics for period {start_date} to {end_date}"
        )

        results: Dict[str, any] = {
            "performance": {},
            "risk": {},
            "ratios": {},
            "market": {},
            "metadata": {},
        }

        # Performance Metrics (18)
        try:
            results["performance"] = self._calculate_performance_metrics(
                portfolio_returns, portfolio_values
            )
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            results["performance"] = {}

        # Risk Metrics (22)
        try:
            results["risk"] = self._calculate_risk_metrics(
                portfolio_returns
            )
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            results["risk"] = {}

        # Risk-Adjusted Ratios (15)
        try:
            results["ratios"] = self._calculate_ratios(portfolio_returns)
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            results["ratios"] = {}

        # Market Metrics (15) - only if benchmark provided
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                results["market"] = self._calculate_market_metrics(
                    portfolio_returns, benchmark_returns
                )
            except Exception as e:
                logger.error(f"Error calculating market metrics: {e}")
                results["market"] = {}

        # Metadata
        calculation_time = time.time() - start_time
        results["metadata"] = {
            "calculation_time_seconds": calculation_time,
            "data_points": len(portfolio_returns),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "benchmark_provided": benchmark_returns is not None,
        }

        # Add raw data for frontend charts
        results["portfolio_returns"] = portfolio_returns
        results["benchmark_returns"] = benchmark_returns
        results["portfolio_values"] = portfolio_values

        logger.info(
            f"Metrics calculation completed in {calculation_time:.3f}s"
        )

        return results

    def _calculate_performance_metrics(
        self,
        returns: pd.Series,
        portfolio_values: Optional[pd.Series],
    ) -> Dict[str, any]:
        """Calculate all 18 performance metrics."""
        metrics: Dict[str, any] = {}

        if returns.empty:
            return metrics

        try:
            # Total Return
            if portfolio_values is not None and len(portfolio_values) >= 2:
                start_value = float(portfolio_values.iloc[0])
                end_value = float(portfolio_values.iloc[-1])
                metrics["total_return"] = performance.calculate_total_return(
                    start_value, end_value
                )
            else:
                # Calculate from returns if values not provided
                total_return = float((1 + returns).prod() - 1)
                metrics["total_return"] = total_return

            # CAGR
            if portfolio_values is not None and len(portfolio_values) >= 2:
                start_value = float(portfolio_values.iloc[0])
                end_value = float(portfolio_values.iloc[-1])
                if portfolio_values.index[0] and portfolio_values.index[-1]:
                    start_dt = portfolio_values.index[0]
                    end_dt = portfolio_values.index[-1]
                    years = (end_dt - start_dt).days / 365.25
                    if years > 0:
                        metrics["cagr"] = performance.calculate_cagr(
                            start_value, end_value, years
                        )

            # Annualized Return
            metrics["annualized_return"] = (
                performance.calculate_annualized_return(returns)
            )

            # Period Returns
            if portfolio_values is not None:
                period_returns = performance.calculate_period_returns(
                    portfolio_values
                )
                metrics.update(period_returns)
            else:
                # Set to None if values not available
                metrics.update({
                    "ytd": None,
                    "mtd": None,
                    "qtd": None,
                    "1m": None,
                    "3m": None,
                    "6m": None,
                    "1y": None,
                    "3y": None,
                    "5y": None,
                })

            # Best/Worst Month
            best_worst = performance.calculate_best_worst_periods(returns)
            metrics["best_month"] = best_worst["best_month"]
            metrics["worst_month"] = best_worst["worst_month"]

            # Win Rate
            metrics["win_rate"] = performance.calculate_win_rate(returns)

            # Payoff Ratio
            metrics["payoff_ratio"] = performance.calculate_payoff_ratio(
                returns
            )

            # Profit Factor
            metrics["profit_factor"] = (
                performance.calculate_profit_factor(returns)
            )

            # Expectancy
            metrics["expectancy"] = performance.calculate_expectancy(
                returns
            )

        except Exception as e:
            logger.warning(f"Error in performance metrics: {e}")

        return metrics

    def _calculate_risk_metrics(
        self, returns: pd.Series
    ) -> Dict[str, any]:
        """Calculate all 22 risk metrics."""
        metrics: Dict[str, any] = {}

        if returns.empty:
            return metrics

        try:
            # Volatility (multiple timeframes)
            volatility = calculate_volatility(returns)
            metrics.update(volatility)

            # Max Drawdown
            max_dd, peak_date, trough_date, duration = (
                calculate_max_drawdown(returns)
            )
            metrics["max_drawdown"] = max_dd
            metrics["max_drawdown_peak_date"] = (
                peak_date.isoformat() if peak_date else None
            )
            metrics["max_drawdown_trough_date"] = (
                trough_date.isoformat() if trough_date else None
            )
            metrics["max_drawdown_duration_days"] = duration

            # Current Drawdown
            metrics["current_drawdown"] = (
                calculate_current_drawdown(returns)
            )

            # Average Drawdown
            metrics["average_drawdown"] = (
                calculate_average_drawdown(returns)
            )

            # Drawdown Duration
            dd_duration = calculate_drawdown_duration(returns)
            metrics.update(dd_duration)

            # Recovery Time
            metrics["recovery_time_days"] = (
                calculate_recovery_time(returns)
            )

            # Ulcer Index
            metrics["ulcer_index"] = calculate_ulcer_index(returns)

            # Pain Index
            metrics["pain_index"] = calculate_pain_index(returns)

            # VaR (90%, 95%, 99%)
            for conf in [0.90, 0.95, 0.99]:
                var_key = f"var_{int(conf * 100)}"
                try:
                    metrics[var_key] = calculate_var(returns, conf)
                except Exception:
                    metrics[var_key] = None

            # CVaR (90%, 95%, 99%)
            for conf in [0.90, 0.95, 0.99]:
                cvar_key = f"cvar_{int(conf * 100)}"
                try:
                    metrics[cvar_key] = calculate_cvar(returns, conf)
                except Exception:
                    metrics[cvar_key] = None

            # Downside Deviation
            metrics["downside_deviation"] = (
                calculate_downside_deviation(returns)
            )

            # Semi-Deviation
            metrics["semi_deviation"] = calculate_semi_deviation(returns)

            # Skewness
            metrics["skewness"] = calculate_skewness(returns)

            # Kurtosis
            metrics["kurtosis"] = calculate_kurtosis(returns)

        except Exception as e:
            logger.warning(f"Error in risk metrics: {e}")

        return metrics

    def _calculate_ratios(self, returns: pd.Series) -> Dict[str, any]:
        """Calculate all 15 risk-adjusted ratios."""
        metrics: Dict[str, any] = {}

        if returns.empty:
            return metrics

        try:
            # Sharpe Ratio
            metrics["sharpe_ratio"] = ratios.calculate_sharpe_ratio(
                returns, self.risk_free_rate
            )

            # Sortino Ratio
            metrics["sortino_ratio"] = ratios.calculate_sortino_ratio(
                returns, self.risk_free_rate
            )

            # Calmar Ratio
            metrics["calmar_ratio"] = ratios.calculate_calmar_ratio(returns)

            # Sterling Ratio
            metrics["sterling_ratio"] = ratios.calculate_sterling_ratio(
                returns
            )

            # Burke Ratio
            metrics["burke_ratio"] = ratios.calculate_burke_ratio(returns)

            # Treynor Ratio (requires beta - set in market metrics)
            # Will be calculated in market metrics section

            # Information Ratio (requires benchmark - set in market metrics)

            # Modigliani M² (requires benchmark - set in market metrics)

            # Omega Ratio
            metrics["omega_ratio"] = ratios.calculate_omega_ratio(returns)

            # Kappa 3
            metrics["kappa_3"] = ratios.calculate_kappa_3(returns)

            # Gain-Pain Ratio
            metrics["gain_pain_ratio"] = (
                ratios.calculate_gain_pain_ratio(returns)
            )

            # Martin Ratio
            metrics["martin_ratio"] = ratios.calculate_martin_ratio(returns)

            # Tail Ratio
            metrics["tail_ratio"] = ratios.calculate_tail_ratio(returns)

            # Common Sense Ratio
            metrics["common_sense_ratio"] = (
                ratios.calculate_common_sense_ratio(returns)
            )

            # Rachev Ratio (requires benchmark - set in market metrics)

        except Exception as e:
            logger.warning(f"Error in ratios: {e}")

        return metrics

    def _calculate_market_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, any]:
        """Calculate all 15 market-related metrics."""
        metrics: Dict[str, any] = {}

        if portfolio_returns.empty or benchmark_returns.empty:
            return metrics

        try:
            # Beta
            metrics["beta"] = market_metrics.calculate_beta(
                portfolio_returns, benchmark_returns
            )

            # Alpha
            metrics["alpha"] = market_metrics.calculate_alpha(
                portfolio_returns,
                benchmark_returns,
                self.risk_free_rate,
            )

            # R-Squared
            metrics["r_squared"] = market_metrics.calculate_r_squared(
                portfolio_returns, benchmark_returns
            )

            # Correlation
            metrics["correlation"] = (
                market_metrics.calculate_correlation(
                    portfolio_returns, benchmark_returns
                )
            )

            # Tracking Error
            metrics["tracking_error"] = (
                market_metrics.calculate_tracking_error(
                    portfolio_returns, benchmark_returns
                )
            )

            # Active Return
            metrics["active_return"] = (
                market_metrics.calculate_active_return(
                    portfolio_returns, benchmark_returns
                )
            )

            # Up Capture
            metrics["up_capture"] = market_metrics.calculate_up_capture(
                portfolio_returns, benchmark_returns
            )

            # Down Capture
            metrics["down_capture"] = (
                market_metrics.calculate_down_capture(
                    portfolio_returns, benchmark_returns
                )
            )

            # Capture Ratio
            metrics["capture_ratio"] = (
                market_metrics.calculate_capture_ratio(
                    portfolio_returns, benchmark_returns
                )
            )

            # Jensen's Alpha (same as Alpha)
            metrics["jensens_alpha"] = (
                market_metrics.calculate_jensens_alpha(
                    portfolio_returns,
                    benchmark_returns,
                    self.risk_free_rate,
                )
            )

            # Active Share (requires weights - not available here)
            metrics["active_share"] = None

            # Batting Average
            metrics["batting_average"] = (
                market_metrics.calculate_batting_average(
                    portfolio_returns, benchmark_returns
                )
            )

            # Benchmark Relative Return
            metrics["benchmark_relative_return"] = (
                market_metrics.calculate_benchmark_relative_return(
                    portfolio_returns, benchmark_returns
                )
            )

            # Rolling Beta
            metrics["rolling_beta"] = market_metrics.calculate_rolling_beta(
                portfolio_returns, benchmark_returns
            )

            # Market Timing Ratio
            metrics["market_timing_ratio"] = (
                market_metrics.calculate_market_timing_ratio(
                    portfolio_returns, benchmark_returns
                )
            )

            # Additional ratios that require benchmark
            metrics["treynor_ratio"] = None
            if metrics["beta"] is not None:
                metrics["treynor_ratio"] = ratios.calculate_treynor_ratio(
                    portfolio_returns,
                    metrics["beta"],
                    self.risk_free_rate,
                )

            metrics["information_ratio"] = (
                ratios.calculate_information_ratio(
                    portfolio_returns, benchmark_returns
                )
            )

            metrics["modigliani_m2"] = ratios.calculate_modigliani_m2(
                portfolio_returns,
                benchmark_returns,
                self.risk_free_rate,
            )

            metrics["rachev_ratio"] = ratios.calculate_rachev_ratio(
                portfolio_returns, benchmark_returns
            )

        except Exception as e:
            logger.warning(f"Error in market metrics: {e}")

        return metrics

