"""Risk Analysis page for advanced risk metrics."""

import logging
from datetime import date, timedelta

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from services.portfolio_service import PortfolioService
from services.risk_service import RiskService
from core.scenario_engine.historical_scenarios import get_all_scenarios
from core.scenario_engine.custom_scenarios import (
    create_custom_scenario,
    validate_scenario,
)
from core.scenario_engine.scenario_chain import create_scenario_chain
from core.exceptions import ValidationError
from core.analytics_engine.risk_metrics import calculate_cvar
from streamlit_app.utils.chart_config import COLORS
from streamlit_app.components.charts import plot_var_distribution

logger = logging.getLogger(__name__)


def _interpret_var_metrics(var_95: float, cvar_95: float, var_99: float, cvar_99: float) -> str:
    """Interpret VaR and CVaR metrics."""
    if var_95 is None or cvar_95 is None:
        return ""
    
    parts = []
    parts.append("**VaR & CVaR Analysis:**")
    
    # VaR 95% interpretation
    if abs(var_95) < 0.02:
        parts.append(f"VaR (95%): {var_95*100:.2f}% - Low daily risk exposure")
    elif abs(var_95) < 0.05:
        parts.append(f"VaR (95%): {var_95*100:.2f}% - Moderate daily risk exposure")
    else:
        parts.append(f"⚠ VaR (95%): {var_95*100:.2f}% - High daily risk exposure")
    
    # CVaR vs VaR comparison
    if cvar_95 and var_95:
        cvar_excess = abs(cvar_95) - abs(var_95)
        if cvar_excess > 0.01:
            parts.append(f"CVaR (95%): {cvar_95*100:.2f}% - Tail risk is {cvar_excess*100:.2f}% higher than VaR, indicating significant tail risk")
        else:
            parts.append(f"CVaR (95%): {cvar_95*100:.2f}% - Tail risk is similar to VaR")
    
    # VaR 99% interpretation
    if var_99 is not None:
        if abs(var_99) < 0.03:
            parts.append(f"VaR (99%): {var_99*100:.2f}% - Low extreme risk exposure")
        elif abs(var_99) < 0.08:
            parts.append(f"VaR (99%): {var_99*100:.2f}% - Moderate extreme risk exposure")
        else:
            parts.append(f"⚠ VaR (99%): {var_99*100:.2f}% - High extreme risk exposure")
    
    # Overall assessment
    if var_95 and var_99:
        if abs(var_99) > abs(var_95) * 1.5:
            parts.append(f"\n⚠ Significant jump from 95% to 99% VaR - Portfolio has fat tails (extreme events risk)")
        else:
            parts.append(f"\nModerate increase from 95% to 99% VaR - Relatively stable tail distribution")
    
    return "\n".join(parts)


def _interpret_var_methods_comparison(var_data: dict, confidence_level: float) -> str:
    """Interpret comparison of different VaR calculation methods."""
    if not var_data:
        return ""
    
    parts = []
    parts.append("**VaR Methods Comparison:**")
    
    methods = ["historical", "parametric", "cornish_fisher", "monte_carlo"]
    method_names = {
        "historical": "Historical",
        "parametric": "Parametric",
        "cornish_fisher": "Cornish-Fisher",
        "monte_carlo": "Monte Carlo",
    }
    
    values = {}
    for method in methods:
        val = var_data.get(method)
        if val is not None and not isinstance(val, str):
            values[method] = val
    
    if not values:
        return ""
    
    # Find min and max
    min_method = min(values, key=values.get)
    max_method = max(values, key=values.get)
    min_val = values[min_method]
    max_val = values[max_method]
    
    spread = max_val - min_val
    
    if spread < 0.01:
        parts.append(f"All methods show similar VaR values (spread: {spread*100:.2f}%) - Consistent risk estimate")
    elif spread < 0.03:
        parts.append(f"Methods show moderate variation (spread: {spread*100:.2f}%) - Some model uncertainty")
    else:
        parts.append(f"⚠ Methods show significant variation (spread: {spread*100:.2f}%) - High model uncertainty, consider using multiple methods")
    
    # Method-specific insights
    if "historical" in values and "parametric" in values:
        hist_val = values["historical"]
        param_val = values["parametric"]
        diff = abs(hist_val - param_val)
        if diff > 0.02:
            parts.append(f"Historical ({hist_val*100:.2f}%) vs Parametric ({param_val*100:.2f}%) differ by {diff*100:.2f}% - Returns may not be normally distributed")
    
    if "cornish_fisher" in values:
        cf_val = values["cornish_fisher"]
        parts.append(f"Cornish-Fisher ({cf_val*100:.2f}%) accounts for skewness and kurtosis - May be more accurate if returns are non-normal")
    
    return "\n".join(parts)


def _interpret_portfolio_var_decomposition(decomposition_data: list, portfolio_var: float) -> str:
    """Interpret VaR decomposition by asset."""
    if not decomposition_data or portfolio_var is None:
        return ""
    
    parts = []
    parts.append("**VaR Decomposition Analysis:**")
    
    # Find largest contributors
    contributions = []
    for item in decomposition_data:
        contrib_pct = float(item.get("Contribution (%)", 0).replace("%", ""))
        ticker = item.get("Asset", "")
        weight = float(item.get("Weight (%)", 0))
        contrib_var = float(item.get("Component VaR (%)", 0).replace("%", ""))
        contributions.append((ticker, weight, contrib_pct, contrib_var))
    
    if not contributions:
        return ""
    
    # Sort by contribution
    contributions.sort(key=lambda x: x[2], reverse=True)
    
    # Largest contributor
    top_ticker, top_weight, top_contrib, top_var = contributions[0]
    parts.append(f"Largest risk contributor: {top_ticker} ({top_weight:.1f}% weight, {top_contrib:.1f}% of VaR)")
    
    # Check concentration
    top_3_contrib = sum(c[2] for c in contributions[:3])
    if top_3_contrib > 70:
        parts.append(f"⚠ High concentration: Top 3 assets contribute {top_3_contrib:.1f}% of total VaR - Consider diversification")
    elif top_3_contrib > 50:
        parts.append(f"Moderate concentration: Top 3 assets contribute {top_3_contrib:.1f}% of total VaR")
    else:
        parts.append(f"✓ Well-diversified: Top 3 assets contribute {top_3_contrib:.1f}% of total VaR")
    
    # Check weight vs risk mismatch
    mismatches = []
    for ticker, weight, contrib, var in contributions:
        if weight > 0:
            risk_weight_ratio = contrib / weight if weight > 0 else 0
            if risk_weight_ratio > 1.5:
                mismatches.append((ticker, weight, contrib, risk_weight_ratio))
    
    if mismatches:
        worst = mismatches[0]
        parts.append(f"⚠ Risk/weight mismatch: {worst[0]} has {worst[1]:.1f}% weight but {worst[2]:.1f}% VaR contribution (ratio: {worst[3]:.1f}x) - Asset is riskier than its weight suggests")
    
    return "\n".join(parts)


def _interpret_monte_carlo_statistics(stats: dict, initial_value: float) -> str:
    """Interpret Monte Carlo simulation statistics."""
    if not stats:
        return ""
    
    parts = []
    parts.append("**Monte Carlo Simulation Analysis:**")
    
    mean = stats.get("mean", 0)
    median = stats.get("median", 0)
    std = stats.get("std", 0)
    min_val = stats.get("min", 0)
    max_val = stats.get("max", 0)
    
    # Return analysis
    mean_return = (mean / initial_value - 1) * 100 if initial_value > 0 else 0
    median_return = (median / initial_value - 1) * 100 if initial_value > 0 else 0
    
    parts.append(f"Expected value: ${mean:,.2f} ({mean_return:+.2f}% return)")
    parts.append(f"Median value: ${median:,.2f} ({median_return:+.2f}% return)")
    
    # Skewness check
    if mean > median * 1.1:
        parts.append(f"Positive skew: Mean > Median - Upside potential exists")
    elif mean < median * 0.9:
        parts.append(f"Negative skew: Mean < Median - Downside risk dominates")
    else:
        parts.append(f"Symmetric distribution: Mean ≈ Median")
    
    # Volatility
    vol_pct = (std / initial_value) * 100 if initial_value > 0 else 0
    if vol_pct < 5:
        parts.append(f"Low volatility: {vol_pct:.1f}% std dev - Relatively stable outcomes")
    elif vol_pct < 15:
        parts.append(f"Moderate volatility: {vol_pct:.1f}% std dev")
    else:
        parts.append(f"⚠ High volatility: {vol_pct:.1f}% std dev - Wide range of possible outcomes")
    
    # Range
    range_pct = ((max_val - min_val) / initial_value) * 100 if initial_value > 0 else 0
    parts.append(f"Outcome range: ${min_val:,.2f} to ${max_val:,.2f} ({range_pct:.1f}% spread)")
    
    return "\n".join(parts)


def _interpret_monte_carlo_percentiles(percentiles: dict, initial_value: float) -> str:
    """Interpret Monte Carlo percentile outcomes."""
    if not percentiles:
        return ""
    
    parts = []
    parts.append("**Percentile Analysis:**")
    
    # Key percentiles - use float keys (5.0, 25.0, etc.) as they are stored in dict
    p5 = percentiles.get(5.0) or percentiles.get(5)
    p25 = percentiles.get(25.0) or percentiles.get(25)
    p50 = percentiles.get(50.0) or percentiles.get(50)
    p75 = percentiles.get(75.0) or percentiles.get(75)
    p95 = percentiles.get(95.0) or percentiles.get(95)
    
    if p5 is not None:
        p5_return = (p5 / initial_value - 1) * 100 if initial_value > 0 else 0
        parts.append(f"5th percentile (worst 5%): ${p5:,.2f} ({p5_return:+.2f}% return) - Downside risk")
    
    if p95 is not None:
        p95_return = (p95 / initial_value - 1) * 100 if initial_value > 0 else 0
        parts.append(f"95th percentile (best 5%): ${p95:,.2f} ({p95_return:+.2f}% return) - Upside potential")
    
    # Interquartile range
    if p25 is not None and p75 is not None:
        iqr = p75 - p25
        iqr_pct = (iqr / initial_value) * 100 if initial_value > 0 else 0
        parts.append(f"Interquartile range (50% of outcomes): {iqr_pct:.1f}% spread - Most likely outcomes")
    
    # Downside vs upside
    if p5 is not None and p95 is not None:
        downside = abs(initial_value - p5)
        upside = abs(p95 - initial_value)
        if downside > upside * 1.5:
            parts.append(f"⚠ Asymmetric risk: Downside ({downside/initial_value*100:.1f}%) > Upside ({upside/initial_value*100:.1f}%) - Negative skew")
        elif upside > downside * 1.5:
            parts.append(f"Positive asymmetry: Upside ({upside/initial_value*100:.1f}%) > Downside ({downside/initial_value*100:.1f}%) - Positive skew")
        else:
            parts.append(f"Balanced: Similar upside and downside potential")
    
    return "\n".join(parts)


def _interpret_monte_carlo_var_comparison(historical_var: float, mc_var: float, confidence: float) -> str:
    """Interpret comparison between historical and Monte Carlo VaR."""
    if historical_var is None or mc_var is None:
        return ""
    
    parts = []
    parts.append("**Historical vs Monte Carlo VaR Comparison:**")
    
    parts.append(f"Historical VaR ({confidence*100:.0f}%): {historical_var*100:.2f}%")
    parts.append(f"Monte Carlo VaR ({confidence*100:.0f}%): {mc_var*100:.2f}%")
    
    diff = mc_var - historical_var
    diff_pct = abs(diff) / abs(historical_var) * 100 if historical_var != 0 else 0
    
    if abs(diff) < 0.01:
        parts.append(f"Methods agree (difference: {diff*100:.2f}%) - Consistent risk estimate")
    elif diff_pct < 20:
        parts.append(f"Moderate difference: {diff*100:.2f}% ({diff_pct:.1f}% relative) - Some model variation")
    else:
        if diff > 0:
            parts.append(f"⚠ Monte Carlo shows higher risk ({diff*100:.2f}% higher) - Forward-looking model suggests more risk than historical data")
        else:
            parts.append(f"Monte Carlo shows lower risk ({abs(diff)*100:.2f}% lower) - Forward-looking model suggests less risk than historical data")
    
    return "\n".join(parts)


def _interpret_extreme_scenarios(extreme_df, initial_value: float) -> str:
    """Interpret extreme scenarios from Monte Carlo."""
    if extreme_df is None or extreme_df.empty:
        return ""
    
    import pandas as pd
    
    parts = []
    parts.append("**Extreme Scenarios Analysis:**")
    
    # Check if it's a DataFrame
    if isinstance(extreme_df, pd.DataFrame):
        if "Scenario" in extreme_df.columns and "Value" in extreme_df.columns:
            # Worst case
            worst_row = extreme_df[extreme_df["Scenario"].str.contains("Worst", case=False, na=False)]
            if not worst_row.empty:
                worst_val_str = worst_row["Value"].iloc[0]
                if isinstance(worst_val_str, str):
                    worst_val = float(worst_val_str.replace("$", "").replace(",", ""))
                else:
                    worst_val = float(worst_val_str)
                worst_return = (worst_val / initial_value - 1) * 100 if initial_value > 0 else 0
                parts.append(f"Worst case: ${worst_val:,.2f} ({worst_return:+.2f}% return) - Maximum loss scenario")
            
            # Best case
            best_row = extreme_df[extreme_df["Scenario"].str.contains("Best", case=False, na=False)]
            if not best_row.empty:
                best_val_str = best_row["Value"].iloc[0]
                if isinstance(best_val_str, str):
                    best_val = float(best_val_str.replace("$", "").replace(",", ""))
                else:
                    best_val = float(best_val_str)
                best_return = (best_val / initial_value - 1) * 100 if initial_value > 0 else 0
                parts.append(f"Best case: ${best_val:,.2f} ({best_return:+.2f}% return) - Maximum gain scenario")
    
    return "\n".join(parts)


def _interpret_scenario_results(results: list) -> str:
    """Interpret stress test or scenario results."""
    if not results:
        return ""
    
    parts = []
    parts.append("**Scenario Analysis:**")
    
    # Calculate statistics
    impacts = [r.get("portfolio_impact_pct", 0) * 100 for r in results]
    worst_impact = min(impacts) if impacts else 0
    best_impact = max(impacts) if impacts else 0
    avg_impact = np.mean(impacts) if impacts else 0
    
    # Worst scenario
    worst_scenario = min(results, key=lambda x: x.get("portfolio_impact_pct", 0))
    worst_name = worst_scenario.get("scenario_name", "Unknown")
    worst_val = worst_scenario.get("portfolio_impact_pct", 0) * 100
    
    parts.append(f"Worst scenario: {worst_name} ({worst_val:.2f}% impact)")
    
    # Best scenario
    best_scenario = max(results, key=lambda x: x.get("portfolio_impact_pct", 0))
    best_name = best_scenario.get("scenario_name", "Unknown")
    best_val = best_scenario.get("portfolio_impact_pct", 0) * 100
    
    if best_val > 0:
        parts.append(f"Best scenario: {best_name} ({best_val:.2f}% gain)")
    else:
        parts.append(f"Least negative: {best_name} ({best_val:.2f}% impact)")
    
    # Average impact
    parts.append(f"Average impact: {avg_impact:.2f}% across all scenarios")
    
    # Risk assessment
    if worst_impact < -30:
        parts.append(f"⚠ High risk: Worst case shows {abs(worst_impact):.1f}% loss - Significant downside exposure")
    elif worst_impact < -15:
        parts.append(f"Moderate risk: Worst case shows {abs(worst_impact):.1f}% loss")
    else:
        parts.append(f"✓ Low risk: Worst case shows {abs(worst_impact):.1f}% loss - Portfolio is resilient")
    
    # Recovery analysis
    recovery_times = [r.get("recovery_time_days") for r in results if r.get("recovery_time_days")]
    if recovery_times:
        avg_recovery = np.mean(recovery_times)
        max_recovery = max(recovery_times)
        parts.append(f"Recovery: Average {avg_recovery:.0f} days, maximum {max_recovery:.0f} days")
    
    return "\n".join(parts)


def _interpret_scenario_recovery(recovery_data: dict) -> str:
    """Interpret portfolio recovery timeline."""
    if not recovery_data or not recovery_data.get("results"):
        return ""
    
    parts = []
    parts.append("**Recovery Timeline Analysis:**")
    
    results = recovery_data.get("results", [])
    scenarios = recovery_data.get("scenarios", {})
    
    # Analyze recovery times
    recovery_times = []
    for r in results:
        recovery_days = r.get("recovery_time_days")
        if recovery_days and recovery_days > 0:
            recovery_times.append({
                "name": r.get("scenario_name", "Unknown"),
                "days": recovery_days,
                "impact": r.get("portfolio_impact_pct", 0) * 100
            })
    
    if recovery_times:
        # Fastest recovery
        fastest = min(recovery_times, key=lambda x: x["days"])
        parts.append(f"Fastest recovery: {fastest['name']} ({fastest['days']:.0f} days) - Most resilient scenario")
        
        # Slowest recovery
        slowest = max(recovery_times, key=lambda x: x["days"])
        parts.append(f"Slowest recovery: {slowest['name']} ({slowest['days']:.0f} days) - Most challenging scenario")
        
        # Average recovery
        avg_recovery = np.mean([r["days"] for r in recovery_times])
        parts.append(f"Average recovery time: {avg_recovery:.0f} days")
        
        # Recovery vs impact relationship
        if len(recovery_times) > 1:
            # Check if larger impacts lead to longer recovery
            impacts = [r["impact"] for r in recovery_times]
            days = [r["days"] for r in recovery_times]
            if abs(impacts[0]) > 0 and abs(days[0]) > 0:
                # Simple correlation check
                larger_impact_longer_recovery = any(
                    abs(impacts[i]) > abs(impacts[j]) and days[i] > days[j]
                    for i in range(len(recovery_times))
                    for j in range(len(recovery_times))
                    if i != j
                )
                if larger_impact_longer_recovery:
                    parts.append(f"Larger impacts generally require longer recovery periods")
    else:
        parts.append("Recovery data not available for analysis")
    
    return "\n".join(parts)


def _interpret_rolling_var(rolling_stats: dict) -> str:
    """Interpret rolling VaR statistics."""
    if not rolling_stats:
        return ""
    
    parts = []
    parts.append("**Rolling VaR Analysis:**")
    
    avg = rolling_stats.get("avg", 0)
    median = rolling_stats.get("median", 0)
    min_var = rolling_stats.get("min", 0)
    max_var = rolling_stats.get("max", 0)
    
    parts.append(f"Average VaR: {avg*100:.2f}%")
    parts.append(f"Median VaR: {median*100:.2f}%")
    
    # Volatility of VaR
    if max_var and min_var:
        var_range = max_var - min_var
        if var_range > 0.05:
            parts.append(f"⚠ High VaR volatility: Range from {min_var*100:.2f}% to {max_var*100:.2f}% ({var_range*100:.2f}% spread) - Risk levels change significantly over time")
        elif var_range > 0.02:
            parts.append(f"Moderate VaR volatility: Range from {min_var*100:.2f}% to {max_var*100:.2f}% ({var_range*100:.2f}% spread)")
        else:
            parts.append(f"Stable VaR: Range from {min_var*100:.2f}% to {max_var*100:.2f}% ({var_range*100:.2f}% spread) - Consistent risk levels")
    
    return "\n".join(parts)


def _interpret_confidence_intervals(final_values: np.ndarray, initial_value: float) -> str:
    """Interpret confidence intervals on distribution."""
    if final_values is None or len(final_values) == 0:
        return ""
    
    parts = []
    parts.append("**Confidence Intervals Analysis:**")
    
    # Calculate confidence intervals
    ci_levels = [0.90, 0.95, 0.99]
    
    for ci in ci_levels:
        lower = np.percentile(final_values, (1 - ci) / 2 * 100)
        upper = np.percentile(final_values, (1 + ci) / 2 * 100)
        
        lower_return = (lower / initial_value - 1) * 100 if initial_value > 0 else 0
        upper_return = (upper / initial_value - 1) * 100 if initial_value > 0 else 0
        range_pct = ((upper - lower) / initial_value) * 100 if initial_value > 0 else 0
        
        parts.append(f"{int(ci*100)}% CI: ${lower:,.2f} to ${upper:,.2f} ({range_pct:.1f}% range, {lower_return:+.1f}% to {upper_return:+.1f}% return)")
    
    # Overall assessment
    ci_90_lower = np.percentile(final_values, 5)
    ci_90_upper = np.percentile(final_values, 95)
    ci_99_lower = np.percentile(final_values, 0.5)
    ci_99_upper = np.percentile(final_values, 99.5)
    
    tail_width_90 = (ci_90_upper - ci_90_lower) / initial_value * 100 if initial_value > 0 else 0
    tail_width_99 = (ci_99_upper - ci_99_lower) / initial_value * 100 if initial_value > 0 else 0
    
    if tail_width_90 < 10:
        parts.append(f"\nNarrow 90% range ({tail_width_90:.1f}%) - Relatively predictable outcomes")
    elif tail_width_90 < 25:
        parts.append(f"\nModerate 90% range ({tail_width_90:.1f}%) - Some outcome variability")
    else:
        parts.append(f"\n⚠ Wide 90% range ({tail_width_90:.1f}%) - High outcome uncertainty")
    
    tail_expansion = tail_width_99 / tail_width_90 if tail_width_90 > 0 else 0
    if tail_expansion > 1.5:
        parts.append(f"Significant tail expansion (99% range is {tail_expansion:.1f}x wider) - Fat tails, extreme events possible")
    else:
        parts.append(f"Moderate tail expansion (99% range is {tail_expansion:.1f}x wider) - Relatively normal distribution")
    
    return "\n".join(parts)


def _interpret_portfolio_var_covariance(portfolio_var: float, confidence: float) -> str:
    """Interpret Portfolio VaR calculated with covariance method."""
    if portfolio_var is None:
        return ""
    
    parts = []
    parts.append("**Portfolio VaR (Covariance Method) Analysis:**")
    
    parts.append(f"Portfolio VaR ({confidence*100:.0f}%): {abs(portfolio_var)*100:.2f}%")
    
    # Risk assessment
    if abs(portfolio_var) < 0.02:
        parts.append(f"Low portfolio risk - Daily loss unlikely to exceed {abs(portfolio_var)*100:.2f}%")
    elif abs(portfolio_var) < 0.05:
        parts.append(f"Moderate portfolio risk - Daily loss may reach {abs(portfolio_var)*100:.2f}%")
    else:
        parts.append(f"⚠ High portfolio risk - Daily loss could exceed {abs(portfolio_var)*100:.2f}%")
    
    parts.append(f"\nThis method accounts for correlations between assets, providing a more accurate portfolio-level risk measure than individual asset VaR.")
    
    return "\n".join(parts)


def _interpret_position_impact_breakdown(position_impacts: dict, scenario_name: str) -> str:
    """Interpret position impact breakdown for a scenario."""
    if not position_impacts:
        return ""
    
    parts = []
    parts.append(f"**Position Impact Analysis ({scenario_name}):**")
    
    # Convert to list for sorting
    impacts_list = [
        (ticker, impact * 100) 
        for ticker, impact in position_impacts.items()
    ]
    
    if not impacts_list:
        return ""
    
    # Sort by impact (most negative first)
    impacts_list.sort(key=lambda x: x[1])
    
    # Worst impacted
    worst_ticker, worst_impact = impacts_list[0]
    parts.append(f"Worst impacted: {worst_ticker} ({worst_impact:.2f}%)")
    
    # Best impacted (if positive)
    best_ticker, best_impact = impacts_list[-1]
    if best_impact > 0:
        parts.append(f"Best impacted: {best_ticker} ({best_impact:.2f}%)")
    
    # Count losses vs gains
    losses = [x for x in impacts_list if x[1] < 0]
    gains = [x for x in impacts_list if x[1] > 0]
    
    if len(losses) > len(gains):
        parts.append(f"Most positions show losses ({len(losses)} vs {len(gains)} gains) - Scenario is broadly negative")
    elif len(gains) > len(losses):
        parts.append(f"Most positions show gains ({len(gains)} vs {len(losses)} losses) - Scenario is broadly positive")
    else:
        parts.append(f"Mixed impact: {len(losses)} losses, {len(gains)} gains")
    
    # Concentration of impact
    if len(impacts_list) > 1:
        abs_impacts = [abs(x[1]) for x in impacts_list]
        top_2_impact = sum(sorted(abs_impacts, reverse=True)[:2])
        total_impact = sum(abs_impacts)
        if total_impact > 0:
            concentration = top_2_impact / total_impact * 100
            if concentration > 70:
                parts.append(f"High impact concentration: Top 2 assets account for {concentration:.1f}% of total impact")
            elif concentration > 50:
                parts.append(f"Moderate impact concentration: Top 2 assets account for {concentration:.1f}% of total impact")
    
    return "\n".join(parts)


def render_risk_analysis_page() -> None:
    """Render risk analysis page."""
    st.title("Risk Analysis")
    st.markdown(
        "Advanced risk analysis including VaR, Monte Carlo simulations, "
        "and stress testing."
    )

    # Initialize services
    portfolio_service = PortfolioService()
    risk_service = RiskService()

    # Get portfolios
    portfolios = portfolio_service.list_portfolios()

    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return

    # Portfolio selection
    portfolio_names = [p.name for p in portfolios]
    selected_name = st.selectbox(
        "Select Portfolio",
        portfolio_names,
        key="risk_analysis_portfolio",
    )

    selected_portfolio = next(
        p for p in portfolios if p.name == selected_name
    )

    # Date range selection
    st.subheader("Analysis Period")
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=365),
            key="risk_start_date",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            key="risk_end_date",
        )

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "VaR Analysis",
            "Monte Carlo Simulation",
            "Historical Scenarios",
            "Custom Scenario",
            "Scenario Chain",
        ]
    )

    with tab1:
        _render_var_analysis(
            risk_service,
            selected_portfolio.id,
            start_date,
            end_date,
        )

    with tab2:
        _render_monte_carlo(
            risk_service,
            selected_portfolio.id,
            start_date,
            end_date,
        )

    with tab3:
        _render_historical_scenarios(
            risk_service,
            selected_portfolio.id,
        )

    with tab4:
        _render_custom_scenario(
            risk_service,
            selected_portfolio.id,
        )

    with tab5:
        _render_scenario_chain(
            risk_service,
            selected_portfolio.id,
        )


def _render_var_analysis(
    risk_service: RiskService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
) -> None:
    """Render VaR analysis section."""
    st.subheader("Value at Risk (VaR) Analysis")

    # Confidence level selector
    confidence_level = st.slider(
        "Confidence Level",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key="var_confidence",
        help="Confidence level for VaR calculation (90%, 95%, or 99%)",
    )
    conf_decimal = confidence_level / 100.0

    # Time horizon
    time_horizon = st.number_input(
        "Time Horizon (days)",
        min_value=1,
        max_value=30,
        value=1,
        key="var_horizon",
    )

    # Calculate VaR
    if st.button("Calculate VaR", key="calculate_var"):
        try:
            with st.spinner("Calculating VaR..."):
                var_results = risk_service.calculate_var_analysis(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date,
                    confidence_level=conf_decimal,
                    include_monte_carlo=True,
                    num_simulations=10000,
                    time_horizon=time_horizon,
                )

                # Display results
                st.success("VaR calculation completed!")

                var_data = var_results["var_results"]

                # Get portfolio returns for visualization
                portfolio_returns = risk_service._get_portfolio_returns(
                    portfolio_id, start_date, end_date
                )

                # Calculate CVaR
                cvar_value = calculate_cvar(
                    portfolio_returns, conf_decimal
                )

                # Calculate metrics for cards (95% and 99%)
                var_95_hist = (
                    var_data.get("historical")
                    if var_data.get("historical")
                    else 0
                )
                cvar_95 = calculate_cvar(portfolio_returns, 0.95)

                var_99_hist = None
                cvar_99 = None
                if (
                    portfolio_returns is not None
                    and not portfolio_returns.empty
                ):
                    from core.analytics_engine.risk_metrics import (
                        calculate_var,
                    )
                    var_99_hist = calculate_var(
                        portfolio_returns, 0.99, "historical"
                    )
                    cvar_99 = calculate_cvar(portfolio_returns, 0.99)

                # Display metrics cards
                st.markdown("### Key Risk Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "VaR (95%)",
                        f"{var_95_hist * 100:.2f}%" if var_95_hist else "N/A",
                        help="Maximum loss at 95% confidence"
                    )
                with col2:
                    st.metric(
                        "CVaR (95%)",
                        f"{cvar_95 * 100:.2f}%" if cvar_95 else "N/A",
                        help="Expected loss beyond VaR at 95% confidence"
                    )
                with col3:
                    st.metric(
                        "VaR (99%)",
                        f"{var_99_hist * 100:.2f}%" if var_99_hist else "N/A",
                        help="Maximum loss at 99% confidence"
                    )
                with col4:
                    st.metric(
                        "CVaR (99%)",
                        f"{cvar_99 * 100:.2f}%" if cvar_99 else "N/A",
                        help="Expected loss beyond VaR at 99% confidence"
                    )

                # Interpretation: VaR metrics
                interpretation = _interpret_var_metrics(var_95_hist, cvar_95, var_99_hist, cvar_99)
                if interpretation:
                    st.info(interpretation)

                st.markdown("---")

                # Create comparison table
                comparison_data = []
                methods = [
                    "historical",
                    "parametric",
                    "cornish_fisher",
                    "monte_carlo",
                ]
                method_names = {
                    "historical": "Historical",
                    "parametric": "Parametric",
                    "cornish_fisher": "Cornish-Fisher",
                    "monte_carlo": "Monte Carlo",
                }

                for method in methods:
                    var_value = var_data.get(method)
                    if (
                        var_value is not None
                        and not isinstance(var_value, str)
                    ):
                        comparison_data.append({
                            "Method": method_names[method],
                            f"VaR ({confidence_level}%)": f"{var_value:.4f}",
                            "VaR (%)": f"{var_value * 100:.2f}%",
                        })
                
                # Add CVaR to comparison table
                if cvar_value is not None:
                    comparison_data.append({
                        "Method": "CVaR (Expected Shortfall)",
                        f"VaR ({confidence_level}%)": (
                            f"{cvar_value:.4f}"
                        ),
                        "VaR (%)": f"{cvar_value * 100:.2f}%",
                    })

                if comparison_data:
                    st.markdown("### VaR Methods Comparison")
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Interpretation: VaR methods comparison
                    interpretation = _interpret_var_methods_comparison(var_data, conf_decimal)
                    if interpretation:
                        st.info(interpretation)

                    # Add distribution chart
                    if (
                        portfolio_returns is not None
                        and not portfolio_returns.empty
                    ):
                        st.markdown("---")
                        st.markdown(
                            "### VaR on Return Distribution"
                        )
                        try:
                            var_hist = var_data.get("historical", 0)
                            if var_hist is not None:
                                fig_dist = plot_var_distribution(
                                    portfolio_returns,
                                    var_hist,
                                    cvar_value,
                                    conf_decimal,
                                )
                                st.plotly_chart(
                                    fig_dist, use_container_width=True
                                )
                                pct_days = int((1 - conf_decimal) * 100)
                                st.caption(
                                    f"**Interpretation:** {pct_days}% of "
                                    f"days have returns less than "
                                    f"{var_hist*100:.2f}%. On those worst "
                                    f"days, the average loss is "
                                    f"{cvar_value*100:.2f}% (CVaR)."
                                )
                        except Exception as e:
                            logger.warning(
                                f"Error plotting VaR distribution: {e}"
                            )

                    # Add VaR Sensitivity Chart
                    st.markdown("---")
                    st.markdown("### VaR Sensitivity Analysis")
                    try:
                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.risk_metrics import (
                                calculate_var,
                            )

                            confidence_levels = [0.90, 0.95, 0.99]
                            var_sensitivity = []
                            cvar_sensitivity = []

                            for cl in confidence_levels:
                                var_val = calculate_var(
                                    portfolio_returns, cl, "historical"
                                )
                                cvar_val = calculate_cvar(
                                    portfolio_returns, cl
                                )
                                var_sensitivity.append(var_val * 100)
                                cvar_sensitivity.append(cvar_val * 100)

                            fig_sensitivity = go.Figure()
                            fig_sensitivity.add_trace(
                                go.Scatter(
                                    x=[
                                        int(cl * 100)
                                        for cl in confidence_levels
                                    ],
                                    y=var_sensitivity,
                                    mode="lines+markers",
                                    name="VaR",
                                    line=dict(
                                        color=COLORS["danger"], width=3
                                    ),
                                    marker=dict(size=10),
                                )
                            )
                            fig_sensitivity.add_trace(
                                go.Scatter(
                                    x=[
                                        int(cl * 100)
                                        for cl in confidence_levels
                                    ],
                                    y=cvar_sensitivity,
                                    mode="lines+markers",
                                    name="CVaR",
                                    line=dict(
                                        color=COLORS["warning"], width=3
                                    ),
                                    marker=dict(size=10),
                                )
                            )

                            fig_sensitivity.update_layout(
                                title=(
                                    "VaR and CVaR at Different "
                                    "Confidence Levels"
                                ),
                                xaxis_title="Confidence Level (%)",
                                yaxis_title="Risk (%)",
                                height=400,
                                template="plotly_dark",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                ),
                            )

                            st.plotly_chart(
                                fig_sensitivity, use_container_width=True
                            )
                            st.caption(
                                "**Interpretation:** Higher confidence "
                                "levels (99%) show more extreme potential "
                                "losses. CVaR is always more negative than "
                                "VaR, representing the average loss in tail "
                                "events."
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error creating VaR sensitivity chart: {e}"
                        )

                    # Add Rolling VaR Chart
                    st.markdown("---")
                    st.markdown("### Rolling VaR Analysis")
                    try:
                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.chart_data import (
                                get_rolling_var_data,
                            )
                            from streamlit_app.components.charts import (
                                plot_rolling_var,
                            )

                            # Window selector
                            rolling_window = st.slider(
                                "Rolling Window (days)",
                                min_value=30,
                                max_value=252,
                                value=63,
                                step=21,
                                key="rolling_var_window",
                                help="Window size for rolling VaR calculation",
                            )

                            var_rolling_data = get_rolling_var_data(
                                portfolio_returns,
                                None,  # No benchmark for now
                                window=rolling_window,
                                confidence_level=conf_decimal,
                            )

                            if var_rolling_data:
                                fig_rolling = plot_rolling_var(
                                    var_rolling_data
                                )
                                st.plotly_chart(
                                    fig_rolling, use_container_width=True
                                )

                                # Statistics
                                stats = var_rolling_data.get(
                                    "portfolio_stats", {}
                                )
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Avg VaR",
                                        f"{stats.get('avg', 0)*100:.2f}%",
                                    )
                                with col2:
                                    st.metric(
                                        "Median VaR",
                                        f"{stats.get('median', 0)*100:.2f}%",
                                    )
                                with col3:
                                    st.metric(
                                        "Min VaR",
                                        f"{stats.get('min', 0)*100:.2f}%",
                                    )
                                with col4:
                                    st.metric(
                                        "Max VaR",
                                        f"{stats.get('max', 0)*100:.2f}%",
                                    )

                                # Interpretation: Rolling VaR
                                interpretation = _interpret_rolling_var(stats)
                                if interpretation:
                                    st.info(interpretation)
                    except Exception as e:
                        logger.warning(
                            f"Error creating rolling VaR chart: {e}"
                        )

                    # Add Portfolio VaR with Covariance Matrix
                    st.markdown("---")
                    st.markdown("### Portfolio VaR with Covariance Matrix")
                    try:
                        # Get portfolio positions and price data
                        portfolio_service = PortfolioService()
                        portfolio = portfolio_service.get_portfolio(
                            portfolio_id
                        )
                        positions = portfolio.positions if portfolio else []

                        if positions:
                            from services.data_service import DataService
                            data_service = DataService()

                            # Get tickers (exclude CASH)
                            tickers = [
                                pos.ticker
                                for pos in positions
                                if pos.ticker != "CASH"
                            ]

                            if len(tickers) >= 2:
                                # Fetch price data
                                all_prices = []
                                for ticker in tickers:
                                    try:
                                        prices = (
                                            data_service.fetch_historical_prices(
                                                ticker,
                                                start_date,
                                                end_date,
                                                use_cache=True,
                                                save_to_db=False,
                                            )
                                        )
                                        if not prices.empty:
                                            prices["Ticker"] = ticker
                                            all_prices.append(prices)
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to fetch {ticker}: {e}"
                                        )

                                if all_prices:
                                    # Combine and pivot
                                    combined = pd.concat(
                                        all_prices, ignore_index=True
                                    )
                                    price_data = combined.pivot_table(
                                        index="Date",
                                        columns="Ticker",
                                        values="Adjusted_Close",
                                        aggfunc="first",
                                    )

                                    # Calculate returns
                                    returns_df = price_data.pct_change().dropna()

                                    # Calculate weights
                                    total_value = sum(
                                        pos.shares
                                        * price_data[pos.ticker].iloc[0]
                                        for pos in positions
                                        if pos.ticker in price_data.columns
                                        and pos.ticker != "CASH"
                                    )

                                    weights = np.array([
                                        (
                                            pos.shares
                                            * price_data[pos.ticker].iloc[0]
                                            / total_value
                                        )
                                        if pos.ticker in price_data.columns
                                        and pos.ticker != "CASH"
                                        else 0.0
                                        for pos in positions
                                        if pos.ticker != "CASH"
                                    ])

                                    # Calculate Portfolio VaR
                                    from core.risk_engine.var_calculator import (
                                        calculate_portfolio_var_covariance,
                                    )

                                    portfolio_var_result = (
                                        calculate_portfolio_var_covariance(
                                            returns_df,
                                            weights,
                                            conf_decimal,
                                            time_horizon,
                                        )
                                    )

                                    # Display Portfolio VaR
                                    st.metric(
                                        "Portfolio VaR (Covariance Method)",
                                        f"{portfolio_var_result['portfolio_var']*100:.2f}%",
                                        help=(
                                            "VaR calculated using covariance "
                                            "matrix, accounting for "
                                            "correlations between assets"
                                        ),
                                    )
                                    
                                    # Interpretation: Portfolio VaR
                                    interpretation = _interpret_portfolio_var_covariance(
                                        portfolio_var_result['portfolio_var'],
                                        conf_decimal
                                    )
                                    if interpretation:
                                        st.info(interpretation)

                                    # VaR Decomposition
                                    st.markdown("#### VaR Decomposition")
                                    decomposition_data = []
                                    for ticker in returns_df.columns:
                                        if ticker in portfolio_var_result[
                                            "component_var"
                                        ]:
                                            decomposition_data.append({
                                                "Asset": ticker,
                                                "Weight (%)": (
                                                    weights[
                                                        list(returns_df.columns).index(ticker)
                                                    ]
                                                    * 100
                                                    if ticker in returns_df.columns
                                                    else 0.0
                                                ),
                                                "Component VaR (%)": (
                                                    portfolio_var_result[
                                                        "component_var"
                                                    ][ticker]
                                                    * 100
                                                ),
                                                "Contribution (%)": (
                                                    portfolio_var_result[
                                                        "var_contribution_pct"
                                                    ][ticker]
                                                ),
                                                "Marginal VaR (%)": (
                                                    portfolio_var_result[
                                                        "marginal_var"
                                                    ][ticker]
                                                    * 100
                                                ),
                                            })

                                    if decomposition_data:
                                        import pandas as pd
                                        decomp_df = pd.DataFrame(
                                            decomposition_data
                                        )
                                        st.dataframe(
                                            decomp_df, use_container_width=True
                                        )
                                        
                                        # Interpretation: VaR decomposition
                                        interpretation = _interpret_portfolio_var_decomposition(
                                            decomposition_data,
                                            portfolio_var_result['portfolio_var']
                                        )
                                        if interpretation:
                                            st.info(interpretation)

                                        # Bar chart of contributions
                                        fig_decomp = go.Figure()
                                        fig_decomp.add_trace(
                                            go.Bar(
                                                x=decomp_df["Asset"],
                                                y=decomp_df["Contribution (%)"],
                                                marker=dict(
                                                    color=COLORS["primary"]
                                                ),
                                                text=[
                                                    f"{v:.1f}%"
                                                    for v in decomp_df[
                                                        "Contribution (%)"
                                                    ]
                                                ],
                                                textposition="outside",
                                            )
                                        )
                                        fig_decomp.update_layout(
                                            title=(
                                                "VaR Contribution by Asset "
                                                "(%)"
                                            ),
                                            xaxis_title="Asset",
                                            yaxis_title="Contribution (%)",
                                            height=400,
                                            template="plotly_dark",
                                        )
                                        st.plotly_chart(
                                            fig_decomp, use_container_width=True
                                        )

                                        st.caption(
                                            "**Component VaR:** Contribution "
                                            "of each asset to total portfolio "
                                            "VaR. **Marginal VaR:** Change in "
                                            "portfolio VaR for a 1% increase "
                                            "in asset weight."
                                        )
                    except Exception as e:
                        logger.warning(
                            f"Error calculating Portfolio VaR: {e}"
                        )

                    # Visual comparison
                    try:
                        fig = go.Figure()
                        methods_list = [d["Method"] for d in comparison_data]

                        # Extract and convert VaR values safely
                        var_values = []
                        for d in comparison_data:
                            var_str = d.get(f"VaR ({confidence_level}%)", "0")
                            try:
                                # Remove commas and convert to float
                                var_float = float(
                                    str(var_str).replace(",", "")
                                )
                                var_values.append(var_float)
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    f"Could not convert VaR value "
                                    f"'{var_str}': {e}"
                                )
                                var_values.append(0.0)

                        if not var_values or all(v == 0.0 for v in var_values):
                            raise ValueError("No valid VaR values to plot")

                        # Format: percentages above, absolute values below
                        var_values_pct = [v * 100 for v in var_values]
                        var_values_abs = [abs(v) for v in var_values]

                        fig.add_trace(
                            go.Bar(
                                x=methods_list,
                                y=var_values_abs,
                                marker_color=COLORS["error"],
                                text=[f"{p:.1f}%" for p in var_values_pct],
                                textposition="outside",
                                textfont=dict(size=12),
                            )
                        )

                        title_text = (
                            f"VaR Comparison ({confidence_level}% Confidence)"
                        )
                        fig.update_layout(
                            title=title_text,
                            xaxis_title="Method",
                            yaxis_title="VaR (absolute value)",
                            height=400,
                            template="plotly_dark",
                        )

                        # Add absolute values below bars
                        fig.add_annotation(
                            text="Values shown: percentages above bars, "
                            "absolute values on Y-axis",
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            font=dict(size=10, color="gray"),
                        )

                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation: VaR comparison chart
                        interpretation = _interpret_var_methods_comparison(
                            var_data, confidence_level / 100
                        )
                        if interpretation:
                            st.info(interpretation)
                    except Exception as chart_error:
                        logger.warning(
                            f"Error creating VaR chart: {chart_error}"
                        )
                        st.warning(
                            "Could not display VaR comparison chart, "
                            "but calculations completed successfully."
                        )
                else:
                    st.warning(
                        "No VaR values could be calculated. "
                        "Please check your portfolio data."
                    )

        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            st.error(f"Error calculating VaR: {error_msg}")
            logger.exception(f"VaR calculation error: {error_msg}")


def _render_monte_carlo(
    risk_service: RiskService,
    portfolio_id: str,
    start_date: date,
    end_date: date,
) -> None:
    """Render Monte Carlo simulation section."""
    st.subheader("Monte Carlo Simulation")

    st.info(
        "Monte Carlo simulation forecasts portfolio value **forward** "
        "in time based on historical return patterns. "
        "Time horizon = number of days **into the future** to simulate."
    )

    # Simulation parameters
    col1, col2 = st.columns(2)

    with col1:
        time_horizon = st.number_input(
            "Time Horizon (days forward)",
            min_value=1,
            max_value=252,
            value=30,
            key="mc_horizon",
            help="Number of days into the future to simulate",
        )

        num_simulations = st.selectbox(
            "Number of Simulations",
            options=[1000, 5000, 10000, 50000],
            index=2,
            key="mc_simulations",
        )

    with col2:
        initial_value = st.number_input(
            "Initial Portfolio Value",
            min_value=1.0,
            value=100000.0,
            step=10000.0,
            key="mc_initial",
        )

        model = st.selectbox(
            "Model",
            options=["gbm", "jump_diffusion"],
            key="mc_model",
            help="GBM: Geometric Brownian Motion\n"
            "Jump Diffusion: Includes jump events",
        )

    # Show paths option
    show_paths = st.checkbox(
        "Show Paths Chart (Spaghetti Chart)",
        value=False,
        key="mc_show_paths",
        help="Display individual simulation paths",
    )

    # Run simulation
    if st.button("Run Simulation", key="run_mc"):
        try:
            with st.spinner(
                f"Running {num_simulations:,} simulations..."
            ):
                results = risk_service.run_monte_carlo_simulation(
                    portfolio_id=portfolio_id,
                    start_date=start_date,
                    end_date=end_date,
                    time_horizon=time_horizon,
                    num_simulations=num_simulations,
                    initial_value=initial_value,
                    model=model,
                )

                st.success("Simulation completed!")

                # Display statistics
                st.markdown("### Simulation Statistics")
                stats = results["statistics"]
                percentiles = results["percentiles"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"${stats['mean']:,.2f}")
                    st.metric("Median", f"${stats['median']:,.2f}")
                with col2:
                    st.metric("Std Dev", f"${stats['std']:,.2f}")
                    st.metric("Min", f"${stats['min']:,.2f}")
                with col3:
                    st.metric("Max", f"${stats['max']:,.2f}")
                
                # Interpretation: Monte Carlo statistics
                interpretation = _interpret_monte_carlo_statistics(stats, initial_value)
                if interpretation:
                    st.info(interpretation)

                # Percentiles
                st.markdown("### Percentile Outcomes")
                percentile_data = [
                    {"Percentile": f"{p}%", "Value": f"${v:,.2f}"}
                    for p, v in percentiles.items()
                ]
                import pandas as pd
                df = pd.DataFrame(percentile_data)
                st.dataframe(df, use_container_width=True)
                
                # Interpretation: Percentiles
                interpretation = _interpret_monte_carlo_percentiles(percentiles, initial_value)
                if interpretation:
                    st.info(interpretation)

                # Distribution histogram - use same style as other distributions
                final_values = results["final_values"]
                final_values_array = np.array(final_values)
                
                # Create histogram data similar to get_return_distribution_data
                counts, edges = np.histogram(final_values_array, bins=50)
                mean_val = float(np.mean(final_values_array))
                std_val = float(np.std(final_values_array))
                
                fig = go.Figure()
                
                # Histogram bars (purple)
                bin_centers = (edges[:-1] + edges[1:]) / 2
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=counts,
                        name="Final Value Distribution",
                        marker_color=COLORS["primary"],
                        opacity=0.7,
                    )
                )
                
                # Mean line (blue dashed)
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color=COLORS["secondary"],  # Blue
                    annotation_text=f"Mean: ${mean_val:,.0f}",
                    annotation_position="top",
                )
                
                # Normal distribution overlay (orange dashed)
                if std_val > 0:
                    from scipy import stats as scipy_stats
                    x_norm = np.linspace(edges[0], edges[-1], 100)
                    pdf_values = scipy_stats.norm.pdf(x_norm, loc=mean_val, scale=std_val)
                    y_norm = pdf_values * len(counts) * (edges[1] - edges[0])
                    fig.add_trace(
                        go.Scatter(
                            x=x_norm,
                            y=y_norm,
                            mode="lines",
                            name="Normal Distribution",
                            line=dict(color=COLORS["warning"], width=2, dash="dash"),  # Orange
                        )
                    )
                
                # Add percentile lines (90%, 95%, 99%) - like other distributions
                # Lower tail (negative side)
                p5 = percentiles.get(5.0) or percentiles.get(5)
                p10 = percentiles.get(10.0) or percentiles.get(10)
                if p5 is not None:
                    fig.add_annotation(
                        x=p5,
                        y=1.0,
                        xref="x",
                        yref="paper",
                        text=f"${p5:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color=COLORS["danger"]),
                        bgcolor="rgba(0,0,0,0.7)",
                    )
                    fig.add_vline(
                        x=p5,
                        line_dash="dot",
                        line_color=COLORS["danger"],
                        annotation_text="5%",
                        annotation_position="bottom",
                        line_width=1,
                    )
                if p10 is not None:
                    fig.add_annotation(
                        x=p10,
                        y=1.0,
                        xref="x",
                        yref="paper",
                        text=f"${p10:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color=COLORS["danger"]),
                        bgcolor="rgba(0,0,0,0.7)",
                    )
                    fig.add_vline(
                        x=p10,
                        line_dash="dot",
                        line_color=COLORS["danger"],
                        annotation_text="10%",
                        annotation_position="bottom",
                        line_width=1,
                    )
                
                # Upper tail (positive side)
                p90 = percentiles.get(90.0) or percentiles.get(90)
                p95 = percentiles.get(95.0) or percentiles.get(95)
                p99 = percentiles.get(99.0) or percentiles.get(99)
                if p90 is not None:
                    fig.add_annotation(
                        x=p90,
                        y=1.0,
                        xref="x",
                        yref="paper",
                        text=f"${p90:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color=COLORS["success"]),
                        bgcolor="rgba(0,0,0,0.7)",
                    )
                    fig.add_vline(
                        x=p90,
                        line_dash="dot",
                        line_color=COLORS["success"],
                        annotation_text="90%",
                        annotation_position="bottom",
                        line_width=1,
                    )
                if p95 is not None:
                    fig.add_annotation(
                        x=p95,
                        y=1.0,
                        xref="x",
                        yref="paper",
                        text=f"${p95:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color=COLORS["success"]),
                        bgcolor="rgba(0,0,0,0.7)",
                    )
                    fig.add_vline(
                        x=p95,
                        line_dash="dot",
                        line_color=COLORS["success"],
                        annotation_text="95%",
                        annotation_position="bottom",
                        line_width=1,
                    )
                if p99 is not None:
                    fig.add_annotation(
                        x=p99,
                        y=1.0,
                        xref="x",
                        yref="paper",
                        text=f"${p99:,.0f}",
                        showarrow=False,
                        font=dict(size=10, color=COLORS["success"]),
                        bgcolor="rgba(0,0,0,0.7)",
                    )
                    fig.add_vline(
                        x=p99,
                        line_dash="dot",
                        line_color=COLORS["success"],
                        annotation_text="99%",
                        annotation_position="bottom",
                        line_width=1,
                    )
                
                # Add initial value line (white)
                fig.add_vline(
                    x=initial_value,
                    line_dash="dash",
                    line_color="white",
                    annotation_text=f"Initial: ${initial_value:,.0f}",
                    annotation_position="top",
                )

                from streamlit_app.utils.chart_config import get_chart_layout
                layout = get_chart_layout(
                    title="Monte Carlo Simulation: Final Value Distribution",
                    xaxis=dict(title="Portfolio Value ($)", tickformat="$,.0f"),
                    yaxis=dict(title="Frequency"),
                    hovermode="x unified",
                )
                fig.update_layout(**layout)

                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation: Final value distribution with confidence intervals
                parts = []
                parts.append("**Final Value Distribution Analysis:**")
                
                # Distribution shape
                statistics = results.get("statistics", {})
                mean_val = statistics.get("mean", 0) if isinstance(statistics, dict) else 0
                median_val = statistics.get("median", 0) if isinstance(statistics, dict) else 0
                if mean_val > median_val * 1.05:
                    parts.append("Right-skewed distribution - More upside potential than downside risk")
                elif mean_val < median_val * 0.95:
                    parts.append("Left-skewed distribution - More downside risk than upside potential")
                else:
                    parts.append("Symmetric distribution - Balanced upside and downside")
                
                # Confidence Intervals Analysis (from second chart)
                final_values_array = np.array(final_values)
                ci_levels = [0.90, 0.95, 0.99]
                
                for ci in ci_levels:
                    lower = np.percentile(final_values_array, (1 - ci) / 2 * 100)
                    upper = np.percentile(final_values_array, (1 + ci) / 2 * 100)
                    
                    lower_return = (lower / initial_value - 1) * 100 if initial_value > 0 else 0
                    upper_return = (upper / initial_value - 1) * 100 if initial_value > 0 else 0
                    range_pct = ((upper - lower) / initial_value) * 100 if initial_value > 0 else 0
                    
                    # Format with HTML non-breaking spaces to prevent awkward line breaks
                    lower_str = f"${lower:,.2f}".replace(" ", "\u00A0")
                    upper_str = f"${upper:,.2f}".replace(" ", "\u00A0")
                    parts.append(
                        f"{int(ci*100)}% CI: {lower_str}\u00A0to\u00A0{upper_str} "
                        f"({range_pct:.1f}% range, "
                        f"{lower_return:+.1f}%\u00A0to\u00A0{upper_return:+.1f}% return)"
                    )
                
                # Overall assessment (from confidence intervals)
                ci_90_lower = np.percentile(final_values_array, 5)
                ci_90_upper = np.percentile(final_values_array, 95)
                ci_99_lower = np.percentile(final_values_array, 0.5)
                ci_99_upper = np.percentile(final_values_array, 99.5)
                
                tail_width_90 = (ci_90_upper - ci_90_lower) / initial_value * 100 if initial_value > 0 else 0
                tail_width_99 = (ci_99_upper - ci_99_lower) / initial_value * 100 if initial_value > 0 else 0
                
                if tail_width_90 < 10:
                    parts.append(f"Narrow 90% range ({tail_width_90:.1f}%) - Relatively predictable outcomes")
                elif tail_width_90 < 25:
                    parts.append(f"Moderate 90% range ({tail_width_90:.1f}%) - Some outcome variability")
                else:
                    parts.append(f"⚠ Wide 90% range ({tail_width_90:.1f}%) - High outcome uncertainty")
                
                tail_expansion = tail_width_99 / tail_width_90 if tail_width_90 > 0 else 0
                if tail_expansion > 1.5:
                    parts.append(f"Significant tail expansion (99% range is {tail_expansion:.1f}x wider) - Fat tails, extreme events possible")
                elif tail_expansion > 1.2:
                    parts.append(f"Moderate tail expansion (99% range is {tail_expansion:.1f}x wider) - Some tail risk")
                else:
                    parts.append(f"Normal tail behavior (99% range is {tail_expansion:.1f}x wider) - Relatively normal distribution")
                
                interpretation = "\n".join(parts)
                if interpretation:
                    st.info(interpretation)

                # VaR/CVaR from simulations
                st.markdown("---")
                st.markdown("### VaR & CVaR from Simulations")
                try:

                    # Calculate returns from final values
                    returns_sim = (
                        np.array(final_values) / initial_value - 1.0
                    )

                    # Calculate VaR and CVaR at different levels
                    confidence_levels_mc = [0.90, 0.95, 0.99]
                    var_cvar_data = []

                    for cl in confidence_levels_mc:
                        # VaR = percentile
                        var_val = np.percentile(
                            returns_sim, (1 - cl) * 100
                        )
                        # CVaR = mean of returns below VaR
                        tail_returns = returns_sim[
                            returns_sim <= var_val
                        ]
                        cvar_val = (
                            tail_returns.mean()
                            if len(tail_returns) > 0
                            else var_val
                        )

                        var_cvar_data.append({
                            "Confidence": f"{int(cl*100)}%",
                            "VaR (%)": f"{var_val*100:.2f}%",
                            "VaR ($)": (
                                f"${var_val*initial_value:,.2f}"
                            ),
                            "CVaR (%)": f"{cvar_val*100:.2f}%",
                            "CVaR ($)": (
                                f"${cvar_val*initial_value:,.2f}"
                            ),
                        })

                    import pandas as pd
                    var_cvar_df = pd.DataFrame(var_cvar_data)
                    st.dataframe(var_cvar_df, use_container_width=True)
                    
                    # Interpretation: VaR & CVaR from simulations
                    parts = []
                    parts.append("**VaR & CVaR Analysis:**")
                    
                    # Get 95% VaR and CVaR
                    var_95_data = next((d for d in var_cvar_data if d["Confidence"] == "95%"), None)
                    if var_95_data:
                        var_95_pct = float(var_95_data["VaR (%)"].replace("%", ""))
                        cvar_95_pct = float(var_95_data["CVaR (%)"].replace("%", ""))
                        
                        parts.append(f"95% VaR: {abs(var_95_pct):.2f}% - Expected loss in worst 5% of scenarios")
                        parts.append(f"95% CVaR: {abs(cvar_95_pct):.2f}% - Average loss in worst 5% of scenarios")
                        
                        # CVaR vs VaR spread
                        cvar_spread = abs(cvar_95_pct) - abs(var_95_pct)
                        if cvar_spread > 2:
                            parts.append(f"Large CVaR-VaR spread ({cvar_spread:.2f}%) - Significant tail risk beyond VaR threshold")
                        elif cvar_spread > 1:
                            parts.append(f"Moderate CVaR-VaR spread ({cvar_spread:.2f}%) - Some tail risk")
                        else:
                            parts.append(f"Small CVaR-VaR spread ({cvar_spread:.2f}%) - Limited tail risk")
                    
                    # Compare across confidence levels
                    if len(var_cvar_data) >= 2:
                        var_90 = float(next((d for d in var_cvar_data if d["Confidence"] == "90%"), {}).get("VaR (%)", "0").replace("%", ""))
                        var_99 = float(next((d for d in var_cvar_data if d["Confidence"] == "99%"), {}).get("VaR (%)", "0").replace("%", ""))
                        if var_90 and var_99:
                            var_expansion = abs(var_99) / abs(var_90) if abs(var_90) > 0 else 0
                            if var_expansion > 2:
                                parts.append(f"⚠ High risk escalation: 99% VaR is {var_expansion:.1f}x higher than 90% VaR - Extreme events significantly increase risk")
                            elif var_expansion > 1.5:
                                parts.append(f"Moderate risk escalation: 99% VaR is {var_expansion:.1f}x higher than 90% VaR")
                    
                    interpretation = "\n".join(parts)
                    if interpretation:
                        st.info(interpretation)

                    # Compare with historical VaR
                    st.markdown("---")
                    st.markdown("### Comparison with Historical VaR")
                    try:
                        portfolio_returns = (
                            risk_service._get_portfolio_returns(
                                portfolio_id, start_date, end_date
                            )
                        )

                        if (
                            portfolio_returns is not None
                            and not portfolio_returns.empty
                        ):
                            from core.analytics_engine.risk_metrics import (
                                calculate_var,
                            )

                            comparison_data = []
                            for cl in confidence_levels_mc:
                                # Historical VaR
                                hist_var = calculate_var(
                                    portfolio_returns, cl, "historical"
                                )
                                # Monte Carlo VaR
                                mc_var = np.percentile(
                                    returns_sim, (1 - cl) * 100
                                )

                                comparison_data.append({
                                    "Confidence": f"{int(cl*100)}%",
                                    "Historical VaR (%)": (
                                        f"{hist_var*100:.2f}%"
                                    ),
                                    "MC VaR (%)": (
                                        f"{mc_var*100:.2f}%"
                                    ),
                                    "Difference": (
                                        f"{(mc_var - hist_var)*100:.2f}%"
                                    ),
                                })

                            comp_df = pd.DataFrame(comparison_data)
                            st.dataframe(comp_df, use_container_width=True)
                            
                            # Interpretation: VaR comparison (use 95% confidence)
                            hist_var_95 = calculate_var(
                                portfolio_returns, 0.95, "historical"
                            )
                            mc_var_95 = np.percentile(
                                returns_sim, (1 - 0.95) * 100
                            )
                            interpretation = _interpret_monte_carlo_var_comparison(
                                hist_var_95, mc_var_95, 0.95
                            )
                            if interpretation:
                                st.info(interpretation)

                            # Visual comparison
                            fig_comp = go.Figure()
                            fig_comp.add_trace(
                                go.Bar(
                                    x=[int(cl * 100) for cl in confidence_levels_mc],
                                    y=[
                                        calculate_var(
                                            portfolio_returns, cl, "historical"
                                        )
                                        * 100
                                        for cl in confidence_levels_mc
                                    ],
                                    name="Historical VaR",
                                    marker_color=COLORS["danger"],
                                )
                            )
                            fig_comp.add_trace(
                                go.Bar(
                                    x=[int(cl * 100) for cl in confidence_levels_mc],
                                    y=[
                                        np.percentile(
                                            returns_sim, (1 - cl) * 100
                                        )
                                        * 100
                                        for cl in confidence_levels_mc
                                    ],
                                    name="Monte Carlo VaR",
                                    marker_color=COLORS["warning"],
                                )
                            )

                            fig_comp.update_layout(
                                title=(
                                    "Historical VaR vs Monte Carlo VaR"
                                ),
                                xaxis_title="Confidence Level (%)",
                                yaxis_title="VaR (%)",
                                barmode="group",
                                height=400,
                                template="plotly_dark",
                            )
                            st.plotly_chart(
                                fig_comp, use_container_width=True
                            )
                            
                            # Interpretation: Historical vs Monte Carlo VaR chart
                            parts = []
                            parts.append("**Historical vs Monte Carlo VaR Comparison (Chart):**")
                            
                            # Calculate differences for all confidence levels
                            differences = []
                            for cl in confidence_levels_mc:
                                hist_var = calculate_var(portfolio_returns, cl, "historical")
                                mc_var = np.percentile(returns_sim, (1 - cl) * 100)
                                diff = mc_var - hist_var
                                differences.append({
                                    "confidence": cl,
                                    "diff": diff,
                                    "hist": hist_var,
                                    "mc": mc_var
                                })
                            
                            # Overall pattern
                            avg_diff = np.mean([d["diff"] for d in differences])
                            if abs(avg_diff) < 0.01:
                                parts.append("Methods agree closely - Historical and Monte Carlo VaR are consistent across confidence levels")
                            elif avg_diff > 0:
                                parts.append(f"Monte Carlo consistently higher ({avg_diff*100:.2f}% avg) - Forward-looking model suggests more risk than historical patterns")
                            else:
                                parts.append(f"Monte Carlo consistently lower ({abs(avg_diff)*100:.2f}% avg) - Forward-looking model suggests less risk than historical patterns")
                            
                            # Confidence level analysis
                            if len(differences) >= 2:
                                low_conf_diff = differences[0]["diff"]
                                high_conf_diff = differences[-1]["diff"]
                                if abs(high_conf_diff) > abs(low_conf_diff) * 1.5:
                                    parts.append("Difference increases at higher confidence levels - Model uncertainty grows for extreme events")
                            
                            interpretation = "\n".join(parts)
                            if interpretation:
                                st.info(interpretation)

                    except Exception as e:
                        logger.warning(
                            f"Error comparing with historical VaR: {e}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error calculating VaR/CVaR from simulations: {e}"
                    )

                # Extreme scenarios analysis
                st.markdown("---")
                st.markdown("### Extreme Scenarios Analysis")
                try:
                    # Worst case scenarios
                    worst_5_pct = np.percentile(final_values, 5)
                    worst_1_pct = np.percentile(final_values, 1)
                    worst_0_1_pct = np.percentile(final_values, 0.1)

                    # Best case scenarios
                    best_5_pct = np.percentile(final_values, 95)
                    best_1_pct = np.percentile(final_values, 99)
                    best_0_1_pct = np.percentile(final_values, 99.9)

                    extreme_data = [
                        {
                            "Scenario": "Worst 5%",
                            "Value": f"${worst_5_pct:,.2f}",
                            "Return": (
                                f"${worst_5_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_5_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Worst 1%",
                            "Value": f"${worst_1_pct:,.2f}",
                            "Return": (
                                f"${worst_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Worst 0.1%",
                            "Value": f"${worst_0_1_pct:,.2f}",
                            "Return": (
                                f"${worst_0_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((worst_0_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 5%",
                            "Value": f"${best_5_pct:,.2f}",
                            "Return": f"${best_5_pct - initial_value:,.2f}",
                            "Return %": (
                                f"{((best_5_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 1%",
                            "Value": f"${best_1_pct:,.2f}",
                            "Return": f"${best_1_pct - initial_value:,.2f}",
                            "Return %": (
                                f"{((best_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                        {
                            "Scenario": "Best 0.1%",
                            "Value": f"${best_0_1_pct:,.2f}",
                            "Return": (
                                f"${best_0_1_pct - initial_value:,.2f}"
                            ),
                            "Return %": (
                                f"{((best_0_1_pct - initial_value)/initial_value)*100:.2f}%"
                            ),
                        },
                    ]

                    extreme_df = pd.DataFrame(extreme_data)
                    st.dataframe(extreme_df, use_container_width=True)
                    
                    # Interpretation: Extreme scenarios
                    interpretation = _interpret_extreme_scenarios(extreme_df, initial_value)
                    if interpretation:
                        st.info(interpretation)

                    # Visual comparison
                    fig_extreme = go.Figure()
                    scenarios = [d["Scenario"] for d in extreme_data]
                    values = [
                        float(d["Value"].replace("$", "").replace(",", ""))
                        for d in extreme_data
                    ]

                    colors_extreme = [
                        COLORS["danger"]
                        if "Worst" in s
                        else COLORS["success"]
                        for s in scenarios
                    ]

                    fig_extreme.add_trace(
                        go.Bar(
                            x=scenarios,
                            y=values,
                            marker_color=colors_extreme,
                            text=[f"${v:,.0f}" for v in values],
                            textposition="outside",
                        )
                    )

                    # Add initial value line (white)
                    fig_extreme.add_hline(
                        y=initial_value,
                        line_dash="dash",
                        line_color="white",
                        annotation_text=f"Initial: ${initial_value:,.0f}",
                    )

                    fig_extreme.update_layout(
                        title="Extreme Scenarios: Best vs Worst Cases",
                        xaxis_title="Scenario",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_extreme, use_container_width=True)

                except Exception as e:
                    logger.warning(
                        f"Error analyzing extreme scenarios: {e}"
                    )

                # Spaghetti chart moved here (after Final Value Distribution)
                if show_paths and results.get("simulated_paths"):
                    st.markdown("---")
                    st.markdown("### Simulation Paths (Spaghetti Chart)")
                    paths = results["simulated_paths"]
                    time_horizon = results["time_horizon"]

                    fig_paths = go.Figure()

                    # Convert paths to numpy array for max/min calculation
                    paths_array = np.array(paths)
                    
                    # Calculate max and min paths
                    max_path = np.max(paths_array, axis=0)
                    min_path = np.min(paths_array, axis=0)

                    # Plot a sample of paths (max 100 for performance)
                    num_paths_to_show = min(100, len(paths))
                    step = (
                        len(paths) // num_paths_to_show
                        if num_paths_to_show > 0
                        else 1
                    )

                    for i in range(0, len(paths), step):
                        path_values = paths[i]
                        days = list(range(time_horizon))
                        fig_paths.add_trace(
                            go.Scatter(
                                x=days,
                                y=path_values,
                                mode="lines",
                                line=dict(
                                    width=0.5,
                                    color="rgba(191, 159, 251, 0.3)",
                                ),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

                    # Add max path (green from palette)
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=max_path,
                            mode="lines",
                            line=dict(width=2, color=COLORS["success"]),  # Green
                            name="Max Path",
                        )
                    )
                    
                    # Add min path (red from palette)
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=min_path,
                            mode="lines",
                            line=dict(width=2, color=COLORS["danger"]),  # Red
                            name="Min Path",
                        )
                    )

                    # Add initial value line (white)
                    fig_paths.add_hline(
                        y=initial_value,
                        line_dash="dash",
                        line_color="white",
                        annotation_text=f"Initial: ${initial_value:,.0f}",
                    )

                    # Add percentile paths
                    from core.risk_engine.monte_carlo import (
                        simulate_portfolio_paths,
                    )

                    # Get returns for percentile calculation
                    returns = risk_service._get_portfolio_returns(
                        portfolio_id, start_date, end_date
                    )

                    # Re-run simulation to get paths for percentiles
                    percentile_result = simulate_portfolio_paths(
                        returns,
                        time_horizon,
                        num_simulations,
                        initial_value,
                        model,
                    )

                    # Add median path (50th percentile)
                    median_path = percentile_result.simulated_paths[
                        np.argmin(
                            np.abs(
                                percentile_result.final_values
                                - percentile_result.percentiles[50.0]
                            )
                        )
                    ]
                    fig_paths.add_trace(
                        go.Scatter(
                            x=list(range(time_horizon)),
                            y=median_path,
                            mode="lines",
                            line=dict(width=2, color=COLORS["secondary"]),  # Blue
                            name="Median (50th percentile)",
                        )
                    )

                    fig_paths.update_layout(
                        title=f"Monte Carlo Simulation: {time_horizon} Days",
                        xaxis_title="Day",
                        yaxis_title="Portfolio Value ($)",
                        yaxis=dict(
                            tickformat="$,.0f",
                            tickmode="linear",
                            tick0=0,
                            dtick=10000,
                        ),
                        height=500,
                        template="plotly_dark",
                    )

                    st.plotly_chart(fig_paths, use_container_width=True)
                    
                    # Interpretation: Simulation paths
                    parts = []
                    parts.append("**Simulation Paths Analysis:**")
                    parts.append(f"Shows {len(paths)} individual simulation paths over {time_horizon} days")
                    
                    # Analyze path convergence/divergence
                    if len(paths) > 0 and len(paths[0]) > 0:
                        final_values_from_paths = [path[-1] for path in paths]
                        path_std = np.std(final_values_from_paths)
                        path_mean = np.mean(final_values_from_paths)
                        path_cv = path_std / path_mean if path_mean > 0 else 0
                        
                        if path_cv > 0.3:
                            parts.append(f"High path divergence (CV: {path_cv:.2f}) - Wide range of possible outcomes, high uncertainty")
                        elif path_cv > 0.15:
                            parts.append(f"Moderate path divergence (CV: {path_cv:.2f}) - Some outcome variability")
                        else:
                            parts.append(f"Low path divergence (CV: {path_cv:.2f}) - Relatively predictable outcomes")
                        
                        # Trend analysis
                        initial_val = paths[0][0] if paths else initial_value
                        positive_paths = sum(1 for fv in final_values_from_paths if fv > initial_val)
                        positive_pct = positive_paths / len(final_values_from_paths) * 100 if final_values_from_paths else 0
                        
                        if positive_pct > 60:
                            parts.append(f"Most paths ({positive_pct:.0f}%) end above initial value - Positive expected outcome")
                        elif positive_pct < 40:
                            parts.append(f"Most paths ({100-positive_pct:.0f}%) end below initial value - Negative expected outcome")
                        else:
                            parts.append(f"Mixed outcomes ({positive_pct:.0f}% positive) - Balanced risk/reward")
                    
                    interpretation = "\n".join(parts)
                    if interpretation:
                        st.info(interpretation)
                elif show_paths:
                    st.warning(
                        "Path data not available. Please ensure simulation "
                        "returns path information."
                    )

        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            logger.exception("Monte Carlo simulation error")


def _render_stress_tests(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render stress testing section."""
    st.subheader("Stress Testing")

    # Get available scenarios
    scenarios = risk_service.get_available_scenarios()

    if not scenarios:
        st.warning("No stress scenarios available.")
        return

    # Scenario selection
    scenario_names = list(scenarios.keys())
    selected_scenarios = st.multiselect(
        "Select Scenarios to Test",
        options=scenario_names,
        key="stress_scenarios",
        help="Select one or more historical scenarios to test",
    )

    if not selected_scenarios:
        st.info("Please select at least one scenario.")
        return

    # Display scenario descriptions
    with st.expander("Scenario Descriptions", expanded=False):
        for key in selected_scenarios:
            scenario = scenarios[key]
            st.markdown(f"**{scenario.name}**")
            st.markdown(f"*{scenario.description}*")
            st.markdown(
                f"Period: {scenario.start_date} to {scenario.end_date}"
            )
            st.markdown(
                f"Market Impact: {scenario.market_impact_pct * 100:.1f}%"
            )
            st.markdown("---")

    # Run stress tests
    if st.button("Run Stress Tests", key="run_stress"):
        try:
            with st.spinner("Running stress tests..."):
                results = risk_service.run_stress_test(
                    portfolio_id=portfolio_id,
                    scenario_names=selected_scenarios,
                )

                st.success("Stress tests completed!")

                # Display results
                st.markdown("### Stress Test Results")

                # Create results table
                import pandas as pd
                results_data = []
                for r in results:
                    # Safely access nested dictionaries
                    worst_pos = r.get("worst_position", {})
                    worst_ticker = worst_pos.get("ticker", "N/A")
                    worst_impact_pct = worst_pos.get("impact_pct", 0.0)

                    results_data.append({
                        "Scenario": r.get("scenario_name", "Unknown"),
                        "Impact %": (
                            f"{r.get('portfolio_impact_pct', 0.0) * 100:.2f}%"
                        ),
                        "Impact ($)": (
                            f"${r.get('portfolio_impact_value', 0.0):,.2f}"
                        ),
                        "Worst Position": worst_ticker,
                        "Worst Impact": (
                            f"{worst_impact_pct * 100:.2f}%"
                        ),
                        "Recovery (days)": r.get("recovery_time_days", "N/A"),
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Interpretation: Stress test results
                interpretation = _interpret_scenario_results(results)
                if interpretation:
                    st.info(interpretation)

                # Visual comparison
                fig = go.Figure()

                scenario_names_list = [r["scenario_name"] for r in results]
                impacts = [r["portfolio_impact_pct"] * 100 for r in results]

                fig.add_trace(
                    go.Bar(
                        x=scenario_names_list,
                        y=impacts,
                        marker_color=[
                            COLORS["error"] if i < 0 else COLORS["success"]
                            for i in impacts
                        ],
                        text=[f"{i:.2f}%" for i in impacts],
                        textposition="outside",
                    )
                )

                fig.update_layout(
                    title="Portfolio Impact by Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=400,
                    template="plotly_dark",
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error running stress tests: {str(e)}")
            logger.exception("Stress test error")


def _render_historical_scenarios(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render historical scenarios section."""
    st.subheader("Historical Scenarios")

    # Get all available scenarios
    scenarios = get_all_scenarios()

    if not scenarios:
        st.warning("No historical scenarios available.")
        return

    # Scenario selection
    scenario_keys = list(scenarios.keys())
    scenario_display = [
        scenarios[key].name for key in scenario_keys
    ]

    selected_indices = st.multiselect(
        "Select Scenarios",
        options=range(len(scenario_keys)),
        format_func=lambda x: scenario_display[x],
        key="historical_scenarios",
    )

    if not selected_indices:
        st.info("Please select at least one scenario.")
        return

    selected_keys = [scenario_keys[i] for i in selected_indices]

    # Display scenario info
    with st.expander("Selected Scenarios", expanded=True):
        for key in selected_keys:
            scenario = scenarios[key]
            st.markdown(f"**{scenario.name}**")
            st.markdown(f"*{scenario.description}*")
            st.markdown(
                f"Period: {scenario.start_date} to {scenario.end_date}"
            )
            st.markdown(
                f"Market Impact: {scenario.market_impact_pct * 100:.1f}%"
            )
            if scenario.recovery_period_days:
                st.markdown(
                    f"Recovery Period: {scenario.recovery_period_days} days"
                )
            st.markdown("---")

    # Run scenarios
    if st.button("Run Scenarios", key="run_historical"):
        try:
            with st.spinner("Running scenarios..."):
                results = risk_service.run_stress_test(
                    portfolio_id=portfolio_id,
                    scenario_names=selected_keys,
                )

                st.success("Scenario analysis completed!")

                # Display results
                st.markdown("### Scenario Results")

                import pandas as pd
                results_data = []
                for r in results:
                    # Safely access nested dictionaries
                    worst_pos = r.get("worst_position", {})
                    worst_ticker = worst_pos.get("ticker", "N/A")
                    worst_impact_pct = worst_pos.get("impact_pct", 0.0)

                    results_data.append({
                        "Scenario": r.get("scenario_name", "Unknown"),
                        "Impact %": (
                            f"{r.get('portfolio_impact_pct', 0.0) * 100:.2f}%"
                        ),
                        "Impact ($)": (
                            f"${r.get('portfolio_impact_value', 0.0):,.2f}"
                        ),
                        "Worst Position": worst_ticker,
                        "Worst Impact": (
                            f"{worst_impact_pct * 100:.2f}%"
                        ),
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Interpretation: Historical scenario results
                interpretation = _interpret_scenario_results(results)
                if interpretation:
                    st.info(interpretation)

                # Visual comparison
                fig = go.Figure()

                scenario_names_list = [r["scenario_name"] for r in results]
                impacts = [r["portfolio_impact_pct"] * 100 for r in results]

                # For negative values, use 'outside' and adjust with
                # annotations if needed
                # go.Bar only supports: 'inside', 'outside', 'auto', 'none'
                fig.add_trace(
                    go.Bar(
                        x=scenario_names_list,
                        y=impacts,
                        marker_color=[
                            COLORS["error"] if i < 0 else COLORS["success"]
                            for i in impacts
                        ],
                        text=[f"{i:.2f}%" for i in impacts],
                        textposition="outside",
                        textfont=dict(size=12),
                    )
                )

                # Remove duplicate annotations - text is already in bars via text parameter
                # No need for additional annotations

                fig.update_layout(
                    title="Portfolio Impact by Historical Scenario",
                    xaxis_title="Scenario",
                    yaxis_title="Impact (%)",
                    height=500,
                    template="plotly_dark",
                    margin=dict(b=100),  # Extra bottom margin for labels
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation: Portfolio impact chart
                interpretation = _interpret_scenario_results(results)
                if interpretation:
                    st.info(interpretation)

                # Portfolio Recovery Chart
                st.markdown("---")
                st.markdown("### Portfolio Recovery Timeline")
                try:
                    fig_recovery = go.Figure()
                    
                    # Use colors from palette for recovery lines
                    scenario_colors = [
                        COLORS["primary"],    # Purple
                        COLORS["secondary"],  # Blue
                        COLORS["success"],    # Green
                        COLORS["warning"],    # Orange
                        COLORS["additional"], # Yellow
                    ]

                    for idx, r in enumerate(results):
                        scenario_name = r.get("scenario_name", "Unknown")
                        impact_pct = r.get("portfolio_impact_pct", 0.0)
                        recovery_days = r.get("recovery_time_days", None)

                        # Simulate recovery path
                        if recovery_days and recovery_days > 0:
                            days = list(
                                range(
                                    0,
                                    recovery_days + 1,
                                    max(1, recovery_days // 20),
                                )
                            )
                            # Linear recovery (simplified)
                            recovery_path = [
                                1.0 + impact_pct * (1 - d / recovery_days)
                                for d in days
                            ]
                            recovery_path = [
                                max(0, v) for v in recovery_path
                            ]  # No negative

                            color = scenario_colors[idx % len(scenario_colors)]
                            
                            fig_recovery.add_trace(
                                go.Scatter(
                                    x=days,
                                    y=[v * 100 for v in recovery_path],
                                    mode="lines+markers",
                                    name=scenario_name,
                                    line=dict(width=2, color=color),
                                    marker=dict(color=color, size=6),
                                )
                            )

                    # Add baseline (100%)
                    fig_recovery.add_hline(
                        y=100,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Baseline (100%)",
                    )

                    fig_recovery.update_layout(
                        title="Portfolio Recovery Timeline",
                        xaxis_title="Days After Event",
                        yaxis_title="Portfolio Value (% of Initial)",
                        height=500,
                        template="plotly_dark",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_recovery, use_container_width=True)
                    st.caption(
                        "**Note:** Recovery timeline is estimated based on "
                        "scenario recovery period. Actual recovery may vary."
                    )
                    
                    # Interpretation: Recovery timeline
                    recovery_data = {
                        "results": results,
                        "scenarios": {key: scenarios[key] for key in selected_keys}
                    }
                    interpretation = _interpret_scenario_recovery(recovery_data)
                    if interpretation:
                        st.info(interpretation)

                except Exception as e:
                    logger.warning(f"Error creating recovery chart: {e}")

                # Enhanced Scenario Comparison
                st.markdown("---")
                st.markdown("### Enhanced Scenario Comparison")
                try:
                    # Get scenario details
                    scenario_details = {}
                    for key in selected_keys:
                        scenario = scenarios[key]
                        scenario_details[key] = {
                            "name": scenario.name,
                            "start_date": scenario.start_date,
                            "end_date": scenario.end_date,
                            "duration_days": (
                                scenario.end_date - scenario.start_date
                            ).days,
                            "market_impact": scenario.market_impact_pct,
                        }

                    # Create comparison table
                    comparison_data = []
                    for r in results:
                        scenario_key = next(
                            (
                                k
                                for k in selected_keys
                                if scenarios[k].name == r.get("scenario_name")
                            ),
                            None,
                        )
                        if scenario_key:
                            details = scenario_details[scenario_key]
                            comparison_data.append({
                                "Scenario": details["name"],
                                "Period": (
                                    f"{details['start_date']} to "
                                    f"{details['end_date']}"
                                ),
                                "Duration (days)": details["duration_days"],
                                "Market Impact": (
                                    f"{details['market_impact']*100:.1f}%"
                                ),
                                "Portfolio Impact": (
                                    f"{r.get('portfolio_impact_pct', 0)*100:.2f}%"
                                ),
                                "Recovery (days)": (
                                    r.get("recovery_time_days", "N/A")
                                ),
                            })

                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)

                        # Visual comparison with multiple metrics
                        fig_comp = go.Figure()

                        scenario_names = [
                            d["Scenario"] for d in comparison_data
                        ]
                        impacts = [
                            float(d["Portfolio Impact"].replace("%", ""))
                            for d in comparison_data
                        ]
                        durations = [
                            d["Duration (days)"] for d in comparison_data
                        ]

                        fig_comp.add_trace(
                            go.Bar(
                                x=scenario_names,
                                y=impacts,
                                name="Portfolio Impact (%)",
                                marker_color=COLORS["danger"],
                                yaxis="y",
                            )
                        )

                        fig_comp.add_trace(
                            go.Scatter(
                                x=scenario_names,
                                y=durations,
                                name="Duration (days)",
                                mode="lines+markers",
                                line=dict(color=COLORS["warning"], width=3),
                                marker=dict(size=10),
                                yaxis="y2",
                            )
                        )

                        fig_comp.update_layout(
                            title=(
                                "Scenario Comparison: Impact vs Duration"
                            ),
                            xaxis_title="Scenario",
                            yaxis=dict(
                                title="Portfolio Impact (%)",
                                side="left",
                            ),
                            yaxis2=dict(
                                title="Duration (days)",
                                side="right",
                                overlaying="y",
                            ),
                            height=500,
                            template="plotly_dark",
                            barmode="group",
                        )

                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Interpretation: Enhanced scenario comparison
                        interpretation = _interpret_scenario_results(results)
                        if interpretation:
                            st.info(interpretation)

                except Exception as e:
                    logger.warning(
                        f"Error creating enhanced comparison: {e}"
                    )

                # Position Breakdown
                st.markdown("---")
                st.markdown("### Position Impact Breakdown")
                try:
                    # Get portfolio positions
                    portfolio_service = PortfolioService()
                    portfolio = portfolio_service.get_portfolio(portfolio_id)
                    positions = portfolio.positions if portfolio else []

                    if positions:
                        # Create breakdown for each scenario
                        for r in results:
                            scenario_name = r.get("scenario_name", "Unknown")
                            details = r.get("details", {})
                            position_impacts = details.get(
                                "position_impacts", {}
                            )

                            if position_impacts:
                                st.markdown(f"#### {scenario_name}")

                                breakdown_data = []
                                for (
                                    ticker,
                                    impact,
                                ) in position_impacts.items():
                                    # Find position
                                    pos = next(
                                        (
                                            p
                                            for p in positions
                                            if p.ticker == ticker
                                        ),
                                        None,
                                    )
                                    if pos:
                                        breakdown_data.append({
                                            "Ticker": ticker,
                                            "Weight": (
                                                f"{pos.weight*100:.2f}%"
                                                if hasattr(pos, "weight")
                                                else "N/A"
                                            ),
                                            "Impact (%)": (
                                                f"{impact*100:.2f}%"
                                            ),
                                            "Impact Type": (
                                                "Loss" if impact < 0 else "Gain"
                                            ),
                                        })

                                if breakdown_data:
                                    breakdown_df = pd.DataFrame(breakdown_data)
                                    st.dataframe(
                                        breakdown_df, use_container_width=True
                                    )

                                    # Bar chart
                                    fig_breakdown = go.Figure()
                                    fig_breakdown.add_trace(
                                        go.Bar(
                                            x=breakdown_df["Ticker"],
                                            y=[
                                                float(v.replace("%", ""))
                                                for v in breakdown_df[
                                                    "Impact (%)"
                                                ]
                                            ],
                                            marker=dict(
                                                color=[
                                                    COLORS["danger"]
                                                    if float(v.replace("%", ""))
                                                    < 0
                                                    else COLORS["success"]
                                                    for v in breakdown_df[
                                                        "Impact (%)"
                                                    ]
                                                ]
                                            ),
                                            text=breakdown_df["Impact (%)"],
                                            textposition="outside",
                                        )
                                    )

                                    fig_breakdown.update_layout(
                                        title=(
                                            f"Position Impact: {scenario_name}"
                                        ),
                                        xaxis_title="Asset",
                                        yaxis_title="Impact (%)",
                                        height=400,
                                        template="plotly_dark",
                                    )

                                    st.plotly_chart(
                                        fig_breakdown, use_container_width=True
                                    )
                                    
                                    # Interpretation: Position impact breakdown
                                    interpretation = _interpret_position_impact_breakdown(
                                        position_impacts, scenario_name
                                    )
                                    if interpretation:
                                        st.info(interpretation)

                except Exception as e:
                    logger.warning(
                        f"Error creating position breakdown: {e}"
                    )

                # Timeline Visualization
                st.markdown("---")
                st.markdown("### Historical Timeline Visualization")
                try:
                    # Create timeline with all scenarios
                    timeline_data = []
                    for key in selected_keys:
                        scenario = scenarios[key]
                        timeline_data.append({
                            "name": scenario.name,
                            "start": scenario.start_date,
                            "end": scenario.end_date,
                            "impact": scenario.market_impact_pct,
                        })

                    # Sort by start date
                    timeline_data.sort(key=lambda x: x["start"])

                    # Create Gantt-like chart
                    fig_timeline = go.Figure()

                    colors_timeline = [
                        COLORS["danger"]
                        if d["impact"] < -0.2
                        else COLORS["warning"]
                        if d["impact"] < -0.1
                        else COLORS["info"]
                        for d in timeline_data
                    ]

                    for i, data in enumerate(timeline_data):
                        duration = (data["end"] - data["start"]).days
                        fig_timeline.add_trace(
                            go.Bar(
                                x=[duration],
                                y=[data["name"]],
                                orientation="h",
                                marker_color=colors_timeline[i],
                                text=(
                                    f"{data['start']} to {data['end']} "
                                    f"({duration} days)"
                                ),
                                textposition="inside",
                                textfont=dict(color="white", size=11),  # White text
                                name=data["name"],
                            )
                        )

                    fig_timeline.update_layout(
                        title="Historical Scenarios Timeline",
                        xaxis_title="Duration (days)",
                        yaxis_title="Scenario",
                        height=300 + len(timeline_data) * 50,
                        template="plotly_dark",
                        showlegend=False,
                    )

                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Timeline with impact magnitude
                    fig_timeline2 = go.Figure()

                    for data in timeline_data:
                        fig_timeline2.add_trace(
                            go.Scatter(
                                x=[data["start"], data["end"]],
                                y=[
                                    data["impact"] * 100,
                                    data["impact"] * 100,
                                ],
                                mode="lines+markers",
                                name=data["name"],
                                line=dict(
                                    width=5,
                                    color=(
                                        COLORS["danger"]
                                        if data["impact"] < -0.2
                                        else COLORS["warning"]
                                        if data["impact"] < -0.1
                                        else COLORS["info"]
                                    ),
                                ),
                                marker=dict(size=10),
                            )
                        )

                    fig_timeline2.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="No Impact",
                    )

                    fig_timeline2.update_layout(
                        title="Market Impact Over Time",
                        xaxis_title="Date",
                        yaxis_title="Market Impact (%)",
                        height=400,
                        template="plotly_dark",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_timeline2, use_container_width=True)
                    
                    # Interpretation: Timeline visualization
                    if timeline_data:
                        parts = []
                        parts.append("**Historical Timeline Analysis:**")
                        parts.append(f"Analyzed {len(timeline_data)} historical scenario(s)")
                        
                        # Find most severe
                        most_severe = min(timeline_data, key=lambda x: x["impact"])
                        parts.append(f"Most severe: {most_severe['name']} ({most_severe['impact']*100:.1f}% market impact, {(most_severe['end'] - most_severe['start']).days} days duration)")
                        
                        # Find longest
                        longest = max(timeline_data, key=lambda x: (x["end"] - x["start"]).days)
                        parts.append(f"Longest duration: {longest['name']} ({(longest['end'] - longest['start']).days} days)")
                        
                        interpretation = "\n".join(parts)
                        if interpretation:
                            st.info(interpretation)

                except Exception as e:
                    logger.warning(
                        f"Error creating timeline visualization: {e}"
                    )

        except KeyError as e:
            error_msg = f"Missing data in results: {str(e)}"
            st.error(f"Error running scenarios: {error_msg}")
            logger.exception(f"Scenario analysis KeyError: {error_msg}")
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            st.error(f"Error running scenarios: {error_msg}")
            logger.exception(f"Scenario analysis error: {error_msg}")


def _render_custom_scenario(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render custom scenario builder."""
    st.subheader("Custom Scenario Builder")

    # Scenario inputs
    scenario_name = st.text_input(
        "Scenario Name",
        key="custom_name",
        placeholder="e.g., Tech Recession 2024",
    )

    scenario_description = st.text_area(
        "Description",
        key="custom_description",
        placeholder="Describe the scenario...",
    )

    # Market impact
    market_impact = st.slider(
        "Market Impact (%)",
        min_value=-100.0,
        max_value=100.0,
        value=-20.0,
        step=1.0,
        key="custom_market",
        help="Overall market impact percentage",
    )

    # Sector impacts (simplified - would need sector mapping)
    st.markdown("### Sector Impacts (Optional)")
    st.info(
        "Sector-specific impacts can be added here. "
        "This is a simplified version."
    )

    # Asset-specific impacts
    st.markdown("### Asset-Specific Impacts (Optional)")
    asset_impacts_text = st.text_area(
        "Asset Impacts",
        key="custom_assets",
        placeholder=(
            "Format: TICKER:IMPACT%\nExample:\nAAPL:-30%\nMSFT:-25%"
        ),
        help="Enter one ticker:impact per line",
    )

    # Parse asset impacts
    asset_impacts = {}
    if asset_impacts_text:
        for line in asset_impacts_text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    ticker = parts[0].strip().upper()
                    try:
                        impact = (
                            float(parts[1].strip().replace("%", "")) / 100.0
                        )
                        asset_impacts[ticker] = impact
                    except ValueError:
                        pass

    # Create and run scenario
    if st.button("Create & Run Scenario", key="run_custom"):
        if not scenario_name:
            st.error("Please enter a scenario name.")
            return

        try:
            # Create custom scenario
            custom_scenario = create_custom_scenario(
                name=scenario_name,
                description=scenario_description,
                market_impact_pct=market_impact / 100.0,
                asset_impacts=asset_impacts,
            )

            # Validate scenario
            is_valid, error_msg = validate_scenario(custom_scenario)
            if not is_valid:
                st.error(f"Invalid scenario: {error_msg}")
                return

            # Run scenario
            with st.spinner("Running custom scenario..."):
                result = risk_service.run_custom_scenario(
                    portfolio_id=portfolio_id,
                    scenario=custom_scenario,
                )

                st.success("Custom scenario completed!")

                # Display results
                st.markdown("### Scenario Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Portfolio Impact",
                        f"{result['portfolio_impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Impact Value",
                        f"${result['portfolio_impact_value']:,.2f}",
                    )
                with col2:
                    st.metric(
                        "Worst Position",
                        result["worst_position"]["ticker"],
                        f"{result['worst_position']['impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Best Position",
                        result["best_position"]["ticker"],
                        f"{result['best_position']['impact_pct'] * 100:.2f}%",
                    )
                
                # Interpretation: Custom scenario results
                interpretation = _interpret_scenario_results([result])
                if interpretation:
                    st.info(interpretation)

        except ValidationError as e:
            st.error(f"Validation error: {str(e)}")
        except Exception as e:
            st.error(f"Error running custom scenario: {str(e)}")
            logger.exception("Custom scenario error")


def _render_scenario_chain(
    risk_service: RiskService,
    portfolio_id: str,
) -> None:
    """Render scenario chain builder."""
    st.subheader("Scenario Chain Builder")

    st.info(
        "Create a chain of scenarios to analyze cumulative portfolio "
        "impacts from multiple sequential events."
    )

    # Get all scenarios
    scenarios = get_all_scenarios()

    if not scenarios:
        st.warning("No scenarios available.")
        return

    # Chain name
    chain_name = st.text_input(
        "Chain Name",
        key="chain_name",
        placeholder="e.g., Double Crisis (2008 + 2020)",
    )

    chain_description = st.text_area(
        "Description",
        key="chain_description",
        placeholder="Describe the scenario chain...",
    )

    # Scenario selection for chain
    scenario_keys = list(scenarios.keys())
    scenario_display = [
        scenarios[key].name for key in scenario_keys
    ]

    selected_indices = st.multiselect(
        "Select Scenarios (in order)",
        options=range(len(scenario_keys)),
        format_func=lambda x: scenario_display[x],
        key="chain_scenarios",
        help="Select scenarios in the order they should be applied",
    )

    if not selected_indices:
        st.info("Please select at least one scenario.")
        return

    selected_keys = [scenario_keys[i] for i in selected_indices]
    selected_scenarios = [scenarios[key] for key in selected_keys]

    # Display chain
    if selected_scenarios:
        st.markdown("### Scenario Chain")
        for i, scenario in enumerate(selected_scenarios, 1):
            st.markdown(f"{i}. **{scenario.name}**")
            st.markdown(f"   Impact: {scenario.market_impact_pct * 100:.1f}%")

    # Run chain
    if st.button("Run Scenario Chain", key="run_chain"):
        if not chain_name:
            st.error("Please enter a chain name.")
            return

        try:
            # Create chain
            chain = create_scenario_chain(
                name=chain_name,
                description=chain_description,
                scenarios=selected_scenarios,
            )

            # Run chain
            with st.spinner("Running scenario chain..."):
                results = risk_service.run_scenario_chain(
                    portfolio_id=portfolio_id,
                    chain=chain,
                )

                st.success("Scenario chain completed!")

                # Display results
                st.markdown("### Chain Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Cumulative Impact",
                        f"{results['cumulative_impact_pct'] * 100:.2f}%",
                    )
                    st.metric(
                        "Final Portfolio Value",
                        f"${results['final_portfolio_value']:,.2f}",
                    )
                with col2:
                    st.metric(
                        "Impact Value",
                        f"${results['cumulative_impact_value']:,.2f}",
                    )
                    st.metric(
                        "Worst Scenario",
                        results.get("worst_scenario", "N/A"),
                    )

                # Individual scenario results (table only, no charts)
                st.markdown("### Individual Scenario Results")
                import pandas as pd
                scenario_results = results["scenario_results"]

                results_data = []
                for r in scenario_results:
                    cum_pct = f"{r['cumulative_impact_pct'] * 100:.2f}%"
                    results_data.append({
                        "Scenario": r["scenario_name"],
                        "Impact %": f"{r['impact_pct'] * 100:.2f}%",
                        "Cumulative %": cum_pct,
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Interpretation: Scenario chain results
                interpretation = _interpret_scenario_results(scenario_results)
                if interpretation:
                    st.info(interpretation)

        except Exception as e:
            st.error(f"Error running scenario chain: {str(e)}")
            logger.exception("Scenario chain error")


# Main entry point
if __name__ == "__main__":
    render_risk_analysis_page()
