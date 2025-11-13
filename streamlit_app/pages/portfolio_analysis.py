"""Portfolio Analysis page - Full implementation according to specification."""

import logging
import os
import tempfile
from datetime import date, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from services.portfolio_service import PortfolioService
from services.analytics_service import AnalyticsService
from services.report_service import ReportService
from core.analytics_engine.chart_data import (
    get_cumulative_returns_data,
    get_return_distribution_data,
    get_monthly_heatmap_data,
    get_rolling_sharpe_data,
    get_rolling_sortino_data,
    get_rolling_volatility_data,
    get_rolling_beta_data,
    get_rolling_alpha_data,
    get_rolling_active_return_data,
    get_bull_bear_analysis_data,
    get_underwater_plot_data,
    get_yearly_returns_data,
    get_period_returns_comparison_data,
    get_three_month_rolling_periods_data,
    get_seasonal_analysis_data,
    get_monthly_active_returns_data,
    get_win_rate_statistics_data,
    get_outlier_analysis_data,
    get_statistical_tests_data,
    get_qq_plot_data,
    get_capture_ratio_data,
    get_risk_return_scatter_data,
    get_drawdown_periods_data,
    get_drawdown_recovery_data,
    get_asset_metrics_data,
    get_asset_impact_on_return_data,
    get_asset_impact_on_risk_data,
    get_risk_vs_weight_comparison_data,
    get_diversification_coefficient_data,
    get_correlation_matrix_data,
    get_correlation_statistics_data,
    get_correlation_with_benchmark_data,
    get_cluster_analysis_data,
    get_asset_price_dynamics_data,
    get_rolling_correlation_with_benchmark_data,
    get_detailed_asset_analysis_data,
)
from core.analytics_engine.advanced_metrics import (
    calculate_expected_returns,
    calculate_common_performance_periods,
    calculate_probabilistic_sharpe_ratio,
    calculate_smart_sharpe,
    calculate_smart_sortino,
    calculate_kelly_criterion,
    calculate_risk_of_ruin,
)
from streamlit_app.components.charts import (
    plot_cumulative_returns,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_rolling_sharpe,
    plot_rolling_sortino,
    plot_rolling_beta,
    plot_rolling_alpha,
    plot_rolling_active_return,
    plot_rolling_volatility,
    plot_bull_bear_returns_comparison,
    plot_bull_bear_rolling_beta,
    plot_underwater,
    plot_yearly_returns,
    plot_asset_allocation,
    plot_sector_allocation,
    plot_active_returns_area,
    plot_period_returns_bar,
    plot_qq_plot,
    plot_return_quantiles_box,
    plot_seasonal_bar,
    plot_outlier_scatter,
    plot_rolling_win_rate,
    plot_capture_ratio,
    plot_risk_return_scatter,
    plot_drawdown_periods,
    plot_drawdown_recovery,
    plot_var_distribution,
    plot_impact_on_return,
    plot_impact_on_risk,
    plot_risk_vs_weight_comparison,
    plot_correlation_matrix,
    plot_correlation_with_benchmark,
    plot_clustered_correlation_matrix,
    plot_dendrogram,
    plot_asset_price_dynamics,
    plot_rolling_correlation_with_benchmark,
    plot_detailed_asset_price_volume,
    plot_asset_correlation_bar,
)
from streamlit_app.components.position_table import render_position_table
from streamlit_app.components.metric_card_comparison import render_metric_cards_row
from streamlit_app.components.comparison_table import render_comparison_table
from streamlit_app.components.assets_metrics import render_assets_table_extended
from streamlit_app.utils.chart_config import COLORS, get_chart_layout
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _interpret_qq_plot(deviations: np.ndarray, lower_tail: float, upper_tail: float, mean_dev: float, std_dev: float) -> str:
    """
    Interpret QQ plot results and provide automatic analysis.
    
    Args:
        deviations: Array of deviations from theoretical line
        lower_tail: Mean deviation in lower tail (bottom 10%)
        upper_tail: Mean deviation in upper tail (top 10%)
        mean_dev: Mean deviation overall
        std_dev: Standard deviation of deviations
        
    Returns:
        Interpretation string
    """
    # Determine overall fit
    abs_mean_dev = abs(mean_dev)
    fit_quality = "excellent" if abs_mean_dev < 0.001 else "good" if abs_mean_dev < 0.005 else "moderate" if abs_mean_dev < 0.01 else "poor"
    
    # Determine tail behavior
    has_fat_tails = abs(lower_tail) > 0.01 or abs(upper_tail) > 0.01
    has_left_tail = lower_tail < -0.005
    has_right_tail = upper_tail > 0.005
    
    # Determine skewness
    is_skewed_left = lower_tail < -0.01 and upper_tail < 0.005
    is_skewed_right = upper_tail > 0.01 and lower_tail > -0.005
    
    interpretation_parts = []
    interpretation_parts.append(f"**Distribution Analysis:**")
    
    # Overall fit
    if fit_quality == "excellent":
        interpretation_parts.append(f"✓ Distribution is very close to normal (excellent fit)")
    elif fit_quality == "good":
        interpretation_parts.append(f"✓ Distribution is approximately normal (good fit)")
    elif fit_quality == "moderate":
        interpretation_parts.append(f"⚠ Distribution deviates moderately from normal")
    else:
        interpretation_parts.append(f"⚠ Distribution deviates significantly from normal")
    
    # Tail analysis
    if has_fat_tails:
        if has_left_tail and has_right_tail:
            interpretation_parts.append(f"⚠ **Fat tails detected:** Both tails deviate from normal. This means extreme gains and losses occur more frequently than expected in a normal distribution.")
        elif has_left_tail:
            interpretation_parts.append(f"⚠ **Left tail risk:** Extreme losses occur more frequently than normal distribution predicts. Higher downside risk.")
        elif has_right_tail:
            interpretation_parts.append(f"✓ **Right tail benefit:** Extreme gains occur more frequently than normal. Potential for larger positive surprises.")
    else:
        interpretation_parts.append(f"✓ Tails are close to normal distribution")
    
    # Skewness
    if is_skewed_left:
        interpretation_parts.append(f"⚠ **Negative skew:** Distribution is skewed left. More frequent small gains but occasional large losses.")
    elif is_skewed_right:
        interpretation_parts.append(f"✓ **Positive skew:** Distribution is skewed right. More frequent small losses but occasional large gains.")
    else:
        interpretation_parts.append(f"✓ Distribution is approximately symmetric")
    
    # Practical implications
    interpretation_parts.append(f"\n**Practical Implications:**")
    if has_fat_tails or fit_quality in ["moderate", "poor"]:
        interpretation_parts.append(f"- VaR and CVaR estimates may underestimate tail risk")
        interpretation_parts.append(f"- Consider using non-parametric risk measures")
        interpretation_parts.append(f"- Portfolio may experience more extreme events than normal models predict")
    else:
        interpretation_parts.append(f"- Standard risk models (VaR, Sharpe) are appropriate")
        interpretation_parts.append(f"- Returns follow approximately normal distribution")
    
    return "\n".join(interpretation_parts)


def _interpret_distribution(mean: float, std: float, skew: float, kurt: float, period: str) -> str:
    """
    Interpret return distribution characteristics.
    
    Args:
        mean: Mean return
        std: Standard deviation
        skew: Skewness
        kurt: Kurtosis (excess)
        period: "daily" or "monthly"
        
    Returns:
        Interpretation string
    """
    period_name = "Daily" if period == "daily" else "Monthly"
    
    parts = []
    
    # Mean interpretation
    if mean > 0:
        parts.append(f"✓ Average {period_name.lower()} return: {mean*100:.2f}% (positive)")
    else:
        parts.append(f"⚠ Average {period_name.lower()} return: {mean*100:.2f}% (negative)")
    
    # Volatility interpretation
    if std < 0.01:
        parts.append(f"Low volatility ({std*100:.2f}%) - stable returns")
    elif std < 0.02:
        parts.append(f"Moderate volatility ({std*100:.2f}%)")
    else:
        parts.append(f"High volatility ({std*100:.2f}%) - wide price swings")
    
    # Skewness interpretation
    if abs(skew) < 0.1:
        parts.append(f"Symmetric distribution (skew: {skew:.2f})")
    elif skew < -0.5:
        parts.append(f"⚠ Left-skewed (skew: {skew:.2f}) - more extreme losses than gains")
    elif skew > 0.5:
        parts.append(f"✓ Right-skewed (skew: {skew:.2f}) - more extreme gains than losses")
    else:
        parts.append(f"Slight {'left' if skew < 0 else 'right'} skew ({skew:.2f})")
    
    # Kurtosis interpretation
    if kurt < -0.5:
        parts.append(f"Thin tails (kurtosis: {kurt:.2f}) - fewer extreme events")
    elif kurt > 0.5:
        parts.append(f"⚠ Fat tails (kurtosis: {kurt:.2f}) - more extreme events than normal")
    else:
        parts.append(f"Normal tail behavior (kurtosis: {kurt:.2f})")
    
    return " | ".join(parts)


def _interpret_cumulative_returns(
    portfolio_cumulative: pd.Series,
    benchmark_cumulative: pd.Series = None
) -> str:
    """Interpret cumulative returns chart."""
    parts = []
    parts.append(f"**Cumulative Returns Analysis:**")
    
    # Calculate total return
    total_return = portfolio_cumulative.iloc[-1] if not portfolio_cumulative.empty else 0
    
    # Determine trend
    if total_return > 0.20:
        trend = f"Strong growth: portfolio gained {total_return*100:.1f}% over the period"
    elif total_return > 0.10:
        trend = f"Moderate growth: portfolio gained {total_return*100:.1f}% over the period"
    elif total_return > 0:
        trend = f"Positive return: portfolio gained {total_return*100:.1f}% over the period"
    elif total_return > -0.10:
        trend = f"Minor decline: portfolio lost {abs(total_return)*100:.1f}% over the period"
    else:
        trend = f"Significant decline: portfolio lost {abs(total_return)*100:.1f}% over the period"
    
    parts.append(trend)
    
    # Benchmark comparison
    if benchmark_cumulative is not None and not benchmark_cumulative.empty:
        bench_return = benchmark_cumulative.iloc[-1]
        outperformance = total_return - bench_return
        
        if abs(outperformance) < 0.01:
            parts.append(f"Portfolio performance similar to benchmark ({bench_return*100:.1f}%)")
        elif outperformance > 0.05:
            parts.append(f"✓ Outperformed benchmark by {outperformance*100:.1f}% ({bench_return*100:.1f}%)")
        elif outperformance > 0:
            parts.append(f"✓ Slightly outperformed benchmark by {outperformance*100:.1f}% ({bench_return*100:.1f}%)")
        elif outperformance > -0.05:
            parts.append(f"⚠ Slightly underperformed benchmark by {abs(outperformance)*100:.1f}% ({bench_return*100:.1f}%)")
        else:
            parts.append(f"⚠ Underperformed benchmark by {abs(outperformance)*100:.1f}% ({bench_return*100:.1f}%)")
    
    # Volatility of path
    if not portfolio_cumulative.empty and len(portfolio_cumulative) > 1:
        returns_series = portfolio_cumulative.pct_change().dropna()
        path_volatility = returns_series.std()
        
        if path_volatility < 0.01:
            parts.append(f"Smooth growth trajectory with low volatility")
        elif path_volatility < 0.02:
            parts.append(f"Moderate volatility in growth path")
        else:
            parts.append(f"Volatile path with significant swings")
    
    return "\n".join(parts)


def _interpret_daily_returns(daily_returns: pd.Series) -> str:
    """Interpret daily returns chart."""
    parts = []
    parts.append(f"**Daily Returns Analysis:**")
    
    if daily_returns.empty:
        return "Insufficient data for analysis"
    
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    pos_days = (daily_returns > 0).sum()
    neg_days = (daily_returns < 0).sum()
    total_days = len(daily_returns)
    pos_pct = (pos_days / total_days * 100) if total_days > 0 else 0
    max_return = daily_returns.max()
    min_return = daily_returns.min()
    
    # Average return
    if mean_return > 0.001:
        parts.append(f"Average daily return: {mean_return*100:.2f}% (positive)")
    elif mean_return > -0.001:
        parts.append(f"Average daily return: {mean_return*100:.2f}% (near zero)")
    else:
        parts.append(f"Average daily return: {mean_return*100:.2f}% (negative)")
    
    # Positive vs negative days
    if pos_pct > 60:
        parts.append(f"More positive days ({pos_days}, {pos_pct:.0f}%) than negative ({neg_days}, {100-pos_pct:.0f}%)")
    elif pos_pct > 50:
        parts.append(f"Slightly more positive days ({pos_days}, {pos_pct:.0f}%) than negative ({neg_days}, {100-pos_pct:.0f}%)")
    elif pos_pct > 40:
        parts.append(f"More negative days ({neg_days}, {100-pos_pct:.0f}%) than positive ({pos_days}, {pos_pct:.0f}%)")
    else:
        parts.append(f"Significantly more negative days ({neg_days}, {100-pos_pct:.0f}%) than positive ({pos_days}, {pos_pct:.0f}%)")
    
    # Extreme days
    parts.append(f"Largest gain: {max_return*100:.2f}%, Largest loss: {min_return*100:.2f}%")
    
    # Volatility
    if std_return < 0.01:
        parts.append(f"Low volatility ({std_return*100:.2f}% daily) - stable returns")
    elif std_return < 0.02:
        parts.append(f"Moderate volatility ({std_return*100:.2f}% daily)")
    else:
        parts.append(f"High volatility ({std_return*100:.2f}% daily) - wide price swings")
    
    return "\n".join(parts)


def _interpret_active_returns(active_returns: pd.Series) -> str:
    """Interpret daily active returns chart."""
    parts = []
    parts.append(f"**Daily Active Returns Analysis:**")
    
    if active_returns.empty:
        return "Insufficient data for analysis"
    
    mean = active_returns.mean()
    std = active_returns.std()
    pos_days = (active_returns > 0).sum()
    total_days = len(active_returns)
    pos_days_pct = (pos_days / total_days * 100) if total_days > 0 else 0
    
    # Average active return
    if mean > 0.001:
        parts.append(f"Average daily active return: {mean*100:.2f}% (positive alpha)")
    elif mean > -0.001:
        parts.append(f"Average daily active return: {mean*100:.2f}% (near zero alpha)")
    else:
        parts.append(f"Average daily active return: {mean*100:.2f}% (negative alpha)")
    
    # Consistency
    if pos_days_pct > 60:
        parts.append(f"Portfolio consistently outperforms benchmark ({pos_days_pct:.0f}% of days positive)")
    elif pos_days_pct > 50:
        parts.append(f"Portfolio performance mixed ({pos_days_pct:.0f}% of days positive)")
    else:
        parts.append(f"Portfolio consistently underperforms benchmark ({pos_days_pct:.0f}% of days positive)")
    
    # Volatility
    if std < 0.01:
        parts.append(f"Low active return volatility ({std*100:.2f}% daily) - stable alpha")
    elif std < 0.02:
        parts.append(f"Moderate active return volatility ({std*100:.2f}% daily)")
    else:
        parts.append(f"High active return volatility ({std*100:.2f}% daily) - volatile alpha")
    
    return "\n".join(parts)


def _interpret_period_returns(periods_df: pd.DataFrame) -> str:
    """Interpret return by periods chart."""
    parts = []
    parts.append(f"**Period Returns Analysis:**")
    
    if periods_df.empty or "Portfolio" not in periods_df.columns:
        return "Insufficient data for analysis"
    
    # Find best and worst periods
    portfolio_returns = periods_df["Portfolio"].dropna()
    if portfolio_returns.empty:
        return "No portfolio returns data available"
    
    best_idx = portfolio_returns.idxmax()
    worst_idx = portfolio_returns.idxmin()
    best_return = portfolio_returns.loc[best_idx]
    worst_return = portfolio_returns.loc[worst_idx]
    best_period = periods_df.loc[best_idx, "Period"] if "Period" in periods_df.columns else str(best_idx)
    worst_period = periods_df.loc[worst_idx, "Period"] if "Period" in periods_df.columns else str(worst_idx)
    
    # Best period
    if best_return > 0.20:
        parts.append(f"Best performing period: {best_period} ({best_return*100:.1f}% - exceptional)")
    elif best_return > 0.10:
        parts.append(f"Best performing period: {best_period} ({best_return*100:.1f}% - strong)")
    elif best_return > 0:
        parts.append(f"Best performing period: {best_period} ({best_return*100:.1f}% - positive)")
    else:
        parts.append(f"Best performing period: {best_period} ({best_return*100:.1f}% - still negative)")
    
    # Worst period
    if worst_return < -0.20:
        parts.append(f"Worst performing period: {worst_period} ({worst_return*100:.1f}% - severe decline)")
    elif worst_return < -0.10:
        parts.append(f"Worst performing period: {worst_period} ({worst_return*100:.1f}% - significant decline)")
    elif worst_return < 0:
        parts.append(f"Worst performing period: {worst_period} ({worst_return*100:.1f}% - minor decline)")
    else:
        parts.append(f"Worst performing period: {worst_period} ({worst_return*100:.1f}% - still positive)")
    
    # Benchmark comparison if available
    if "Benchmark" in periods_df.columns and "Portfolio" in periods_df.columns:
        aligned = pd.DataFrame({
            "Portfolio": periods_df["Portfolio"],
            "Benchmark": periods_df["Benchmark"]
        }).dropna()
        
        if not aligned.empty:
            outperformed = (aligned["Portfolio"] > aligned["Benchmark"]).sum()
            total_periods = len(aligned)
            outperformed_pct = (outperformed / total_periods * 100) if total_periods > 0 else 0
            
            if outperformed_pct > 70:
                parts.append(f"Portfolio outperformed benchmark in {outperformed_pct:.0f}% of periods ({outperformed}/{total_periods})")
            elif outperformed_pct > 50:
                parts.append(f"Portfolio outperformed benchmark in {outperformed_pct:.0f}% of periods ({outperformed}/{total_periods})")
            else:
                parts.append(f"Portfolio underperformed benchmark in {100-outperformed_pct:.0f}% of periods ({total_periods-outperformed}/{total_periods})")
    
    return "\n".join(parts)


def _interpret_yearly_returns(yearly_df: pd.DataFrame) -> str:
    """Interpret annual returns chart."""
    parts = []
    parts.append(f"**Annual Returns Analysis:**")
    
    if yearly_df.empty or "Portfolio" not in yearly_df.columns:
        return "Insufficient data for analysis"
    
    portfolio_returns = yearly_df["Portfolio"].dropna()
    if portfolio_returns.empty:
        return "No portfolio returns data available"
    
    # Find best and worst years
    best_year = portfolio_returns.idxmax()
    worst_year = portfolio_returns.idxmin()
    best_return = portfolio_returns.loc[best_year] / 100  # Convert from % to decimal
    worst_return = portfolio_returns.loc[worst_year] / 100
    avg_return = portfolio_returns.mean() / 100
    
    # Best year
    if best_return > 0.30:
        parts.append(f"Best year: {best_year} ({best_return*100:.1f}% - exceptional)")
    elif best_return > 0.20:
        parts.append(f"Best year: {best_year} ({best_return*100:.1f}% - strong)")
    elif best_return > 0:
        parts.append(f"Best year: {best_year} ({best_return*100:.1f}%)")
    else:
        parts.append(f"Best year: {best_year} ({best_return*100:.1f}% - still negative)")
    
    # Worst year
    if worst_return < -0.30:
        parts.append(f"Worst year: {worst_year} ({worst_return*100:.1f}% - severe decline)")
    elif worst_return < -0.20:
        parts.append(f"Worst year: {worst_year} ({worst_return*100:.1f}% - significant decline)")
    elif worst_return < 0:
        parts.append(f"Worst year: {worst_year} ({worst_return*100:.1f}% - decline)")
    else:
        parts.append(f"Worst year: {worst_year} ({worst_return*100:.1f}% - still positive)")
    
    # Average
    parts.append(f"Average annual return: {avg_return*100:.1f}%")
    
    # Benchmark comparison
    if "Benchmark" in yearly_df.columns:
        aligned = pd.DataFrame({
            "Portfolio": yearly_df["Portfolio"],
            "Benchmark": yearly_df["Benchmark"]
        }).dropna()
        
        if not aligned.empty:
            outperformed = (aligned["Portfolio"] > aligned["Benchmark"]).sum()
            total_years = len(aligned)
            if outperformed > total_years / 2:
                parts.append(f"Portfolio outperformed benchmark in {outperformed}/{total_years} years")
            elif outperformed > 0:
                parts.append(f"Portfolio outperformed benchmark in {outperformed}/{total_years} years")
            else:
                parts.append(f"Portfolio underperformed benchmark in all {total_years} years")
    
    return "\n".join(parts)


def _interpret_monthly_heatmap(heatmap_df: pd.DataFrame) -> str:
    """Interpret monthly returns heatmap - returns facts for caption."""
    if heatmap_df.empty:
        return ""
    
    # Flatten heatmap to find best/worst months
    all_values = heatmap_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        return ""
    
    # Find best and worst months across all years
    best_value = np.nanmax(all_values)
    worst_value = np.nanmin(all_values)
    
    # Find which month has best average
    month_means = heatmap_df.mean(axis=0)
    best_month_idx = month_means.idxmax()
    worst_month_idx = month_means.idxmin()
    best_month_avg = month_means.loc[best_month_idx]
    worst_month_avg = month_means.loc[worst_month_idx]
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    best_month_name = month_names[best_month_idx - 1] if best_month_idx in range(1, 13) else f"Month {best_month_idx}"
    worst_month_name = month_names[worst_month_idx - 1] if worst_month_idx in range(1, 13) else f"Month {worst_month_idx}"
    
    return f"**Best month:** {best_month_name} ({best_month_avg:.1f}% average) | **Worst month:** {worst_month_name} ({worst_month_avg:.1f}% average)"


def _interpret_monthly_active_returns(heatmap_df: pd.DataFrame) -> str:
    """Interpret monthly active returns heatmap."""
    parts = []
    parts.append(f"**Monthly Active Returns Analysis:**")
    
    if heatmap_df.empty:
        return "Insufficient data for analysis"
    
    # Flatten heatmap
    all_values = heatmap_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        return "No active returns data available"
    
    mean_active = np.mean(all_values)
    positive_values = all_values[all_values > 0]
    total_months = len(all_values)
    positive_months = len(positive_values)
    best_value = np.max(all_values)
    worst_value = np.min(all_values)
    
    # Find best and worst months
    month_means = heatmap_df.mean(axis=0)
    best_month_idx = month_means.idxmax()
    worst_month_idx = month_means.idxmin()
    best_month_avg = month_means.loc[best_month_idx]
    worst_month_avg = month_means.loc[worst_month_idx]
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    best_month_name = month_names[best_month_idx - 1] if best_month_idx in range(1, 13) else f"Month {best_month_idx}"
    worst_month_name = month_names[worst_month_idx - 1] if worst_month_idx in range(1, 13) else f"Month {worst_month_idx}"
    
    # Average active return
    parts.append(f"Average monthly active return: {mean_active:.2f}%")
    
    # Consistency
    positive_pct = (positive_months / total_months * 100) if total_months > 0 else 0
    if positive_pct > 60:
        parts.append(f"Portfolio consistently outperforms benchmark ({positive_months}/{total_months} positive months)")
    elif positive_pct > 40:
        parts.append(f"Portfolio performance mixed ({positive_months}/{total_months} positive months)")
    else:
        parts.append(f"Portfolio consistently underperforms benchmark ({positive_months}/{total_months} positive months)")
    
    # Best/worst months
    parts.append(f"Best active month: {best_month_name} ({best_month_avg:.1f}%), Worst: {worst_month_name} ({worst_month_avg:.1f}%)")
    
    return "\n".join(parts)


def _interpret_seasonal_pattern(seasonal_df: pd.DataFrame, pattern_type: str) -> str:
    """Interpret seasonal pattern (day of week, month, or quarter)."""
    parts = []
    
    if seasonal_df.empty or "Portfolio" not in seasonal_df.columns:
        return "Insufficient data for analysis"
    
    portfolio_values = seasonal_df["Portfolio"].dropna()
    if portfolio_values.empty:
        return "No portfolio data available"
    
    # Find best and worst
    best_idx = portfolio_values.idxmax()
    worst_idx = portfolio_values.idxmin()
    best_return = portfolio_values.loc[best_idx] / 100  # Convert from % to decimal
    worst_return = portfolio_values.loc[worst_idx] / 100
    range_value = best_return - worst_return
    
    # Pattern type specific
    if pattern_type == "day_of_week":
        parts.append(f"**Day of Week Analysis:**")
        pattern_threshold = 0.001  # 0.1% difference
    elif pattern_type == "month":
        parts.append(f"**Monthly Pattern Analysis:**")
        pattern_threshold = 0.002  # 0.2% difference
    else:  # quarter
        parts.append(f"**Quarterly Pattern Analysis:**")
        pattern_threshold = 0.003  # 0.3% difference
    
    # Best
    parts.append(f"Best {pattern_type.replace('_', ' ')}: {best_idx} ({best_return*100:.2f}% average)")
    
    # Worst
    parts.append(f"Worst {pattern_type.replace('_', ' ')}: {worst_idx} ({worst_return*100:.2f}% average)")
    
    # Pattern detection
    if range_value > pattern_threshold:
        parts.append(f"Strong {pattern_type.replace('_', '-')} pattern detected")
    else:
        parts.append(f"No significant {pattern_type.replace('_', '-')} pattern")
    
    return "\n".join(parts)


def _interpret_quantiles_box(portfolio_returns: pd.Series, benchmark_returns: pd.Series = None) -> str:
    """Interpret return quantiles box plots."""
    parts = []
    parts.append(f"**Return Quantiles Analysis:**")
    
    if portfolio_returns.empty:
        return "Insufficient data for analysis"
    
    # Calculate quantiles
    q25 = portfolio_returns.quantile(0.25)
    q50 = portfolio_returns.quantile(0.50)  # median
    q75 = portfolio_returns.quantile(0.75)
    iqr = q75 - q25
    
    # Median
    parts.append(f"Portfolio median return: {q50*100:.2f}%")
    
    # Benchmark comparison
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_q50 = benchmark_returns.quantile(0.50)
        if q50 > bench_q50 + 0.001:
            parts.append(f"Portfolio shows higher returns than benchmark (median: {bench_q50*100:.2f}%)")
        elif q50 < bench_q50 - 0.001:
            parts.append(f"Portfolio shows lower returns than benchmark (median: {bench_q50*100:.2f}%)")
        else:
            parts.append(f"Portfolio shows similar returns to benchmark (median: {bench_q50*100:.2f}%)")
    
    # IQR
    parts.append(f"Interquartile range: {iqr*100:.2f}% ({q25*100:.2f}% to {q75*100:.2f}%)")
    
    # Volatility interpretation
    if iqr > 0.05:  # 5% IQR
        parts.append(f"Wide spread indicates high volatility")
    elif iqr > 0.02:  # 2% IQR
        parts.append(f"Moderate spread indicates moderate volatility")
    else:
        parts.append(f"Narrow spread indicates stable returns")
    
    return "\n".join(parts)


def _interpret_win_rate_stats(port_stats: dict, bench_stats: dict = None) -> str:
    """Interpret win rate statistics."""
    parts = []
    parts.append(f"**Win Rate Analysis:**")
    
    win_rate_daily = port_stats.get('win_days_pct', 0)
    avg_up_day = port_stats.get('avg_up_day', 0)
    avg_down_day = port_stats.get('avg_down_day', 0)
    
    # Daily win rate
    if win_rate_daily > 60:
        parts.append(f"Daily win rate: {win_rate_daily:.1f}% (strong positive bias)")
    elif win_rate_daily > 50:
        parts.append(f"Daily win rate: {win_rate_daily:.1f}% (positive bias)")
    elif win_rate_daily > 40:
        parts.append(f"Daily win rate: {win_rate_daily:.1f}% (slight negative bias)")
    else:
        parts.append(f"Daily win rate: {win_rate_daily:.1f}% (more losing days)")
    
    # Average win/loss
    if avg_up_day != 0 and avg_down_day != 0:
        parts.append(f"Average win: {avg_up_day:.2f}%, Average loss: {avg_down_day:.2f}%")
        ratio = abs(avg_up_day / avg_down_day) if avg_down_day != 0 else 0
        if ratio > 1.2:
            parts.append(f"Positive expectancy (wins are {ratio:.2f}x larger than losses)")
        elif ratio > 0.8:
            parts.append(f"Balanced expectancy (wins are {ratio:.2f}x losses)")
        else:
            parts.append(f"Negative expectancy (wins are {ratio:.2f}x losses)")
    
    # Benchmark comparison
    if bench_stats is not None and bench_stats.get('win_rate_daily') is not None:
        bench_win_rate = bench_stats.get('win_rate_daily', 0) * 100
        diff = win_rate_daily - bench_win_rate
        if abs(diff) > 5:
            if diff > 0:
                parts.append(f"Portfolio win rate {diff:.1f}% higher than benchmark ({bench_win_rate:.1f}%)")
            else:
                parts.append(f"Portfolio win rate {abs(diff):.1f}% lower than benchmark ({bench_win_rate:.1f}%)")
    
    return "\n".join(parts)


def _interpret_rolling_win_rate(rolling_win_rate: pd.Series, bench_rolling: pd.Series = None) -> str:
    """Interpret rolling win rate chart."""
    parts = []
    parts.append(f"**Rolling Win Rate Analysis:**")
    
    if rolling_win_rate.empty:
        return "Insufficient data for analysis"
    
    current_win_rate = rolling_win_rate.iloc[-1] if len(rolling_win_rate) > 0 else 0
    avg_win_rate = rolling_win_rate.mean()
    
    # Current win rate
    parts.append(f"Current 12-month win rate: {current_win_rate:.1f}%")
    
    # Average
    parts.append(f"Average win rate: {avg_win_rate:.1f}%")
    
    # Trend
    if len(rolling_win_rate) > 1:
        recent_avg = rolling_win_rate.iloc[-6:].mean() if len(rolling_win_rate) >= 6 else rolling_win_rate.iloc[-3:].mean()
        early_avg = rolling_win_rate.iloc[:6].mean() if len(rolling_win_rate) >= 6 else rolling_win_rate.iloc[:3].mean()
        trend = recent_avg - early_avg
        
        if trend > 5:
            parts.append(f"Win rate improving over period (+{trend:.1f}%)")
        elif trend < -5:
            parts.append(f"Win rate declining over period ({trend:.1f}%)")
        else:
            parts.append(f"Win rate stable over period")
    
    # Benchmark comparison
    if bench_rolling is not None and not bench_rolling.empty:
        bench_current = bench_rolling.iloc[-1] if len(bench_rolling) > 0 else 0
        diff = current_win_rate - bench_current
        if abs(diff) > 5:
            if diff > 0:
                parts.append(f"Portfolio win rate {diff:.1f}% higher than benchmark ({bench_current:.1f}%)")
            else:
                parts.append(f"Portfolio win rate {abs(diff):.1f}% lower than benchmark ({bench_current:.1f}%)")
    
    return "\n".join(parts)


def _interpret_smart_ratios(observed_sharpe: float, smart_sharpe: float, observed_sortino: float, smart_sortino: float) -> str:
    """Interpret Smart Sharpe & Sortino ratios."""
    parts = []
    parts.append(f"**Smart Ratios Analysis:**")
    
    sharpe_adjustment = observed_sharpe - smart_sharpe
    sortino_adjustment = observed_sortino - smart_sortino
    sortino_conservative = observed_sortino / np.sqrt(2)
    
    # Sharpe adjustment
    parts.append(f"Smart Sharpe adjusts for autocorrelation: {sharpe_adjustment:+.2f} difference from observed")
    if abs(sharpe_adjustment) > 0.3:
        parts.append(f"Autocorrelation significantly impacts Sharpe ratio")
    elif abs(sharpe_adjustment) > 0.1:
        parts.append(f"Moderate autocorrelation impact on Sharpe ratio")
    else:
        parts.append(f"Minimal autocorrelation impact on Sharpe ratio")
    
    # Sortino comparison
    parts.append(f"Smart Sortino: {smart_sortino:.2f} vs Observed: {observed_sortino:.2f} (adjustment: {sortino_adjustment:+.2f})")
    parts.append(f"Conservative Sortino estimate: {sortino_conservative:.2f}")
    
    return "\n".join(parts)


def _interpret_risk_return_scatter(scatter_data: dict) -> str:
    """Interpret risk/return scatter plot."""
    parts = []
    parts.append(f"**Risk/Return Analysis:**")
    
    portfolio_info = scatter_data.get("portfolio", {})
    portfolio_return = portfolio_info.get("return", 0)
    portfolio_risk = portfolio_info.get("volatility", 0)
    
    benchmark_info = scatter_data.get("benchmark", {})
    benchmark_return = benchmark_info.get("return") if benchmark_info else None
    benchmark_risk = benchmark_info.get("volatility") if benchmark_info else None
    
    # Portfolio position
    parts.append(f"Portfolio position: {portfolio_return*100:.1f}% return, {portfolio_risk*100:.1f}% volatility")
    
    # Benchmark comparison
    if benchmark_return is not None and benchmark_risk is not None:
        parts.append(f"Benchmark position: {benchmark_return*100:.1f}% return, {benchmark_risk*100:.1f}% volatility")
        
        # Sharpe comparison
        risk_free_rate = scatter_data.get("risk_free_rate", 0)
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        benchmark_sharpe = (benchmark_return - risk_free_rate) / benchmark_risk if benchmark_risk > 0 else 0
        
        if portfolio_sharpe > benchmark_sharpe + 0.1:
            parts.append(f"Portfolio offers better risk-adjusted returns (Sharpe: {portfolio_sharpe:.2f} vs {benchmark_sharpe:.2f})")
        elif portfolio_sharpe < benchmark_sharpe - 0.1:
            parts.append(f"Portfolio risk-adjusted returns lower than benchmark (Sharpe: {portfolio_sharpe:.2f} vs {benchmark_sharpe:.2f})")
        else:
            parts.append(f"Portfolio risk-adjusted returns similar to benchmark (Sharpe: {portfolio_sharpe:.2f} vs {benchmark_sharpe:.2f})")
    
    # Efficient frontier (simplified check)
    if portfolio_return > 0 and portfolio_risk > 0:
        # Simple check: if return/risk ratio is high, likely above frontier
        risk_return_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        if risk_return_ratio > 0.5:
            parts.append(f"Portfolio appears above the efficient frontier")
        elif risk_return_ratio > 0.3:
            parts.append(f"Portfolio appears on the efficient frontier")
        else:
            parts.append(f"Portfolio appears below the efficient frontier")
    
    return "\n".join(parts)


def _interpret_kelly_criterion(kelly_data: dict) -> str:
    """Interpret Kelly Criterion results."""
    parts = []
    parts.append(f"**Kelly Criterion Analysis:**")
    
    kelly_full = kelly_data.get("kelly_full", 0)
    kelly_half = kelly_data.get("kelly_half", 0)
    kelly_quarter = kelly_data.get("kelly_quarter", 0)
    
    # Interpretation
    if kelly_full > 0.5:
        parts.append(f"Full Kelly ({kelly_full*100:.1f}%) suggests high leverage - very aggressive")
        parts.append(f"Half-Kelly ({kelly_half*100:.1f}%) recommended for most investors")
    elif kelly_full > 0.2:
        parts.append(f"Full Kelly ({kelly_full*100:.1f}%) suggests moderate leverage")
        parts.append(f"Half-Kelly ({kelly_half*100:.1f}%) provides conservative position sizing")
    elif kelly_full > 0:
        parts.append(f"Full Kelly ({kelly_full*100:.1f}%) suggests low leverage")
        parts.append(f"Half-Kelly ({kelly_half*100:.1f}%) provides very conservative position sizing")
    else:
        parts.append(f"Negative Kelly suggests avoiding this strategy")
    
    parts.append(f"Quarter-Kelly ({kelly_quarter*100:.1f}%) for very risk-averse investors")
    
    return "\n".join(parts)


def _interpret_single_drawdown(dd: dict) -> str:
    """Interpret a single drawdown recovery."""
    parts = []
    number = dd.get('number', 0)
    depth = dd.get('depth', 0)
    duration_days = dd.get('duration_days', 0)
    recovery_days = dd.get('recovery_days')
    total_days = duration_days + (recovery_days if recovery_days else 0)
    
    parts.append(f"**Drawdown #{number} Analysis:**")
    
    # Depth interpretation
    if depth > 0.20:
        parts.append(f"Depth: {depth*100:.2f}% (Severe decline)")
    elif depth > 0.10:
        parts.append(f"Depth: {depth*100:.2f}% (Significant decline)")
    else:
        parts.append(f"Depth: {depth*100:.2f}% (Moderate decline)")
    
    # Duration interpretation
    if duration_days > 180:
        parts.append(f"Duration: {duration_days} days (Long period)")
    elif duration_days > 90:
        parts.append(f"Duration: {duration_days} days (Moderate period)")
    else:
        parts.append(f"Duration: {duration_days} days (Short period)")
    
    # Recovery interpretation
    if recovery_days:
        if recovery_days < 60:
            parts.append(f"Recovery time: {recovery_days} days (Fast recovery)")
        elif recovery_days < 180:
            parts.append(f"Recovery time: {recovery_days} days (Moderate recovery)")
        else:
            parts.append(f"Recovery time: {recovery_days} days (Slow recovery)")
        parts.append(f"Total impact: {total_days} days from peak to recovery")
    else:
        parts.append(f"Not yet recovered - ongoing drawdown")
    
    return "\n".join(parts)


def _interpret_drawdown_chart(max_dd: float, current_dd: float, drawdown_series: pd.Series) -> str:
    """Interpret underwater plot / drawdown chart."""
    parts = []
    parts.append(f"**Drawdown Analysis:**")
    
    # Max drawdown
    if max_dd < -0.20:
        parts.append(f"⚠ **Severe maximum drawdown:** {max_dd*100:.1f}% - Portfolio lost more than 20% from peak")
    elif max_dd < -0.10:
        parts.append(f"⚠ **Significant maximum drawdown:** {max_dd*100:.1f}% - Portfolio lost 10-20% from peak")
    else:
        parts.append(f"✓ **Moderate maximum drawdown:** {max_dd*100:.1f}% - Losses stayed below 10%")
    
    # Current drawdown
    if current_dd < -0.05:
        parts.append(f"⚠ **Currently in drawdown:** {current_dd*100:.1f}% - Portfolio is below recent peak")
    elif current_dd < -0.01:
        parts.append(f"⚠ **Slight current drawdown:** {current_dd*100:.1f}% - Minor decline from peak")
    else:
        parts.append(f"✓ **At or near peak:** Portfolio is performing well")
    
    # Drawdown frequency
    if not drawdown_series.empty:
        dd_below_5pct = (drawdown_series < -0.05).sum()
        dd_frequency = dd_below_5pct / len(drawdown_series) * 100
        if dd_frequency > 20:
            parts.append(f"⚠ **Frequent drawdowns:** Portfolio spent {dd_frequency:.1f}% of time in >5% drawdown")
        elif dd_frequency > 10:
            parts.append(f"**Moderate drawdown frequency:** Portfolio spent {dd_frequency:.1f}% of time in >5% drawdown")
        else:
            parts.append(f"✓ **Low drawdown frequency:** Portfolio spent only {dd_frequency:.1f}% of time in >5% drawdown")
    
    return "\n".join(parts)


def _interpret_drawdown_periods(periods: list, avg_depth: float, avg_duration: float) -> str:
    """Interpret drawdown periods chart."""
    parts = []
    
    if len(periods) == 0:
        return "No significant drawdown periods detected (>5% threshold)"
    
    parts.append(f"**Found {len(periods)} drawdown period(s):**")
    parts.append(f"Average depth: {avg_depth*100:.1f}% | Average duration: {avg_duration:.0f} days")
    
    if avg_depth < -0.15:
        parts.append("⚠ Deep drawdowns - portfolio experiences significant losses")
    elif avg_duration > 100:
        parts.append("⚠ Long recovery times - portfolio takes time to recover from losses")
    else:
        parts.append("✓ Drawdowns are manageable in depth and duration")
    
    return " | ".join(parts)


def _interpret_var_distribution(var_hist: float, cvar: float, conf_level: float, var_param: float, var_cf: float) -> str:
    """Interpret VaR distribution chart."""
    parts = []
    parts.append(f"**Value at Risk Analysis ({int(conf_level*100)}% confidence):**")
    
    # Historical VaR
    parts.append(f"**Historical VaR:** {var_hist*100:.2f}% - Worst expected loss on {int((1-conf_level)*100)}% of days")
    
    # CVaR
    parts.append(f"**Conditional VaR (Expected Shortfall):** {cvar*100:.2f}% - Average loss on worst days")
    
    # Method comparison
    if abs(var_hist - var_param) < 0.001:
        parts.append(f"✓ Historical and Parametric VaR are similar - returns are close to normal")
    elif var_param < var_hist:
        parts.append(f"⚠ Parametric VaR ({var_param*100:.2f}%) underestimates risk vs Historical ({var_hist*100:.2f}%)")
    else:
        parts.append(f"Parametric VaR ({var_param*100:.2f}%) is more conservative than Historical")
    
    # Cornish-Fisher
    if abs(var_cf - var_param) > 0.002:
        parts.append(f"⚠ Cornish-Fisher VaR ({var_cf*100:.2f}%) differs from Parametric - non-normal distribution detected")
    
    # Practical implications
    parts.append(f"\n**Practical Implications:**")
    if var_hist < -0.05:
        parts.append(f"- Portfolio can lose more than 5% on worst {int((1-conf_level)*100)}% of days")
        parts.append(f"- On those worst days, average loss is {cvar*100:.2f}%")
        parts.append(f"- Consider reducing position sizes or adding hedging")
    else:
        parts.append(f"- Daily losses typically stay below 5%")
        parts.append(f"- Risk level is manageable for most investors")
    
    return "\n".join(parts)


def _interpret_var_benchmark_comparison(var_hist: float, cvar: float, bench_var_hist: float, bench_cvar: float, confidence_level: int) -> str:
    """Interpret VaR/CVaR benchmark comparison."""
    parts = []
    parts.append(f"**Benchmark Comparison Analysis ({confidence_level}% confidence):**")
    
    # VaR comparison
    var_diff = var_hist - bench_var_hist
    var_diff_pct = var_diff * 100
    
    if abs(var_diff) < 0.001:
        parts.append(f"Portfolio VaR ({var_hist*100:.2f}%) is similar to benchmark ({bench_var_hist*100:.2f}%) - similar risk levels")
    elif var_diff < 0:
        parts.append(f"Portfolio VaR ({var_hist*100:.2f}%) is lower than benchmark ({bench_var_hist*100:.2f}%) by {abs(var_diff_pct):.2f}% - portfolio has lower risk")
    else:
        parts.append(f"Portfolio VaR ({var_hist*100:.2f}%) is higher than benchmark ({bench_var_hist*100:.2f}%) by {var_diff_pct:.2f}% - portfolio has higher risk")
    
    # CVaR comparison
    cvar_diff = cvar - bench_cvar
    cvar_diff_pct = cvar_diff * 100
    
    if abs(cvar_diff) < 0.001:
        parts.append(f"Portfolio CVaR ({cvar*100:.2f}%) is similar to benchmark ({bench_cvar*100:.2f}%) - similar tail risk")
    elif cvar_diff < 0:
        parts.append(f"Portfolio CVaR ({cvar*100:.2f}%) is lower than benchmark ({bench_cvar*100:.2f}%) by {abs(cvar_diff_pct):.2f}% - portfolio has lower tail risk")
    else:
        parts.append(f"Portfolio CVaR ({cvar*100:.2f}%) is higher than benchmark ({bench_cvar*100:.2f}%) by {cvar_diff_pct:.2f}% - portfolio has higher tail risk")
    
    return "\n".join(parts)


def _interpret_rolling_metric(metric_series: pd.Series, metric_name: str, direction: str, window: int, threshold: float = None) -> str:
    """Interpret rolling metric chart."""
    if metric_series.empty:
        return ""
    
    parts = []
    current = metric_series.iloc[-1] if len(metric_series) > 0 else 0
    mean_val = metric_series.mean()
    std_val = metric_series.std()
    min_val = metric_series.min()
    max_val = metric_series.max()
    
    parts.append(f"**{metric_name} ({window}-day rolling):**")
    parts.append(f"Current: {current:.3f} | Average: {mean_val:.3f} | Range: {min_val:.3f} to {max_val:.3f}")
    
    # Trend analysis
    if len(metric_series) > 10:
        recent_avg = metric_series.iloc[-10:].mean()
        earlier_avg = metric_series.iloc[:10].mean()
        trend = "improving" if (direction == "higher" and recent_avg > earlier_avg) or (direction == "lower" and recent_avg < earlier_avg) else "declining"
        parts.append(f"Trend: {trend} (recent average: {recent_avg:.3f} vs earlier: {earlier_avg:.3f})")
    
    # Threshold comparison
    if threshold is not None:
        above_threshold = (metric_series > threshold).sum() / len(metric_series) * 100 if direction == "higher" else (metric_series < threshold).sum() / len(metric_series) * 100
        if direction == "higher":
            parts.append(f"Time above {threshold:.2f}: {above_threshold:.1f}%")
            if current > threshold:
                parts.append(f"✓ Currently above threshold ({threshold:.2f})")
            else:
                parts.append(f"⚠ Currently below threshold ({threshold:.2f})")
        else:
            parts.append(f"Time below {threshold:.2f}: {above_threshold:.1f}%")
            if current < threshold:
                parts.append(f"✓ Currently below threshold ({threshold:.2f})")
            else:
                parts.append(f"⚠ Currently above threshold ({threshold:.2f})")
    
    return " | ".join(parts)


def _interpret_rolling_beta(beta_series: pd.Series, window: int) -> str:
    """Interpret rolling beta chart."""
    if beta_series.empty:
        return ""
    
    parts = []
    current = beta_series.iloc[-1] if len(beta_series) > 0 else 0
    mean_val = beta_series.mean()
    min_val = beta_series.min()
    max_val = beta_series.max()
    
    parts.append(f"**Rolling Beta ({window}-day rolling):**")
    parts.append(f"Current: {current:.3f} | Average: {mean_val:.3f} | Range: {min_val:.3f} to {max_val:.3f}")
    
    # Beta interpretation
    if abs(current - 1.0) < 0.1:
        parts.append(f"Current beta ({current:.3f}) is close to 1.0 - portfolio moves similarly to benchmark")
    elif current > 1.0:
        parts.append(f"Current beta ({current:.3f}) > 1.0 - portfolio is more volatile than benchmark")
    else:
        parts.append(f"Current beta ({current:.3f}) < 1.0 - portfolio is less volatile than benchmark")
    
    # Trend analysis
    if len(beta_series) > 10:
        recent_avg = beta_series.iloc[-10:].mean()
        earlier_avg = beta_series.iloc[:10].mean()
        if abs(recent_avg - earlier_avg) < 0.05:
            parts.append(f"Beta is relatively stable (recent: {recent_avg:.3f} vs earlier: {earlier_avg:.3f})")
        elif recent_avg > earlier_avg:
            parts.append(f"Beta is increasing (recent: {recent_avg:.3f} vs earlier: {earlier_avg:.3f}) - portfolio becoming more sensitive to market")
        else:
            parts.append(f"Beta is decreasing (recent: {recent_avg:.3f} vs earlier: {earlier_avg:.3f}) - portfolio becoming less sensitive to market")
    
    return "\n".join(parts)


def _interpret_bull_bear_analysis(bull: dict, bear: dict) -> str:
    """Interpret bull/bear market analysis."""
    parts = []
    parts.append("**Bull/Bear Market Analysis:**")
    
    # Portfolio returns comparison
    bull_port_return = bull.get('portfolio_return', 0)
    bear_port_return = bear.get('portfolio_return', 0)
    return_diff = bull_port_return - bear_port_return
    
    if abs(return_diff) < 0.01:
        parts.append(f"Portfolio performs similarly in bull ({bull_port_return:.2f}%) and bear ({bear_port_return:.2f}%) markets")
    elif return_diff > 0:
        parts.append(f"Portfolio performs better in bull markets ({bull_port_return:.2f}%) than in bear markets ({bear_port_return:.2f}%) - difference: {return_diff:.2f}%")
    else:
        parts.append(f"Portfolio performs better in bear markets ({bear_port_return:.2f}%) than in bull markets ({bull_port_return:.2f}%) - difference: {abs(return_diff):.2f}%")
    
    # Beta comparison
    bull_beta = bull.get('beta', 0)
    bear_beta = bear.get('beta', 0)
    beta_diff = bull_beta - bear_beta
    
    if abs(beta_diff) < 0.1:
        parts.append(f"Beta is similar in bull ({bull_beta:.2f}) and bear ({bear_beta:.2f}) markets - consistent market sensitivity")
    elif beta_diff > 0:
        parts.append(f"Beta is higher in bull markets ({bull_beta:.2f}) than in bear markets ({bear_beta:.2f}) - more sensitive during uptrends")
    else:
        parts.append(f"Beta is higher in bear markets ({bear_beta:.2f}) than in bull markets ({bull_beta:.2f}) - more sensitive during downtrends")
    
    # Outperformance comparison
    bull_outperf = bull.get('difference', 0)
    bear_outperf = bear.get('difference', 0)
    
    if bull_outperf > 0 and bear_outperf > 0:
        parts.append(f"Portfolio outperforms benchmark in both bull ({bull_outperf:.2f}%) and bear ({bear_outperf:.2f}%) markets")
    elif bull_outperf > 0:
        parts.append(f"Portfolio outperforms in bull markets ({bull_outperf:.2f}%) but underperforms in bear markets ({bear_outperf:.2f}%)")
    elif bear_outperf > 0:
        parts.append(f"Portfolio underperforms in bull markets ({bull_outperf:.2f}%) but outperforms in bear markets ({bear_outperf:.2f}%)")
    else:
        parts.append(f"Portfolio underperforms benchmark in both bull ({bull_outperf:.2f}%) and bear ({bear_outperf:.2f}%) markets")
    
    return "\n".join(parts)


def _interpret_impact_on_return(impact_data: dict) -> str:
    """Interpret impact on return chart."""
    if not impact_data or not impact_data.get("tickers"):
        return ""
    
    parts = []
    tickers = impact_data["tickers"]
    contributions = impact_data["contributions"]
    
    if not tickers or not contributions:
        return ""
    
    # Top contributor
    top_ticker = tickers[0]
    top_contrib = contributions[0]
    
    parts.append(f"**Impact on Total Return Analysis:**")
    parts.append(f"Top contributor: {top_ticker} ({top_contrib:.2f}% of portfolio return)")
    
    # Concentration analysis
    total_contrib = sum(abs(c) for c in contributions)
    top_contrib_pct = abs(top_contrib) / total_contrib * 100 if total_contrib > 0 else 0
    
    if top_contrib_pct > 50:
        parts.append(f"⚠ High concentration: {top_ticker} accounts for {top_contrib_pct:.1f}% of total contribution")
        parts.append(f"Portfolio return is heavily dependent on single asset")
    elif top_contrib_pct > 30:
        parts.append(f"Moderate concentration: {top_ticker} accounts for {top_contrib_pct:.1f}% of total contribution")
    else:
        parts.append(f"✓ Well-distributed: top contributor accounts for {top_contrib_pct:.1f}% of total contribution")
    
    # Top 3 contributors
    if len(tickers) >= 3:
        top3_contrib = sum(abs(c) for c in contributions[:3])
        top3_pct = top3_contrib / total_contrib * 100 if total_contrib > 0 else 0
        parts.append(f"Top 3 assets account for {top3_pct:.1f}% of total return contribution")
    
    return "\n".join(parts)


def _interpret_impact_on_risk(impact_data: dict) -> str:
    """Interpret impact on risk chart."""
    if not impact_data or not impact_data.get("tickers"):
        return ""
    
    parts = []
    tickers = impact_data["tickers"]
    risk_contributions = impact_data["risk_contributions"]
    
    if not tickers or not risk_contributions:
        return ""
    
    # Top risk contributor
    top_ticker = tickers[0]
    top_risk = risk_contributions[0]
    
    parts.append(f"**Impact on Portfolio Risk Analysis:**")
    parts.append(f"Biggest risk contributor: {top_ticker} ({top_risk:.1f}% of portfolio risk)")
    
    # Concentration analysis
    total_risk = sum(abs(r) for r in risk_contributions)
    top_risk_pct = abs(top_risk) / total_risk * 100 if total_risk > 0 else 0
    
    if top_risk_pct > 50:
        parts.append(f"⚠ High risk concentration: {top_ticker} accounts for {top_risk_pct:.1f}% of total risk")
        parts.append(f"Portfolio risk is heavily dependent on single asset - consider diversification")
    elif top_risk_pct > 30:
        parts.append(f"Moderate risk concentration: {top_ticker} accounts for {top_risk_pct:.1f}% of total risk")
    else:
        parts.append(f"✓ Well-distributed risk: top contributor accounts for {top_risk_pct:.1f}% of total risk")
    
    # Top 3 contributors
    if len(tickers) >= 3:
        top3_risk = sum(abs(r) for r in risk_contributions[:3])
        top3_pct = top3_risk / total_risk * 100 if total_risk > 0 else 0
        parts.append(f"Top 3 assets account for {top3_pct:.1f}% of total risk contribution")
    
    return "\n".join(parts)


def _interpret_risk_vs_weight_comparison(comparison_data: dict) -> str:
    """Interpret risk vs weight comparison chart."""
    if not comparison_data or not comparison_data.get("tickers"):
        return ""
    
    parts = []
    tickers = comparison_data["tickers"]
    risk_impact = comparison_data.get("risk_impact", [])
    return_impact = comparison_data.get("return_impact", [])
    weights = comparison_data.get("weights", [])
    
    if not tickers or not risk_impact or not weights:
        return ""
    
    parts.append(f"**Risk vs Weight Comparison Analysis:**")
    
    # Find assets with significant imbalances
    imbalances = []
    for i, ticker in enumerate(tickers):
        if i < len(risk_impact) and i < len(weights):
            risk = abs(risk_impact[i])
            weight = abs(weights[i])
            
            if weight > 0:
                risk_ratio = risk / weight if weight > 0 else 0
                if risk_ratio > 1.5:
                    imbalances.append((ticker, "risk", risk_ratio, risk, weight))
                elif risk_ratio < 0.5:
                    imbalances.append((ticker, "low_risk", risk_ratio, risk, weight))
    
    # Find assets with return/weight imbalances
    return_imbalances = []
    if return_impact:
        for i, ticker in enumerate(tickers):
            if i < len(return_impact) and i < len(weights):
                ret = abs(return_impact[i])
                weight = abs(weights[i])
                
                if weight > 0:
                    return_ratio = ret / weight if weight > 0 else 0
                    if return_ratio > 1.5:
                        return_imbalances.append((ticker, return_ratio, ret, weight))
    
    if imbalances:
        high_risk = [x for x in imbalances if x[1] == "risk"]
        if high_risk:
            top_imbalance = high_risk[0]
            parts.append(f"⚠ {top_imbalance[0]}: Risk impact ({top_imbalance[3]:.1f}%) >> Weight ({top_imbalance[4]:.1f}%) - {top_imbalance[2]:.1f}x higher")
            parts.append(f"This asset contributes more risk than its portfolio weight suggests")
    
    if return_imbalances:
        top_return = return_imbalances[0]
        parts.append(f"✓ {top_return[0]}: Return impact ({top_return[2]:.2f}%) >> Weight ({top_return[3]:.1f}%) - {top_return[1]:.1f}x higher")
        parts.append(f"This asset contributes more return than its portfolio weight suggests")
    
    if not imbalances and not return_imbalances:
        parts.append(f"✓ Portfolio shows balanced risk/return distribution relative to weights")
        parts.append(f"Assets contribute proportionally to their weights")
    
    return "\n".join(parts)


def _interpret_correlation_matrix(corr_matrix_data: dict) -> str:
    """Interpret correlation matrix chart."""
    if not corr_matrix_data:
        return ""
    
    corr_matrix = corr_matrix_data.get("correlation_matrix")
    if corr_matrix is None or (hasattr(corr_matrix, 'empty') and corr_matrix.empty):
        return ""
    
    parts = []
    
    # Calculate statistics
    # Exclude diagonal (1.0 values)
    mask = ~np.eye(len(corr_matrix), dtype=bool)
    values = corr_matrix.values[mask]
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return ""
    
    avg_corr = np.mean(values)
    median_corr = np.median(values)
    max_corr = np.max(values)
    min_corr = np.min(values)
    
    # Count high/low correlations
    high_corr = np.sum(values > 0.8)
    low_corr = np.sum(values < 0.2)
    
    parts.append(f"**Correlation Matrix Analysis:**")
    
    # Overall assessment
    if avg_corr < 0.3:
        parts.append(f"✓ Low average correlation ({avg_corr:.2f}) - Excellent diversification potential")
    elif avg_corr < 0.5:
        parts.append(f"Moderate average correlation ({avg_corr:.2f}) - Good diversification potential")
    else:
        parts.append(f"⚠ High average correlation ({avg_corr:.2f}) - Limited diversification")
    
    # High correlation pairs
    if high_corr > 0:
        parts.append(f"⚠ Found {high_corr} pair(s) with correlation > 0.8 - Risk of concentration")
    else:
        parts.append(f"✓ No highly correlated pairs (>0.8) found")
    
    # Low correlation pairs
    if low_corr > 0:
        parts.append(f"✓ Found {low_corr} pair(s) with correlation < 0.2 - Good diversification opportunities")
    else:
        parts.append(f"⚠ No low correlation pairs (<0.2) found - Consider adding assets with lower correlation")
    
    # Range
    parts.append(f"Correlation range: {min_corr:.2f} to {max_corr:.2f}")
    
    return "\n".join(parts)


def _interpret_correlation_with_benchmark(corr_bench_data: dict) -> str:
    """Interpret correlation with benchmark chart."""
    if not corr_bench_data or not corr_bench_data.get("tickers"):
        return ""
    
    parts = []
    tickers = corr_bench_data["tickers"]
    correlations = corr_bench_data["correlations"]
    betas = corr_bench_data.get("betas", [])
    
    if not tickers or not correlations:
        return ""
    
    # Calculate average correlation
    avg_corr = np.mean(correlations)
    
    # Find highest and lowest correlations
    max_idx = np.argmax(correlations)
    min_idx = np.argmin(correlations)
    
    parts.append(f"**Correlation with Benchmark Analysis:**")
    parts.append(f"Average correlation: {avg_corr:.2f}")
    
    # Highest correlation
    parts.append(f"Highest correlation: {tickers[max_idx]} ({correlations[max_idx]:.2f})")
    if betas and max_idx < len(betas):
        parts.append(f"Beta: {betas[max_idx]:.2f} - {'More volatile' if betas[max_idx] > 1.0 else 'Less volatile'} than benchmark")
    
    # Lowest correlation
    if min_idx != max_idx:
        parts.append(f"Lowest correlation: {tickers[min_idx]} ({correlations[min_idx]:.2f})")
        if betas and min_idx < len(betas):
            parts.append(f"Beta: {betas[min_idx]:.2f} - {'More volatile' if betas[min_idx] > 1.0 else 'Less volatile'} than benchmark")
    
    # Overall assessment
    if avg_corr > 0.8:
        parts.append(f"⚠ Portfolio is highly sensitive to benchmark movements")
    elif avg_corr > 0.5:
        parts.append(f"Portfolio shows moderate sensitivity to benchmark")
    else:
        parts.append(f"✓ Portfolio shows low sensitivity to benchmark - good diversification")
    
    return "\n".join(parts)


def _interpret_average_correlation_to_portfolio(avg_corr_data: dict) -> str:
    """Interpret average correlation to portfolio chart."""
    if not avg_corr_data:
        return ""
    
    tickers = avg_corr_data.get("tickers")
    avg_correlations = avg_corr_data.get("avg_correlations")
    diversification_scores = avg_corr_data.get("diversification_scores", [])
    
    if not tickers or not avg_correlations or len(tickers) == 0 or len(avg_correlations) == 0:
        return ""
    
    parts = []
    
    # Find highest and lowest average correlations
    max_idx = np.argmax(avg_correlations)
    min_idx = np.argmin(avg_correlations)
    
    parts.append(f"**Average Correlation to Portfolio Analysis:**")
    
    # Highest correlation (worst for diversification)
    parts.append(f"⚠ Highest average correlation: {tickers[max_idx]} ({avg_correlations[max_idx]:.3f})")
    if diversification_scores and max_idx < len(diversification_scores):
        parts.append(f"Diversification score: {diversification_scores[max_idx]:.3f} - Consider reducing weight")
    
    # Lowest correlation (best for diversification)
    if min_idx != max_idx:
        parts.append(f"✓ Lowest average correlation: {tickers[min_idx]} ({avg_correlations[min_idx]:.3f})")
        if diversification_scores and min_idx < len(diversification_scores):
            parts.append(f"Diversification score: {diversification_scores[min_idx]:.3f} - Good for diversification")
    
    # Count assets with good/bad diversification
    good_div = sum(1 for c in avg_correlations if c < 0.3)
    bad_div = sum(1 for c in avg_correlations if c > 0.7)
    
    if good_div > 0:
        parts.append(f"✓ {good_div} asset(s) with good diversification (correlation < 0.3)")
    if bad_div > 0:
        parts.append(f"⚠ {bad_div} asset(s) with poor diversification (correlation > 0.7) - Consider reducing exposure")
    
    return "\n".join(parts)


def _interpret_rolling_correlations(rolling_corr_data: dict, window: int) -> str:
    """Interpret rolling correlations chart."""
    if not rolling_corr_data:
        return ""
    
    rolling_correlations = rolling_corr_data.get("rolling_correlations")
    if not rolling_correlations or (isinstance(rolling_correlations, dict) and len(rolling_correlations) == 0):
        return ""
    
    parts = []
    
    parts.append(f"**Rolling Correlations Analysis ({window}-day window):**")
    
    # Analyze each pair
    stability_analysis = []
    for pair_name, corr_series in rolling_correlations.items():
        if corr_series.empty:
            continue
        
        # Calculate statistics
        mean_corr = corr_series.mean()
        std_corr = corr_series.std()
        min_corr = corr_series.min()
        max_corr = corr_series.max()
        
        # Stability (low std = stable)
        is_stable = std_corr < 0.2
        
        stability_analysis.append({
            "pair": pair_name,
            "mean": mean_corr,
            "std": std_corr,
            "min": min_corr,
            "max": max_corr,
            "stable": is_stable
        })
    
    if not stability_analysis:
        return ""
    
    # Sort by mean correlation
    stability_analysis.sort(key=lambda x: abs(x["mean"]), reverse=True)
    
    # Top pair
    top_pair = stability_analysis[0]
    parts.append(f"Highest average correlation: {top_pair['pair']} ({top_pair['mean']:.2f})")
    
    if top_pair["stable"]:
        parts.append(f"Correlation is relatively stable (std: {top_pair['std']:.2f})")
    else:
        parts.append(f"⚠ Correlation is volatile (std: {top_pair['std']:.2f}) - Range: {top_pair['min']:.2f} to {top_pair['max']:.2f}")
    
    # Check for correlation spikes (high max relative to mean)
    high_spikes = [p for p in stability_analysis if p["max"] - p["mean"] > 0.3]
    if high_spikes:
        parts.append(f"⚠ {len(high_spikes)} pair(s) show correlation spikes during market stress")
    
    # Overall trend (if we have enough data points)
    if len(stability_analysis) > 0:
        avg_mean = np.mean([p["mean"] for p in stability_analysis])
        if avg_mean < 0.3:
            parts.append(f"✓ Overall low correlations - Good diversification maintained over time")
        elif avg_mean > 0.7:
            parts.append(f"⚠ Overall high correlations - Limited diversification over time")
    
    return "\n".join(parts)


def _interpret_asset_price_dynamics(price_dynamics_data: dict) -> str:
    """Interpret asset price dynamics chart."""
    if not price_dynamics_data or not price_dynamics_data.get("price_series"):
        return ""
    
    parts = []
    price_series = price_dynamics_data["price_series"]
    
    if not price_series:
        return ""
    
    # Calculate final returns for each asset
    final_returns = {}
    for ticker, series in price_series.items():
        if not series.empty:
            final_returns[ticker] = series.iloc[-1]
    
    if not final_returns:
        return ""
    
    # Find best and worst performers
    best_ticker = max(final_returns, key=final_returns.get)
    worst_ticker = min(final_returns, key=final_returns.get)
    best_return = final_returns[best_ticker]
    worst_return = final_returns[worst_ticker]
    
    parts.append(f"**Asset Price Dynamics Analysis:**")
    parts.append(f"Best performer: {best_ticker} ({best_return:+.2f}%)")
    parts.append(f"Worst performer: {worst_ticker} ({worst_return:+.2f}%)")
    
    # Calculate spread
    spread = best_return - worst_return
    if spread > 50:
        parts.append(f"⚠ Large performance spread ({spread:.2f}%) - High dispersion in asset returns")
    elif spread > 20:
        parts.append(f"Moderate performance spread ({spread:.2f}%)")
    else:
        parts.append(f"✓ Relatively tight performance spread ({spread:.2f}%) - Assets moving similarly")
    
    # Count positive/negative returns
    positive_count = sum(1 for r in final_returns.values() if r > 0)
    negative_count = len(final_returns) - positive_count
    
    if positive_count == len(final_returns):
        parts.append(f"✓ All assets show positive returns")
    elif negative_count == len(final_returns):
        parts.append(f"⚠ All assets show negative returns")
    else:
        parts.append(f"Mixed performance: {positive_count} positive, {negative_count} negative")
    
    return "\n".join(parts)


def _interpret_rolling_correlation_with_benchmark(rolling_corr_data: dict, window: int) -> str:
    """Interpret rolling correlation with benchmark chart."""
    if not rolling_corr_data:
        return ""
    
    rolling_correlations = rolling_corr_data.get("rolling_correlations")
    if not rolling_correlations or (isinstance(rolling_correlations, dict) and len(rolling_correlations) == 0):
        return ""
    
    parts = []
    
    parts.append(f"**Rolling Correlation with Benchmark Analysis ({window}-day window):**")
    
    # Analyze each asset
    correlation_analysis = []
    for ticker, corr_series in rolling_correlations.items():
        if corr_series.empty:
            continue
        
        mean_corr = corr_series.mean()
        std_corr = corr_series.std()
        min_corr = corr_series.min()
        max_corr = corr_series.max()
        
        correlation_analysis.append({
            "ticker": ticker,
            "mean": mean_corr,
            "std": std_corr,
            "min": min_corr,
            "max": max_corr,
            "stable": std_corr < 0.2
        })
    
    if not correlation_analysis:
        return ""
    
    # Sort by mean correlation
    correlation_analysis.sort(key=lambda x: abs(x["mean"]), reverse=True)
    
    # Highest correlation
    top_asset = correlation_analysis[0]
    parts.append(f"Highest average correlation: {top_asset['ticker']} ({top_asset['mean']:.2f})")
    
    if top_asset["stable"]:
        parts.append(f"Correlation is relatively stable (std: {top_asset['std']:.2f})")
    else:
        parts.append(f"⚠ Correlation is volatile (std: {top_asset['std']:.2f}) - Range: {top_asset['min']:.2f} to {top_asset['max']:.2f}")
    
    # Overall assessment
    avg_mean = np.mean([a["mean"] for a in correlation_analysis])
    if avg_mean > 0.8:
        parts.append(f"⚠ Portfolio assets are highly correlated with benchmark - Limited diversification")
    elif avg_mean > 0.5:
        parts.append(f"Moderate correlation with benchmark")
    else:
        parts.append(f"✓ Low correlation with benchmark - Good diversification")
    
    return "\n".join(parts)


def _interpret_price_volume_chart(detailed_data: dict, ticker: str) -> str:
    """Interpret price and volume chart."""
    if not detailed_data:
        return ""
    
    parts = []
    prices = detailed_data.get("prices")
    ma50 = detailed_data.get("ma50")
    ma200 = detailed_data.get("ma200")
    
    if prices is None or prices.empty:
        return ""
    
    # Calculate price change
    price_start = prices.iloc[0] if len(prices) > 0 else None
    price_end = prices.iloc[-1] if len(prices) > 0 else None
    
    if price_start and price_end and price_start > 0:
        price_change = ((price_end / price_start) - 1) * 100
        
        parts.append(f"**Price and Volume Analysis - {ticker}:**")
        parts.append(f"Price change: {price_change:+.2f}%")
        
        # Trend assessment
        if price_change > 20:
            parts.append(f"✓ Strong upward trend")
        elif price_change > 5:
            parts.append(f"Moderate upward trend")
        elif price_change < -20:
            parts.append(f"⚠ Strong downward trend")
        elif price_change < -5:
            parts.append(f"Moderate downward trend")
        else:
            parts.append(f"Relatively flat price movement")
        
        # MA analysis
        if ma50 is not None and not ma50.empty and ma200 is not None and not ma200.empty:
            current_price = price_end
            current_ma50 = ma50.iloc[-1] if len(ma50) > 0 else None
            current_ma200 = ma200.iloc[-1] if len(ma200) > 0 else None
            
            if current_ma50 and current_ma200:
                if current_price > current_ma50 > current_ma200:
                    parts.append(f"✓ Price above both MA50 and MA200 - Bullish signal")
                elif current_price < current_ma50 < current_ma200:
                    parts.append(f"⚠ Price below both MA50 and MA200 - Bearish signal")
                elif current_price > current_ma50:
                    parts.append(f"Price above MA50 but below MA200 - Mixed signal")
                else:
                    parts.append(f"Price below MA50 - Potential weakness")
    
    return "\n".join(parts)


def _interpret_comparison_of_return(detailed_data: dict, ticker: str) -> str:
    """Interpret comparison of return chart."""
    if not detailed_data or not detailed_data.get("cumulative_returns"):
        return ""
    
    parts = []
    cum_returns = detailed_data["cumulative_returns"]
    
    asset_cum = cum_returns.get("asset")
    portfolio_cum = cum_returns.get("portfolio")
    benchmark_cum = cum_returns.get("benchmark")
    
    if asset_cum is None or asset_cum.empty:
        return ""
    
    # Calculate final returns
    asset_final = asset_cum.iloc[-1] if len(asset_cum) > 0 else 0
    
    parts.append(f"**Return Comparison Analysis - {ticker}:**")
    parts.append(f"Asset total return: {asset_final*100:+.2f}%")
    
    if portfolio_cum is not None and not portfolio_cum.empty:
        portfolio_final = portfolio_cum.iloc[-1] if len(portfolio_cum) > 0 else 0
        diff_portfolio = (asset_final - portfolio_final) * 100
        parts.append(f"Portfolio total return: {portfolio_final*100:+.2f}%")
        
        if abs(diff_portfolio) < 1:
            parts.append(f"Asset performance similar to portfolio ({diff_portfolio:+.2f}% difference)")
        elif diff_portfolio > 0:
            parts.append(f"✓ Asset outperforms portfolio by {diff_portfolio:.2f}%")
        else:
            parts.append(f"⚠ Asset underperforms portfolio by {abs(diff_portfolio):.2f}%")
    
    if benchmark_cum is not None and not benchmark_cum.empty:
        benchmark_final = benchmark_cum.iloc[-1] if len(benchmark_cum) > 0 else 0
        diff_benchmark = (asset_final - benchmark_final) * 100
        parts.append(f"Benchmark total return: {benchmark_final*100:+.2f}%")
        
        if abs(diff_benchmark) < 1:
            parts.append(f"Asset performance similar to benchmark ({diff_benchmark:+.2f}% difference)")
        elif diff_benchmark > 0:
            parts.append(f"✓ Asset outperforms benchmark by {diff_benchmark:.2f}%")
        else:
            parts.append(f"⚠ Asset underperforms benchmark by {abs(diff_benchmark):.2f}%")
    
    return "\n".join(parts)


def _interpret_asset_correlations(other_corrs: dict, ticker: str) -> str:
    """Interpret correlations with other assets chart."""
    if not other_corrs:
        return ""
    
    parts = []
    
    # Sort by correlation value
    sorted_corrs = sorted(other_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    if not sorted_corrs:
        return ""
    
    parts.append(f"**Correlation Analysis - {ticker}:**")
    
    # Highest correlation
    highest_ticker, highest_corr = sorted_corrs[0]
    parts.append(f"Highest correlation: {highest_ticker} ({highest_corr:.2f})")
    
    if highest_corr > 0.8:
        parts.append(f"⚠ Very high correlation - Assets move together, limited diversification benefit")
    elif highest_corr > 0.5:
        parts.append(f"Moderate correlation - Some diversification benefit")
    else:
        parts.append(f"✓ Low correlation - Good diversification potential")
    
    # Lowest correlation
    if len(sorted_corrs) > 1:
        lowest_ticker, lowest_corr = sorted_corrs[-1]
        parts.append(f"Lowest correlation: {lowest_ticker} ({lowest_corr:.2f})")
        
        if lowest_corr < 0:
            parts.append(f"✓ Negative correlation - Natural hedging opportunity")
        elif lowest_corr < 0.2:
            parts.append(f"✓ Very low correlation - Excellent diversification")
    
    # Count high/low correlations
    high_corr_count = sum(1 for _, corr in sorted_corrs if abs(corr) > 0.7)
    low_corr_count = sum(1 for _, corr in sorted_corrs if abs(corr) < 0.3)
    
    if high_corr_count > 0:
        parts.append(f"⚠ {high_corr_count} asset(s) with high correlation (>0.7) - Consider reducing exposure to similar assets")
    if low_corr_count > 0:
        parts.append(f"✓ {low_corr_count} asset(s) with low correlation (<0.3) - Good diversification opportunities")
    
    return "\n".join(parts)


def show():
    """Render Portfolio Analysis page."""
    st.title("Portfolio Analysis")

    # Services
    portfolio_service = PortfolioService()
    analytics_service = AnalyticsService()

    # Get portfolios
    portfolios = portfolio_service.list_portfolios()

    if not portfolios:
        st.warning("No portfolios found. Please create a portfolio first.")
        return

    # === ANALYSIS PARAMETERS SECTION ===
    st.subheader("Analysis Parameters")

    # Row 1: Start Date (left) + End Date (right)
    col1, col2 = st.columns([2, 2])

    portfolio_names = {p.name: p.id for p in portfolios}
    default_end = date.today()
    default_start = default_end - timedelta(days=365)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=default_end,
            key="start_date_input",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end,
            min_value=start_date,
            max_value=default_end,
            key="end_date_input",
        )

    # Row 2: Portfolio (left) + Comparison selector (right)
    col1, col2 = st.columns([2, 2])

    with col1:
        selected_name = st.selectbox(
            "Portfolio",
            options=list(portfolio_names.keys()),
            key="portfolio_analysis_selector",
        )
        portfolio_id = portfolio_names[selected_name]

    with col2:
        st.markdown("**Comparison**")
        cmp_type = st.radio(
            "",
            options=["None", "Index ETF", "Portfolio"],
            horizontal=True,
            label_visibility="collapsed",
            key="cmp_type_radio",
        )
        comparison_type = None
        comparison_value = None
        if cmp_type == "Index ETF":
            presets = ["SPY", "QQQ", "VTI", "DIA", "IWM"]
            comparison_value = st.selectbox("Index ETF", options=presets, key="cmp_etf_select")
            comparison_type = "ticker"
        elif cmp_type == "Portfolio":
            other_names = [name for name in portfolio_names.keys() if name != selected_name]
            if other_names:
                sel_other = st.selectbox("Portfolio", options=other_names, key="cmp_portfolio_select")
                comparison_value = portfolio_names[sel_other]
                comparison_type = "portfolio"
            else:
                st.info("No other portfolios available for comparison")
        else:
            st.caption("No comparison selected")
        benchmark_ticker = None

    # Row 3: Risk-Free Rate + Buttons
    col1, col2, col3 = st.columns([2, 3, 3])

    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.35,
            step=0.01,
            key="risk_free_rate_input",
        ) / 100

    with col2:
        if st.button("Calculate Metrics", type="primary", use_container_width=True):
            with st.spinner("Calculating analytics..."):
                try:
                    result = analytics_service.calculate_portfolio_metrics(
                        portfolio_id=portfolio_id,
                        benchmark_ticker=benchmark_ticker,
                        comparison_type=comparison_type,
                        comparison_value=comparison_value,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    st.session_state.portfolio_analytics = result
                    st.session_state.portfolio_id = portfolio_id
                    st.session_state.selected_name = selected_name
                    st.success("Analytics calculated!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error calculating analytics: {e}")
                    st.error(f"Error: {str(e)}")

    with col3:
        if st.button("Update Prices", type="secondary", use_container_width=True):
            st.info("Price update functionality coming soon...")

    st.markdown("---")

    # Check if analytics available
    if "portfolio_analytics" not in st.session_state:
        st.info("Click 'Calculate Metrics' to start analysis")
        return

    analytics = st.session_state.portfolio_analytics
    portfolio_id = st.session_state.get("portfolio_id", portfolio_id)
    selected_name = st.session_state.get("selected_name", selected_name)
    
    # Extract data
    portfolio_returns = analytics.get("portfolio_returns")
    # Use comparison series as benchmark for rendering
    comparison_returns = analytics.get("comparison_returns")
    benchmark_returns = comparison_returns if comparison_returns is not None else analytics.get("benchmark_returns")
    portfolio_values = analytics.get("portfolio_values")
    perf = analytics.get("performance", {})
    risk = analytics.get("risk", {})
    ratios = analytics.get("ratios", {})
    market = analytics.get("market", {})
    
    # Benchmark metrics will be calculated in _render_overview_tab
    
    # Get portfolio positions
    try:
        portfolio = portfolio_service.get_portfolio(portfolio_id)
        positions = portfolio.get_all_positions() if portfolio else []
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        positions = []
    
    # === MAIN TABS ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Performance",
        "Risk",
        "Assets & Correlations",
        "Export & Reports"
    ])
    
    # === TAB 1: OVERVIEW ===
    with tab1:
        _render_overview_tab(
            perf, risk, ratios, market,
            portfolio_returns, benchmark_returns, portfolio_values,
            positions, start_date, end_date, risk_free_rate
        )
    
    # === TAB 2: PERFORMANCE ===
    with tab2:
        _render_performance_tab(
            perf, portfolio_returns, benchmark_returns, portfolio_values,
            risk_free_rate, start_date, end_date
        )
    
    # === TAB 3: RISK ===
    with tab3:
        _render_risk_tab(
            risk, ratios, market,
            portfolio_returns, benchmark_returns, portfolio_values,
            risk_free_rate, start_date, end_date
        )
    
    # === TAB 4: ASSETS & CORRELATIONS ===
    with tab4:
        _render_assets_tab(
            positions, portfolio_returns, benchmark_returns,
            portfolio_id, portfolio_service
        )
    
    # === TAB 5: EXPORT & REPORTS ===
    with tab5:
        _render_export_tab(
            selected_name, perf, risk, ratios, market,
            portfolio_returns, benchmark_returns, portfolio_values,
            positions, start_date, end_date, risk_free_rate
        )


def _render_overview_tab(
    perf, risk, ratios, market,
    portfolio_returns, benchmark_returns, portfolio_values,
    positions, start_date, end_date, risk_free_rate=0.0435
):
    """Render Overview tab content."""
    # Section 1.1: Key Performance Cards
    st.subheader("Key Performance Metrics")
    
    # Prepare benchmark metrics for cards (reuse calc below when possible)
    bm_for_cards = {}
    common_idx = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        # Align by overlapping dates only, no zero-filling to avoid distortions
        try:
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_bench_cards = benchmark_returns.loc[common_idx]
        except Exception:
            aligned_bench_cards = benchmark_returns.reindex(
                portfolio_returns.index, method="ffill"
            ).dropna()
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.risk_metrics import (
            calculate_volatility,
            calculate_max_drawdown,
        )
        from core.analytics_engine.ratios import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
        )
        max_dd_cards = calculate_max_drawdown(aligned_bench_cards)
        bm_for_cards = {
            "total_return": float((1 + aligned_bench_cards).prod() - 1),
            "annualized_return": float(calculate_annualized_return(aligned_bench_cards)),
            "volatility": float(calculate_volatility(aligned_bench_cards).get("annual", 0.0)),
            "max_drawdown": float(max_dd_cards[0] if isinstance(max_dd_cards, tuple) else max_dd_cards),
            "sharpe_ratio": float(calculate_sharpe_ratio(aligned_bench_cards, risk_free_rate=risk_free_rate) or 0),
            "sortino_ratio": float(calculate_sortino_ratio(aligned_bench_cards, risk_free_rate=risk_free_rate) or 0),
        }

    # Prepare portfolio metrics for cards/table early
    portfolio_metrics_flat = {
        "total_return": perf.get("total_return", 0),
        "cagr": perf.get("cagr", perf.get("annualized_return", 0)),
        "annualized_return": perf.get("annualized_return", 0),
        "volatility": (risk.get("volatility", {}) or {}).get("annual", risk.get("volatility", 0)) if isinstance(risk.get("volatility", 0), dict) else risk.get("volatility", 0),
        "sharpe_ratio": ratios.get("sharpe_ratio", 0),
        "sortino_ratio": ratios.get("sortino_ratio", 0),
        "max_drawdown": risk.get("max_drawdown", 0),
        "calmar_ratio": ratios.get("calmar_ratio", 0),
        "beta": market.get("beta", 0),
        "alpha": market.get("alpha", 0),
        "up_capture": market.get("up_capture", None),
        "down_capture": market.get("down_capture", None),
    }

    # Fallbacks if backend returned zeros: compute from portfolio_returns
    try:
        if portfolio_returns is not None and not portfolio_returns.empty:
            from core.analytics_engine.performance import calculate_annualized_return
            from core.analytics_engine.ratios import calculate_sharpe_ratio
            if abs(portfolio_metrics_flat.get("total_return", 0)) < 1e-8:
                portfolio_metrics_flat["total_return"] = float((1 + portfolio_returns).prod() - 1)
            if abs(portfolio_metrics_flat.get("annualized_return", 0)) < 1e-8:
                portfolio_metrics_flat["annualized_return"] = float(
                    calculate_annualized_return(portfolio_returns)
                )
            if abs(portfolio_metrics_flat.get("sharpe_ratio", 0)) < 1e-6:
                portfolio_metrics_flat["sharpe_ratio"] = float(
                    calculate_sharpe_ratio(portfolio_returns, risk_free_rate=risk_free_rate) or 0
                )
            # Ensure CAGR equals annualized return for consistency in table
            portfolio_metrics_flat["cagr"] = portfolio_metrics_flat.get("annualized_return", portfolio_metrics_flat.get("cagr", 0))
            # Volatility & MaxDD fallback from returns for strict equality with benchmark
            try:
                from core.analytics_engine.risk_metrics import (
                    calculate_volatility as _calc_vol,
                    calculate_max_drawdown as _calc_dd,
                )
                # If benchmark exists, recalculate using common date range
                series_for_calc = (
                    portfolio_returns.loc[common_idx]
                    if common_idx is not None and len(common_idx) > 1
                    else portfolio_returns
                )
                vol = _calc_vol(series_for_calc)
                if isinstance(vol, dict):
                    vol_val = float(vol.get("annual", 0.0))
                else:
                    vol_val = float(vol)
                if vol_val == 0 or portfolio_metrics_flat.get("volatility") in (0, None):
                    portfolio_metrics_flat["volatility"] = vol_val
                dd = _calc_dd(series_for_calc)
                dd_val = float(dd[0] if isinstance(dd, tuple) else dd)
                if dd_val != 0 and portfolio_metrics_flat.get("max_drawdown", 0) == 0:
                    portfolio_metrics_flat["max_drawdown"] = dd_val
            except Exception:
                pass
    except Exception:
        pass

    # Row 1: Total Return, CAGR, Volatility, Max Drawdown
    metrics_row1 = [
        {
            "label": "Total Return",
            "portfolio_value": portfolio_metrics_flat.get("total_return", 0),
            "benchmark_value": bm_for_cards.get("total_return"),
            "format": "percent",
            "higher_is_better": True,
            "help_text": "Cumulative return from start to end date.",
        },
        {
            "label": "CAGR",
            "portfolio_value": portfolio_metrics_flat.get("annualized_return", 0),
            "benchmark_value": bm_for_cards.get("annualized_return"),
            "format": "percent",
            "higher_is_better": True,
            "help_text": "Average annual return assuming reinvestment.",
        },
        {
            "label": "Volatility",
            "portfolio_value": portfolio_metrics_flat.get("volatility", 0),
            "benchmark_value": bm_for_cards.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
            "help_text": "Annualized standard deviation of returns.",
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": risk.get("max_drawdown", 0),
            "benchmark_value": bm_for_cards.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
            "help_text": "Largest peak-to-trough decline.",
        },
    ]
    render_metric_cards_row(metrics_row1, columns_per_row=4)

    # Row 2: Sharpe Ratio, Sortino Ratio, Beta, Alpha
    st.markdown("---")
    metrics_row2 = [
        {
            "label": "Sharpe Ratio",
            "portfolio_value": portfolio_metrics_flat.get("sharpe_ratio", 0),
            "benchmark_value": bm_for_cards.get("sharpe_ratio"),
            "format": "ratio",
            "higher_is_better": True,
            "help_text": "Risk-adjusted return.",
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "benchmark_value": bm_for_cards.get("sortino_ratio"),
            "format": "ratio",
            "higher_is_better": True,
            "help_text": "Like Sharpe but only penalizes downside volatility.",
        },
        {
            "label": "Beta",
            "portfolio_value": market.get("beta", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "ratio",
            "higher_is_better": None,  # Special: closer to 1.0 is better
            "help_text": "Sensitivity to market movements.",
        },
        {
            "label": "Alpha",
            "portfolio_value": market.get("alpha", 0),
            "benchmark_value": 0.0,
            "format": "percent",
            "higher_is_better": True,
            "help_text": "Excess return above benchmark. Positive = outperformance, negative = underperformance.",
        },
    ]
    render_metric_cards_row(metrics_row2, columns_per_row=4)
    
    # Section 1.2: Portfolio Dynamics (3 graphs stacked)
    st.markdown("---")
    st.subheader("Portfolio Performance")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        # Enforce selected period on series for plotting (normalize tz)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        try:
            pr_index = portfolio_returns.index.tz_localize(None)
        except Exception:
            pr_index = portfolio_returns.index
        pr = portfolio_returns[(pr_index >= start_ts) & (pr_index < end_ts)]
        br = None
        if benchmark_returns is not None and not benchmark_returns.empty:
            try:
                br_index = benchmark_returns.index.tz_localize(None)
            except Exception:
                br_index = benchmark_returns.index
            br = benchmark_returns[(br_index >= start_ts) & (br_index < end_ts)]
        # Cumulative Returns
        cum_data = get_cumulative_returns_data(pr, br)
        if cum_data:
            fig = plot_cumulative_returns(cum_data)
            st.plotly_chart(fig, use_container_width=True, key="overview_cumulative_returns")
            
            # Automatic interpretation
            portfolio_cum = cum_data.get("portfolio")
            benchmark_cum = cum_data.get("benchmark")
            if portfolio_cum is not None and not portfolio_cum.empty:
                interpretation = _interpret_cumulative_returns(portfolio_cum, benchmark_cum)
                st.info(interpretation)
        
        # Underwater Plot (Drawdowns)
        if portfolio_values is not None and not portfolio_values.empty:
            # Calculate benchmark values from benchmark returns if available
            benchmark_values = None
            if br is not None and not br.empty:
                aligned_bench = br.reindex(
                    portfolio_values.index, method="ffill"
                ).fillna(0)
                # Start from same initial value as portfolio
                initial_value = float(portfolio_values.iloc[0])
                benchmark_values = (1 + aligned_bench).cumprod() * initial_value
            
            underwater_data = get_underwater_plot_data(
                portfolio_values, benchmark_values
            )
            if underwater_data:
                fig = plot_underwater(underwater_data)
                st.plotly_chart(fig, use_container_width=True, key="overview_underwater")
                
                # Automatic interpretation
                portfolio_dd = underwater_data.get("underwater", pd.Series())
                if not portfolio_dd.empty:
                    max_dd = portfolio_dd.min() / 100  # Convert from % to decimal
                    current_dd = portfolio_dd.iloc[-1] / 100 if len(portfolio_dd) > 0 else 0
                    interpretation = _interpret_drawdown_chart(max_dd, current_dd, portfolio_dd)
                    st.info(interpretation)
        
        # Daily Returns (bar chart) - without benchmark
        st.subheader("Daily Returns")
        daily_df = pd.DataFrame({
            "Date": pr.index,
            "Return": pr.values * 100,
        })
        daily_df["Color"] = daily_df["Return"].apply(
            lambda x: COLORS["success"] if x >= 0 else COLORS["danger"]
        )
        
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_df["Date"],
            y=daily_df["Return"],
            marker_color=daily_df["Color"],
            name="Daily Returns",
        ))
        fig.update_layout(
            title="Daily Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            hovermode="x unified",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_daily_returns")
        
        # Automatic interpretation
        interpretation = _interpret_daily_returns(pr)
        st.info(interpretation)
    
    # Section 1.3: Portfolio Structure
    st.markdown("---")
    st.subheader("Portfolio Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution by Assets**")
        if positions:
            # Use weight_target if available, otherwise equal weight
            weights = []
            for pos in positions:
                if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                    weights.append(pos.weight_target)
                else:
                    weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
            
            total_weight = sum(weights)
            if total_weight > 0:
                # Build mapping ticker -> weight % for donut
                alloc_data = {}
                for pos, w in zip(positions, weights):
                    pct = (w / total_weight * 100)
                    alloc_data[pos.ticker] = alloc_data.get(pos.ticker, 0.0) + pct
                fig = plot_asset_allocation(alloc_data)
                st.plotly_chart(fig, use_container_width=True, key="overview_asset_allocation")
                
                # Show top asset
                if alloc_data:
                    top_ticker = max(alloc_data, key=alloc_data.get)
                    top_weight = alloc_data[top_ticker]
                    st.caption(f"**Top asset:** {top_ticker} - {top_weight:.1f}% of portfolio")
        else:
            st.info("No positions available")

    with col2:
        st.markdown("**Distribution by Sectors**")
        from core.data_manager.ticker_validator import TickerValidator
        validator = TickerValidator()
        sector_to_weight: dict[str, float] = {}
        if positions:
            # Build weights like in asset allocation
            weights = []
            for pos in positions:
                if hasattr(pos, 'weight_target') and pos.weight_target is not None:
                    weights.append(pos.weight_target)
                else:
                    weights.append(1.0 / len(positions) if len(positions) > 0 else 0.0)
            total_weight = sum(weights)
            tickers = [pos.ticker for pos in positions]
            for tkr, w in zip(tickers, weights):
                if tkr == "CASH":
                    sector = "Cash"
                else:
                    try:
                        info = validator.get_ticker_info(tkr)
                        sector = info.sector or "Other"
                    except Exception:
                        sector = "Other"
                pct = (w / total_weight * 100) if total_weight > 0 else 0.0
                sector_to_weight[sector] = sector_to_weight.get(sector, 0.0) + pct
        if sector_to_weight:
            fig = plot_sector_allocation(sector_to_weight)
            st.plotly_chart(fig, use_container_width=True, key="overview_sector_allocation")
            
            # Show top sector
            top_sector = max(sector_to_weight, key=sector_to_weight.get)
            top_sector_weight = sector_to_weight[top_sector]
            st.caption(f"**Top sector:** {top_sector} - {top_sector_weight:.1f}% of portfolio")
        else:
            st.info("No sector data available")
    
    # Section 1.4: Comparison Table
    st.markdown("---")
    st.subheader("Portfolio vs Comparison")
    
    # Calculate benchmark metrics if benchmark data available
    from core.analytics_engine.performance import (
        calculate_annualized_return,
    )
    from core.analytics_engine.risk_metrics import (
        calculate_volatility,
        calculate_max_drawdown,
    )
    from core.analytics_engine.ratios import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )
    
    # portfolio_metrics_flat already prepared above
    
    # Calculate comparison metrics
    benchmark_metrics_flat = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            # Align by strict intersection to avoid distortions
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            aligned_bench = benchmark_returns.loc[common_idx]
            if not aligned_bench.empty:
                # Calculate benchmark metrics
                max_dd_result = calculate_max_drawdown(aligned_bench)
                max_dd_value = max_dd_result[0] if isinstance(
                    max_dd_result, tuple
                ) else max_dd_result
                
                benchmark_metrics_flat = {
                    "total_return": float((1 + aligned_bench).prod() - 1),
                    "annualized_return": float(
                        calculate_annualized_return(aligned_bench)
                    ),
                    "cagr": float(calculate_annualized_return(aligned_bench)),
                    "volatility": float(calculate_volatility(aligned_bench).get("annual", 0.0)),
                    "max_drawdown": float(max_dd_value),
                    "sharpe_ratio": float(
                        calculate_sharpe_ratio(
                            aligned_bench, risk_free_rate=risk_free_rate
                        ) or 0
                    ),
                    "sortino_ratio": float(
                        calculate_sortino_ratio(
                            aligned_bench, risk_free_rate=risk_free_rate
                        ) or 0
                    ),
                    "calmar_ratio": float((calculate_annualized_return(aligned_bench))/abs(max_dd_value)) if max_dd_value not in (0, None) else 0.0,
                    "beta": 1.0,  # Benchmark beta is always 1.0 vs itself
                    "alpha": 0.0,  # Benchmark alpha is 0 vs itself
                    "up_capture": 1.0,
                    "down_capture": 1.0,
                }
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
    
    render_comparison_table(
        portfolio_metrics=portfolio_metrics_flat,
        benchmark_metrics=benchmark_metrics_flat if benchmark_metrics_flat else None,
        title="Key Metrics Comparison",
        height=480,
    )
    
    # Section 1.5: Time in Market Metadata
    st.markdown("---")
    st.subheader("Analysis Metadata")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        trading_days = len(portfolio_returns)
        total_days = (end_date - start_date).days + 1
        data_quality = (trading_days / total_days * 100) if total_days > 0 else 0
        
        metadata_text = f"""
**Analysis Period:** {start_date} to {end_date} ({total_days} days)
**Trading Days:** {trading_days}
**Time in Market:** {trading_days}/{total_days} days ({data_quality:.1f}%)
**Data Quality:** {data_quality:.1f}% (no missing data)
**Last Updated:** {date.today()} {pd.Timestamp.now().strftime('%H:%M:%S')}
        """
        st.markdown(metadata_text)


def _render_performance_tab(perf, portfolio_returns, benchmark_returns, portfolio_values, risk_free_rate, start_date, end_date):
    """Render Performance tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Returns Analysis",
        "Periodic Analysis",
        "Distribution"
    ])
    
    # Calculate benchmark values if needed
    benchmark_values = None
    if benchmark_returns is not None and not benchmark_returns.empty and portfolio_values is not None:
        aligned_bench = benchmark_returns.reindex(portfolio_values.index, method="ffill").fillna(0)
        initial_value = float(portfolio_values.iloc[0])
        benchmark_values = (1 + aligned_bench).cumprod() * initial_value
    
    with sub_tab1:
        _render_returns_analysis(perf, portfolio_returns, benchmark_returns, portfolio_values, benchmark_values)
    
    with sub_tab2:
        _render_periodic_analysis(portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_distribution_analysis(portfolio_returns, benchmark_returns)


def _render_returns_analysis(perf, portfolio_returns, benchmark_returns, portfolio_values, benchmark_values):
    """Sub-tab 2.1: Returns Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return

    # Graph 2.1.1: Cumulative Returns
    st.subheader("Cumulative Returns")
    cum_data = get_cumulative_returns_data(portfolio_returns, benchmark_returns)
    if cum_data:
        fig = plot_cumulative_returns(cum_data)
        st.plotly_chart(fig, use_container_width=True, key="returns_cumulative")
        
        # Automatic interpretation
        portfolio_cum = cum_data.get("portfolio")
        benchmark_cum = cum_data.get("benchmark")
        if portfolio_cum is not None and not portfolio_cum.empty:
            interpretation = _interpret_cumulative_returns(portfolio_cum, benchmark_cum)
            st.info(interpretation)
    
    # Graph 2.1.2: Daily Active Returns (Area Chart)
    st.markdown("---")
    st.subheader("Daily Active Returns")
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = benchmark_returns.reindex(portfolio_returns.index, method="ffill").fillna(0)
        active_returns = portfolio_returns - aligned
        
        # Area chart
        fig = plot_active_returns_area(active_returns)
        st.plotly_chart(fig, use_container_width=True, key="returns_active_returns_area")

        # Stats box - display values as metrics
        pos_days = (active_returns > 0).sum()
        total_days = len(active_returns)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Avg Daily Active Return", f"{active_returns.mean()*100:.2f}%")
        with col2:
            st.metric("Positive Days", f"{pos_days} ({pos_days/total_days*100:.1f}%)")
        with col3:
            st.metric("Negative Days", f"{total_days - pos_days} ({(total_days-pos_days)/total_days*100:.1f}%)")
        with col4:
            st.metric("Max Daily Alpha", f"{active_returns.max()*100:.2f}%")
        with col5:
            st.metric("Min Daily Alpha", f"{active_returns.min()*100:.2f}%")
        
        # Automatic interpretation
        interpretation = _interpret_active_returns(active_returns)
        st.info(interpretation)
    
    # Table 2.1.3: Return by Periods
    st.markdown("---")
    st.subheader("Return by Periods")
    periods_data = get_period_returns_comparison_data(
        portfolio_returns, benchmark_returns, portfolio_values, benchmark_values
    )
    if periods_data.get("periods") is not None:
        periods_df = periods_data["periods"].copy()
        
        # Format for display (keep original values for calculations)
        from streamlit_app.components.comparison_table import (
            _calculate_percentage_change,
        )
        
        display_data = []
        for _, row in periods_df.iterrows():
            port_val = row["Portfolio"]
            bench_val = row["Benchmark"]
            
            # Format values (already in decimal format)
            port_formatted = f"{port_val * 100:.2f}%" if pd.notna(port_val) else "—"
            bench_formatted = f"{bench_val * 100:.2f}%" if pd.notna(bench_val) else "—"
            
            # Calculate relative difference like in comparison_table
            diff_formatted = "—"
            if pd.notna(port_val) and pd.notna(bench_val):
                # Use same logic as comparison_table for returns
                diff_pct = _calculate_percentage_change(
                    "total_return", port_val, bench_val
                )
                # Format without emoji circles
                if abs(diff_pct) < 0.01:
                    diff_formatted = "—"
                else:
                    sign = "+" if diff_pct > 0 else ""
                    diff_formatted = f"{sign}{diff_pct:.2f}%"
            
            display_data.append({
                "Period": row["Period"],
                "Portfolio": port_formatted,
                "Benchmark": bench_formatted,
                "Difference": diff_formatted,
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Bar chart (use original values, not multiplied)
        fig = plot_period_returns_bar(periods_df)
        st.plotly_chart(fig, use_container_width=True, key="returns_periods_bar")
        
        # Automatic interpretation
        interpretation = _interpret_period_returns(periods_df)
        st.info(interpretation)
    
    # Table 2.1.4: Expected Returns
    st.markdown("---")
    st.subheader("Expected Returns (Mean Historical)")
    expected_port = calculate_expected_returns(portfolio_returns)
    expected_bench = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        expected_bench = calculate_expected_returns(benchmark_returns)
    
    if expected_port:
        timeframes = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        port_vals = [
            expected_port.get("expected_daily", 0) * 100,
            expected_port.get("expected_weekly", 0) * 100,
            expected_port.get("expected_monthly", 0) * 100,
            expected_port.get("expected_quarterly", 0) * 100,
            expected_port.get("expected_yearly", 0) * 100,
        ]
        bench_vals = [0.0] * 5
        if expected_bench:
            bench_vals = [
                expected_bench.get("expected_daily", 0) * 100,
                expected_bench.get("expected_weekly", 0) * 100,
                expected_bench.get("expected_monthly", 0) * 100,
                expected_bench.get("expected_quarterly", 0) * 100,
                expected_bench.get("expected_yearly", 0) * 100,
            ]
        
        expected_df = pd.DataFrame({
            "Timeframe": timeframes,
            "Portfolio": [f"{v:.2f}%" for v in port_vals],
            "Benchmark": [f"{v:.2f}%" for v in bench_vals],
            "Difference": [f"{p - b:.2f}%" for p, b in zip(port_vals, bench_vals)],
        })
        st.dataframe(expected_df, use_container_width=True, hide_index=True)
        st.caption("Note: Based on arithmetic mean of historical returns")
    
    # Metric 2.1.5: Common Performance Periods (CPP)
    st.markdown("---")
    st.subheader("Common Performance Periods (CPP)")
    if benchmark_returns is not None and not benchmark_returns.empty:
        cpp_data = calculate_common_performance_periods(portfolio_returns, benchmark_returns)
        if cpp_data:
            same_dir_pct = cpp_data.get("same_direction_pct", 0) * 100
            cpp_index = cpp_data.get("cpp_index", 0)
            
            # Display values as metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Same Direction", f"{same_dir_pct:.1f}%")
                st.caption("Portfolio and Benchmark moved in same direction")
            with col2:
                st.metric("CPP Index", f"{cpp_index:.2f}")
                st.caption("Correlation of directional moves")
            
            # Interpretation
            correlation_level = 'highly' if cpp_index > 0.7 else 'moderately' if cpp_index > 0.4 else 'lowly'
            st.info(f"Portfolio is {correlation_level} correlated with market direction.")
    
    # Table 2.1.6: Best/Worst Periods (3-month rolling)
    st.markdown("---")
    st.subheader("The Best and Worst Periods")
    
    # Best 3-Month Periods
    st.markdown("**Best 3-Month Periods:**")
    rolling_periods = get_three_month_rolling_periods_data(
        portfolio_returns, benchmark_returns, top_n=3
    )
    if rolling_periods.get("best") is not None and not rolling_periods["best"].empty:
        best_df = rolling_periods["best"].copy()
        best_df["#"] = range(1, len(best_df) + 1)
        best_df["Portfolio"] = best_df["Portfolio"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Benchmark"] = best_df["Benchmark"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Difference"] = best_df["Difference"].apply(lambda x: f"{x*100:.2f}%")
        best_df["Start"] = best_df["Start"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        best_df["End"] = best_df["End"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        display_best = best_df[["#", "Start", "End", "Portfolio", "Benchmark", "Difference"]]
        st.dataframe(display_best, use_container_width=True, hide_index=True)
    
    # Worst 3-Month Periods
    st.markdown("**Worst 3-Month Periods:**")
    if rolling_periods.get("worst") is not None and not rolling_periods["worst"].empty:
        worst_df = rolling_periods["worst"].copy()
        worst_df["#"] = range(1, len(worst_df) + 1)
        worst_df["Portfolio"] = worst_df["Portfolio"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Benchmark"] = worst_df["Benchmark"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Difference"] = worst_df["Difference"].apply(lambda x: f"{x*100:.2f}%")
        worst_df["Start"] = worst_df["Start"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        worst_df["End"] = worst_df["End"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x))
        display_worst = worst_df[["#", "Start", "End", "Portfolio", "Benchmark", "Difference"]]
        st.dataframe(display_worst, use_container_width=True, hide_index=True)


def _render_periodic_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.2: Periodic Analysis."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Graph 2.2.1: Annual Returns (EOY)
    st.subheader("Annual Returns (EOY)")
    yearly_data = get_yearly_returns_data(portfolio_returns, benchmark_returns)
    if yearly_data.get("yearly") is not None and not yearly_data["yearly"].empty:
        fig = plot_yearly_returns({"yearly": yearly_data["yearly"]})
        st.plotly_chart(fig, use_container_width=True, key="periodic_yearly")
        
        # Automatic interpretation
        interpretation = _interpret_yearly_returns(yearly_data["yearly"])
        st.info(interpretation)
    
    # Heatmap 2.2.2: Monthly Returns Calendar
    st.markdown("---")
    st.subheader("Monthly Returns Calendar (%)")
    heatmap_data = get_monthly_heatmap_data(portfolio_returns)
    if heatmap_data.get("heatmap") is not None and not heatmap_data["heatmap"].empty:
        fig = plot_monthly_heatmap({"heatmap": heatmap_data["heatmap"]})
        st.plotly_chart(fig, use_container_width=True, key="periodic_monthly")
        
        # Facts for caption
        facts = _interpret_monthly_heatmap(heatmap_data["heatmap"])
        if facts:
            st.caption(facts)
    
    # Heatmap 2.2.3: Monthly Active Returns
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.markdown("---")
        st.subheader("Monthly Active Returns (%) - Portfolio vs Benchmark")
        active_heatmap_data = get_monthly_active_returns_data(portfolio_returns, benchmark_returns)
        if active_heatmap_data.get("heatmap") is not None and not active_heatmap_data["heatmap"].empty:
            fig = plot_monthly_heatmap({"heatmap": active_heatmap_data["heatmap"]})
            st.plotly_chart(fig, use_container_width=True, key="periodic_monthly_active")
            
            # Automatic interpretation
            interpretation = _interpret_monthly_active_returns(active_heatmap_data["heatmap"])
            st.info(interpretation)
    
    # Charts 2.2.4: Seasonal Analysis
    st.markdown("---")
    st.subheader("Seasonal Analysis")
    seasonal_data = get_seasonal_analysis_data(portfolio_returns, benchmark_returns)
    
    if seasonal_data.get("day_of_week") is not None and not seasonal_data["day_of_week"].empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = plot_seasonal_bar(seasonal_data["day_of_week"], "Avg Return by Day of Week (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_day")
            
            # Automatic interpretation
            interpretation = _interpret_seasonal_pattern(seasonal_data["day_of_week"], "day_of_week")
            st.info(interpretation)
        
        with col2:
            fig = plot_seasonal_bar(seasonal_data["month"], "Avg Return by Month (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_month")
            
            # Automatic interpretation
            interpretation = _interpret_seasonal_pattern(seasonal_data["month"], "month")
            st.info(interpretation)
        
        with col3:
            fig = plot_seasonal_bar(seasonal_data["quarter"], "Avg Return by Quarter (%)")
            st.plotly_chart(fig, use_container_width=True, key="seasonal_quarter")
            
            # Automatic interpretation
            interpretation = _interpret_seasonal_pattern(seasonal_data["quarter"], "quarter")
            st.info(interpretation)


def _render_distribution_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 2.3: Distribution."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Charts 2.3.1: Return Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Daily Returns")
        dist_data_daily = get_return_distribution_data(portfolio_returns, bins=50)
        if dist_data_daily:
            fig = plot_return_distribution(dist_data_daily, bar_color="blue")
            st.plotly_chart(fig, use_container_width=True, key="distribution_daily")
            
            # Automatic interpretation
            if dist_data_daily.get("mean") is not None:
                mean = dist_data_daily.get("mean", 0)
                std = dist_data_daily.get("std", 0)
                skew = dist_data_daily.get("skewness", 0)
                kurt = dist_data_daily.get("kurtosis", 0)
                interpretation = _interpret_distribution(mean, std, skew, kurt, "daily")
                st.info(interpretation)
    
    with col2:
        st.subheader("Distribution of Monthly Returns")
        monthly_returns = portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        if not monthly_returns.empty:
            dist_data_monthly = get_return_distribution_data(monthly_returns, bins=30)
            if dist_data_monthly:
                fig = plot_return_distribution(dist_data_monthly, bar_color="blue")
                st.plotly_chart(fig, use_container_width=True, key="distribution_monthly")
                
                # Automatic interpretation
                if dist_data_monthly.get("mean") is not None:
                    mean = dist_data_monthly.get("mean", 0)
                    std = dist_data_monthly.get("std", 0)
                    skew = dist_data_monthly.get("skewness", 0)
                    kurt = dist_data_monthly.get("kurtosis", 0)
                    interpretation = _interpret_distribution(mean, std, skew, kurt, "monthly")
                    st.info(interpretation)
    
    # Chart 2.3.2: Q-Q Plot
    st.markdown("---")
    st.subheader("Q-Q Plot")
    qq_data = get_qq_plot_data(portfolio_returns)
    if qq_data:
        fig = plot_qq_plot(qq_data)
        st.plotly_chart(fig, use_container_width=True, key="distribution_qq")
        
        # Automatic interpretation of QQ plot
        theoretical = qq_data.get("theoretical", np.array([]))
        sample = qq_data.get("sample", np.array([]))
        if len(theoretical) > 0 and len(sample) > 0:
            # Calculate deviations from the line
            deviations = sample - theoretical
            mean_dev = np.mean(deviations)
            std_dev = np.std(deviations)
            
            # Calculate tail behavior
            lower_tail = np.mean(deviations[:len(deviations)//10])  # Bottom 10%
            upper_tail = np.mean(deviations[-len(deviations)//10:])  # Top 10%
            
            # Interpretation
            interpretation = _interpret_qq_plot(deviations, lower_tail, upper_tail, mean_dev, std_dev)
            st.info(interpretation)
    
    # Chart 2.3.3: Return Quantiles Box Plots
    st.markdown("---")
    st.subheader("Return Quantiles Box Plots")
    fig = plot_return_quantiles_box(portfolio_returns, benchmark_returns)
    st.plotly_chart(fig, use_container_width=True, key="distribution_box")
    
    # Automatic interpretation
    interpretation = _interpret_quantiles_box(portfolio_returns, benchmark_returns)
    st.info(interpretation)
    
    # Table 2.3.4: Win Rate Statistics (Comprehensive)
    st.markdown("---")
    st.subheader("Win Rate Statistics - Comprehensive")
    win_rate_data = get_win_rate_statistics_data(portfolio_returns, benchmark_returns)
    
    if win_rate_data.get("stats"):
        stats = win_rate_data["stats"]
        port_stats = stats.get("portfolio", {})
        bench_stats = stats.get("benchmark", {})
        
        # Build benchmark and difference columns
        bench_values = [
            f"{bench_stats.get('win_rate_daily', 0)*100:.1f}%" if bench_stats.get('win_rate_daily') is not None else "N/A",
            f"{bench_stats.get('win_rate_weekly', 0)*100:.1f}%" if bench_stats.get('win_rate_weekly') is not None else "N/A",
            f"{bench_stats.get('win_rate_monthly', 0)*100:.1f}%" if bench_stats.get('win_rate_monthly') is not None else "N/A",
            f"{bench_stats.get('win_rate_quarterly', 0)*100:.1f}%" if bench_stats.get('win_rate_quarterly') is not None else "N/A",
            f"{bench_stats.get('win_rate_yearly', 0)*100:.1f}%" if bench_stats.get('win_rate_yearly') is not None else "N/A",
            f"{bench_stats.get('avg_win_daily', 0)*100:.2f}%" if bench_stats.get('avg_win_daily') is not None else "N/A",
            f"{bench_stats.get('avg_loss_daily', 0)*100:.2f}%" if bench_stats.get('avg_loss_daily') is not None else "N/A",
            f"{bench_stats.get('avg_win_monthly', 0)*100:.2f}%" if bench_stats.get('avg_win_monthly') is not None else "N/A",
            f"{bench_stats.get('avg_loss_monthly', 0)*100:.2f}%" if bench_stats.get('avg_loss_monthly') is not None else "N/A",
            f"{bench_stats.get('best_daily', 0)*100:.2f}%" if bench_stats.get('best_daily') is not None else "N/A",
            f"{bench_stats.get('worst_daily', 0)*100:.2f}%" if bench_stats.get('worst_daily') is not None else "N/A",
            f"{bench_stats.get('best_monthly', 0)*100:.2f}%" if bench_stats.get('best_monthly') is not None else "N/A",
            f"{bench_stats.get('worst_monthly', 0)*100:.2f}%" if bench_stats.get('worst_monthly') is not None else "N/A",
        ]
        
        # Calculate differences
        diff_values = []
        port_vals = [
            port_stats.get('win_days_pct', 0),
            port_stats.get('win_weeks_pct', 0),
            port_stats.get('win_months_pct', 0),
            port_stats.get('win_quarters_pct', 0),
            port_stats.get('win_years_pct', 0),
            port_stats.get('avg_up_day', 0),
            port_stats.get('avg_down_day', 0),
            port_stats.get('avg_up_month', 0),
            port_stats.get('avg_down_month', 0),
            port_stats.get('best_day', 0),
            port_stats.get('worst_day', 0),
            port_stats.get('best_month', 0),
            port_stats.get('worst_month', 0),
        ]
        bench_vals = [
            bench_stats.get('win_rate_daily', 0) * 100 if bench_stats.get('win_rate_daily') is not None else None,
            bench_stats.get('win_rate_weekly', 0) * 100 if bench_stats.get('win_rate_weekly') is not None else None,
            bench_stats.get('win_rate_monthly', 0) * 100 if bench_stats.get('win_rate_monthly') is not None else None,
            bench_stats.get('win_rate_quarterly', 0) * 100 if bench_stats.get('win_rate_quarterly') is not None else None,
            bench_stats.get('win_rate_yearly', 0) * 100 if bench_stats.get('win_rate_yearly') is not None else None,
            bench_stats.get('avg_win_daily', 0) * 100 if bench_stats.get('avg_win_daily') is not None else None,
            bench_stats.get('avg_loss_daily', 0) * 100 if bench_stats.get('avg_loss_daily') is not None else None,
            bench_stats.get('avg_win_monthly', 0) * 100 if bench_stats.get('avg_win_monthly') is not None else None,
            bench_stats.get('avg_loss_monthly', 0) * 100 if bench_stats.get('avg_loss_monthly') is not None else None,
            bench_stats.get('best_daily', 0) * 100 if bench_stats.get('best_daily') is not None else None,
            bench_stats.get('worst_daily', 0) * 100 if bench_stats.get('worst_daily') is not None else None,
            bench_stats.get('best_monthly', 0) * 100 if bench_stats.get('best_monthly') is not None else None,
            bench_stats.get('worst_monthly', 0) * 100 if bench_stats.get('worst_monthly') is not None else None,
        ]
        
        for p, b in zip(port_vals, bench_vals):
            if b is not None:
                diff = p - b
                diff_values.append(f"{diff:+.2f}%")
            else:
                diff_values.append("N/A")
        
        win_rate_df = pd.DataFrame({
            "Timeframe": ["Win Days %", "Win Weeks %", "Win Months %", "Win Quarters %", "Win Years %",
                          "Avg Up Day", "Avg Down Day", "Avg Up Month", "Avg Down Month",
                          "Best Day", "Worst Day", "Best Month", "Worst Month"],
            "Portfolio": [
                f"{port_stats.get('win_days_pct', 0):.1f}%",
                f"{port_stats.get('win_weeks_pct', 0):.1f}%",
                f"{port_stats.get('win_months_pct', 0):.1f}%",
                f"{port_stats.get('win_quarters_pct', 0):.1f}%",
                f"{port_stats.get('win_years_pct', 0):.1f}%",
                f"{port_stats.get('avg_up_day', 0):.2f}%",
                f"{port_stats.get('avg_down_day', 0):.2f}%",
                f"{port_stats.get('avg_up_month', 0):.2f}%",
                f"{port_stats.get('avg_down_month', 0):.2f}%",
                f"{port_stats.get('best_day', 0):.2f}%",
                f"{port_stats.get('worst_day', 0):.2f}%",
                f"{port_stats.get('best_month', 0):.2f}%",
                f"{port_stats.get('worst_month', 0):.2f}%",
            ],
            "Benchmark": bench_values,
            "Difference": diff_values,
        })
        # Calculate required height (13 rows + header)
        row_height = 35  # Approximate height per row
        header_height = 40
        total_height = 13 * row_height + header_height
        st.dataframe(win_rate_df, use_container_width=True, hide_index=True, height=total_height)
        
        # Automatic interpretation
        interpretation = _interpret_win_rate_stats(port_stats, bench_stats)
        st.info(interpretation)
        
        # 12-Month Rolling Win Rate Chart
        rolling_win_rate = win_rate_data.get("rolling", pd.Series())
        if not rolling_win_rate.empty:
            bench_rolling = None
            if benchmark_returns is not None:
                # Calculate benchmark rolling win rate
                bench_win_rate_data = get_win_rate_statistics_data(benchmark_returns)
                bench_rolling = bench_win_rate_data.get("rolling", pd.Series())
            
            fig = plot_rolling_win_rate(rolling_win_rate, bench_rolling)
            st.plotly_chart(fig, use_container_width=True, key="distribution_rolling_win_rate")
            
            # Automatic interpretation
            interpretation = _interpret_rolling_win_rate(rolling_win_rate, bench_rolling)
            st.info(interpretation)
    
    # Section 2.3.5: Outlier Analysis
    st.markdown("---")
    st.subheader("Outlier Analysis - Tail Events")
    outlier_data = get_outlier_analysis_data(portfolio_returns, outlier_threshold=2.0)
    
    if outlier_data.get("stats"):
        stats = outlier_data["stats"]
        
        # Display values as metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Outlier Win Ratio", f"{stats.get('outlier_win_ratio', 0):.2f}")
            st.caption("Avg outlier win / Avg normal win")
        with col2:
            st.metric("Outlier Loss Ratio", f"{stats.get('outlier_loss_ratio', 0):.2f}")
            st.caption("Avg outlier loss / Avg normal loss")
        with col3:
            outlier_count = stats.get('outlier_count', 0)
            total_count = stats.get('total_count', 1)
            outlier_pct = (outlier_count / total_count * 100) if total_count > 0 else 0
            st.metric("Outlier Count", f"{outlier_count} ({outlier_pct:.1f}%)")
            st.caption("Beyond 2 standard deviations")
        
        # Interpretation
        win_ratio = stats.get('outlier_win_ratio', 0)
        loss_ratio = stats.get('outlier_loss_ratio', 0)
        interpretation_text = f"Big wins are {win_ratio:.2f}x larger than typical wins. Big losses are {loss_ratio:.2f}x larger than typical losses."
        st.info(interpretation_text)
        
        # Outlier Scatter Plot
        fig = plot_outlier_scatter(portfolio_returns, outlier_data)
        st.plotly_chart(fig, use_container_width=True, key="distribution_outlier_scatter")
    
    # Table 2.3.6: Statistical Tests
    st.markdown("---")
    st.subheader("Statistical Tests - Distribution Analysis")
    stats_tests = get_statistical_tests_data(portfolio_returns)
    
    if stats_tests:
        shapiro = stats_tests.get("shapiro_wilk", {})
        jb = stats_tests.get("jarque_bera", {})
        skewness = stats_tests.get("skewness", 0)
        kurtosis = stats_tests.get("kurtosis", 0)
        sample_size = stats_tests.get("sample_size", 0)
        
        # Format p-values more accurately
        def format_pvalue(p):
            if p < 0.0001:
                return f"{p:.2e}"  # Scientific notation
            else:
                return f"{p:.4f}"
        
        tests_df = pd.DataFrame({
            "Test": ["Shapiro-Wilk", "Jarque-Bera", "Skewness", "Kurtosis (Excess)"],
            "Statistic": [
                f"{shapiro.get('statistic', 0):.4f}",
                f"{jb.get('statistic', 0):.4f}",
                f"{skewness:.3f}",
                f"{kurtosis:+.3f}",
            ],
            "p-value": [
                format_pvalue(shapiro.get('pvalue', 1)),
                format_pvalue(jb.get('pvalue', 1)),
                "-",
                "-",
            ],
        })
        st.dataframe(tests_df, use_container_width=True, hide_index=True)
        
        # Show sample size info
        shapiro_sample = shapiro.get('sample_size', sample_size)
        if sample_size > 1000:
            st.caption(f"Sample size: {sample_size:,} observations" + 
                      (f" (Shapiro-Wilk used random sample of {shapiro_sample:,})" if shapiro_sample < sample_size else ""))
        
        # Interpretation
        shapiro_p = shapiro.get("pvalue", 1.0)
        jb_p = jb.get("pvalue", 1.0)
        is_normal = shapiro_p >= 0.05 and jb_p >= 0.05
        
        skew_interpretation = "Slight negative skew" if skewness < -0.1 else "Slight positive skew" if skewness > 0.1 else "Symmetric"
        kurtosis_interpretation = "Leptokurtic (fat tails)" if kurtosis > 0.5 else "Platykurtic (thin tails)" if kurtosis < -0.5 else "Normal tails"
        
        # Add note about large sample sensitivity
        interpretation_text = f"""
**Interpretation:**  
**Shapiro-Wilk:** Distribution is {'NOT' if shapiro_p < 0.05 else ''} normal (p {'<' if shapiro_p < 0.05 else '≥'} 0.05)  
**Jarque-Bera:** Distribution is {'NOT' if jb_p < 0.05 else ''} normal (p {'<' if jb_p < 0.05 else '≥'} 0.05)  
**Skewness:** {skew_interpretation} ({skewness:.3f})  
**Kurtosis:** {kurtosis_interpretation} ({kurtosis:+.3f})
"""
        
        if sample_size > 1000 and not is_normal:
            interpretation_text += """
**Note:** Large samples (>1000) make normality tests very sensitive - they detect even minor deviations.  
Financial returns typically show fat tails and slight skewness, so rejection of normality is expected and not necessarily a concern.
"""
        
        st.info(interpretation_text)


def _render_risk_tab(risk, ratios, market, portfolio_returns, benchmark_returns, portfolio_values, risk_free_rate, start_date, end_date):
    """Render Risk tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Key Metrics",
        "Drawdown Analysis",
        "VaR & CVaR",
        "Rolling Risk Metrics"
    ])
    
    # Calculate benchmark values if needed
    benchmark_values = None
    if benchmark_returns is not None and not benchmark_returns.empty and portfolio_values is not None:
        aligned_bench = benchmark_returns.reindex(portfolio_values.index, method="ffill").fillna(0)
        initial_value = float(portfolio_values.iloc[0])
        benchmark_values = (1 + aligned_bench).cumprod() * initial_value
    
    with sub_tab1:
        _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate)
    
    with sub_tab2:
        _render_drawdown_analysis(risk, portfolio_values, benchmark_values, portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_var_analysis(portfolio_returns, benchmark_returns)
    
    with sub_tab4:
        _render_rolling_risk(portfolio_returns, benchmark_returns, risk_free_rate)


def _render_risk_key_metrics(risk, ratios, market, benchmark_returns, portfolio_returns, risk_free_rate):
    """Sub-tab 3.1: Key Risk Metrics."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # Recalculate portfolio risk metrics from returns to ensure correctness
    from core.analytics_engine.risk_metrics import (
        calculate_volatility,
        calculate_max_drawdown,
        calculate_var,
        calculate_cvar,
    )
    
    portfolio_vol = calculate_volatility(portfolio_returns)
    portfolio_max_dd_tuple = calculate_max_drawdown(portfolio_returns)
    portfolio_max_dd = portfolio_max_dd_tuple[0] if isinstance(portfolio_max_dd_tuple, tuple) else portfolio_max_dd_tuple
    
    # Calculate benchmark metrics for comparison
    benchmark_risk_metrics = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            from core.analytics_engine.ratios import (
                calculate_sortino_ratio,
                calculate_calmar_ratio,
            )
            
            bench_vol = calculate_volatility(benchmark_returns)
            bench_vol_annual = bench_vol.get("annual", 0.0) if isinstance(bench_vol, dict) else bench_vol
            bench_max_dd = calculate_max_drawdown(benchmark_returns)
            bench_max_dd_val = bench_max_dd[0] if isinstance(bench_max_dd, tuple) else bench_max_dd
            
            benchmark_risk_metrics = {
                "volatility": float(bench_vol_annual),
                "max_drawdown": float(bench_max_dd_val),
                "sortino_ratio": float(calculate_sortino_ratio(benchmark_returns, risk_free_rate) or 0),
                "calmar_ratio": float(calculate_calmar_ratio(benchmark_returns) or 0),
                "var_95": float(calculate_var(benchmark_returns, confidence_level=0.95, method="historical") or 0),
                "cvar_95": float(calculate_cvar(benchmark_returns, confidence_level=0.95) or 0),
                "up_capture": 1.0,
                "down_capture": 1.0,
            }
        except Exception as e:
            logger.warning(f"Error calculating benchmark risk metrics: {e}")
    
    # Section 3.1.1: Risk Metric Cards (8 cards in 2 rows)
    st.subheader("Risk Metrics")
    
    # Use recalculated values
    vol_annual = portfolio_vol.get("annual", 0.0) if isinstance(portfolio_vol, dict) else portfolio_vol
    
    # Row 1: Volatility, Max Drawdown, Sortino Ratio, Calmar Ratio
    risk_metrics_row1 = [
        {
            "label": "Volatility",
            "portfolio_value": float(vol_annual),
            "benchmark_value": benchmark_risk_metrics.get("volatility"),
            "format": "percent",
            "higher_is_better": False,
            "help_text": "Annualized standard deviation of returns.",
        },
        {
            "label": "Max Drawdown",
            "portfolio_value": float(portfolio_max_dd),
            "benchmark_value": benchmark_risk_metrics.get("max_drawdown"),
            "format": "percent",
            "higher_is_better": False,
            "help_text": "Largest peak-to-trough decline.",
        },
        {
            "label": "Sortino Ratio",
            "portfolio_value": ratios.get("sortino_ratio", 0),
            "benchmark_value": benchmark_risk_metrics.get("sortino_ratio"),
            "format": "ratio",
            "higher_is_better": True,
            "help_text": "Like Sharpe but only penalizes downside volatility.",
        },
        {
            "label": "Calmar Ratio",
            "portfolio_value": ratios.get("calmar_ratio", 0),
            "benchmark_value": benchmark_risk_metrics.get("calmar_ratio"),
            "format": "ratio",
            "higher_is_better": True,
            "help_text": "Annual return / Max Drawdown.",
        },
    ]
    render_metric_cards_row(risk_metrics_row1, columns_per_row=4)
    
    st.markdown("---")
    
    # Row 2: VaR (95%), CVaR (95%), Up Capture, Down Capture
    risk_metrics_row2 = [
        {
            "label": "VaR (95%)",
            "portfolio_value": risk.get("var_95", 0),
            "benchmark_value": benchmark_risk_metrics.get("var_95"),
            "format": "percent",
            "higher_is_better": False,  # Less negative is better
            "help_text": "Worst expected loss on 95% of days.",
        },
        {
            "label": "CVaR (95%)",
            "portfolio_value": risk.get("cvar_95", 0),
            "benchmark_value": benchmark_risk_metrics.get("cvar_95"),
            "format": "percent",
            "higher_is_better": False,
            "help_text": "Average loss on worst 5% of days. More conservative than VaR.",
        },
        {
            "label": "Up Capture",
            "portfolio_value": market.get("up_capture", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "percent",
            "higher_is_better": True,
            "help_text": "Portfolio return when benchmark is up.",
        },
        {
            "label": "Down Capture",
            "portfolio_value": market.get("down_capture", 0),
            "benchmark_value": 1.0 if benchmark_returns is not None else None,
            "format": "percent",
            "higher_is_better": False,  # Lower is better
            "help_text": "Portfolio return when benchmark is down.",
        },
    ]
    render_metric_cards_row(risk_metrics_row2, columns_per_row=4)
    
    # Section 3.1.2: Probabilistic Sharpe Ratio
    st.markdown("---")
    st.subheader("Probabilistic Sharpe Ratio")
    
    psr_95 = calculate_probabilistic_sharpe_ratio(portfolio_returns, risk_free_rate, benchmark_sharpe=1.0)
    psr_99 = calculate_probabilistic_sharpe_ratio(portfolio_returns, risk_free_rate, benchmark_sharpe=0.0)
    observed_sharpe = ratios.get("sharpe_ratio", 0)
    
    if psr_95 is not None:
        psr_95_pct = psr_95 * 100
        psr_99_pct = psr_99 * 100 if psr_99 is not None else 0
        
        # Display values as metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observed Sharpe Ratio", f"{observed_sharpe:.2f}")
        with col2:
            st.metric("PSR (95% confidence)", f"{psr_95_pct:.1f}%")
        with col3:
            st.metric("PSR (99% confidence)", f"{psr_99_pct:.1f}%")
        
        # Interpretation based on result
        if psr_95 > 0.85:
            st.success(
                f"{psr_95_pct:.1f}% probability that true Sharpe > 1.0. "
                f"High statistical significance. Sharpe is likely NOT due to luck."
            )
        elif psr_95 > 0.50:
            st.info(
                f"{psr_95_pct:.1f}% probability that true Sharpe > 1.0. "
                f"Moderate statistical significance. Sharpe may be influenced by luck."
            )
        else:
            st.warning(
                f"{psr_95_pct:.1f}% probability that true Sharpe > 1.0. "
                f"Low statistical significance. Sharpe may be influenced by luck."
            )
    else:
        st.warning("Insufficient data for Probabilistic Sharpe Ratio calculation")
    
    # Section 3.1.3: Smart Sharpe & Sortino
    st.markdown("---")
    st.subheader("Smart Sharpe & Sortino")
    
    smart_sharpe = calculate_smart_sharpe(portfolio_returns, risk_free_rate)
    smart_sortino = calculate_smart_sortino(portfolio_returns, risk_free_rate)
    observed_sortino = ratios.get("sortino_ratio", 0)
    
    if smart_sharpe is not None and smart_sortino is not None:
        sharpe_adjustment = observed_sharpe - smart_sharpe
        sortino_adjustment = observed_sortino - smart_sortino
        sortino_conservative = observed_sortino / np.sqrt(2)
        
        smart_ratios_df = pd.DataFrame({
            "Ratio": [
                "Sharpe Ratio",
                "Smart Sharpe (Autocorrelation adj.)",
                "Sortino Ratio",
                "Smart Sortino",
                "Sortino/√2 (Conservative Est.)",
        ],
        "Value": [
                f"{observed_sharpe:.2f}",
                f"{smart_sharpe:.2f}",
                f"{observed_sortino:.2f}",
                f"{smart_sortino:.2f}",
                f"{sortino_conservative:.2f}",
            ],
            "Adjustment": [
                "—",
                f"{sharpe_adjustment:+.2f}",
                "—",
                f"{sortino_adjustment:+.2f}",
                "—",
            ],
        })
        
        st.dataframe(smart_ratios_df, use_container_width=True, hide_index=True, height=230)
        st.caption("Note: Smart ratios adjust for autocorrelation and non-normality")
        
        # Automatic interpretation
        if observed_sharpe is not None and smart_sharpe is not None:
            interpretation = _interpret_smart_ratios(observed_sharpe, smart_sharpe, observed_sortino, smart_sortino)
            st.info(interpretation)
    else:
        st.warning("Insufficient data for Smart Sharpe/Sortino calculation")
    
    # Chart 3.1.4: Capture Ratio Visualization
    st.markdown("---")
    st.subheader("Capture Ratio Visualization")
    
    up_capture = market.get("up_capture")
    down_capture = market.get("down_capture")
    
    if up_capture is not None and down_capture is not None:
        capture_data = get_capture_ratio_data(up_capture, down_capture)
        if capture_data:
            fig = plot_capture_ratio(capture_data)
            st.plotly_chart(fig, use_container_width=True, key="risk_capture_ratio")
            
            capture_ratio = capture_data.get("capture_ratio", 0)
            
            # Display values as metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Capture Ratio", f"{capture_ratio:.2f}")
            with col2:
                st.metric("Up Capture", f"{up_capture*100:.0f}%")
            with col3:
                st.metric("Down Capture", f"{down_capture*100:.0f}%")
            
            # Interpretation based on capture_ratio
            if capture_ratio > 1.2:
                st.success(
                    f"Strong asymmetry ({capture_ratio:.2f}). "
                    f"Portfolio captures {up_capture*100:.0f}% of market upside and only {down_capture*100:.0f}% of market downside. "
                    f"Favorable risk/reward profile."
                )
            elif capture_ratio > 1.1:
                st.success(
                    f"Moderate asymmetry ({capture_ratio:.2f}). "
                    f"Portfolio captures {up_capture*100:.0f}% of market upside and {down_capture*100:.0f}% of market downside. "
                    f"Favorable risk/reward profile."
                )
            elif capture_ratio < 0.9:
                st.warning(
                    f"Unfavorable asymmetry ({capture_ratio:.2f}). "
                    f"Portfolio captures {up_capture*100:.0f}% of market upside and {down_capture*100:.0f}% of market downside. "
                    f"Portfolio may be underperforming in up markets."
                )
            else:
                st.info(
                    f"Neutral asymmetry ({capture_ratio:.2f}). "
                    f"Portfolio captures {up_capture*100:.0f}% of market upside and {down_capture*100:.0f}% of market downside. "
                    f"Neutral risk/reward profile."
                )
    else:
        st.info("Capture ratios require benchmark comparison")
    
    # Chart 3.1.5: Risk/Return Scatter
    st.markdown("---")
    st.subheader("Risk/Return Scatter")
    
    scatter_data = get_risk_return_scatter_data(
        portfolio_returns, benchmark_returns, risk_free_rate
    )
    if scatter_data:
        fig = plot_risk_return_scatter(scatter_data)
        st.plotly_chart(fig, use_container_width=True, key="risk_return_scatter")
        
        # Automatic interpretation
        interpretation = _interpret_risk_return_scatter(scatter_data)
        st.info(interpretation)
    else:
        st.warning("Unable to generate risk/return scatter plot")
    
    # Section 3.1.6: Information Ratio Breakdown
    st.markdown("---")
    st.subheader("Information Ratio Breakdown")
    
    # Recalculate IR from scratch to ensure correctness
    if benchmark_returns is not None and not benchmark_returns.empty:
        from core.analytics_engine.performance import calculate_annualized_return
        from core.analytics_engine.ratios import calculate_information_ratio
        from core.analytics_engine.market_metrics import calculate_tracking_error
        
        try:
            # Calculate components
            port_return = calculate_annualized_return(portfolio_returns)
            bench_return = calculate_annualized_return(benchmark_returns)
            active_return = port_return - bench_return
            
            # Recalculate tracking error and IR
            tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns) or 0
            info_ratio = calculate_information_ratio(portfolio_returns, benchmark_returns) or 0
            
            # Display values as metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Return", f"{active_return*100:+.2f}%")
            with col2:
                st.metric("Tracking Error", f"{tracking_error*100:.2f}%")
            with col3:
                st.metric("Information Ratio", f"{info_ratio:.2f}")
            
            # Stacked bar visualization
            import plotly.graph_objects as go
            from streamlit_app.utils.chart_config import COLORS, get_chart_layout
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=["Return Breakdown"],
                x=[bench_return * 100],
                orientation="h",
                name="Benchmark Return",
                marker=dict(color=COLORS["secondary"]),
                text=[f"{bench_return*100:.1f}%"],
                textposition="inside",
            ))
            fig.add_trace(go.Bar(
                y=["Return Breakdown"],
                x=[active_return * 100],
                orientation="h",
                name="Active Return",
                marker=dict(color=COLORS["primary"]),  # Purple - always purple
                text=[f"{active_return*100:+.1f}%"],
                textposition="inside",
            ))
            
            layout = get_chart_layout(
                title="Return Breakdown",
                xaxis=dict(title="Return (%)", tickformat=",.1f"),
                yaxis=dict(title=""),
                barmode="stack",
                height=200,
            )
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, key="info_ratio_breakdown")
            
            # Interpretation based on IR
            active_return_pct = abs(active_return)/abs(port_return)*100 if port_return != 0 else 0
            if info_ratio > 1.0:
                st.success(
                    f"High Information Ratio ({info_ratio:.2f}) indicates consistent alpha generation. "
                    f"Active return represents {active_return_pct:.0f}% of total return."
                )
            elif info_ratio > 0.75:
                st.info(
                    f"Moderate Information Ratio ({info_ratio:.2f}) indicates some alpha generation. "
                    f"Active return represents {active_return_pct:.0f}% of total return."
                )
            else:
                st.warning(
                    f"Low Information Ratio ({info_ratio:.2f}) indicates inconsistent alpha generation. "
                    f"Active return represents {active_return_pct:.0f}% of total return."
                )
        except Exception as e:
            logger.warning(f"Error calculating Information Ratio: {e}")
            st.info("Unable to calculate Information Ratio")
    else:
        st.info("Information Ratio requires benchmark comparison")
    
    # Section 3.1.7: Kelly Criterion & Risk of Ruin
    st.markdown("---")
    st.subheader("Kelly Criterion & Risk of Ruin")
    
    kelly_data = calculate_kelly_criterion(portfolio_returns)
    risk_of_ruin = calculate_risk_of_ruin(portfolio_returns)
    
    if kelly_data:
        kelly_full = kelly_data.get("kelly_full", 0) * 100
        kelly_half = kelly_data.get("kelly_half", 0) * 100
        kelly_quarter = kelly_data.get("kelly_quarter", 0) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Kelly Criterion - Position Sizing**")
            
            # Display values as metrics
            st.metric("Full Kelly", f"{kelly_full:.1f}%")
            st.caption("Optimal leverage for max growth")
            st.metric("Half-Kelly", f"{kelly_half:.1f}%")
            st.caption("Conservative, reduces volatility")
            st.metric("Quarter-Kelly", f"{kelly_quarter:.1f}%")
            st.caption("Very conservative")
            
            # Automatic interpretation
            interpretation = _interpret_kelly_criterion(kelly_data)
            st.info(interpretation)
        
        with col2:
            if risk_of_ruin:
                st.markdown("**Risk of Ruin Analysis**")
                ruin_df = pd.DataFrame({
                    "Drawdown": ["-10%", "-20%", "-25%", "-30%", "-50%"],
                    "Probability": [
                        f"{risk_of_ruin.get('ruin_10pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_20pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_25pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_30pct', 0)*100:.1f}%",
                        f"{risk_of_ruin.get('ruin_50pct', 0)*100:.1f}%",
                    ],
                    "Est. Recovery": ["~3 mo", "~8 mo", "~12 mo", "~18 mo", "~5 yr"],
                })
                st.dataframe(ruin_df, use_container_width=True, hide_index=True, height=210)
                st.caption("Note: Recovery times are approximate estimates")
    else:
        st.warning("Insufficient data for Kelly Criterion / Risk of Ruin calculation")
    
    # Table 3.1.8: Complete Risk Metrics Table (28 metrics as per spec)
    st.markdown("---")
    st.subheader("Complete Risk Metrics Table")
    
    # Recalculate all metrics from portfolio returns
    from core.analytics_engine.risk_metrics import (
        calculate_volatility as calc_vol,
        calculate_max_drawdown as calc_dd,
        calculate_var,
        calculate_cvar,
        calculate_downside_deviation,
    )
    
    # Portfolio metrics
    port_vol = calc_vol(portfolio_returns)
    port_dd_tuple = calc_dd(portfolio_returns)
    port_dd_val = port_dd_tuple[0] if isinstance(port_dd_tuple, tuple) else port_dd_tuple
    port_dd_date = port_dd_tuple[1] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 1 else None
    port_dd_trough = port_dd_tuple[2] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 2 else None
    port_dd_duration = port_dd_tuple[3] if isinstance(port_dd_tuple, tuple) and len(port_dd_tuple) > 3 else None
    
    # Calculate all VaR/CVaR metrics
    port_var_90 = calculate_var(portfolio_returns, 0.90, method="historical") or 0
    port_var_95 = calculate_var(portfolio_returns, 0.95, method="historical") or 0
    port_var_99 = calculate_var(portfolio_returns, 0.99, method="historical") or 0
    port_var_95_param = calculate_var(portfolio_returns, 0.95, method="parametric") or 0
    port_var_95_cf = calculate_var(portfolio_returns, 0.95, method="cornish_fisher") or 0
    
    port_cvar_90 = calculate_cvar(portfolio_returns, 0.90) or 0
    port_cvar_95 = calculate_cvar(portfolio_returns, 0.95) or 0
    port_cvar_99 = calculate_cvar(portfolio_returns, 0.99) or 0
    
    port_downside = calculate_downside_deviation(portfolio_returns) or 0
    
    # Calculate remaining metrics
    from core.analytics_engine.risk_metrics import (
        calculate_semi_deviation,
        calculate_skewness,
        calculate_kurtosis,
        calculate_top_drawdowns,
        calculate_current_drawdown,
        calculate_average_drawdown,
        calculate_ulcer_index,
        calculate_pain_index,
    )
    
    port_semi = calculate_semi_deviation(portfolio_returns) or 0
    port_skew = calculate_skewness(portfolio_returns) or 0
    port_kurt = calculate_kurtosis(portfolio_returns) or 0
    
    # Calculate drawdown metrics using top_drawdowns
    port_current_dd = calculate_current_drawdown(portfolio_returns) or 0
    port_avg_dd = calculate_average_drawdown(portfolio_returns) or 0
    port_ulcer = calculate_ulcer_index(portfolio_returns) or 0
    port_pain = calculate_pain_index(portfolio_returns) or 0
    
    # Calculate avg DD duration and recovery from top drawdowns
    top_dds = calculate_top_drawdowns(portfolio_returns, top_n=5)
    avg_dd_duration = 0
    avg_recovery_time = 0
    max_recovery_time = 0
    if top_dds:
        avg_dd_duration = sum(dd['duration_days'] for dd in top_dds) / len(top_dds)
        recoveries = [dd['recovery_days'] for dd in top_dds if dd['recovery_days']]
        if recoveries:
            avg_recovery_time = sum(recoveries) / len(recoveries)
            max_recovery_time = max(recoveries)
    
    # Benchmark metrics
    bench_metrics = {}
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            bench_vol = calc_vol(benchmark_returns)
            bench_dd_tuple = calc_dd(benchmark_returns)
            bench_dd_val = bench_dd_tuple[0] if isinstance(bench_dd_tuple, tuple) else bench_dd_tuple
            bench_dd_date = bench_dd_tuple[1] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 1 else None
            bench_dd_trough = bench_dd_tuple[2] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 2 else None
            bench_dd_duration = bench_dd_tuple[3] if isinstance(bench_dd_tuple, tuple) and len(bench_dd_tuple) > 3 else None
            
            # Calculate all benchmark metrics
            bench_current_dd = calculate_current_drawdown(benchmark_returns) or 0
            bench_avg_dd = calculate_average_drawdown(benchmark_returns) or 0
            bench_ulcer = calculate_ulcer_index(benchmark_returns) or 0
            bench_pain = calculate_pain_index(benchmark_returns) or 0
            bench_semi = calculate_semi_deviation(benchmark_returns) or 0
            bench_skew = calculate_skewness(benchmark_returns) or 0
            bench_kurt = calculate_kurtosis(benchmark_returns) or 0
            
            # Calculate benchmark drawdown durations
            bench_top_dds = calculate_top_drawdowns(benchmark_returns, top_n=5)
            bench_avg_dd_duration = 0
            bench_avg_recovery = 0
            bench_max_recovery = 0
            if bench_top_dds:
                bench_avg_dd_duration = sum(dd['duration_days'] for dd in bench_top_dds) / len(bench_top_dds)
                bench_recoveries = [dd['recovery_days'] for dd in bench_top_dds if dd['recovery_days']]
                if bench_recoveries:
                    bench_avg_recovery = sum(bench_recoveries) / len(bench_recoveries)
                    bench_max_recovery = max(bench_recoveries)
            
            bench_metrics = {
                'daily_vol': bench_vol.get('daily', 0),
                'weekly_vol': bench_vol.get('weekly', 0),
                'monthly_vol': bench_vol.get('monthly', 0),
                'annual_vol': bench_vol.get('annual', 0),
                'max_dd': bench_dd_val,
                'max_dd_date': bench_dd_date,
                'max_dd_trough': bench_dd_trough,
                'max_dd_duration': bench_dd_duration,
                'current_dd': bench_current_dd,
                'avg_dd': bench_avg_dd,
                'avg_dd_duration': bench_avg_dd_duration,
                'avg_recovery': bench_avg_recovery,
                'max_recovery': bench_max_recovery,
                'ulcer': bench_ulcer,
                'pain': bench_pain,
                'var_90': calculate_var(benchmark_returns, 0.90, method="historical") or 0,
                'var_95': calculate_var(benchmark_returns, 0.95, method="historical") or 0,
                'var_99': calculate_var(benchmark_returns, 0.99, method="historical") or 0,
                'var_95_param': calculate_var(benchmark_returns, 0.95, method="parametric") or 0,
                'var_95_cf': calculate_var(benchmark_returns, 0.95, method="cornish_fisher") or 0,
                'cvar_90': calculate_cvar(benchmark_returns, 0.90) or 0,
                'cvar_95': calculate_cvar(benchmark_returns, 0.95) or 0,
                'cvar_99': calculate_cvar(benchmark_returns, 0.99) or 0,
                'downside': calculate_downside_deviation(benchmark_returns) or 0,
                'semi': bench_semi,
                'skew': bench_skew,
                'kurt': bench_kurt,
            }
        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics for table: {e}")
    
    # Extract portfolio volatilities
    p_daily = port_vol.get('daily', 0) if isinstance(port_vol, dict) else 0
    p_weekly = port_vol.get('weekly', 0) if isinstance(port_vol, dict) else 0
    p_monthly = port_vol.get('monthly', 0) if isinstance(port_vol, dict) else 0
    p_annual = port_vol.get('annual', 0) if isinstance(port_vol, dict) else 0
    
    # Build table with all 28 metrics
    metrics_data = [
        ("Daily Volatility", f"{p_daily*100:.2f}%", 
         f"{bench_metrics.get('daily_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_daily - bench_metrics.get('daily_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Weekly Volatility", f"{p_weekly*100:.2f}%",
         f"{bench_metrics.get('weekly_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_weekly - bench_metrics.get('weekly_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Monthly Volatility", f"{p_monthly*100:.2f}%",
         f"{bench_metrics.get('monthly_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_monthly - bench_metrics.get('monthly_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Annual Volatility", f"{p_annual*100:.2f}%",
         f"{bench_metrics.get('annual_vol', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(p_annual - bench_metrics.get('annual_vol', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Max Drawdown", f"{port_dd_val*100:.2f}%",
         f"{bench_metrics.get('max_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_dd_val - bench_metrics.get('max_dd', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Max DD Peak Date", str(port_dd_date)[:10] if port_dd_date else "—",
         str(bench_metrics.get('max_dd_date'))[:10] if bench_metrics and bench_metrics.get('max_dd_date') else "—",
         "—"),
        ("Max DD Trough Date", str(port_dd_trough)[:10] if port_dd_trough else "—",
         str(bench_metrics.get('max_dd_trough'))[:10] if bench_metrics and bench_metrics.get('max_dd_trough') else "—",
         "—"),
        ("Max DD Duration (days)", str(port_dd_duration) if port_dd_duration else "—",
         str(bench_metrics.get('max_dd_duration')) if bench_metrics and bench_metrics.get('max_dd_duration') else "—",
         f"{int(port_dd_duration or 0) - int(bench_metrics.get('max_dd_duration') or 0):+d}" if bench_metrics and port_dd_duration and bench_metrics.get('max_dd_duration') else "—"),
        ("Current Drawdown", f"{port_current_dd*100:.2f}%",
         f"{bench_metrics.get('current_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_current_dd - bench_metrics.get('current_dd', 0))*100:+.2f}%" if bench_metrics and port_current_dd != 0 else "—"),
        ("Average Drawdown", f"{port_avg_dd*100:.2f}%",
         f"{bench_metrics.get('avg_dd', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_avg_dd - bench_metrics.get('avg_dd', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Avg DD Duration", f"{int(avg_dd_duration)} days" if avg_dd_duration else "—",
         f"{int(bench_metrics.get('avg_dd_duration', 0))} days" if bench_metrics and bench_metrics.get('avg_dd_duration') else "—",
         f"{int(avg_dd_duration or 0) - int(bench_metrics.get('avg_dd_duration') or 0):+d} days" if bench_metrics and avg_dd_duration and bench_metrics.get('avg_dd_duration') else "—"),
        ("Avg Recovery Time", f"{int(avg_recovery_time)} days" if avg_recovery_time else "—",
         f"{int(bench_metrics.get('avg_recovery', 0))} days" if bench_metrics and bench_metrics.get('avg_recovery') else "—",
         f"{int(avg_recovery_time or 0) - int(bench_metrics.get('avg_recovery') or 0):+d} days" if bench_metrics and avg_recovery_time and bench_metrics.get('avg_recovery') else "—"),
        ("Max Recovery Time", f"{int(max_recovery_time)} days" if max_recovery_time else "—",
         f"{int(bench_metrics.get('max_recovery', 0))} days" if bench_metrics and bench_metrics.get('max_recovery') else "—",
         f"{int(max_recovery_time or 0) - int(bench_metrics.get('max_recovery') or 0):+d} days" if bench_metrics and max_recovery_time and bench_metrics.get('max_recovery') else "—"),
        ("Ulcer Index", f"{port_ulcer*100:.2f}%",
         f"{bench_metrics.get('ulcer', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_ulcer - bench_metrics.get('ulcer', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Pain Index", f"{port_pain*100:.2f}%",
         f"{bench_metrics.get('pain', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_pain - bench_metrics.get('pain', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 90% (Historical)", f"{port_var_90*100:.2f}%",
         f"{bench_metrics.get('var_90', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_90 - bench_metrics.get('var_90', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Historical)", f"{port_var_95*100:.2f}%",
         f"{bench_metrics.get('var_95', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95 - bench_metrics.get('var_95', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 99% (Historical)", f"{port_var_99*100:.2f}%",
         f"{bench_metrics.get('var_99', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_99 - bench_metrics.get('var_99', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Parametric)", f"{port_var_95_param*100:.2f}%",
         f"{bench_metrics.get('var_95_param', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95_param - bench_metrics.get('var_95_param', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("VaR 95% (Cornish-Fisher)", f"{port_var_95_cf*100:.2f}%",
         f"{bench_metrics.get('var_95_cf', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_var_95_cf - bench_metrics.get('var_95_cf', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 90%", f"{port_cvar_90*100:.2f}%",
         f"{bench_metrics.get('cvar_90', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_90 - bench_metrics.get('cvar_90', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 95%", f"{port_cvar_95*100:.2f}%",
         f"{bench_metrics.get('cvar_95', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_95 - bench_metrics.get('cvar_95', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("CVaR 99%", f"{port_cvar_99*100:.2f}%",
         f"{bench_metrics.get('cvar_99', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_cvar_99 - bench_metrics.get('cvar_99', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Downside Deviation", f"{port_downside*100:.2f}%",
         f"{bench_metrics.get('downside', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_downside - bench_metrics.get('downside', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Semi-Deviation", f"{port_semi*100:.2f}%",
         f"{bench_metrics.get('semi', 0)*100:.2f}%" if bench_metrics else "—",
         f"{(port_semi - bench_metrics.get('semi', 0))*100:+.2f}%" if bench_metrics else "—"),
        ("Skewness", f"{port_skew:.3f}",
         f"{bench_metrics.get('skew', 0):.3f}" if bench_metrics else "—",
         f"{(port_skew - bench_metrics.get('skew', 0)):+.3f}" if bench_metrics else "—"),
        ("Kurtosis (Excess)", f"{port_kurt:+.3f}",
         f"{bench_metrics.get('kurt', 0):+.3f}" if bench_metrics else "—",
         f"{(port_kurt - bench_metrics.get('kurt', 0)):+.3f}" if bench_metrics else "—"),
    ]
    
    complete_risk_df = pd.DataFrame(metrics_data, columns=["Metric", "Portfolio", "Benchmark", "Difference"])
    
    # Calculate height for all rows (no scroll)
    row_height = 35
    header_height = 40
    total_height = len(complete_risk_df) * row_height + header_height
    
    st.dataframe(complete_risk_df, use_container_width=True, hide_index=True, height=total_height)
    
    # Facts for caption
    facts_parts = []
    facts_parts.append(f"**Key Insights:** Portfolio volatility: {p_annual*100:.1f}%")
    facts_parts.append(f"Max drawdown: {port_dd_val*100:.1f}%")
    if port_current_dd != 0:
        facts_parts.append(f"Current drawdown: {port_current_dd*100:.1f}%")
    st.caption(" | ".join(facts_parts))


def _render_drawdown_analysis(risk, portfolio_values, benchmark_values, portfolio_returns, benchmark_returns):
    """Sub-tab 3.2: Drawdown Analysis."""
    st.subheader("Drawdown Analysis")
    
    if portfolio_values is None or portfolio_values.empty:
        st.warning("No portfolio values data available for drawdown analysis")
        return
    
    # Import needed functions
    from core.analytics_engine.risk_metrics import calculate_top_drawdowns, calculate_drawdown_duration
    from core.analytics_engine.performance import calculate_annualized_return
    
    # Chart 3.2.1: Underwater Plot
    st.markdown("### Underwater Plot")
    underwater_data = get_underwater_plot_data(portfolio_values, benchmark_values)
    if underwater_data:
        fig = plot_underwater(underwater_data)
        st.plotly_chart(fig, use_container_width=True, key="drawdown_underwater")
        
        # Automatic interpretation
        portfolio_dd = underwater_data.get("underwater", pd.Series())
        if not portfolio_dd.empty:
            max_dd = portfolio_dd.min() / 100  # Convert from % to decimal
            current_dd = portfolio_dd.iloc[-1] / 100 if len(portfolio_dd) > 0 else 0
            interpretation = _interpret_drawdown_chart(max_dd, current_dd, portfolio_dd)
            st.info(interpretation)
    
    st.markdown("---")
    
    # Chart 3.2.2: Drawdown Periods
    st.markdown("### Drawdown Periods")
    drawdown_periods_data = get_drawdown_periods_data(portfolio_values, threshold=0.05)
    if drawdown_periods_data:
        fig = plot_drawdown_periods(drawdown_periods_data)
        st.plotly_chart(fig, use_container_width=True, key="drawdown_periods")
        
        # Automatic interpretation
        periods = drawdown_periods_data.get("periods", [])
        if periods:
            avg_depth = sum(p.get("depth", 0) for p in periods) / len(periods)
            avg_duration = sum(p.get("duration_days", 0) for p in periods) / len(periods)
            interpretation = _interpret_drawdown_periods(periods, avg_depth, avg_duration)
            st.info(interpretation)
    
    st.markdown("---")
    
    # Chart 3.2.3: Drawdown Recovery Visualization
    st.markdown("### Drawdown Recovery Timeline")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        recovery_data = get_drawdown_recovery_data(portfolio_returns, top_n=3)
        
        if recovery_data:
            for dd in recovery_data:
                # Create expander for each drawdown
                with st.expander(
                    f"Drawdown #{dd['number']}: {dd['start_date']} to {dd['recovery_date'] if dd['recovery_date'] else 'Ongoing'}",
                    expanded=(dd['number'] == 1)  # Expand first one
                ):
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Depth", f"{dd['depth']*100:.2f}%")
                    with col2:
                        st.metric("Duration", f"{dd['duration_days']} days")
                    with col3:
                        if dd['recovery_days']:
                            st.metric("Recovery Time", f"{dd['recovery_days']} days")
                        else:
                            st.metric("Recovery Time", "Not recovered")
                    with col4:
                        total_days = dd['duration_days'] + (dd['recovery_days'] if dd['recovery_days'] else 0)
                        st.metric("Total Duration", f"{total_days} days")
                    
                    # Show timeline chart
                    fig = plot_drawdown_recovery(dd)
                    st.plotly_chart(fig, use_container_width=True, key=f"recovery_{dd['number']}")
                    
                    # Show value information
                    if dd['peak_value'] and dd['trough_value']:
                        st.caption(
                            f"**Values:** Peak: {dd['peak_value']:.2f} → "
                            f"Trough: {dd['trough_value']:.2f} ({dd['depth']*100:.2f}%)"
                            + (f" → Recovery: {dd['recovery_value']:.2f}" if dd['recovery_value'] else " → Not recovered")
                        )
                    
                    # Automatic interpretation
                    interpretation = _interpret_single_drawdown(dd)
                    st.info(interpretation)
        else:
            st.info("No significant drawdowns found (threshold: 0.1%)")
    
    st.markdown("---")
    
    # Table 3.2.4: Top 5 Drawdowns
    st.markdown("### Top 5 Drawdowns")
    
    if portfolio_returns is not None and not portfolio_returns.empty:
        top_drawdowns = calculate_top_drawdowns(portfolio_returns, top_n=5)
        
        if top_drawdowns:
            # Create DataFrame for top drawdowns
            dd_data = []
            for i, dd in enumerate(top_drawdowns, 1):
                dd_data.append({
                    "#": i,
                    "Start (Peak)": dd['start_date'].strftime("%Y-%m-%d") if dd['start_date'] else "—",
                    "Bottom (Trough)": dd['bottom_date'].strftime("%Y-%m-%d") if dd['bottom_date'] else "—",
                    "Recovery (End)": dd['recovery_date'].strftime("%Y-%m-%d") if dd['recovery_date'] else "Ongoing",
                    "Depth (%)": f"{dd['depth']*100:.2f}%",
                    "Duration (days)": dd['duration_days'],
                    "Recovery (days)": dd['recovery_days'] if dd['recovery_days'] else "—",
                })
            
            dd_df = pd.DataFrame(dd_data)
            st.dataframe(dd_df, use_container_width=True, hide_index=True)
            
            # Facts for caption
            worst_depth = max(dd['depth'] for dd in top_drawdowns)
            longest_duration = max(dd['duration_days'] for dd in top_drawdowns)
            longest_recovery = max((dd['recovery_days'] for dd in top_drawdowns if dd['recovery_days']), default=None)
            recovery_text = f"{longest_recovery} days" if longest_recovery else "N/A"
            st.caption(f"**Summary:** Worst drawdown: {worst_depth*100:.2f}% | Longest duration: {longest_duration} days | Longest recovery: {recovery_text}")
            
            # Benchmark Comparison (if available)
            st.markdown("#### Benchmark Comparison")
            
            comparison_data = []
            
            # Portfolio metrics
            avg_dd_depth = sum(dd['depth'] for dd in top_drawdowns) / len(top_drawdowns) if top_drawdowns else 0
            avg_dd_duration = sum(dd['duration_days'] for dd in top_drawdowns) / len(top_drawdowns) if top_drawdowns else 0
            avg_recovery = sum(dd['recovery_days'] for dd in top_drawdowns if dd['recovery_days']) / len([dd for dd in top_drawdowns if dd['recovery_days']]) if any(dd['recovery_days'] for dd in top_drawdowns) else None
            max_recovery = max((dd['recovery_days'] for dd in top_drawdowns if dd['recovery_days']), default=None)
            
            comparison_data.append({
                "#": "1",
                "Metric": "Avg Drawdown Depth",
                "Portfolio": f"{avg_dd_depth*100:.2f}%",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "2",
                "Metric": "Avg Drawdown Duration",
                "Portfolio": f"{avg_dd_duration:.0f} days",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "3",
                "Metric": "Avg Recovery Time",
                "Portfolio": f"{avg_recovery:.0f} days" if avg_recovery else "—",
                "Benchmark": "—"
            })
            comparison_data.append({
                "#": "4",
                "Metric": "Max Recovery Time",
                "Portfolio": f"{max_recovery:.0f} days" if max_recovery else "—",
                "Benchmark": "—"
            })
            
            # Try to calculate benchmark metrics
            if benchmark_returns is not None and not benchmark_returns.empty:
                try:
                    bench_drawdowns = calculate_top_drawdowns(benchmark_returns, top_n=5)
                    
                    if bench_drawdowns:
                        bench_avg_dd_depth = sum(dd['depth'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_dd_duration = sum(dd['duration_days'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_recovery = sum(dd['recovery_days'] for dd in bench_drawdowns if dd['recovery_days']) / len([dd for dd in bench_drawdowns if dd['recovery_days']]) if any(dd['recovery_days'] for dd in bench_drawdowns) else None
                        bench_max_recovery = max((dd['recovery_days'] for dd in bench_drawdowns if dd['recovery_days']), default=None)
                        
                        comparison_data[0]["Benchmark"] = f"{bench_avg_dd_depth*100:.2f}%"
                        comparison_data[1]["Benchmark"] = f"{bench_avg_dd_duration:.0f} days"
                        comparison_data[2]["Benchmark"] = f"{bench_avg_recovery:.0f} days" if bench_avg_recovery else "—"
                        comparison_data[3]["Benchmark"] = f"{bench_max_recovery:.0f} days" if bench_max_recovery else "—"
                except Exception as e:
                    logger.warning(f"Error calculating benchmark drawdown metrics: {e}")
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Automatic interpretation
            if benchmark_returns is not None and not benchmark_returns.empty:
                try:
                    bench_drawdowns = calculate_top_drawdowns(benchmark_returns, top_n=5)
                    if bench_drawdowns:
                        bench_avg_dd_depth = sum(dd['depth'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_dd_duration = sum(dd['duration_days'] for dd in bench_drawdowns) / len(bench_drawdowns)
                        bench_avg_recovery = sum(dd['recovery_days'] for dd in bench_drawdowns if dd['recovery_days']) / len([dd for dd in bench_drawdowns if dd['recovery_days']]) if any(dd['recovery_days'] for dd in bench_drawdowns) else None
                        
                        parts = []
                        parts.append(f"**Benchmark Comparison Analysis:**")
                        parts.append(f"Average drawdown depth: Portfolio {avg_dd_depth*100:.2f}% vs Benchmark {bench_avg_dd_depth*100:.2f}%")
                        
                        if abs(avg_dd_depth - bench_avg_dd_depth) < 0.01:
                            parts.append(f"Similar drawdown depths")
                        elif avg_dd_depth < bench_avg_dd_depth:
                            parts.append(f"Portfolio shows deeper drawdowns")
                        else:
                            parts.append(f"Portfolio shows shallower drawdowns")
                        
                        if avg_recovery and bench_avg_recovery:
                            parts.append(f"Average recovery time: Portfolio {avg_recovery:.0f} days vs Benchmark {bench_avg_recovery:.0f} days")
                            if abs(avg_recovery - bench_avg_recovery) < 10:
                                parts.append(f"Similar recovery times")
                            elif avg_recovery < bench_avg_recovery:
                                parts.append(f"Portfolio recovers faster")
                            else:
                                parts.append(f"Portfolio recovers slower")
                        
                        interpretation = "\n".join(parts)
                        st.info(interpretation)
                except Exception as e:
                    logger.warning(f"Error interpreting benchmark comparison: {e}")
        else:
            st.info("No significant drawdowns found")
    else:
        st.warning("No portfolio returns data available")


def _render_var_analysis(portfolio_returns, benchmark_returns):
    """Sub-tab 3.3: VaR & CVaR."""
    st.subheader("Value at Risk (VaR) & Conditional VaR")
    
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    from core.analytics_engine.risk_metrics import calculate_var, calculate_cvar
    
    # Control 3.3.1: Trust Level Slider
    st.markdown("### Confidence Level")
    confidence_level = st.slider(
        "Select Confidence Level",
        min_value=90,
        max_value=99,
        value=95,
        step=1,
        key="var_confidence_level",
        help="Adjust confidence level to see how VaR and CVaR change"
    )
    
    conf_decimal = confidence_level / 100.0
    
    # Calculate VaR and CVaR for selected confidence level
    var_hist = calculate_var(portfolio_returns, conf_decimal, method="historical") or 0
    var_param = calculate_var(portfolio_returns, conf_decimal, method="parametric") or 0
    var_cf = calculate_var(portfolio_returns, conf_decimal, method="cornish_fisher") or 0
    cvar = calculate_cvar(portfolio_returns, conf_decimal) or 0
    
    # Calculate benchmark metrics if available
    bench_var_hist = 0
    bench_cvar = 0
    if benchmark_returns is not None and not benchmark_returns.empty:
        try:
            bench_var_hist = calculate_var(benchmark_returns, conf_decimal, method="historical") or 0
            bench_cvar = calculate_cvar(benchmark_returns, conf_decimal) or 0
        except Exception as e:
            logger.warning(f"Error calculating benchmark VaR/CVaR: {e}")
    
    st.markdown("---")
    
    # Table 3.3.2: VaR Comparison
    st.markdown("### VaR Methods Comparison")
    
    # Main comparison table
    var_comparison_data = [
        ("Historical", f"{var_hist*100:.2f}%", f"{int((1-conf_decimal)*100)}% of days worse"),
        ("Parametric", f"{var_param*100:.2f}%", "Assumes normal dist"),
        ("Cornish-Fisher", f"{var_cf*100:.2f}%", "Adj. for skew/kurt"),
        (f"CVaR (ES) {confidence_level}%", f"{cvar*100:.2f}%", "Avg beyond VaR"),
    ]
    
    var_comparison_df = pd.DataFrame(
        var_comparison_data,
        columns=["Method", f"VaR ({confidence_level}%)", "Interpretation"]
    )
    st.dataframe(var_comparison_df, use_container_width=True, hide_index=True)
    
    # Benchmark Comparison table
    if benchmark_returns is not None and not benchmark_returns.empty:
        st.markdown("#### Benchmark Comparison")
        
        benchmark_comparison_data = [
            (f"VaR {confidence_level}% (Hist)",
             f"{var_hist*100:.2f}%",
             f"{bench_var_hist*100:.2f}%",
             f"{(var_hist - bench_var_hist)*100:+.2f}%"),
            (f"CVaR {confidence_level}%",
             f"{cvar*100:.2f}%",
             f"{bench_cvar*100:.2f}%",
             f"{(cvar - bench_cvar)*100:+.2f}%"),
        ]
        
        benchmark_comparison_df = pd.DataFrame(
            benchmark_comparison_data,
            columns=["Method", "Portfolio", "Benchmark", "Difference"]
        )
        st.dataframe(benchmark_comparison_df, use_container_width=True, hide_index=True)
        
        # Add interpretation
        interpretation = _interpret_var_benchmark_comparison(
            var_hist, cvar, bench_var_hist, bench_cvar, confidence_level
        )
        st.info(interpretation)
    
    st.markdown("---")
    
    # Chart 3.3.3: VaR Visualization on Distribution
    st.markdown("### VaR on Return Distribution")
    
    try:
        fig = plot_var_distribution(
            portfolio_returns,
            var_hist,
            cvar,
            conf_decimal
        )
        st.plotly_chart(fig, use_container_width=True, key="var_distribution")
        
        # Add interpretation
        interpretation = _interpret_var_distribution(var_hist, cvar, conf_decimal, var_param, var_cf)
        st.info(interpretation)
    except Exception as e:
        logger.error(f"Error plotting VaR distribution: {e}")
        st.error("Could not generate VaR distribution chart")


def _render_rolling_risk(portfolio_returns, benchmark_returns, risk_free_rate):
    """Sub-tab 3.4: Rolling Risk Metrics."""
    if portfolio_returns is None or portfolio_returns.empty:
        st.warning("No portfolio returns data available")
        return
    
    # ===== Control 3.4.1: Window Size Slider =====
    st.markdown("### Rolling Window Size")
    window_size = st.slider(
        "Rolling Window Size (days)",
        min_value=21,
        max_value=252,
        value=63,
        step=21,
        key="rolling_window_size",
        help="Select the window size for rolling metrics calculation"
    )
    
    st.markdown(f"**Selected:** {window_size} days (≈{window_size // 21} months)")
    
    # ===== Chart 3.4.2: Rolling Volatility =====
    st.markdown("---")
    st.subheader("Rolling Volatility")
    try:
        vol_data = get_rolling_volatility_data(
            portfolio_returns,
            benchmark_returns,
            window=window_size
        )
        if vol_data:
            fig = plot_rolling_volatility(vol_data)
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="rolling_volatility_chart"
            )
            
            # Automatic interpretation
            if vol_data and "portfolio" in vol_data:
                port_vol = vol_data["portfolio"]
                interpretation = _interpret_rolling_metric(port_vol, "Volatility", "lower", window_size)
                st.info(interpretation)
            
            # Volatility Statistics as Comparison Cards
            st.markdown("**Volatility Statistics**")
            
            # Prepare metrics data for comparison cards
            port_stats = vol_data.get('portfolio_stats', {})
            bench_stats = vol_data.get('benchmark_stats')
            
            metrics_data = [
                {
                    "label": "Avg Volatility",
                    "portfolio_value": port_stats.get('avg', 0),
                    "benchmark_value": (
                        bench_stats.get('avg', 0)
                        if bench_stats else None
                    ),
                    "format": "percent",
                    "higher_is_better": False
                },
                {
                    "label": "Median Volatility",
                    "portfolio_value": port_stats.get('median', 0),
                    "benchmark_value": (
                        bench_stats.get('median', 0)
                        if bench_stats else None
                    ),
                    "format": "percent",
                    "higher_is_better": False
                },
                {
                    "label": "Min Volatility",
                    "portfolio_value": port_stats.get('min', 0),
                    "benchmark_value": (
                        bench_stats.get('min', 0)
                        if bench_stats else None
                    ),
                    "format": "percent",
                    "higher_is_better": False
                },
                {
                    "label": "Max Volatility",
                    "portfolio_value": port_stats.get('max', 0),
                    "benchmark_value": (
                        bench_stats.get('max', 0)
                        if bench_stats else None
                    ),
                    "format": "percent",
                    "higher_is_better": False
                }
            ]
            
            render_metric_cards_row(metrics_data, columns_per_row=4)
    except Exception as e:
        logger.error(f"Error plotting rolling volatility: {e}")
        st.error("Could not generate rolling volatility chart")
    
    # ===== Chart 3.4.3: Rolling Sharpe Ratio =====
    st.markdown("---")
    st.subheader("Rolling Sharpe Ratio")
    try:
        sharpe_data = get_rolling_sharpe_data(
            portfolio_returns,
            benchmark_returns,
            window=window_size,
            risk_free_rate=risk_free_rate
        )
        if sharpe_data:
            fig = plot_rolling_sharpe(sharpe_data, window=window_size)
            st.plotly_chart(fig, use_container_width=True, key="rolling_sharpe_chart")
            
            # Automatic interpretation
            if sharpe_data and "portfolio" in sharpe_data:
                port_sharpe = sharpe_data["portfolio"]
                interpretation = _interpret_rolling_metric(port_sharpe, "Sharpe Ratio", "higher", window_size, threshold=1.0)
                st.info(interpretation)
    except Exception as e:
        logger.error(f"Error plotting rolling Sharpe ratio: {e}")
        st.error("Could not generate rolling Sharpe ratio chart")
    
    # ===== Chart 3.4.4: Rolling Sortino Ratio =====
    st.markdown("---")
    st.subheader("Rolling Sortino Ratio")
    try:
        sortino_data = get_rolling_sortino_data(
            portfolio_returns,
            benchmark_returns,
            window=window_size,
            risk_free_rate=risk_free_rate
        )
        if sortino_data:
            fig = plot_rolling_sortino(sortino_data, window=window_size)
            st.plotly_chart(fig, use_container_width=True, key="rolling_sortino_chart")
            
            # Automatic interpretation
            if sortino_data and "portfolio" in sortino_data:
                port_sortino = sortino_data["portfolio"]
                interpretation = _interpret_rolling_metric(port_sortino, "Sortino Ratio", "higher", window_size, threshold=1.0)
                st.info(interpretation)
    except Exception as e:
        logger.error(f"Error plotting rolling Sortino ratio: {e}")
        st.error("Could not generate rolling Sortino ratio chart")
    
    # Only show beta/alpha charts if benchmark is available
    if benchmark_returns is not None and not benchmark_returns.empty:
        # ===== Chart 3.4.5: Rolling Beta =====
        st.markdown("---")
        st.subheader("Rolling Beta")
        try:
            beta_data = get_rolling_beta_data(
                portfolio_returns,
                benchmark_returns,
                window=window_size
            )
            if beta_data:
                fig = plot_rolling_beta(beta_data)
                st.plotly_chart(fig, use_container_width=True, key="rolling_beta_chart")
                
                # Automatic interpretation
                if beta_data and "beta" in beta_data:
                    port_beta = beta_data["beta"]
                    interpretation = _interpret_rolling_beta(port_beta, window_size)
                    if interpretation:
                        st.info(interpretation)
        except Exception as e:
            logger.error(f"Error plotting rolling beta: {e}")
            st.error("Could not generate rolling beta chart")
        
        # ===== Chart 3.4.6: Rolling Alpha =====
        st.markdown("---")
        st.subheader("Rolling Alpha")
        try:
            alpha_data = get_rolling_alpha_data(
                portfolio_returns,
                benchmark_returns,
                window=window_size,
                risk_free_rate=risk_free_rate
            )
            if alpha_data:
                fig = plot_rolling_alpha(alpha_data)
                st.plotly_chart(fig, use_container_width=True, key="rolling_alpha_chart")
                
                # Automatic interpretation
                if alpha_data and "alpha" in alpha_data:
                    port_alpha = alpha_data["alpha"]
                    interpretation = _interpret_rolling_metric(port_alpha, "Alpha", "higher", window_size, threshold=0.0)
                    if interpretation:
                        st.info(interpretation)
        except Exception as e:
            logger.error(f"Error plotting rolling alpha: {e}")
            st.error("Could not generate rolling alpha chart")
        
        # ===== Chart 3.4.7: Rolling Active Return =====
        st.markdown("---")
        st.subheader("Rolling Active Return")
        try:
            active_return_data = get_rolling_active_return_data(
                portfolio_returns,
                benchmark_returns,
                window=window_size
            )
            if active_return_data:
                fig = plot_rolling_active_return(active_return_data)
                st.plotly_chart(fig, use_container_width=True, key="rolling_active_return_chart")
                
                # Display statistics
                stats = active_return_data.get("stats", {})
                st.markdown("**Active Return Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Active Return", f"{stats.get('avg', 0)*100:.2f}%")
                col2.metric("Periods with Positive Alpha", f"{stats.get('pct_positive', 0):.1f}%")
                col3.metric("Max Alpha", f"{stats.get('max', 0)*100:.2f}%")
                col4.metric("Min Alpha", f"{stats.get('min', 0)*100:.2f}%")
                
                # Automatic interpretation
                if active_return_data and "active_return" in active_return_data:
                    port_active = active_return_data["active_return"]
                    interpretation = _interpret_rolling_metric(port_active, "Active Return", "higher", window_size, threshold=0.0)
                    if interpretation:
                        st.info(interpretation)
        except Exception as e:
            logger.error(f"Error plotting rolling active return: {e}")
            st.error("Could not generate rolling active return chart")
        
        # ===== Section 3.4.8: Bull/Bear Market Analysis =====
        st.markdown("---")
        st.subheader("Bull/Bear Market Analysis")
        st.markdown("*Separate Analysis of Bullish and Bearish Periods*")
        
        try:
            bull_bear_data = get_bull_bear_analysis_data(
                portfolio_returns,
                benchmark_returns,
                window=126  # Use 126 days (6 months) for bull/bear analysis
            )
            if bull_bear_data:
                bull = bull_bear_data.get("bull", {})
                bear = bull_bear_data.get("bear", {})
                
                # Statistics Table
                st.markdown(
                    "**Performance in Different Market Conditions**"
                )
                st.caption(
                    "Median daily return when benchmark is up (bullish) "
                    "vs down (bearish)"
                )
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    "Metric": [
                        "Portfolio Avg Daily Return (%)",
                        "Benchmark Avg Daily Return (%)",
                        "Beta",
                        "Outperformance (%)"
                    ],
                    "Bullish Market": [
                        f"{bull.get('portfolio_return', 0):.2f}",
                        f"{bull.get('benchmark_return', 0):.2f}",
                        f"{bull.get('beta', 0):.2f}",
                        f"{bull.get('difference', 0):.2f}"
                    ],
                    "Bearish Market": [
                        f"{bear.get('portfolio_return', 0):.2f}",
                        f"{bear.get('benchmark_return', 0):.2f}",
                        f"{bear.get('beta', 0):.2f}",
                        f"{bear.get('difference', 0):.2f}"
                    ],
                    "Difference": [
                        f"{bull.get('portfolio_return', 0) - bear.get('portfolio_return', 0):.2f}",
                        f"{bull.get('benchmark_return', 0) - bear.get('benchmark_return', 0):.2f}",
                        f"{bull.get('beta', 0) - bear.get('beta', 0):.2f}",
                        f"{bull.get('difference', 0) - bear.get('difference', 0):.2f}"
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Returns Comparison Bar Chart
                st.markdown("**Average Annualized Returns Comparison**")
                st.caption(
                    "Median daily returns in bullish vs bearish market days"
                )
                fig_comparison = plot_bull_bear_returns_comparison(
                    bull_bear_data
                )
                st.plotly_chart(
                    fig_comparison,
                    use_container_width=True,
                    key="bull_bear_comparison"
                )
                
                # Rolling Beta in Different Market Periods
                st.markdown("**Rolling Beta in Different Market Periods (126 days)**")
                fig_rolling = plot_bull_bear_rolling_beta(bull_bear_data)
                st.plotly_chart(fig_rolling, use_container_width=True, key="bull_bear_rolling_beta")
                
                # Add interpretation
                interpretation = _interpret_bull_bear_analysis(bull, bear)
                st.info(interpretation)
        except Exception as e:
            logger.error(f"Error in bull/bear market analysis: {e}")
            st.error("Could not generate bull/bear market analysis")
    else:
        st.info("Benchmark data required for Beta, Alpha, Active Return, and Bull/Bear analysis")


def _render_assets_tab(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service):
    """Render Assets & Correlations tab with sub-tabs."""
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Asset Overview & Impact",
        "Correlations",
        "Asset Details & Dynamics"
    ])
    
    with sub_tab1:
        _render_asset_overview(positions)
    
    with sub_tab2:
        _render_correlations(positions, portfolio_returns, benchmark_returns)
    
    with sub_tab3:
        _render_asset_details(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service)


def _render_asset_overview(positions):
    """Sub-tab 4.1: Asset Overview & Impact."""
    if not positions:
        st.info("No positions found")
        return
    
    # === Section 4.1.1: Assets Table (Extended) ===
    st.subheader("Assets Overview - Full Details")
    st.caption(
        "**Change%** shows daily price change "
        "(today vs previous trading day)"
    )
    
    # Fetch price data for asset metrics
    try:
        from services.data_service import DataService
        from datetime import date, timedelta
        
        data_service = DataService()
        end_date = date.today()
        start_date = end_date - timedelta(days=5)  # Last 5 days for price change
        
        # Fetch prices
        tickers = [pos.ticker for pos in positions]
        all_prices = []
        for ticker in tickers:
            try:
                if ticker == "CASH":
                    # Cash always 1.0
                    dr = pd.bdate_range(start=start_date, end=end_date)
                    prices = pd.DataFrame({
                        "Date": dr,
                        "Adjusted_Close": 1.0,
                        "Ticker": "CASH",
                    })
                else:
                    prices = data_service.fetch_historical_prices(
                        ticker, start_date, end_date, use_cache=True, save_to_db=False
                    )
                    prices["Ticker"] = ticker
                
                if not prices.empty:
                    all_prices.append(prices)
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        
        if all_prices:
            combined = pd.concat(all_prices, ignore_index=True)
            price_data = combined.pivot_table(
                index="Date",
                columns="Ticker",
                values="Adjusted_Close",
                aggfunc="last",
            )
        else:
            price_data = None
        
        # Get asset metrics data
        asset_data = get_asset_metrics_data(positions, price_data)
        
        if asset_data is not None and not asset_data.empty:
            render_assets_table_extended(asset_data)
        else:
            # Fallback to basic table
            render_position_table(positions)
    
    except Exception as e:
        logger.warning(f"Error fetching asset data: {e}")
        # Fallback to basic table
        render_position_table(positions)
    
    # === Fetch full price data for all impact analyses ===
    price_data_full = None
    start_date_impact = None
    end_date_impact = None
    
    try:
        analytics = st.session_state.get("portfolio_analytics", {})
        portfolio_returns = analytics.get("portfolio_returns")
        
        if portfolio_returns is not None and not portfolio_returns.empty:
            # Get date range from portfolio returns
            start_date_impact = portfolio_returns.index.min()
            end_date_impact = portfolio_returns.index.max()
            
            # Fetch full price data for impact calculation
            all_prices_full = []
            for ticker in tickers:
                try:
                    if ticker == "CASH":
                        dr = pd.bdate_range(
                            start=start_date_impact, end=end_date_impact
                        )
                        prices_full = pd.DataFrame({
                            "Date": dr,
                            "Adjusted_Close": 1.0,
                            "Ticker": "CASH",
                        })
                    else:
                        prices_full = data_service.fetch_historical_prices(
                            ticker, start_date_impact, end_date_impact,
                            use_cache=True, save_to_db=False
                        )
                        prices_full["Ticker"] = ticker
                    
                    if not prices_full.empty:
                        all_prices_full.append(prices_full)
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch full prices for {ticker}: {e}"
                    )
            
            if all_prices_full:
                combined_full = pd.concat(all_prices_full, ignore_index=True)
                price_data_full = combined_full.pivot_table(
                    index="Date",
                    columns="Ticker",
                    values="Adjusted_Close",
                    aggfunc="last",
                )
    except Exception as e:
        logger.error(f"Error fetching price data for impact analysis: {e}")
    
    # === Section 4.1.2: Chart - Impact on Total Return ===
    st.markdown("---")
    st.subheader("Impact on Assets to Total Return")
    
    try:
        # Check if we have required data
        if price_data_full is None or price_data_full.empty:
            st.info(
                "Please calculate analytics first to see impact analysis. "
                "Price data is required for this calculation."
            )
        elif start_date_impact is None or end_date_impact is None:
            st.info(
                "Please calculate analytics first to see impact analysis. "
                "Portfolio returns date range is required."
            )
        else:
            # Calculate impact on return
            impact_return_data = get_asset_impact_on_return_data(
                positions, price_data_full, start_date_impact, end_date_impact
            )
            
            if impact_return_data and impact_return_data.get("tickers"):
                fig = plot_impact_on_return(impact_return_data)
                st.plotly_chart(
                    fig, use_container_width=True, key="impact_on_return"
                )
                
                # Show top contributor
                top_ticker = impact_return_data["tickers"][0]
                top_contrib = impact_return_data["contributions"][0]
                st.caption(
                    f"**Top contributor:** {top_ticker} - "
                    f"{top_contrib:.2f}% weighted contribution "
                    f"to portfolio return"
                )
                
                # Automatic interpretation
                interpretation = _interpret_impact_on_return(impact_return_data)
                if interpretation:
                    st.info(interpretation)
            else:
                st.info("Insufficient data for impact on return analysis. "
                       "Please ensure all positions have valid price data.")
    
    except Exception as e:
        logger.error(f"Error calculating impact on return: {e}", exc_info=True)
        st.error(
            f"Unable to calculate impact on return: {str(e)}. "
            "Please check the logs for details."
        )
    
    # === Section 4.1.3: Chart - Impact on Risk ===
    st.markdown("---")
    st.subheader("Impact on Assets to Overall Portfolio Risk")
    
    try:
        if price_data_full is not None and not price_data_full.empty:
            impact_risk_data = get_asset_impact_on_risk_data(positions, price_data_full)
            
            if impact_risk_data:
                fig = plot_impact_on_risk(impact_risk_data)
                st.plotly_chart(fig, use_container_width=True, key="impact_on_risk")
                
                # Show biggest contributor
                if impact_risk_data["tickers"]:
                    top_ticker = impact_risk_data["tickers"][0]
                    top_contrib = impact_risk_data["risk_contributions"][0]
                    st.caption(
                        f"**Biggest risk contributor:** {top_ticker} - "
                        f"{top_contrib:.1f}% of portfolio risk"
                    )
                
                # Automatic interpretation
                interpretation = _interpret_impact_on_risk(impact_risk_data)
                if interpretation:
                    st.info(interpretation)
            else:
                st.info("Insufficient data for impact on risk analysis")
        else:
            st.info("Price data not available for risk analysis")
    
    except Exception as e:
        logger.error(f"Error calculating impact on risk: {e}")
        st.info("Unable to calculate impact on risk")
    
    # === Section 4.1.4: Chart - Comparison Risk vs Weight ===
    st.markdown("---")
    st.subheader("Comparison of Risk & Return Impact and Asset Weighting")
    
    try:
        if (price_data_full is not None and not price_data_full.empty 
            and start_date_impact is not None):
            comparison_data = get_risk_vs_weight_comparison_data(
                positions, price_data_full, start_date_impact, end_date_impact
            )
            
            if comparison_data:
                fig = plot_risk_vs_weight_comparison(comparison_data)
                st.plotly_chart(fig, use_container_width=True, key="risk_vs_weight")
                
                st.caption(
                    "**For well-diversified portfolio:** bars should be similar. "
                    "**Red bars:** Risk impact, **Green bars:** Return impact, "
                    "**Orange bars:** Portfolio weight. "
                    "If red bar >> orange bar: asset contributes more risk than weight "
                    "(high beta/volatility). "
                    "If green bar >> orange bar: asset contributes more return than weight."
                )
                
                # Automatic interpretation
                interpretation = _interpret_risk_vs_weight_comparison(comparison_data)
                if interpretation:
                    st.info(interpretation)
            else:
                st.info("Insufficient data for comparison analysis")
        else:
            st.info(
                "Please calculate analytics first to see comparison analysis. "
                "Price data and date range are required."
            )
    
    except Exception as e:
        logger.error(f"Error calculating risk vs weight comparison: {e}", exc_info=True)
        st.error(
            f"Unable to calculate comparison: {str(e)}. "
            "Please check the logs for details."
        )
    
    # === Section 4.1.5: Diversification Coefficient ===
    st.markdown("---")
    st.subheader("Diversification Assessment")
    
    try:
        if price_data_full is not None and not price_data_full.empty:
            div_data = get_diversification_coefficient_data(positions, price_data_full)
            
            if div_data:
                coef = div_data["diversification_coefficient"]
                vol_reduction = div_data["volatility_reduction_pct"]
                is_diversified = div_data["is_diversified"]
                
                # Display coefficient in styled box
                st.markdown("━" * 60)
                st.markdown(f"### Diversification Coefficient: **{coef:.2f}**")
                st.markdown("")
                st.markdown("**Formula:** Weighted sum of volatilities / Portfolio volatility")
                st.markdown("")
                st.markdown("**Interpretation:**")
                
                if is_diversified:
                    st.success(
                        f"✓ Value > 1.0 indicates positive diversification effect\n\n"
                        f"✓ {coef:.2f} means {vol_reduction:.1f}% volatility reduction from diversification\n\n"
                        f"✓ Portfolio is well-diversified"
                    )
                else:
                    st.warning(
                        f"Value <= 1.0 indicates little to no diversification benefit\n\n"
                        f"Portfolio may be under-diversified or concentrated"
                    )
                
                st.info(
                    "The diversification coefficient shows the ratio of the "
                    "weighted sum of individual volatilities to total portfolio "
                    "volatility. A value above 1 indicates positive effect."
                )
                st.markdown("━" * 60)
            else:
                st.info("Insufficient data for diversification analysis")
        else:
            st.info("Price data not available")
    
    except Exception as e:
        logger.error(f"Error calculating diversification coefficient: {e}")
        st.info("Unable to calculate diversification coefficient")
    
    # === Section 4.1.6: Factor Exposure Analysis ===
    st.markdown("---")
    st.subheader("Factor Exposure Analysis")
    
    st.info(
        "**Factor Analysis** using market index proxies\n\n"
        "Requires factor data API (e.g., French Data Library)\n\n"
        "This analysis estimates portfolio factor exposures using "
        "regression on market indices as factor proxies."
    )
    
    try:
        analytics = st.session_state.get("portfolio_analytics", {})
        portfolio_returns = analytics.get("portfolio_returns")
        
        if (portfolio_returns is not None and not portfolio_returns.empty 
            and len(portfolio_returns) > 30):
            
            # Try to fetch Fama-French factors first, fallback to ETF proxies
            from core.data_manager.factor_data import get_fama_french_factors
            from services.data_service import DataService
            
            factor_data = {}
            
            start_factor = portfolio_returns.index.min()
            end_factor = portfolio_returns.index.max()
            
            # Convert datetime index to date if needed
            if hasattr(start_factor, 'date'):
                start_factor_date = start_factor.date()
            else:
                start_factor_date = start_factor
            if hasattr(end_factor, 'date'):
                end_factor_date = end_factor.date()
            else:
                end_factor_date = end_factor
            
            with st.spinner("Fetching factor data..."):
                # First, try to fetch real Fama-French factors
                try:
                    ff_factors = get_fama_french_factors(
                        start_date=start_factor_date,
                        end_date=end_factor_date,
                        include_momentum=True
                    )
                    
                    if ff_factors:
                        factor_data.update(ff_factors)
                        logger.info(
                            f"Successfully loaded {len(ff_factors)} Fama-French factors"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch Fama-French factors: {e}. "
                        "Falling back to ETF proxies."
                    )
                
                # If we don't have enough factors, use ETF proxies as fallback
                if len(factor_data) < 3:
                    logger.info("Using ETF proxies for factor construction")
                    data_service = DataService()
                    
                    # Fetch base ETFs for factor construction
                    base_etfs = {}
                    etf_tickers = ["SPY", "IWM", "VTV", "VUG", "MTUM", "QUAL"]
                    
                    for ticker in etf_tickers:
                        try:
                            prices = data_service.fetch_historical_prices(
                                ticker, start_factor_date, end_factor_date,
                                use_cache=True, save_to_db=False
                            )
                            if not prices.empty:
                                returns = prices.set_index("Date")[
                                    "Adjusted_Close"
                                ].pct_change().dropna()
                                base_etfs[ticker] = returns
                        except Exception as e:
                            logger.warning(f"Failed to fetch {ticker}: {e}")
                    
                    # Construct factors from ETFs (only if not already loaded)
                    if "Market (Mkt-RF)" not in factor_data and "SPY" in base_etfs:
                        factor_data["Market (Mkt-RF)"] = base_etfs["SPY"]
                    
                    if "Size (SMB)" not in factor_data:
                        if "IWM" in base_etfs and "SPY" in base_etfs:
                            smb = base_etfs["IWM"] - base_etfs["SPY"]
                            factor_data["Size (SMB)"] = smb
                    
                    if "Value (HML)" not in factor_data:
                        if "VTV" in base_etfs and "VUG" in base_etfs:
                            hml = base_etfs["VTV"] - base_etfs["VUG"]
                            factor_data["Value (HML)"] = hml
                    
                    if "Momentum (MOM)" not in factor_data and "MTUM" in base_etfs:
                        factor_data["Momentum (MOM)"] = base_etfs["MTUM"]
                    
                    # Quality (QMJ): QUAL (if available)
                    if "QUAL" in base_etfs:
                        factor_data["Quality (QMJ)"] = base_etfs["QUAL"]
            
            if len(factor_data) >= 1:
                # Align all data - use inner join to keep only dates with all data
                aligned_data = pd.DataFrame({
                    "Portfolio": portfolio_returns
                })
                
                # Add each factor with proper alignment
                for factor_name, returns in factor_data.items():
                    # Align returns to portfolio dates
                    aligned_returns = returns.reindex(
                        portfolio_returns.index, method=None
                    )
                    aligned_data[factor_name] = aligned_returns
                
                # Drop rows where any factor or portfolio is NaN
                aligned_data = aligned_data.dropna()
                
                # Debug: log how many factors we have
                logger.info(
                    f"Factor analysis: {len(factor_data)} factors loaded, "
                    f"{len(aligned_data)} aligned observations. "
                    f"Factors: {list(factor_data.keys())}"
                )
                
                # Show info about loaded factors
                if len(factor_data) > 0:
                    factor_names = ", ".join(list(factor_data.keys())[:5])
                    if len(factor_data) > 5:
                        factor_names += f" (+{len(factor_data) - 5} more)"
                    st.caption(
                        f"**Loaded {len(factor_data)} factor(s):** {factor_names}"
                    )
                
                if len(aligned_data) > 20:
                    # Run regression
                    from scipy import stats
                    
                    y = aligned_data["Portfolio"].values
                    X_factors = aligned_data.drop(columns=["Portfolio"])
                    
                    # Add constant (intercept)
                    X = np.column_stack([
                        np.ones(len(X_factors)), X_factors.values
                    ])
                    
                    # Perform regression
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    # Calculate residuals and statistics
                    y_pred = X @ beta
                    residuals = y - y_pred
                    n = len(y)
                    k = X.shape[1]
                    mse = np.sum(residuals**2) / (n - k)
                    
                    # Standard errors
                    var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
                    se_beta = np.sqrt(var_beta)
                    
                    # t-statistics
                    t_stats = beta / se_beta
                    
                    # p-values
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
                    
                    # R-squared
                    ss_total = np.sum((y - np.mean(y))**2)
                    ss_residual = np.sum(residuals**2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Calculate contributions
                    explained_var = np.var(y_pred)
                    factor_contributions = []
                    for i in range(1, len(beta)):
                        contrib = (
                            beta[i] * X_factors.iloc[:, i-1].values
                        ).var() / explained_var * 100
                        factor_contributions.append(contrib)
                    
                    # Normalize contributions
                    total_contrib = sum(factor_contributions)
                    if total_contrib > 0:
                        factor_contributions = [
                            c / total_contrib * 100 
                            for c in factor_contributions
                        ]
                    
                    # Create results table (sorted by contribution)
                    results = []
                    for i, factor_name in enumerate(X_factors.columns):
                        sig = ""
                        if p_values[i+1] < 0.001:
                            sig = "***"
                        elif p_values[i+1] < 0.01:
                            sig = "**"
                        elif p_values[i+1] < 0.05:
                            sig = "*"
                        
                        results.append({
                            "Factor": factor_name,
                            "Exposure": beta[i+1],
                            "t-stat": t_stats[i+1],
                            "p-value": p_values[i+1],
                            "Contribution": factor_contributions[i],
                        })
                    
                    # Sort by contribution (descending by absolute value)
                    results = sorted(
                        results, 
                        key=lambda x: abs(x["Contribution"]), 
                        reverse=True
                    )
                    
                    # Format for display table
                    display_results = []
                    for r in results:
                        # Format t-stat with significance
                        t_stat_str = f"{r['t-stat']:.1f}{'***' if r['p-value'] < 0.001 else '**' if r['p-value'] < 0.01 else '*' if r['p-value'] < 0.05 else ''}"
                        
                        display_results.append({
                            "Factor": r["Factor"],
                            "Exposure": f"{r['Exposure']:.2f}",
                            "t-stat": t_stat_str,
                            "Contribution (%)": f"{r['Contribution']:.1f}%",
                        })
                    
                    results_df = pd.DataFrame(display_results)
                    
                    # Display results table (styled as per spec)
                    st.markdown("**Factor Analysis - Style Attribution**")
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    st.caption(
                        "*Significance: *** p<0.001, ** p<0.01, * p<0.05*"
                    )
                    
                    # PIE chart showing % contribution - better for small values
                    import plotly.graph_objects as go
                    
                    # Sort by absolute contribution (largest first) for better visualization
                    chart_data = sorted(
                        results, 
                        key=lambda x: abs(x["Contribution"]), 
                        reverse=True
                    )
                    
                    # Colors for factors (match spec style)
                    # Use colors from unified palette
                    from streamlit_app.utils.chart_config import COLORS
                    factor_colors = {
                        "Market (Mkt-RF)": COLORS["secondary"],  # Blue
                        "Size (SMB)": COLORS["success"],  # Green
                        "Value (HML)": COLORS["warning"],  # Orange
                        "Momentum (MOM)": COLORS["primary"],  # Purple
                        "Quality (QMJ)": COLORS["additional"],  # Yellow
                    }
                    
                    # Prepare data for pie chart
                    # Use absolute values for pie chart, but show sign in labels
                    labels = []
                    values = []
                    colors_list = []
                    text_info = []
                    hover_data = []
                    
                    for r in chart_data:
                        factor_name = r["Factor"]
                        contrib = r["Contribution"]
                        exposure = r["Exposure"]
                        abs_contrib = abs(contrib)
                        
                        # Only include factors with non-zero contribution
                        if abs_contrib > 0.01:  # Threshold: 0.01%
                            labels.append(factor_name)
                            values.append(abs_contrib)
                            colors_list.append(
                                factor_colors.get(factor_name, "#888888")
                            )
                            
                            # Create label with sign indicator
                            sign = "+" if contrib > 0 else "-"
                            text_info.append(
                                f"{sign}{abs_contrib:.1f}%"
                            )
                            # Store actual contribution and exposure for hover
                            hover_data.append([contrib, exposure])
                    
                    # Create pie chart
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,  # Donut chart style
                        marker=dict(
                            colors=colors_list,
                            line=dict(color='#1A1C20', width=2)
                        ),
                        text=text_info,
                        textposition='outside',
                        textinfo='text+percent',
                        hovertemplate=(
                            "<b>%{label}</b><br>" +
                            "Contribution: %{customdata[0]:+.2f}%<br>" +
                            "Exposure: %{customdata[1]:.2f}<br>" +
                            "<extra></extra>"
                        ),
                        customdata=hover_data,
                    )])
                    
                    from streamlit_app.utils.chart_config import (
                        get_chart_layout
                    )
                    
                    # Custom layout for pie chart
                    layout = get_chart_layout(
                        title="Factor Attribution Chart",
                        showlegend=True,
                    )
                    # Remove axis-specific settings for pie chart
                    layout.pop('xaxis', None)
                    layout.pop('yaxis', None)
                    layout.pop('barmode', None)
                    layout.pop('hovermode', None)
                    
                    fig.update_layout(**layout)
                    
                    # Add note if there are negative contributions
                    negative_factors = [
                        r for r in chart_data 
                        if r["Contribution"] < 0 and abs(r["Contribution"]) > 0.01
                    ]
                    if negative_factors:
                        neg_names = ", ".join([r["Factor"] for r in negative_factors])
                        st.caption(
                            f"**Note:** Negative contributions shown with minus sign: {neg_names}"
                        )
                    
                    st.plotly_chart(
                        fig, use_container_width=True, 
                        key="factor_contribution"
                    )
                    
                    # Model statistics (compact)
                    st.markdown("**Model Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{r_squared:.3f}")
                    with col2:
                        st.metric("Observations", f"{n}")
                    with col3:
                        alpha_annual = beta[0]*252*100
                        st.metric(
                            "Alpha (Annual)", 
                            f"{alpha_annual:.2f}%"
                        )
                    
                    # === Automatic Interpretation ===
                    st.markdown("---")
                    st.subheader("Automatic Interpretation")
                    
                    # Analyze dominant factors
                    significant_factors = [
                        r for r in results 
                        if abs(r["t-stat"]) > 2 and abs(r["Contribution"]) > 1.0
                    ]
                    dominant_factors = [
                        r for r in results 
                        if abs(r["Contribution"]) > 10.0
                    ]
                    
                    # Style analysis
                    value_exposure = None
                    size_exposure = None
                    momentum_exposure = None
                    quality_exposure = None
                    market_exposure = None
                    
                    for r in results:
                        factor_name = r["Factor"]
                        exposure = r["Exposure"]
                        if "Market" in factor_name or "Mkt-RF" in factor_name:
                            market_exposure = exposure
                        elif "Value" in factor_name or "HML" in factor_name:
                            value_exposure = exposure
                        elif "Size" in factor_name or "SMB" in factor_name:
                            size_exposure = exposure
                        elif "Momentum" in factor_name or "MOM" in factor_name:
                            momentum_exposure = exposure
                        elif "Quality" in factor_name or "QMJ" in factor_name:
                            quality_exposure = exposure
                    
                    # Build interpretation
                    interpretation_parts = []
                    
                    # Model quality
                    if r_squared > 0.9:
                        interpretation_parts.append(
                            f"✅ **Excellent model fit** (R² = {r_squared:.1%}). "
                            "The model explains most of the portfolio's return variance."
                        )
                    elif r_squared > 0.7:
                        interpretation_parts.append(
                            f"✅ **Good model fit** (R² = {r_squared:.1%}). "
                            "The model explains a substantial portion of returns."
                        )
                    elif r_squared > 0.5:
                        interpretation_parts.append(
                            f"⚠️ **Moderate model fit** (R² = {r_squared:.1%}). "
                            "The model explains about half of the variance. "
                            "Consider if additional factors are needed."
                        )
                    else:
                        interpretation_parts.append(
                            f"⚠️ **Weak model fit** (R² = {r_squared:.1%}). "
                            "The model explains less than half of the variance. "
                            "Portfolio may have significant idiosyncratic risk or "
                            "require different factor model."
                        )
                    
                    # Alpha interpretation
                    if abs(alpha_annual) > 5:
                        if alpha_annual > 0:
                            interpretation_parts.append(
                                f"✅ **Strong positive alpha** ({alpha_annual:+.2f}% annual). "
                                "Portfolio shows significant outperformance beyond "
                                "factor exposures."
                            )
                        else:
                            interpretation_parts.append(
                                f"⚠️ **Negative alpha** ({alpha_annual:+.2f}% annual). "
                                "Portfolio underperforms after accounting for factor exposures."
                            )
                    elif abs(alpha_annual) > 2:
                        if alpha_annual > 0:
                            interpretation_parts.append(
                                f"✅ **Positive alpha** ({alpha_annual:+.2f}% annual). "
                                "Portfolio generates excess returns beyond factors."
                            )
                        else:
                            interpretation_parts.append(
                                f"⚠️ **Slight negative alpha** ({alpha_annual:+.2f}% annual). "
                                "Portfolio slightly underperforms factor-adjusted returns."
                            )
                    else:
                        interpretation_parts.append(
                            f"ℹ️ **Neutral alpha** ({alpha_annual:+.2f}% annual). "
                            "Returns are well explained by factor exposures."
                        )
                    
                    # Dominant factors
                    if dominant_factors:
                        top_factor = dominant_factors[0]
                        interpretation_parts.append(
                            f"📈 **Dominant factor:** {top_factor['Factor']} "
                            f"contributes {abs(top_factor['Contribution']):.1f}% of returns. "
                            f"Exposure: {top_factor['Exposure']:.2f} "
                            f"(t-stat: {top_factor['t-stat']:.1f})."
                        )
                    
                    if len(dominant_factors) > 1:
                        second_factor = dominant_factors[1]
                        interpretation_parts.append(
                            f"📊 **Secondary factor:** {second_factor['Factor']} "
                            f"contributes {abs(second_factor['Contribution']):.1f}% of returns."
                        )
                    
                    # Style classification
                    style_classification = []
                    
                    if value_exposure is not None:
                        if value_exposure > 0.3:
                            style_classification.append("**Value-oriented**")
                        elif value_exposure < -0.3:
                            style_classification.append("**Growth-oriented**")
                        else:
                            style_classification.append("**Blend**")
                    
                    if size_exposure is not None:
                        if size_exposure > 0.2:
                            style_classification.append("**Small-cap tilt**")
                        elif size_exposure < -0.2:
                            style_classification.append("**Large-cap tilt**")
                        else:
                            style_classification.append("**Market-cap neutral**")
                    
                    if momentum_exposure is not None:
                        if momentum_exposure > 0.2:
                            style_classification.append("**Momentum strategy**")
                        elif momentum_exposure < -0.2:
                            style_classification.append("**Contrarian strategy**")
                    
                    if quality_exposure is not None:
                        if quality_exposure > 0.2:
                            style_classification.append("**Quality focus**")
                        elif quality_exposure < -0.2:
                            style_classification.append("**Lower quality exposure**")
                    
                    if style_classification:
                        interpretation_parts.append(
                            f"🎯 **Portfolio style:** {', '.join(style_classification)}."
                        )
                    
                    # Market exposure
                    if market_exposure is not None:
                        if market_exposure > 1.1:
                            interpretation_parts.append(
                                f"📊 **High market sensitivity** (β = {market_exposure:.2f}). "
                                "Portfolio moves more than the market."
                            )
                        elif market_exposure < 0.9:
                            interpretation_parts.append(
                                f"📊 **Lower market sensitivity** (β = {market_exposure:.2f}). "
                                "Portfolio moves less than the market."
                            )
                        else:
                            interpretation_parts.append(
                                f"📊 **Market-like sensitivity** (β = {market_exposure:.2f}). "
                                "Portfolio moves in line with the market."
                            )
                    
                    # Statistical significance
                    highly_significant = [
                        r for r in results 
                        if abs(r["t-stat"]) > 3 and abs(r["p-value"]) < 0.01
                    ]
                    if len(highly_significant) >= 2:
                        interpretation_parts.append(
                            f"✅ {len(highly_significant)} factors show **highly significant** "
                            "exposures (p < 0.01), indicating robust factor loadings."
                        )
                    
                    # Display interpretation
                    for part in interpretation_parts:
                        st.markdown(part)
                        st.markdown("")
                    
                    # Summary recommendation
                    st.markdown("---")
                    st.markdown("### 💡 Summary")
                    
                    summary_points = []
                    
                    # Factor concentration
                    top_2_contrib = sum([
                        abs(r["Contribution"]) for r in results[:2]
                    ])
                    if top_2_contrib > 80:
                        summary_points.append(
                            "⚠️ Portfolio is **highly concentrated** in 1-2 factors. "
                            "Consider diversifying factor exposures."
                        )
                    elif top_2_contrib > 60:
                        summary_points.append(
                            "ℹ️ Portfolio shows **moderate factor concentration**. "
                            "Some diversification across factors is present."
                        )
                    else:
                        summary_points.append(
                            "✅ Portfolio shows **good factor diversification** "
                            "across multiple factors."
                        )
                    
                    # Risk assessment
                    if market_exposure is not None and market_exposure > 1.2:
                        summary_points.append(
                            "⚠️ **High market beta** suggests portfolio may experience "
                            "significant volatility during market downturns."
                        )
                    elif market_exposure is not None and market_exposure < 0.8:
                        summary_points.append(
                            "✅ **Lower market beta** provides some downside protection "
                            "during market declines."
                        )
                    
                    # Alpha assessment
                    if alpha_annual > 3:
                        summary_points.append(
                            f"✅ **Positive alpha** ({alpha_annual:+.2f}%) suggests "
                            "portfolio management adds value beyond factor exposures."
                        )
                    elif alpha_annual < -3:
                        summary_points.append(
                            f"⚠️ **Negative alpha** ({alpha_annual:+.2f}%) suggests "
                            "portfolio may benefit from review of strategy or holdings."
                        )
                    
                    for point in summary_points:
                        st.markdown(f"- {point}")
                    
                    # Interpretation guide
                    with st.expander("📖 How to interpret", expanded=False):
                        st.markdown("""
                        **Factor Exposure (Beta):**
                        - Shows how the portfolio responds to each factor
                        - Example: Market exposure of 1.0 means portfolio 
                          moves 1:1 with market
                        
                        **t-statistic:**
                        - Measures statistical significance
                        - |t| > 2 typically indicates significance
                        - |t| > 3 indicates high significance
                        
                        **Contribution (%):**
                        - Percentage of explained return variance from 
                          each factor
                        - Sum = 100% of explained variance
                        
                        **R²:**
                        - Proportion of portfolio variance explained by 
                          factors
                        - Higher R² = better model fit
                        - R² > 0.9 = excellent, > 0.7 = good, < 0.5 = weak
                        
                        **Alpha (Intercept):**
                        - Excess return not explained by factors
                        - Positive alpha = outperformance
                        - Negative alpha = underperformance
                        """)
                else:
                    st.warning("Insufficient aligned data for regression")
            else:
                st.warning(
                    "Unable to fetch factor proxy data. "
                    "Please check internet connection."
                )
        else:
            st.info(
                "Please calculate analytics with at least 30 days of data "
                "to run factor analysis"
            )
    
    except Exception as e:
        logger.error(f"Error in factor analysis: {e}")
        st.error(f"Error calculating factor analysis: {str(e)}")


def _render_correlations(positions, portfolio_returns, benchmark_returns):
    """Sub-tab 4.2: Correlations."""
    st.subheader("Correlation Analysis")
    
    if not positions:
        st.info("No positions available for correlation analysis")
        return
    
    # Fetch price data
    try:
        from services.data_service import DataService
        
        analytics = st.session_state.get("portfolio_analytics", {})
        portfolio_returns = analytics.get("portfolio_returns")
        
        if portfolio_returns is None or portfolio_returns.empty:
            st.info("Please calculate analytics first to see correlation analysis")
            return
        
        # Get date range
        start_date = portfolio_returns.index.min()
        end_date = portfolio_returns.index.max()
        
        data_service = DataService()
        tickers = [pos.ticker for pos in positions]
        
        # Fetch price data
        all_prices = []
        for ticker in tickers:
            try:
                if ticker == "CASH":
                    dr = pd.bdate_range(start=start_date, end=end_date)
                    prices = pd.DataFrame({
                        "Date": dr,
                        "Adjusted_Close": 1.0,
                        "Ticker": "CASH",
                    })
                else:
                    prices = data_service.fetch_historical_prices(
                        ticker, start_date, end_date,
                        use_cache=True, save_to_db=False
                    )
                    prices["Ticker"] = ticker
                
                if not prices.empty:
                    all_prices.append(prices)
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        
        if not all_prices:
            st.info("Unable to fetch price data for correlation analysis")
            return
        
        # Combine and pivot
        combined = pd.concat(all_prices, ignore_index=True)
        price_data = combined.pivot_table(
            index="Date",
            columns="Ticker",
            values="Adjusted_Close",
            aggfunc="last",
        )
        
        # === Section 4.2.1: Correlation Matrix ===
        st.subheader("Correlation Matrix - All Assets + Benchmark")
        
        corr_matrix_data = get_correlation_matrix_data(
            positions, price_data, benchmark_returns
        )
        
        if corr_matrix_data and corr_matrix_data.get("correlation_matrix") is not None:
            correlation_matrix = corr_matrix_data["correlation_matrix"]
            fig = plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")
            
            # Automatic interpretation
            interpretation = _interpret_correlation_matrix(corr_matrix_data)
            if interpretation:
                st.info(interpretation)
        else:
            st.info("Insufficient data for correlation matrix")
        
        # === Section 4.2.2: Correlation Statistics ===
        st.markdown("---")
        st.subheader("Correlation Statistics")
        
        if corr_matrix_data and corr_matrix_data.get("correlation_matrix") is not None:
            corr_stats = get_correlation_statistics_data(
                corr_matrix_data["correlation_matrix"]
            )
            
            if corr_stats:
                # Row 1: Average Correlation | Median Correlation
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Correlation", f"{corr_stats['average_correlation']:.2f}")
                with col2:
                    st.metric("Median Correlation", f"{corr_stats['median_correlation']:.2f}")
                
                # Row 2: Min Correlation | Max Correlation
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Min Correlation", 
                             f"{corr_stats['min_correlation']:.2f}",
                             help=f"Between {corr_stats['min_pair'][0]} and {corr_stats['min_pair'][1]}")
                with col4:
                    st.metric("Max Correlation", 
                             f"{corr_stats['max_correlation']:.2f}",
                             help=f"Between {corr_stats['max_pair'][0]} and {corr_stats['max_pair'][1]}")
                
                # Row 3: Pairs > 0.8 (high) | Pairs < 0.2 (low)
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Pairs > 0.8 (high)", corr_stats['high_corr_count'])
                with col6:
                    st.metric("Pairs < 0.2 (low)", corr_stats['low_corr_count'])
                
                # Interpretation and Recommendations
                avg_corr = corr_stats['average_correlation']
                high_corr_count = corr_stats['high_corr_count']
                low_corr_count = corr_stats['low_corr_count']
                min_corr = corr_stats['min_correlation']
                
                st.markdown("---")
                st.subheader("Correlation Analysis & Recommendations")
                
                # Overall assessment
                if avg_corr < 0.3:
                    st.success(
                        "✓ Low average correlation - Excellent "
                        "diversification potential"
                    )
                elif avg_corr < 0.5:
                    st.success(
                        "✓ Moderate average correlation - Good "
                        "diversification potential"
                    )
                else:
                    st.warning(
                        "⚠ High average correlation - Limited "
                        "diversification"
                    )
                
                # Detailed recommendations
                recommendations = []
                
                # Check for high correlation pairs
                if high_corr_count > 0:
                    recommendations.append({
                        "type": "warning",
                        "message": (
                            f"⚠ Found {high_corr_count} pair(s) with "
                            f"correlation > 0.8 - Risk of concentration. "
                            f"Consider reducing weight of highly correlated "
                            f"assets."
                        )
                    })
                
                # Check for low correlation opportunities
                if low_corr_count > 0:
                    recommendations.append({
                        "type": "info",
                        "message": (
                            f"✓ Found {low_corr_count} pair(s) with "
                            f"correlation < 0.2 - Good diversification "
                            f"opportunities present."
                        )
                    })
                else:
                    recommendations.append({
                        "type": "warning",
                        "message": (
                            "⚠ No pairs with correlation < 0.2 found. "
                            "Consider adding assets with lower correlation "
                            "for better diversification."
                        )
                    })
                
                # Check for negative correlation (hedging)
                if min_corr < 0:
                    recommendations.append({
                        "type": "success",
                        "message": (
                            f"✓ Found negative correlation ({min_corr:.2f}) - "
                            f"Natural hedging opportunity present."
                        )
                    })
                else:
                    recommendations.append({
                        "type": "info",
                        "message": (
                            "ℹ No negative correlations found. Consider "
                            "adding assets with negative correlation "
                            "(e.g., bonds vs stocks) for hedging."
                        )
                    })
                
                # Check average correlation level
                if avg_corr > 0.5:
                    recommendations.append({
                        "type": "warning",
                        "message": (
                            f"⚠ Average correlation ({avg_corr:.2f}) is high. "
                            f"Portfolio may be under-diversified. Consider "
                            f"adding assets from different sectors or asset "
                            f"classes."
                        )
                    })
                
                # Display recommendations
                for rec in recommendations:
                    if rec["type"] == "success":
                        st.success(rec["message"])
                    elif rec["type"] == "warning":
                        st.warning(rec["message"])
                    else:
                        st.info(rec["message"])
        
        # === Section 4.2.3: Correlation with Benchmark ===
        st.markdown("---")
        st.subheader("Asset Correlation with SPY")
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            corr_bench_data = get_correlation_with_benchmark_data(
                positions, price_data, benchmark_returns
            )
            
            if corr_bench_data and corr_bench_data.get("tickers"):
                # Bar chart
                fig = plot_correlation_with_benchmark(corr_bench_data)
                st.plotly_chart(fig, use_container_width=True, key="corr_with_benchmark_chart")
                
                # Table
                corr_table_data = []
                for i, ticker in enumerate(corr_bench_data["tickers"]):
                    corr_table_data.append({
                        "Ticker": ticker,
                        "Correlation (SPY)": f"{corr_bench_data['correlations'][i]:.2f}",
                        "Beta": f"{corr_bench_data['betas'][i]:.2f}",
                    })
                
                corr_df = pd.DataFrame(corr_table_data)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
                
                # Automatic interpretation
                interpretation = _interpret_correlation_with_benchmark(corr_bench_data)
                if interpretation:
                    st.info(interpretation)
            else:
                st.info("Insufficient data for correlation with benchmark analysis")
        else:
            st.info("Benchmark data not available")
        
        # === Section 4.2.4: Average Correlation to Portfolio ===
        st.markdown("---")
        st.subheader("Average Correlation to Portfolio")
        
        if corr_matrix_data and price_data is not None:
            from core.analytics_engine.chart_data import (
                get_average_correlation_to_portfolio_data,
            )
            
            avg_corr_data = get_average_correlation_to_portfolio_data(
                positions, price_data
            )
            
            if avg_corr_data:
                # Bar chart
                fig_avg = go.Figure()
                fig_avg.add_trace(
                    go.Bar(
                        x=avg_corr_data["tickers"],
                        y=[c * 100 for c in avg_corr_data["avg_correlations"]],
                        marker=dict(
                            color=[
                                COLORS["success"] if c < 0.3
                                else COLORS["warning"] if c < 0.5
                                else COLORS["danger"]
                                for c in avg_corr_data["avg_correlations"]
                            ]
                        ),
                        text=[
                            f"{c*100:.1f}%"
                            for c in avg_corr_data["avg_correlations"]
                        ],
                        textposition="outside",
                    )
                )
                
                fig_avg.update_layout(
                    title="Average Correlation to Other Assets",
                    xaxis_title="Asset",
                    yaxis_title="Average Correlation (%)",
                    height=400,
                    template="plotly_dark",
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
                
                # Table with diversification scores
                avg_corr_df = pd.DataFrame({
                    "Asset": avg_corr_data["tickers"],
                    "Avg Correlation": [
                        f"{c:.3f}"
                        for c in avg_corr_data["avg_correlations"]
                    ],
                    "Diversification Score": [
                        f"{d:.3f}"
                        for d in avg_corr_data["diversification_scores"]
                    ],
                })
                st.dataframe(avg_corr_df, use_container_width=True)
                
                st.caption(
                    "**Diversification Score = 1 - Avg Correlation.** "
                    "Higher score = better diversification. "
                    "Target: < 0.3 correlation for good diversification."
                )
                
                # Automatic interpretation
                interpretation = _interpret_average_correlation_to_portfolio(avg_corr_data)
                if interpretation:
                    st.info(interpretation)
        
        # === Section 4.2.5: Rolling Correlations ===
        st.markdown("---")
        st.subheader("Rolling Correlations Between Assets")
        
        if price_data is not None:
            from core.analytics_engine.chart_data import (
                get_rolling_correlations_data,
            )
            
            # Window selector
            rolling_window_corr = st.slider(
                "Rolling Window (days)",
                min_value=30,
                max_value=252,
                value=60,
                step=30,
                key="rolling_corr_window_assets",
                help="Window size for rolling correlation calculation",
            )
            
            # Get top pairs to display (highest and lowest correlation)
            if corr_matrix_data and corr_matrix_data.get("correlation_matrix") is not None:
                corr_matrix = corr_matrix_data["correlation_matrix"]
                
                # Get all pairs with their correlations
                pairs_data = []
                tickers_list = list(corr_matrix.columns)
                for i, ticker1 in enumerate(tickers_list):
                    for ticker2 in tickers_list[i+1:]:
                        corr_val = corr_matrix.loc[ticker1, ticker2]
                        if not np.isnan(corr_val):
                            pairs_data.append({
                                "pair": (ticker1, ticker2),
                                "correlation": corr_val,
                            })
                
                # Sort by absolute correlation
                pairs_data.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                # Select top 5 pairs to display
                top_pairs = [p["pair"] for p in pairs_data[:5]]
                
                rolling_corr_data = get_rolling_correlations_data(
                    positions,
                    price_data,
                    window=rolling_window_corr,
                    selected_pairs=top_pairs,
                )
                
                if rolling_corr_data:
                    fig_rolling = go.Figure()
                    
                    for pair_name, corr_series in rolling_corr_data["rolling_correlations"].items():
                        if not corr_series.empty:
                            fig_rolling.add_trace(
                                go.Scatter(
                                    x=corr_series.index,
                                    y=corr_series.values * 100,
                                    mode="lines",
                                    name=pair_name,
                                    line=dict(width=2),
                                )
                            )
                    
                    fig_rolling.update_layout(
                        title=(
                            f"Rolling Correlations "
                            f"({rolling_window_corr} days)"
                        ),
                        xaxis_title="Date",
                        yaxis_title="Correlation (%)",
                        height=500,
                        template="plotly_dark",
                        hovermode="x unified",
                    )
                    
                    # Add horizontal lines for reference
                    fig_rolling.add_hline(
                        y=30,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Good diversification (< 0.3)",
                    )
                    fig_rolling.add_hline(
                        y=70,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="High correlation (> 0.7)",
                    )
                    
                    st.plotly_chart(fig_rolling, use_container_width=True)
                    
                    st.caption(
                        "**Interpretation:** Shows how correlations between "
                        "asset pairs change over time. Spikes in correlations "
                        "often occur during market crises."
                    )
                    
                    # Automatic interpretation
                    interpretation = _interpret_rolling_correlations(rolling_corr_data, rolling_window_corr)
                    if interpretation:
                        st.info(interpretation)
        
        # === Section 4.2.6: Cluster Analysis ===
        st.markdown("---")
        st.subheader("Cluster Analysis of Correlations")
        
        if corr_matrix_data and corr_matrix_data.get("correlation_matrix") is not None:
            # First, get cluster data to determine default number of clusters
            cluster_data = get_cluster_analysis_data(
                corr_matrix_data["correlation_matrix"]
            )
            
            if cluster_data:
                # Allow user to adjust number of clusters
                num_assets = len(corr_matrix_data["tickers"])
                max_clusters = min(num_assets, 6)  # Limit to 6 for readability
                
                col_cluster1, col_cluster2 = st.columns([2, 1])
                with col_cluster1:
                    st.caption(
                        "Adjust number of clusters to see different groupings. "
                        "More clusters = finer granularity, fewer clusters = broader groups."
                    )
                with col_cluster2:
                    n_clusters_user = st.slider(
                        "Number of Clusters",
                        min_value=2,
                        max_value=max_clusters,
                        value=cluster_data.get("n_clusters", 3),
                        key="cluster_count_slider"
                    )
                
                # Override n_clusters with user selection
                cluster_data["n_clusters"] = n_clusters_user
                # Clustered correlation matrix
                st.markdown("**Clustered Correlation Matrix Heatmap**")
                st.caption(
                    "Same as correlation matrix but reordered by clusters. "
                    "Assets grouped by similar behavior."
                )
                fig = plot_clustered_correlation_matrix(
                    cluster_data["clustered_matrix"]
                )
                st.plotly_chart(fig, use_container_width=True, key="clustered_matrix")
                
                # Dendrogram
                st.markdown("**Asset Clustering Dendrogram**")
                st.caption(
                    "Dendrogram showing hierarchical clustering of assets based on correlation. "
                    "Yellow dashed line indicates optimal cut point for cluster formation."
                )
                fig = plot_dendrogram(
                    cluster_data["linkage_matrix"],
                    cluster_data["reordered_tickers"],
                    cluster_data["n_clusters"]
                )
                st.plotly_chart(fig, use_container_width=True, key="dendrogram")
                
                # Show cluster assignments
                clusters = {}
                try:
                    from scipy.cluster.hierarchy import fcluster
                    cluster_assignments = fcluster(
                        cluster_data["linkage_matrix"],
                        cluster_data["n_clusters"],
                        criterion="maxclust"
                    )
                    
                    # Group tickers by cluster
                    for i, ticker in enumerate(cluster_data["reordered_tickers"]):
                        cluster_id = int(cluster_assignments[i])
                        if cluster_id not in clusters:
                            clusters[cluster_id] = []
                        clusters[cluster_id].append(ticker)
                    
                except Exception as e:
                    logger.warning(f"Error calculating cluster assignments: {e}")
                
                # Display clusters (always show, even if empty)
                if clusters:
                    st.markdown("**Cluster Assignments:**")
                    cluster_text = ""
                    for cluster_id in sorted(clusters.keys()):
                        tickers_str = ", ".join(clusters[cluster_id])
                        cluster_text += f"**Cluster {cluster_id}:** {tickers_str}  \n"
                    st.markdown(cluster_text)
                    
                    # Show cluster quality info
                    if len(clusters) > 1:
                        cluster_sizes = [len(clusters[cid]) for cid in clusters.keys()]
                        balanced = max(cluster_sizes) / min(cluster_sizes) < 2.5
                        
                        if balanced:
                            st.success(
                                f"✓ Balanced clustering: {len(clusters)} clusters with "
                                f"relatively even distribution"
                            )
                        else:
                            st.info(
                                f"ℹ Uneven clustering: {len(clusters)} clusters detected. "
                                f"Consider adjusting number of clusters for better balance."
                            )
                else:
                    st.info("Unable to calculate cluster assignments")
                
                # Interpretation guide (always show)
                st.markdown("---")
                with st.expander("📖 How to Interpret Cluster Analysis", expanded=False):
                    st.markdown(f"""
                    **Understanding the Dendrogram:**
                    
                    1. **Distance Axis (X-axis):** Shows how dissimilar assets are. 
                       - Lower distance = more similar (higher correlation)
                       - Higher distance = more different (lower correlation)
                    
                    2. **Tree Structure:** 
                       - Assets on the left are individual assets
                       - Branches connect similar assets/groups (growing right)
                       - Height of branch = distance at which groups merge
                       - **Shorter branches = more similar assets**
                    
                    3. **Cut Line (Yellow Dashed):** 
                       - Shows current number of clusters ({cluster_data["n_clusters"]})
                       - Draw a vertical line at this distance
                       - All branches crossing the line form separate clusters
                       - **Adjust slider above to see different cluster configurations**
                    
                    4. **Cluster Interpretation:**
                       - **Assets in same cluster** = move together (high correlation)
                       - **Assets in different clusters** = move independently (low correlation)
                       - Use this to identify diversification opportunities
                    
                    **Practical Use:**
                    - **All assets in one cluster** → portfolio is not diversified
                    - **Multiple balanced clusters** → good diversification
                    - **Consider reducing exposure** to assets in the same cluster
                    - **Adjust number of clusters** to find optimal grouping for your analysis
                    
                    **Note:** The number of clusters ({cluster_data["n_clusters"]}) is a starting point. 
                    Use the slider above to explore different groupings and find the most 
                    meaningful clusters for your portfolio.
                    """)
            else:
                st.info("Unable to perform cluster analysis")
        else:
            st.info("Correlation matrix not available for cluster analysis")
    
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}", exc_info=True)
        st.error(f"Error calculating correlations: {str(e)}")


def _render_asset_details(positions, portfolio_returns, benchmark_returns, portfolio_id, portfolio_service):
    """Sub-tab 4.3: Asset Details & Dynamics."""
    st.subheader("Asset Price Dynamics")
    
    if not positions:
        st.info("No positions available")
        return
    
    # Fetch price data
    try:
        from services.data_service import DataService
        from streamlit_app.utils.formatters import format_percentage
        
        analytics = st.session_state.get("portfolio_analytics", {})
        portfolio_returns = analytics.get("portfolio_returns")
        
        if portfolio_returns is None or portfolio_returns.empty:
            st.info("Please calculate analytics first to see asset details")
            return
        
        # Get date range
        start_date = portfolio_returns.index.min()
        end_date = portfolio_returns.index.max()
        
        data_service = DataService()
        ticker_list = [pos.ticker for pos in positions]
        
        # Fetch price data
        all_prices = []
        for ticker in ticker_list:
            try:
                if ticker == "CASH":
                    dr = pd.bdate_range(start=start_date, end=end_date)
                    prices = pd.DataFrame({
                        "Date": dr,
                        "Adjusted_Close": 1.0,
                        "Ticker": "CASH",
                    })
                else:
                    prices = data_service.fetch_historical_prices(
                        ticker, start_date, end_date,
                        use_cache=True, save_to_db=False
                    )
                    prices["Ticker"] = ticker
                
                if not prices.empty:
                    all_prices.append(prices)
            except Exception as e:
                logger.warning(f"Failed to fetch prices for {ticker}: {e}")
        
        if not all_prices:
            st.info("Unable to fetch price data")
            return
        
        # Combine and pivot
        combined = pd.concat(all_prices, ignore_index=True)
        price_data = combined.pivot_table(
            index="Date",
            columns="Ticker",
            values="Adjusted_Close",
            aggfunc="last",
        )
        
        # === Section 4.3.1: Asset Multi-Select ===
        # Initialize session state if not exists
        if "asset_selector" not in st.session_state:
            st.session_state["asset_selector"] = (
                ticker_list[:5] if len(ticker_list) > 5 else ticker_list
            )
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            select_all_clicked = st.button("Select All", key="select_all_assets")
            deselect_all_clicked = st.button("Deselect All", key="deselect_all_assets")
        
        # Handle button clicks BEFORE creating widget
        if select_all_clicked:
            st.session_state["asset_selector"] = ticker_list
            st.rerun()
        
        if deselect_all_clicked:
            st.session_state["asset_selector"] = []
            st.rerun()
        
        with col1:
            selected_tickers = st.multiselect(
                "Select Assets for Comparison",
                options=ticker_list,
                default=st.session_state["asset_selector"],
                key="asset_selector"
            )
        
        if not selected_tickers:
            st.info("Please select at least one asset for comparison")
            return
        
        # === Section 4.3.2: Asset Price Dynamics ===
        st.markdown("---")
        st.subheader("Asset Price Change (% from Start Date)")
        
        price_dynamics_data = get_asset_price_dynamics_data(
            positions, price_data, benchmark_returns, selected_tickers
        )
        
        if price_dynamics_data and price_dynamics_data.get("price_series"):
            fig = plot_asset_price_dynamics(price_dynamics_data)
            st.plotly_chart(fig, use_container_width=True, key="price_dynamics")
            
            # Show final returns in legend-style format
            price_series = price_dynamics_data["price_series"]
            final_returns_text = "**Final Returns:**\n"
            for ticker, series in price_series.items():
                if not series.empty:
                    final_return = series.iloc[-1]
                    final_returns_text += f"{ticker}: {final_return:+.2f}%  "
            
            st.markdown(final_returns_text)
            
            # Automatic interpretation
            interpretation = _interpret_asset_price_dynamics(price_dynamics_data)
            if interpretation:
                st.info(interpretation)
        
        # === Section 4.3.3: Rolling Correlation with Benchmark ===
        st.markdown("---")
        st.subheader("Rolling Correlation with SPY")
        
        if benchmark_returns is not None and not benchmark_returns.empty:
            window = st.slider(
                "Window size (days)",
                min_value=30,
                max_value=120,
                value=60,
                step=10,
                key="rolling_corr_window"
            )
            
            rolling_corr_data = get_rolling_correlation_with_benchmark_data(
                positions, price_data, benchmark_returns,
                portfolio_returns, window, selected_tickers
            )
            
            if rolling_corr_data and rolling_corr_data.get("rolling_correlations"):
                fig = plot_rolling_correlation_with_benchmark(rolling_corr_data)
                st.plotly_chart(fig, use_container_width=True, key="rolling_corr")
                
                # Automatic interpretation
                interpretation = _interpret_rolling_correlation_with_benchmark(rolling_corr_data, window)
                if interpretation:
                    st.info(interpretation)
            else:
                st.info("Insufficient data for rolling correlation analysis")
        else:
            st.info("Benchmark data not available")
        
        # === Section 4.3.4: Detailed Analysis of Single Asset ===
        st.markdown("---")
        st.subheader("Detailed Asset Analysis")
        
        selected_ticker = st.selectbox(
            "Select asset for detailed analysis",
            options=ticker_list,
            key="detailed_asset_selector"
        )
        
        if selected_ticker:
            detailed_data = get_detailed_asset_analysis_data(
                selected_ticker, positions, price_data,
                portfolio_returns, benchmark_returns
            )
            
            if detailed_data:
                # Calculate benchmark metrics for comparison
                # Use same alignment as overview (by portfolio returns) to ensure consistency
                benchmark_metrics = {}
                if benchmark_returns is not None and not benchmark_returns.empty:
                    try:
                        # Align benchmark returns with portfolio returns (same as overview)
                        # This ensures benchmark metrics match overview tab
                        common_idx = portfolio_returns.index.intersection(
                            benchmark_returns.index
                        )
                        aligned_bench = benchmark_returns.loc[common_idx]
                        
                        if len(aligned_bench) >= 2:
                            from core.analytics_engine.performance import (
                                calculate_annualized_return,
                            )
                            from core.analytics_engine.risk_metrics import (
                                calculate_volatility,
                            )
                            from core.analytics_engine.ratios import (
                                calculate_sharpe_ratio,
                            )
                            
                            # Calculate benchmark metrics (same method as overview)
                            bench_total_return = (1 + aligned_bench).prod() - 1
                            bench_annual_return = calculate_annualized_return(
                                aligned_bench
                            )
                            bench_volatility = calculate_volatility(
                                aligned_bench
                            ).get("annual", 0.0)
                            bench_sharpe = (
                                calculate_sharpe_ratio(
                                    aligned_bench, risk_free_rate=0.0435
                                ) or 0.0
                            )
                            
                            benchmark_metrics = {
                                "total_return": float(bench_total_return),
                                "annual_return": float(bench_annual_return),
                                "volatility": float(bench_volatility),
                                "sharpe_ratio": float(bench_sharpe),
                            }
                    except Exception as e:
                        logger.warning(f"Error calculating benchmark metrics: {e}")
                
                # Metrics cards with benchmark comparison (like in overview)
                metrics = detailed_data["metrics"]
                portfolio_metrics = detailed_data["portfolio_metrics"]
                
                metrics_data = [
                    {
                        "label": "Total Return",
                        "portfolio_value": metrics.get("total_return", 0),
                        "benchmark_value": benchmark_metrics.get("total_return"),
                        "format": "percent",
                        "higher_is_better": True,
                    },
                    {
                        "label": "Annual Return",
                        "portfolio_value": metrics.get("annual_return", 0),
                        "benchmark_value": benchmark_metrics.get("annual_return"),
                        "format": "percent",
                        "higher_is_better": True,
                    },
                    {
                        "label": "Volatility",
                        "portfolio_value": metrics.get("volatility", 0),
                        "benchmark_value": benchmark_metrics.get("volatility"),
                        "format": "percent",
                        "higher_is_better": False,  # Lower is better
                    },
                    {
                        "label": "Sharpe Ratio",
                        "portfolio_value": metrics.get("sharpe_ratio", 0),
                        "benchmark_value": benchmark_metrics.get("sharpe_ratio"),
                        "format": "ratio",
                        "higher_is_better": True,
                    },
                ]
                
                render_metric_cards_row(metrics_data, columns_per_row=4)
                
                # Price and Volume Chart
                st.markdown("---")
                st.subheader(f"{selected_ticker} - Price and Volume Chart")
                
                fig = plot_detailed_asset_price_volume(
                    detailed_data["prices"],
                    detailed_data["returns"],
                    detailed_data.get("ma50"),
                    detailed_data.get("ma200"),
                    selected_ticker
                )
                st.plotly_chart(fig, use_container_width=True, key="asset_price_volume")
                
                # Automatic interpretation
                interpretation = _interpret_price_volume_chart(detailed_data, selected_ticker)
                if interpretation:
                    st.info(interpretation)
                
                # Comparison of Return
                st.markdown("---")
                st.subheader(f"Comparison of Return - {selected_ticker} vs Portfolio vs SPY")
                
                cum_returns = detailed_data["cumulative_returns"]
                
                # Align all series to common dates
                asset_cum = cum_returns["asset"]
                portfolio_cum = cum_returns["portfolio"]
                benchmark_cum = cum_returns.get("benchmark")
                
                # Find common dates
                common_dates = asset_cum.index.intersection(portfolio_cum.index)
                if benchmark_cum is not None and not benchmark_cum.empty:
                    common_dates = common_dates.intersection(benchmark_cum.index)
                
                if len(common_dates) >= 2:
                    # Align all series to common dates
                    asset_aligned = asset_cum.loc[common_dates] * 100
                    portfolio_aligned = portfolio_cum.loc[common_dates] * 100
                    
                    # Create data dictionary for plot_cumulative_returns
                    # Note: plot_cumulative_returns expects 'portfolio' and 'benchmark' keys
                    # But we want to show asset, portfolio, and benchmark
                    # So we'll create a custom plot
                    fig = go.Figure()
                    
                    # Asset line (green)
                    fig.add_trace(
                        go.Scatter(
                            x=asset_aligned.index,
                            y=asset_aligned.values,
                            mode="lines",
                            name=selected_ticker,
                            line=dict(color=COLORS["success"], width=2),  # Green
                        )
                    )
                    
                    # Portfolio line (purple)
                    fig.add_trace(
                        go.Scatter(
                            x=portfolio_aligned.index,
                            y=portfolio_aligned.values,
                            mode="lines",
                            name="Portfolio",
                            line=dict(color=COLORS["primary"], width=2),  # Purple - portfolio
                        )
                    )
                    
                    # Benchmark line (solid blue)
                    if benchmark_cum is not None and not benchmark_cum.empty:
                        benchmark_aligned = benchmark_cum.loc[common_dates] * 100
                        fig.add_trace(
                            go.Scatter(
                                x=benchmark_aligned.index,
                                y=benchmark_aligned.values,
                                mode="lines",
                                name="SPY (Benchmark)",
                                line=dict(color=COLORS["secondary"], width=2),  # Solid blue
                            )
                        )
                    
                    layout = get_chart_layout(
                        title="Comparison of Return",
                        yaxis=dict(
                            title="Cumulative Return (%)",
                            tickformat=",.1f",
                        ),
                        xaxis=dict(title="Date"),
                        hovermode="x unified",
                    )
                    
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True, key="asset_cumulative")
                    
                    # Automatic interpretation
                    interpretation = _interpret_comparison_of_return(detailed_data, selected_ticker)
                    if interpretation:
                        st.info(interpretation)
                else:
                    st.info("Insufficient overlapping dates for comparison chart")
                
                # Correlations with Other Assets
                st.markdown("---")
                st.subheader(f"Correlations with Other Assets - {selected_ticker}")
                
                other_corrs = detailed_data.get("other_correlations", {})
                if other_corrs:
                    fig = plot_asset_correlation_bar(other_corrs, selected_ticker)
                    st.plotly_chart(fig, use_container_width=True, key="asset_correlations")
                    
                    # Automatic interpretation
                    interpretation = _interpret_asset_correlations(other_corrs, selected_ticker)
                    if interpretation:
                        st.info(interpretation)
                else:
                    st.info("No correlation data available with other assets")
            else:
                st.info(f"Unable to calculate detailed analysis for {selected_ticker}")
    
    except Exception as e:
        logger.error(f"Error in asset details: {e}", exc_info=True)
        st.error(f"Error calculating asset details: {str(e)}")


def _render_export_tab(
    portfolio_name: str,
    perf: dict,
    risk: dict,
    ratios: dict,
    market: dict,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    portfolio_values: pd.Series | None,
    positions: list,
    start_date: date,
    end_date: date,
    risk_free_rate: float,
) -> None:
    """Render Export & Reports tab with PDF generation."""
    import tempfile
    import os
    
    st.header("📄 Export & Reports")
    st.markdown("Generate comprehensive PDF reports with full page screenshots.")
    
    report_service = ReportService()
    
    # PDF Generation Section
    st.subheader("PDF Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Report Sections**")
        tabs_config = {
            "Overview": st.checkbox("Overview", value=True, key="pdf_overview"),
            "Performance": st.checkbox("Performance", value=True, key="pdf_performance"),
            "Risk": st.checkbox("Risk", value=True, key="pdf_risk"),
            "Assets & Correlations": st.checkbox("Assets & Correlations", value=False, key="pdf_assets"),
        }
    
    with col2:
        st.markdown("**Settings**")
        streamlit_url = st.text_input(
            "Streamlit URL",
            value="http://localhost:8501",
            key="pdf_streamlit_url",
            help="Base URL of your Streamlit app (default: http://localhost:8501)"
        )
    
    st.markdown("---")
    
    # Viewport settings
    st.markdown("**Page Settings**")
    col1, col2 = st.columns(2)
    with col1:
        viewport_width = st.number_input(
            "Viewport Width (px)",
            min_value=800,
            max_value=3840,
            value=1920,
            step=160,
            key="pdf_viewport_width",
            help="Width of the page for screenshot (default: 1920px)"
        )
    with col2:
        viewport_height = st.number_input(
            "Initial Viewport Height (px)",
            min_value=600,
            max_value=2160,
            value=1080,
            step=120,
            key="pdf_viewport_height",
            help="Initial height (will expand automatically)"
        )
    
    st.markdown("---")
    
    # Generate PDF button
    if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
        if not portfolio_returns.empty:
            # Count how many pages will be generated
            total_pages = 0
            for tab_name, enabled in tabs_config.items():
                if not enabled:
                    continue
                if tab_name == "Overview":
                    total_pages += 1
                elif tab_name == "Performance":
                    total_pages += 3  # 3 sub-tabs
                elif tab_name == "Risk":
                    total_pages += 4  # 4 sub-tabs
                elif tab_name == "Assets & Correlations":
                    total_pages += 3  # 3 sub-tabs
            
            if total_pages == 0:
                st.warning("⚠️ Please select at least one section to generate PDF.")
            else:
                with st.spinner(f"Generating PDF report from {total_pages} tab(s)... This may take a minute."):
                    try:
                        # Create temporary file for PDF
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                            pdf_path = tmp.name
                        
                        # Generate PDF from Streamlit tabs
                        success = report_service.generate_pdf_from_streamlit_tabs(
                            streamlit_url=streamlit_url,
                            output_path=pdf_path,
                            tabs_config=tabs_config,
                            viewport_width=int(viewport_width),
                            viewport_height=int(viewport_height),
                            wait_timeout=5000,
                        )
                        
                        if success and os.path.exists(pdf_path):
                            # Read PDF file
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            # Get file size
                            file_size = len(pdf_bytes) / (1024 * 1024)  # MB
                            
                            # Download button
                            st.success(f"✅ PDF generated successfully! ({file_size:.2f} MB, {total_pages} pages)")
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"{portfolio_name.replace(' ', '_')}_report_{start_date}_{end_date}.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True,
                            )
                            
                            # Clean up
                            try:
                                os.unlink(pdf_path)
                            except Exception:
                                pass
                        else:
                            st.error("❌ Failed to generate PDF. Please check the logs for details.")
                            
                    except Exception as e:
                        logger.error(f"Error generating PDF: {e}", exc_info=True)
                        st.error(f"Error generating PDF: {str(e)}")
        else:
            st.warning("⚠️ No portfolio returns data available. Please calculate metrics first.")
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About PDF Reports"):
        st.markdown("""
        **Streamlit Tab Screenshot PDF Generation**
        
        This feature generates PDF reports by:
        1. Opening your Streamlit app in a headless browser
        2. Taking full page screenshots of each selected tab
        3. Combining all screenshots into a single PDF document
        
        **How it works:**
        - Each selected main tab = one or more PDF pages
        - Overview tab = 1 page
        - Performance tab = 3 pages (Returns Analysis, Periodic Analysis, Distribution)
        - Risk tab = 4 pages (Key Metrics, Drawdown Analysis, VaR & CVaR, Rolling Risk Metrics)
        - Assets & Correlations tab = 3 pages
        
        **Features:**
        - ✅ Real Streamlit page screenshots (exact visual representation)
        - ✅ Full page capture (includes all scrollable content)
        - ✅ Dark theme preservation
        - ✅ High-quality charts and visualizations
        - ✅ Automatic tab switching and screenshot capture
        
        **Requirements:**
        - Streamlit app must be running and accessible at the specified URL
        - Default URL: http://localhost:8501
        - Make sure you're on the Portfolio Analysis page when generating
        
        **Note:** The first generation may take longer as the browser needs to load and navigate through tabs.
        """)

