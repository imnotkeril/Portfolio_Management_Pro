"""Comparison table component for Portfolio vs Benchmark."""

import streamlit as st
import pandas as pd
from typing import Dict, Optional


def render_comparison_table(
    portfolio_metrics: Dict[str, float],
    benchmark_metrics: Optional[Dict[str, float]] = None,
    title: str = "Portfolio vs Benchmark",
    categories: Optional[Dict[str, list]] = None,
    height: Optional[int] = None,
) -> None:
    """
    Render comparison table between portfolio and benchmark.

    Args:
        portfolio_metrics: Dictionary of portfolio metrics
        benchmark_metrics: Optional dictionary of benchmark metrics
        title: Table title
        categories: Optional dict of metric categories for grouping
    """
    st.subheader(title)

    # Default metric order (flat, no category header rows)
    metric_order = [
        "total_return",
        "cagr",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "beta",
        "alpha",
        "up_capture",
        "down_capture",
    ]

    # Build comparison DataFrame
    comparison_data = []

    for metric in metric_order:
        if metric in portfolio_metrics:
            portfolio_value = portfolio_metrics[metric]
            portfolio_formatted = _format_metric(metric, portfolio_value)

            benchmark_formatted = ""
            diff_formatted = ""
            better_formatted = ""
            if benchmark_metrics and metric in benchmark_metrics:
                benchmark_value = benchmark_metrics[metric]
                benchmark_formatted = _format_metric(metric, benchmark_value)
                if isinstance(portfolio_value, (int, float)) and isinstance(benchmark_value, (int, float)):
                    diff = portfolio_value - benchmark_value
                    diff_formatted = _format_difference(metric, diff)
                    better = _is_better(metric, portfolio_value, benchmark_value)
                    if better is True:
                        better_formatted = "✓"
                    elif better is False:
                        better_formatted = "✗"
                    else:
                        better_formatted = "~"

            comparison_data.append({
                "Metric": _format_metric_name(metric),
                "Portfolio": portfolio_formatted,
                "Benchmark": benchmark_formatted,
                "Difference": diff_formatted,
                "Better": better_formatted,
            })

    df = pd.DataFrame(comparison_data)
    
    # Display table with custom styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height or min(600, 40 * (len(df) + 1)),
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Portfolio": st.column_config.TextColumn("Portfolio", width="small"),
            "Benchmark": st.column_config.TextColumn("Benchmark", width="small"),
            "Difference": st.column_config.TextColumn("Δ", width="small"),
            "Better": st.column_config.TextColumn("Better", width="small"),
        },
    )


def _format_metric_name(metric: str) -> str:
    """Format metric name for display."""
    name_mapping = {
        "total_return": "Total Return",
        "annualized_return": "Annualized Return",
        "cagr": "CAGR",
        "best_day": "Best Day",
        "worst_day": "Worst Day",
        "best_month": "Best Month",
        "worst_month": "Worst Month",
        "volatility": "Volatility",
        "downside_deviation": "Downside Deviation",
        "max_drawdown": "Max Drawdown",
        "max_drawdown_duration": "Max DD Duration (days)",
        "ulcer_index": "Ulcer Index",
        "var_95": "VaR (95%)",
        "cvar_95": "CVaR (95%)",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "information_ratio": "Information Ratio",
        "treynor_ratio": "Treynor Ratio",
        "beta": "Beta",
        "alpha": "Alpha",
        "correlation": "Correlation",
        "up_capture": "Up Capture",
        "down_capture": "Down Capture",
    }
    return name_mapping.get(metric, metric.replace("_", " ").title())


def _format_metric(metric: str, value) -> str:
    """Format metric value for display."""
    if value is None or value == "":
        return "—"

    # Percentage metrics
    percent_metrics = [
        "total_return", "annualized_return", "cagr",
        "best_day", "worst_day", "best_month", "worst_month",
        "volatility", "downside_deviation", "max_drawdown",
        "var_95", "cvar_95", "up_capture", "down_capture",
        "alpha",
    ]

    # Ratio metrics
    ratio_metrics = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "information_ratio", "treynor_ratio", "ulcer_index",
    ]

    # Correlation metrics
    corr_metrics = ["correlation", "beta"]

    # Duration metrics (integer days)
    duration_metrics = ["max_drawdown_duration"]

    if metric in percent_metrics:
        # Values are decimals (e.g., 0.1234 = 12.34%) → convert to percent
        try:
            return f"{value * 100:.2f}%"
        except Exception:
            return f"{value:.2f}%"
    elif metric in ratio_metrics:
        return f"{value:.3f}"
    elif metric in corr_metrics:
        return f"{value:.3f}"
    elif metric in duration_metrics:
        return f"{int(value)}"
    else:
        # Default formatting
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)


def _format_difference(metric: str, diff: float) -> str:
    """Format difference with color indicator."""
    if abs(diff) < 0.001:
        return "—"

    # Metrics where higher is better
    higher_is_better = [
        "total_return", "annualized_return", "cagr",
        "best_day", "best_month",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "information_ratio", "treynor_ratio",
        "up_capture",
    ]

    # Metrics where lower is better
    lower_is_better = [
        "volatility", "downside_deviation", "max_drawdown",
        "var_95", "cvar_95", "ulcer_index",
        "worst_day", "worst_month", "down_capture",
        "max_drawdown_duration",
    ]

    # Determine color
    is_good = False
    if metric in higher_is_better:
        is_good = diff > 0
    elif metric in lower_is_better:
        is_good = diff < 0

    # Format with sign
    sign = "+" if diff > 0 else ""
    
    # Percentage metrics
    if metric in ["total_return", "annualized_return", "cagr",
                  "best_day", "worst_day", "best_month", "worst_month",
                  "volatility", "downside_deviation", "max_drawdown",
                  "var_95", "cvar_95", "up_capture", "down_capture", "alpha"]:
        formatted = f"{sign}{diff:.2f}%"
    elif metric in ["max_drawdown_duration"]:
        formatted = f"{sign}{int(diff)}"
    else:
        formatted = f"{sign}{diff:.3f}"

    # Color-code
    if is_good:
        return f"🟢 {formatted}"
    else:
        return f"🔴 {formatted}"


def _is_better(metric: str, portfolio_value: float, benchmark_value: float):
    """Return True if portfolio better than benchmark, False if worse, None if neutral."""
    try:
        # Metrics where higher is better
        higher_is_better = {
            "total_return", "annualized_return", "cagr",
            "best_day", "best_month",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "information_ratio", "treynor_ratio",
            "up_capture",
        }
        # Metrics where lower is better
        lower_is_better = {
            "volatility", "downside_deviation", "max_drawdown",
            "var_95", "cvar_95", "ulcer_index",
            "worst_day", "worst_month", "down_capture",
            "max_drawdown_duration",
        }

        if abs(portfolio_value - benchmark_value) < 1e-9:
            return None
        if metric in higher_is_better:
            return portfolio_value > benchmark_value
        if metric in lower_is_better:
            return portfolio_value < benchmark_value
        # Neutral metrics (beta ~ 1, alpha higher better)
        if metric == "beta":
            # closer to 1 is better → treat difference by absolute distance
            return abs(portfolio_value - 1.0) < abs(benchmark_value - 1.0)
        if metric == "alpha":
            return portfolio_value > benchmark_value
        return None
    except Exception:
        return None


def render_simple_metrics_table(
    metrics: Dict[str, float],
    title: str = "Metrics",
    format_type: str = "auto",
) -> None:
    """
    Render simple metrics table (single column).

    Args:
        metrics: Dictionary of metrics
        title: Table title
        format_type: 'auto', 'percent', 'ratio', or 'number'
    """
    st.subheader(title)

    data = []
    for key, value in metrics.items():
        metric_name = _format_metric_name(key)
        if format_type == "auto":
            formatted_value = _format_metric(key, value)
        elif format_type == "percent":
            formatted_value = f"{value:.2f}%" if value is not None else "—"
        elif format_type == "ratio":
            formatted_value = f"{value:.3f}" if value is not None else "—"
        else:
            formatted_value = f"{value:.2f}" if value is not None else "—"

        data.append({"Metric": metric_name, "Value": formatted_value})

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
