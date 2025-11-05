"""Metric card component with comparison to benchmark."""

import streamlit as st
from typing import Optional, Union


def render_metric_cards_row(
    metrics: list[dict],
    columns_per_row: int = 4,
) -> None:
    """
    Render a row of metric cards with comparison to benchmark.

    Args:
        metrics: List of metric dictionaries with keys:
                 - label: str
                 - portfolio_value: float
                 - benchmark_value: float (optional)
                 - format: 'percent', 'ratio', 'number', 'days'
                 - higher_is_better: bool (default True)
        columns_per_row: Number of cards per row
    """
    cols = st.columns(columns_per_row)

    for idx, metric_data in enumerate(metrics):
        col_idx = idx % columns_per_row
        with cols[col_idx]:
            render_metric_card(
                label=metric_data["label"],
                portfolio_value=metric_data["portfolio_value"],
                benchmark_value=metric_data.get("benchmark_value"),
                format_type=metric_data.get("format", "percent"),
                higher_is_better=metric_data.get("higher_is_better", True),
            )


def render_metric_card(
    label: str,
    portfolio_value: float,
    benchmark_value: Optional[float] = None,
    format_type: str = "percent",
    higher_is_better: Union[bool, None] = True,
) -> None:
    """
    Render single metric card with comparison.

    Args:
        label: Metric label
        portfolio_value: Portfolio metric value
        benchmark_value: Optional benchmark value for comparison
        format_type: 'percent', 'ratio', 'number', or 'days'
        higher_is_better: Whether higher value is better (True/False),
                         or None for special handling (e.g., Beta)
    """
    # Format portfolio value
    portfolio_formatted = _format_value(portfolio_value, format_type)

    # Custom card layout: label, portfolio value, benchmark comparison
    st.markdown(f"**{label}**")
    st.markdown(f"### {portfolio_formatted}")

    # Show benchmark value with colored indicator
    if benchmark_value is not None:
        # Convert to float to handle numpy types
        try:
            portfolio_value = float(portfolio_value)
            benchmark_value = float(benchmark_value)
        except (ValueError, TypeError):
            # If conversion fails, show white circle
            is_better = None
            benchmark_formatted = _format_value(benchmark_value, format_type)
            st.markdown(f"⚪ {benchmark_formatted}")
            return
        
        benchmark_formatted = _format_value(benchmark_value, format_type)

        # Determine if portfolio is better than benchmark
        # Special handling for Beta (closer to 1.0 is better)
        if label == "Beta":
            # Compare absolute distance from 1.0
            port_dist = abs(portfolio_value - 1.0)
            bench_dist = abs(benchmark_value - 1.0)
            delta = bench_dist - port_dist
            # Very strict epsilon for Beta
            if abs(delta) <= 0.0001:
                is_better = None  # White
            else:
                is_better = port_dist < bench_dist
        elif label == "Max Drawdown":
            # For Max Drawdown: less negative is better
            # Both are negative, so closer to 0 (higher value) is better
            # portfolio > benchmark (less negative) = better
            delta = portfolio_value - benchmark_value
            # Very strict epsilon for drawdown
            if abs(delta) <= 0.0001:
                is_better = None  # White
            else:
                is_better = portfolio_value > benchmark_value
        else:
            delta = portfolio_value - benchmark_value
            # More strict epsilon for white circle
            # White only when difference is truly negligible
            percent_eps = 0.0001  # 0.01% difference (stricter)
            ratio_eps = 0.001  # 0.001 difference (stricter)
            number_eps = 0.001  # 0.001 difference (stricter)
            if format_type == "percent":
                eps = percent_eps
            elif format_type == "ratio":
                eps = ratio_eps
            else:
                eps = number_eps

            if abs(delta) <= eps:
                # Difference is negligible - show white
                is_better = None
            else:
                # Significant difference - determine better/worse
                is_better = (
                    (higher_is_better and delta > 0) or
                    (not higher_is_better and delta < 0)
                )

        if is_better is True:
            color_indicator = "🟢"
        elif is_better is False:
            color_indicator = "🔴"
        else:
            color_indicator = "⚪"

        st.markdown(f"{color_indicator} {benchmark_formatted}")


def _format_value(
    value: float, format_type: str, include_sign: bool = False
) -> str:
    """Format value based on type."""
    if value is None:
        return "—"

    sign = ""
    if include_sign and value > 0:
        sign = "+"

    if format_type == "percent":
        # Convert decimal to percent
        pct = value * 100.0
        if abs(pct) < 0.005:  # <0.01%
            return f"{sign}<0.01%" if include_sign else "<0.01%"
        return f"{sign}{pct:.2f}%"
    elif format_type == "ratio":
        return f"{sign}{value:.3f}"
    elif format_type == "days":
        return f"{sign}{int(value)}"
    else:  # number
        return f"{sign}{value:.2f}"


def render_comparison_grid(
    title: str,
    metrics: dict,
) -> None:
    """
    Render grid of comparison metrics (2 rows x 4 columns).

    Args:
        title: Section title
        metrics: Dictionary with metric definitions
    """
    st.subheader(title)

    # Row 1
    row1 = metrics.get("row1", [])
    if row1:
        render_metric_cards_row(row1, columns_per_row=4)

    # Row 2
    row2 = metrics.get("row2", [])
    if row2:
        render_metric_cards_row(row2, columns_per_row=4)
