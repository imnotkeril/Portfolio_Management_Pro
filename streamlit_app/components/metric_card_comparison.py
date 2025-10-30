"""Metric card component with comparison to benchmark."""

import streamlit as st
from typing import Optional


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
    higher_is_better: bool = True,
) -> None:
    """
    Render single metric card with comparison.

    Args:
        label: Metric label
        portfolio_value: Portfolio metric value
        benchmark_value: Optional benchmark value for comparison
        format_type: 'percent', 'ratio', 'number', or 'days'
        higher_is_better: Whether higher value is better (for coloring)
    """
    # Format portfolio value
    portfolio_formatted = _format_value(portfolio_value, format_type)

    # Calculate delta if benchmark available
    delta_text = None
    delta_color = "off"

    if benchmark_value is not None:
        delta = portfolio_value - benchmark_value
        delta_formatted = _format_value(delta, format_type, include_sign=True)

        # Determine color
        # Thresholds: suppress near-zero noise per type
        percent_eps = 0.0001  # 0.01%
        ratio_eps = 0.001
        number_eps = 0.001
        eps = percent_eps if format_type == "percent" else (ratio_eps if format_type == "ratio" else number_eps)

        if abs(delta) <= eps:
            # Hide delta when effectively zero
            delta_text = None
            delta_color = "off"
        else:
            if (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
                delta_color = "normal"  # Green (good)
            else:
                delta_color = "inverse"  # Red (bad)
            delta_text = f"vs Benchmark: {delta_formatted}"

    # Render metric
    st.metric(
        label=label,
        value=portfolio_formatted,
        delta=delta_text,
        delta_color=delta_color,
    )


def _format_value(value: float, format_type: str, include_sign: bool = False) -> str:
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
